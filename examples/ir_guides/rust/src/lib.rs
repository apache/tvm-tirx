/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Functions that build IR, and that read, traverse, and rewrite `add_one`, from Rust.
//! See docs/ir_guides/working_with_ir_in_rust.rst. Each function is registered as the global
//! function `tirx_ir_guide_examples.rust.<name>`; working_with_ir_in_rust.py calls the object
//! system functions without arguments and the others on `add_one`.

use std::collections::HashMap;

use tvm_tirx::ir::prim::{ex, make_const, Add, AddObj, Mul};
use tvm_tirx::ir::{
    Expr, For, ForObj, IntImm, PrimType, TensorStore, TensorStoreObj, Var, VarObj, UNKNOWN_LOC,
};
use tvm_tirx::script::script;
use tvm_tirx::tirx::Function;
use tvm_tirx::tvm_ffi::{
    self, dispatch, match_any, Any, Array, DefRegionKind, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, VisitCallbacks, VisitContext, VisitInterrupt, WalkOrder, WalkResult,
};

/// A Python list of strings.
type Strings = Array<tvm_ffi::String>;

/// The lines of a printout, for Python.
fn lines(items: impl IntoIterator<Item = String>) -> Strings {
    items.into_iter().map(tvm_ffi::String::from).collect()
}

// [construct.begin]
fn x_times_2_plus_1() -> Result<(Var, Expr)> {
    let x = Var::new("x", PrimType::int(32), UNKNOWN_LOC);
    let expr = (ex(&x) * 2 + 1).expr()?;
    Ok((x, expr))
}

fn construct() -> Result<tvm_ffi::String> {
    let (_, expr) = x_times_2_plus_1()?;
    script(&expr)
}
// [construct.end]

// [objects.begin]
fn objects() -> Result<Strings> {
    let (_, expr) = x_times_2_plus_1()?;
    let add: Add = expr.try_cast()?;
    let alias = add.clone();
    Ok(lines([
        format!("a = {}", script(&add.a)?.as_str()),
        format!("b = {}", script(&add.b)?.as_str()),
        format!("same object: {}", alias.same_as(&add)),
    ]))
}
// [objects.end]

// [casts.begin]
fn casts() -> Result<Strings> {
    let (_, expr) = x_times_2_plus_1()?;
    let add: Add = expr.try_cast()?;
    let value: Expr = add.into();
    let again: Add = value.clone().try_cast()?;
    let mut out = vec![format!("back to Add: {}", script(&again)?.as_str())];
    if let Err(err) = value.try_cast::<Mul>() {
        out.push(format!("as Mul: {}", err.message()));
    }
    Ok(lines(out))
}
// [casts.end]

// [describe.begin]
fn describe(expr: &Expr) -> Result<String> {
    let text = match_any! {
        expr.clone() {
            Add(add) => format!("({} + {})", describe(&add.a)?, describe(&add.b)?),
            Mul(mul) => format!("({} * {})", describe(&mul.a)?, describe(&mul.b)?),
            Var(var) => var.name.as_str().to_owned(),
            IntImm(imm) => imm.value.to_string(),
            _ => "<other>".to_owned(),
        }
    };
    Ok(text)
}
// [describe.end]

// [matching.begin]
fn matching() -> Result<Strings> {
    let (x, expr) = x_times_2_plus_1()?;
    let difference = (ex(&x) - 1).expr()?;
    Ok(lines([describe(&expr)?, describe(&difference)?]))
}
// [matching.end]

// [walk.begin]
fn written_tensors(func: Function) -> Result<Strings> {
    let mut writes = Vec::new();
    let on_store = |store: TensorStore| -> Result<WalkResult> {
        let dest: Var = store.dest.clone().try_cast()?;
        writes.push(dest.name.clone());
        Ok(WalkResult::Advance)
    };
    tvm_ffi::structural_walk(&func, on_store, WalkOrder::PostOrder)?;
    Ok(writes.into_iter().collect())
}
// [walk.end]

// [stop.begin]
fn first_write(func: Function) -> Result<Option<TensorStore>> {
    let find_write = |store: TensorStore| WalkResult::interrupt_with(store);
    let found = tvm_ffi::structural_walk(&func, find_write, WalkOrder::PostOrder)?;
    match found {
        Some(interrupt) => Ok(Some(interrupt.value.try_into()?)),
        None => Ok(None),
    }
}
// [stop.end]

// [specialize.begin]
fn specialize(func: Function, n: Var, value: i64) -> Result<Function> {
    let specialize_var = |var: Var| -> Result<Expr> {
        if var.same_as(&n) {
            let ty: PrimType = var.ty.clone().try_cast()?;
            return make_const(ty, value);
        }
        Ok(var.into())
    };
    tvm_ffi::structural_map(func, specialize_var, WalkOrder::PostOrder)?.try_into()
}
// [specialize.end]

// [definitions.begin]
fn definitions(func: Function) -> Result<Strings> {
    let mut sites = Vec::new();
    let show_definition = |var: Var, kind: DefRegionKind| {
        if kind != DefRegionKind::None {
            sites.push(format!("{} {:?}", var.name.as_str(), kind));
        }
        WalkResult::Advance
    };
    tvm_ffi::structural_walk(&func, show_definition, WalkOrder::PreOrder)?;
    Ok(sites.iter().map(tvm_ffi::String::from).collect())
}
// [definitions.end]

// [equal.begin]
fn compare_specialized(func: Function, n: Var) -> Result<Array<bool>> {
    let first = specialize(func.clone(), n.clone(), 128)?;
    let second = specialize(func, n, 128)?;
    let (map_free_vars, skip_tensor_content) = (false, false);
    let equal: bool = tvm_ffi::Function::get_global("ffi.StructuralEqual")?
        .call_tuple((&first, &second, map_free_vars, skip_tensor_content))?
        .try_into()?;
    Ok([first.same_as(&second), equal].into_iter().collect())
}
// [equal.end]

// [counter.begin]
#[derive(Default)]
struct Counter {
    vars: usize,
    adds: usize,
}

#[dispatch(walk)]
impl Counter {
    fn walk_var(&mut self, _var: &VarObj) -> WalkResult {
        self.vars += 1;
        WalkResult::Advance
    }

    fn walk_add(&mut self, _add: &AddObj) -> WalkResult {
        self.adds += 1;
        WalkResult::Advance
    }
}
// [counter.end]

// [count.begin]
fn count() -> Result<tvm_ffi::String> {
    let (_, expr) = x_times_2_plus_1()?;
    let mut counter = Counter::default();
    tvm_ffi::structural_walk(&expr, &mut counter, WalkOrder::PreOrder)?;
    Ok(format!("variables: {}, additions: {}", counter.vars, counter.adds).into())
}
// [count.end]

// [visit.begin]
#[derive(Default)]
struct LoopNest {
    loops: Vec<String>,
    writes: Vec<String>,
}

fn visit_for(
    loop_: &ForObj,
    visitor: &mut VisitContext<'_, LoopNest>,
) -> Result<Option<VisitInterrupt>> {
    visitor.state_mut().loops.push(loop_.var.name.as_str().to_owned());
    let result = visitor.visit_children()?;
    visitor.state_mut().loops.pop();
    Ok(result)
}

fn visit_store(store: &TensorStoreObj, visitor: &mut VisitContext<'_, LoopNest>) -> Result<()> {
    let loops = visitor.state().loops.join(", ");
    let write = format!("{} inside [{}]", script(&store.dest)?.as_str(), loops);
    visitor.state_mut().writes.push(write);
    Ok(())
}

fn loops_around_writes(func: Function) -> Result<Strings> {
    let mut visitor = VisitCallbacks::new(LoopNest::default(), (visit_for, visit_store));
    tvm_ffi::structural_visit(&func, &mut visitor)?;
    let state = visitor.into_state();
    assert!(state.loops.is_empty());
    Ok(state.writes.iter().map(tvm_ffi::String::from).collect())
}
// [visit.end]

// [mutate.begin]
#[derive(Default)]
struct ReverseLoops {
    reversed_index: HashMap<ObjectIdentity, Expr>,
}

#[dispatch(mutate)]
impl ReverseLoops {
    fn mutate_for(&mut self, loop_: For, mutator: &mut Mutator) -> Result<Any> {
        let index = (ex(&loop_.end) - 1 - &loop_.var).expr()?;
        self.reversed_index.insert(ObjectIdentity::of(&loop_.var), index);
        mutator.default_mutate(self, &loop_)
    }

    fn mutate_var(&mut self, var: Var, mutator: &mut Mutator) -> Expr {
        let is_use = mutator.def_region_kind() == DefRegionKind::None;
        match self.reversed_index.get(&ObjectIdentity::of(&var)) {
            Some(index) if is_use => index.clone(),
            _ => var.into(),
        }
    }
}

fn reverse_loops(func: Function) -> Result<Function> {
    tvm_ffi::structural_mutate(func, ReverseLoops::default())?.try_into()
}
// [mutate.end]

tvm_ffi::static_init_block! {
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.construct",
        construct,
        "Build x * 2 + 1.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.objects",
        objects,
        "Read the fields of an addition.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.casts",
        casts,
        "Cast an expression up and down.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.matching",
        matching,
        "Describe expressions by matching their nodes.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.count",
        count,
        "Count variables and additions with a dispatch struct.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.written_tensors",
        written_tensors,
        "List the tensors a function writes.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.first_write",
        first_write,
        "Find the first write.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.specialize",
        specialize,
        "Replace a variable with a constant.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.definitions",
        definitions,
        "List the definition sites.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.compare_specialized",
        compare_specialized,
        "Compare two specializations.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.loops_around_writes",
        loops_around_writes,
        "List the loops around each write.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.reverse_loops",
        reverse_loops,
        "Reverse every loop.",
    )?;
}
