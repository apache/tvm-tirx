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

//! Functions that read, traverse, and rewrite `add_one` from Rust.
//! See docs/ir_guides/working_with_ir_in_rust.rst. Each function is registered as the global
//! function `tirx_ir_guide_examples.rust.<name>`; working_with_ir_in_rust.py calls each on `add_one`.

use std::collections::HashMap;

use tvm_tirx::ir::prim::{ex, make_const};
use tvm_tirx::ir::{
    Expr, For, ForObj, PrimType, Stmt, TensorStore, TensorStoreObj, Var, UNKNOWN_LOC,
};
use tvm_tirx::script::script;
use tvm_tirx::tirx::Function;
use tvm_tirx::tvm_ffi::{
    self, dispatch, match_any, Any, Array, DefRegionKind, Mutator, ObjectIdentity, ObjectRefCast,
    ObjectRefCore, Result, VisitCallbacks, VisitContext, VisitInterrupt, WalkOrder, WalkResult,
};

/// A Python list of strings.
type Strings = Array<tvm_ffi::String>;

// [inspect.begin]
fn inspect(func: Function) -> Result<tvm_ffi::String> {
    let mut lines = Vec::new();
    for param in func.params.iter() {
        lines.push(format!("param {}", param.name.as_str()));
    }
    let body = func.body.as_ref().expect("add_one has a body");
    for stmt in body.seq.iter() {
        describe(stmt, "", &mut lines)?;
    }
    Ok(lines.join("\n").into())
}

fn describe(stmt: Stmt, indent: &str, lines: &mut Vec<String>) -> Result<()> {
    match_any! {
        stmt {
            For(loop_) => {
                let (var, end) = (loop_.var.name.as_str(), script(&loop_.end)?);
                lines.push(format!("{indent}loop over {var} up to {}", end.as_str()));
                for child in loop_.body.seq.iter() {
                    describe(child, &format!("{indent}  "), lines)?;
                }
            },
            TensorStore(store) => {
                let (dest, value) = (script(&store.dest)?, script(&store.value)?);
                lines.push(format!("{indent}store to {}: {}", dest.as_str(), value.as_str()));
            },
            _ => lines.push(format!("{indent}other statement")),
        }
    }
    Ok(())
}
// [inspect.end]

// [create.begin]
fn double_store(func: Function) -> Result<TensorStore> {
    let body = func.body.as_ref().expect("add_one has a body");
    let loop_: For = body.seq.get(0)?.try_cast()?;
    let store: TensorStore = loop_.body.seq.get(0)?.try_cast()?;
    let doubled = ex(&store.value) * 2;
    TensorStore::new(&store.dest, store.indices.iter(), doubled, UNKNOWN_LOC)
}
// [create.end]

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

// [dispatch_walk.begin]
#[derive(Default)]
struct StatementTrace {
    events: Vec<String>,
}

#[dispatch(walk)]
impl StatementTrace {
    fn walk_for(&mut self, loop_: &ForObj) -> WalkResult {
        self.events.push(format!("loop {}", loop_.var.name.as_str()));
        WalkResult::Advance
    }

    fn walk_store(&mut self, store: &TensorStoreObj) -> Result<WalkResult> {
        self.events.push(format!("write {}", script(&store.dest)?.as_str()));
        Ok(WalkResult::Advance)
    }
}

fn trace_statements(func: Function, pre_order: bool) -> Result<Strings> {
    let order = if pre_order { WalkOrder::PreOrder } else { WalkOrder::PostOrder };
    let mut trace = StatementTrace::default();
    tvm_ffi::structural_walk(&func, &mut trace, order)?;
    Ok(trace.events.iter().map(tvm_ffi::String::from).collect())
}
// [dispatch_walk.end]

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
        "tirx_ir_guide_examples.rust.inspect",
        inspect,
        "Read the fields of a function.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.double_store",
        double_store,
        "Build a store that writes twice the value.",
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
        "tirx_ir_guide_examples.rust.trace_statements",
        trace_statements,
        "Trace loops and writes.",
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
