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

use tvm_tirx::ir::prim::{Add, Mul, make_const};
use tvm_tirx::ir::{
    Expr, For, If, IntImm, PrimExpr, PrimType, PrimVar, TensorStore, UNKNOWN_LOC, Var,
};
use tvm_tirx::script::script;
use tvm_tirx::sym::{Analyzer, ConstraintContext, ProofStrength};
use tvm_tirx::tirx::Function;
use tvm_tirx::tirx::structural_map_with_constraint_context::structural_map_with_constraint_context;
use tvm_tirx::tirx::structural_walk_with_constraint_context::structural_walk_with_constraint_context;
use tvm_tirx::tvm_ffi::object::ObjectIdentity;
use tvm_tirx::tvm_ffi::{
    self, Any, AnyView, Array, DefRegionKind, MutationView, ObjectRefCore, Result,
    StructuralMutatorView, StructuralVisitorView, Unchanged, UnchangedOr, VisitInterrupt,
    WalkOrder, WalkResult, dispatch, match_any,
};

/// A Python list of strings.
type Strings = Array<tvm_ffi::String>;

/// The lines of a printout, for Python.
fn lines(items: impl IntoIterator<Item = String>) -> Strings {
    items.into_iter().map(tvm_ffi::String::from).collect()
}

// [construct.begin]
fn x_times_2_plus_1() -> Result<(PrimVar, PrimExpr)> {
    let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
    let expr = (&x * 2 + 1).try_into()?;
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
    let add: Add = expr.try_into()?;
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
    let add: Add = expr.try_into()?;
    let value: Expr = add.into();
    let again = Add::try_from(&value)?;
    let mut out = vec![format!("back to Add: {}", script(&again)?.as_str())];
    // A wrong guess is a `TypeError` naming both types.
    if let Err(err) = Mul::try_from(&value) {
        out.push(format!("as Mul: {}", err.message()));
    }
    Ok(lines(out))
}
// [casts.end]

// [describe.begin]
fn describe(expr: &PrimExpr) -> Result<String> {
    let text = match_any! {
        AnyView::from(expr) {
            Add(ref add) => format!("({} + {})", describe(&add.a)?, describe(&add.b)?),
            Mul(ref mul) => format!("({} * {})", describe(&mul.a)?, describe(&mul.b)?),
            Var(ref var) => var.name.as_str().to_owned(),
            IntImm(ref imm) => imm.value.to_string(),
            _ => "<other>".to_owned(),
        }
    };
    Ok(text)
}
// [describe.end]

// [matching.begin]
fn matching() -> Result<Strings> {
    let (x, expr) = x_times_2_plus_1()?;
    let difference = (&x - 1).try_into()?;
    Ok(lines([describe(&expr)?, describe(&difference)?]))
}
// [matching.end]

// [printer.begin]
struct Printer;

#[dispatch(print, (AnyView<'_>, bool), String)]
impl Printer {
    fn print_add(&self, add: &Add, nested: bool) -> String {
        let text = format!("{} + {}", self.show(&add.a, true), self.show(&add.b, true));
        if nested { format!("({text})") } else { text }
    }

    fn print_var(&self, var: &Var) -> String {
        var.name.as_str().to_owned()
    }

    fn print_imm(&self, imm: &IntImm) -> String {
        imm.value.to_string()
    }

    fn show(&self, expr: &PrimExpr, nested: bool) -> String {
        let text = self.dispatch(AnyView::from(expr), nested);
        text.unwrap_or_else(|| "<other>".to_owned())
    }
}
// [printer.end]

// [print.begin]
fn print() -> Result<tvm_ffi::String> {
    let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
    let expr = (&x + 1 + 2).try_into()?;
    Ok(Printer.show(&expr, false).into())
}
// [print.end]

// [walk.begin]
fn written_tensors(func: Function) -> Result<Strings> {
    let mut writes = Vec::new();
    let on_store = |store: TensorStore| -> Result<WalkResult> {
        let dest = Var::try_from(&store.dest)?;
        writes.push(dest.name.clone());
        Ok(WalkResult::Advance)
    };
    tvm_ffi::structural_walk(&func, on_store, WalkOrder::PostOrder)?;
    Ok(writes.into_iter().collect())
}
// [walk.end]

// [stop.begin]
fn first_write(func: Function) -> Result<Option<TensorStore>> {
    let find_write = |store: TensorStore| WalkResult::InterruptWith(store.into());
    let found = tvm_ffi::structural_walk(&func, find_write, WalkOrder::PostOrder)?;
    match found {
        Some(interrupt) => Ok(Some(interrupt.value.clone().try_into()?)),
        None => Ok(None),
    }
}
// [stop.end]

// [specialize.begin]
fn specialize(func: Function, n: Var, value: i64) -> Result<Function> {
    let specialize_var = |var: Var| -> Result<Expr> {
        if var.same_as(&n) {
            let ty = PrimType::try_from(&var.ty)?;
            return Ok(make_const(ty, value)?.into());
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
    let equal = tvm_ffi::structural_equal(&first, &second)?;
    Ok([first.same_as(&second), equal].into_iter().collect())
}
// [equal.end]

// [counter.begin]
#[derive(Default)]
struct Counter {
    vars: usize,
    adds: usize,
}

#[dispatch(walk, (AnyView<'_>, DefRegionKind), Result<WalkResult>)]
impl Counter {
    fn walk_var(&mut self, _var: &Var, _kind: DefRegionKind) -> Result<WalkResult> {
        self.vars += 1;
        Ok(WalkResult::Advance)
    }

    fn walk_add(&mut self, _add: &Add) -> Result<WalkResult> {
        self.adds += 1;
        Ok(WalkResult::Advance)
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

#[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorView<'_, Self>), Result<Option<VisitInterrupt>>)]
impl LoopNest {
    fn visit_for(
        &mut self,
        loop_: &For,
        visitor: &mut StructuralVisitorView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        self.loops.push(loop_.var.name.as_str().to_owned());
        let result = visitor.default_visit(self, loop_);
        self.loops.pop();
        result
    }

    fn visit_store(&mut self, store: &TensorStore) -> Result<Option<VisitInterrupt>> {
        let loops = self.loops.join(", ");
        let write = format!("{} inside [{}]", script(&store.dest)?.as_str(), loops);
        self.writes.push(write);
        Ok(None)
    }
}

fn loops_around_writes(func: Function) -> Result<Strings> {
    let mut visitor = LoopNest::default();
    tvm_ffi::structural_visit(&func, &mut visitor)?;
    assert!(visitor.loops.is_empty());
    Ok(visitor.writes.iter().map(tvm_ffi::String::from).collect())
}
// [visit.end]

// [mutate.begin]
#[derive(Default)]
struct ReverseLoops {
    reversed_index: HashMap<ObjectIdentity, Expr>,
}

#[dispatch(mutate, (MutationView<'_>, &mut StructuralMutatorView<'_, Self>), Result<UnchangedOr<Any>>)]
impl ReverseLoops {
    fn mutate_for(
        &mut self,
        value: MutationView<'_, For>,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<UnchangedOr<Any>> {
        let index = (&value.end - 1 - &value.var).try_into()?;
        self.reversed_index.insert(ObjectIdentity::of(&value.var), index);
        mutator.default_mutate(self, value)
    }

    fn mutate_var(
        &mut self,
        var: Var,
        mutator: &mut StructuralMutatorView<'_, Self>,
    ) -> Result<UnchangedOr<Any>> {
        let is_use = mutator.def_region_kind() == DefRegionKind::None;
        Ok(match self.reversed_index.get(&ObjectIdentity::of(&var)) {
            Some(index) if is_use => index.clone().into(),
            _ => Unchanged.into(),
        })
    }
}

fn reverse_loops(func: Function) -> Result<Function> {
    tvm_ffi::structural_mutate(func, ReverseLoops::default())?.try_into()
}
// [mutate.end]

// [check_guard.begin]
fn proven_guards(func: Function) -> Result<Array<bool>> {
    let analyzer = Analyzer::new();
    let mut proven = Vec::new();
    structural_walk_with_constraint_context(
        &func,
        |branch: &If, constraints: &ConstraintContext| -> Result<WalkResult> {
            analyzer.sync_constraint_context(constraints)?;
            proven.push(analyzer.can_prove(&branch.condition, ProofStrength::Default)?);
            Ok(WalkResult::Advance)
        },
        WalkOrder::PreOrder,
    )?;
    Ok(proven.into_iter().collect())
}
// [check_guard.end]

// [remove_guard.begin]
fn remove_redundant_guards(func: Function) -> Result<Function> {
    let analyzer = Analyzer::new();
    structural_map_with_constraint_context(
        func,
        |branch: If, constraints: &ConstraintContext| -> Result<UnchangedOr<Any>> {
            analyzer.sync_constraint_context(constraints)?;
            Ok(if analyzer.can_prove(&branch.condition, ProofStrength::Default)? {
                branch.then_case.clone().into()
            } else {
                Unchanged.into()
            })
        },
        WalkOrder::PreOrder,
    )?
    .try_into()
}
// [remove_guard.end]

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
        "tirx_ir_guide_examples.rust.print",
        print,
        "Print an addition with a dispatch struct.",
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
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.proven_guards",
        proven_guards,
        "Check each guard against the facts at that point.",
    )?;
    tvm_ffi::Function::register_global_typed(
        "tirx_ir_guide_examples.rust.remove_redundant_guards",
        remove_redundant_guards,
        "Remove guards that the facts already prove.",
    )?;
}
