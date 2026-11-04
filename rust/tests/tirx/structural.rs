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

//! Constraint-aware Rust-engine descent through an owned or borrowed dispatcher.

use tvm_tirx::ir::*;
use tvm_tirx::sym::{
    Analyzer, BindingConstraintObj, Constraint, ConstraintContext, PredicateConstraintObj,
    RangeConstraintObj,
};
use tvm_tirx::tirx::structural_visit_with_constraint_context::{
    structural_visit_with_constraint_context as visit, StructuralVisitorWithConstraintContextView,
};
use tvm_tirx::tirx::structural_walk_with_constraint_context::structural_walk_with_constraint_context as walk;
use tvm_tirx::tvm_ffi::{
    dispatch, AnyView, Array, DefRegionKind, Error, Map, ObjectRefCore, Result, VisitInterrupt,
    WalkOrder, WalkResult, RUNTIME_ERROR,
};

use super::int32;
use crate::common::serial;

#[derive(Default)]
struct Observe {
    facts: Vec<(i64, Vec<Constraint>)>,
    definitions: Vec<(String, DefRegionKind)>,
    live: Option<ConstraintContext>,
    snapshot: Option<ConstraintContext>,
    array: Option<Array<Constraint>>,
}

#[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorWithConstraintContextView<'_, Self>), Result<Option<VisitInterrupt>>)]
impl Observe {
    fn visit_evaluate(
        &mut self,
        node: &EvaluateObj,
        visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        if let Some(value) = node.value.downcast_ref::<IntImmObj>() {
            let context = visitor.base().constraints();
            if self.live.is_none() {
                self.snapshot = Some(context.snapshot()?);
                self.array = Some(context.to_array());
                self.live = Some(context.clone());
            }
            self.facts.push((value.value.to_i64().unwrap(), context.to_array().iter().collect()));
        }
        visitor.default_visit(self, node)
    }
    fn visit_var(
        &mut self,
        node: &VarObj,
        visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        self.definitions.push((node.name.as_str().into(), visitor.def_region_kind()));
        visitor.default_visit(self, node)
    }
}

fn eval(value: i64) -> Stmt {
    Evaluate::new(int32(value), UNKNOWN_LOC).into()
}
fn variable(name: &str) -> Var {
    Var::new(name, PrimType::int(32), UNKNOWN_LOC)
}
fn fact_counts(seen: &Observe) -> Vec<(i64, usize)> {
    seen.facts.iter().map(|(n, facts)| (*n, facts.len())).collect()
}
#[test]
fn typed_walk_callbacks_collect_facts_and_restore_analyzer_scopes() {
    let _serial = serial();
    let (x, i) = (variable("typed_x"), variable("typed_i"));
    let bound = int32(4);
    let condition = Var::new("typed_condition", PrimType::bool(), UNKNOWN_LOC);
    let body =
        For::new(&i, int32(2), int32(8), int32(2), ForKind::Default, eval(11), None, UNKNOWN_LOC)
            .unwrap();
    let branch = If::new(&condition, &body, Some(&eval(12)), UNKNOWN_LOC).unwrap();
    let empty_loops =
        [(0, 0, 1), (4, 0, -1)].into_iter().enumerate().map(|(n, (begin, end, step))| {
            Stmt::from(
                For::new(
                    &i,
                    int32(begin),
                    int32(end),
                    int32(step),
                    ForKind::Default,
                    eval(14 + n as i64),
                    None,
                    UNKNOWN_LOC,
                )
                .unwrap(),
            )
        });
    let root = SeqStmt::new(
        [Stmt::from(Bind::new(&x, &bound, UNKNOWN_LOC).unwrap()), branch.into(), eval(13)]
            .into_iter()
            .chain(empty_loops),
        UNKNOWN_LOC,
    );
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let analyzer = Analyzer::new();
        let mut facts = Vec::new();
        let mut loop_bounds = Vec::new();
        let mut definitions = Vec::new();
        walk(
            &root,
            (
                |node: &VarObj, _: &ConstraintContext, kind: DefRegionKind| {
                    definitions.push((node.name.as_str().to_owned(), kind));
                    WalkResult::Advance
                },
                (
                    |node: &EvaluateObj, context: &ConstraintContext| -> Result<WalkResult> {
                        let marker = &node.value.downcast_ref::<IntImmObj>().unwrap().value;
                        facts.push((
                            marker.to_i64().unwrap(),
                            context.to_array().iter().collect::<Vec<_>>(),
                        ));
                        analyzer.sync_constraint_context(context)?;
                        loop_bounds.push(if analyzer.const_int_bound_is_bound(&i)? {
                            let bounds = analyzer.const_int_bound(&i)?;
                            Some((bounds.min_value, bounds.max_value))
                        } else {
                            None
                        });
                        Ok(WalkResult::Advance)
                    },
                    |_: AnyView<'_>| WalkResult::Advance,
                ),
            ),
            order,
        )
        .unwrap();
        assert_eq!(
            facts.iter().map(|(marker, facts)| (*marker, facts.len())).collect::<Vec<_>>(),
            [(11, 5), (12, 2), (13, 1), (14, 1), (15, 1)]
        );
        assert_eq!(loop_bounds, [Some((2, 7)), None, None, None, None]);
        let binding = facts[0].1[0].downcast_ref::<BindingConstraintObj>().unwrap();
        assert!(binding.var.same_as(&x));
        assert!(binding.value.same_as(&bound));
        assert!(facts[0].1[0].same_as(&facts[2].1[0]));
        let positive = facts[0].1[1].downcast_ref::<PredicateConstraintObj>().unwrap();
        assert!(positive.condition.same_as(&condition));
        assert!(!positive.assume_always);
        let negative = facts[1].1[1].downcast_ref::<PredicateConstraintObj>().unwrap();
        assert!(negative.condition.downcast_ref::<prim::NotObj>().unwrap().a.same_as(&condition));
        assert!(!negative.assume_always);
        let range = facts[0].1[2].downcast_ref::<RangeConstraintObj>().unwrap();
        assert!(range.var.same_as(&i));
        assert_eq!(range.range.begin.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(2));
        assert_eq!(range.range.extent.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(6));
        assert!(definitions
            .iter()
            .any(|(name, kind)| name == "typed_i" && *kind == DefRegionKind::Simple));
    }
}

#[test]
fn live_context_and_snapshot_have_distinct_prefix_semantics() {
    let _serial = serial();
    let (x, y) = (variable("snapshot_x"), variable("snapshot_y"));
    let root = SeqStmt::new(
        [
            Stmt::from(Bind::new(&x, int32(1), UNKNOWN_LOC).unwrap()),
            eval(10),
            Bind::new(&y, int32(2), UNKNOWN_LOC).unwrap().into(),
            eval(20),
        ],
        UNKNOWN_LOC,
    );
    let mut seen = Observe::default();
    visit(&root, &mut seen).unwrap();
    let live = seen.live.unwrap();
    assert_eq!(live.local_size(), 2);
    assert_eq!(live.to_array().len(), 2);
    assert_eq!(seen.snapshot.unwrap().to_array().len(), 1);
    assert_eq!(seen.array.unwrap().len(), 1);
    assert_eq!(seen.facts[0].1.len(), 1);
    assert_eq!(seen.facts[1].1.len(), 2);
}

#[test]
fn explicit_reentry_uses_borrowed_state_and_restores_definition_regions() {
    let _serial = serial();
    struct Reenter<'a> {
        seen: &'a mut Vec<(i64, DefRegionKind)>,
        depth: usize,
    }
    #[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorWithConstraintContextView<'_, Self>), Result<Option<VisitInterrupt>>)]
    impl<'a> Reenter<'a> {
        fn visit_integer(
            &mut self,
            value: i64,
            visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.seen.push((value, visitor.def_region_kind()));
            if self.depth == 0 {
                self.depth += 1;
                visitor.with_def_region_kind(DefRegionKind::Pattern, |visitor| {
                    visitor.with_def_region_kind(DefRegionKind::Simple, |visitor| {
                        visitor.visit(self, &(value + 1))
                    })
                })?;
                self.depth -= 1;
                self.seen.push((value + 2, visitor.def_region_kind()));
            }
            Ok(None)
        }
    }
    let mut observed = Vec::new();
    visit(&1_i64, Reenter { seen: &mut observed, depth: 0 }).unwrap();
    assert_eq!(
        observed,
        [(1, DefRegionKind::None), (2, DefRegionKind::Pattern), (3, DefRegionKind::None)]
    );
}

#[test]
fn interrupts_restore_native_scope_and_errors_terminate() {
    let _serial = serial();
    struct Halt {
        fail: bool,
        seen: usize,
        restored: bool,
    }
    #[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorWithConstraintContextView<'_, Self>), Result<Option<VisitInterrupt>>)]
    impl Halt {
        fn visit_for(
            &mut self,
            node: &ForObj,
            visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            // A shared base reference remains valid across native context/flag
            // mutation because every such field has interior mutability.
            let base = visitor.base();
            let result = visitor.default_visit(self, node);
            if result.is_ok() {
                self.restored = base.constraints().to_array().is_empty();
            }
            result
        }
        fn visit_evaluate(
            &mut self,
            _: &EvaluateObj,
            visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            self.seen += 1;
            assert_eq!(visitor.base().constraints().to_array().len(), 3);
            if self.fail {
                Err(Error::new(RUNTIME_ERROR, "constraint stop", ""))
            } else {
                Ok(Some(VisitInterrupt::with(17_i64)))
            }
        }
    }
    let i = variable("halt_i");
    let body =
        For::new(&i, int32(0), int32(4), int32(1), ForKind::Default, eval(11), None, UNKNOWN_LOC)
            .unwrap();
    let root = SeqStmt::new([Stmt::from(body), eval(12)], UNKNOWN_LOC);
    let mut state = Halt { fail: false, seen: 0, restored: false };
    let interrupt = visit(&root, &mut state).unwrap().unwrap();
    assert_eq!(i64::try_from(interrupt.value.clone()).unwrap(), 17);
    assert_eq!(state.seen, 1);
    assert!(state.restored);
    state.fail = true;
    assert!(visit(&root, &mut state).err().unwrap().to_string().contains("constraint stop"));
    assert_eq!(state.seen, 2);

    // Public walk order and Skip: pre-order prunes the leaf, post-order observes it first.
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut trace = Vec::new();
        walk(
            &root,
            |value: AnyView<'_>| {
                if let Some(node) = value.downcast_ref::<EvaluateObj>() {
                    let marker =
                        node.value.downcast_ref::<IntImmObj>().unwrap().value.to_i64().unwrap();
                    trace.push(("evaluate", marker));
                    return WalkResult::Skip;
                }
                if let Some(node) = value.downcast_ref::<IntImmObj>() {
                    let marker = node.value.to_i64().unwrap();
                    if [11, 12].contains(&marker) {
                        trace.push(("leaf", marker));
                    }
                }
                WalkResult::Advance
            },
            order,
        )
        .unwrap();
        let expected = match order {
            WalkOrder::PreOrder => vec![("evaluate", 11), ("evaluate", 12)],
            WalkOrder::PostOrder => {
                vec![("leaf", 11), ("evaluate", 11), ("leaf", 12), ("evaluate", 12)]
            }
        };
        assert_eq!(trace, expected);
        let mut calls = 0;
        let interrupted = walk(
            &root,
            |_: &EvaluateObj| {
                calls += 1;
                WalkResult::interrupt_with(19_i64)
            },
            order,
        )
        .unwrap()
        .unwrap();
        assert_eq!(i64::try_from(interrupted.value.clone()).unwrap(), 19);
        assert_eq!(calls, 1);
        let failure = walk(
            &root,
            |_: &EvaluateObj| -> Result<WalkResult> {
                Err(Error::new(RUNTIME_ERROR, "walk stop", ""))
            },
            order,
        )
        .err()
        .unwrap();
        assert!(failure.to_string().contains("walk stop"));
    }
}

#[test]
fn lazy_call_arms_gain_predicates_but_eager_select_arms_do_not() {
    let _serial = serial();
    #[derive(Default)]
    struct Leaves(Vec<(i64, Vec<Constraint>)>);
    #[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorWithConstraintContextView<'_, Self>), Result<Option<VisitInterrupt>>)]
    impl Leaves {
        fn visit_integer(
            &mut self,
            node: &IntImmObj,
            visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>,
        ) -> Result<Option<VisitInterrupt>> {
            let value = node.value.to_i64().unwrap();
            if [21, 22, 31, 32].contains(&value) {
                self.0.push((value, visitor.base().constraints().to_array().iter().collect()));
            }
            Ok(None)
        }
    }
    let condition = Var::new("lazy_condition", PrimType::bool(), UNKNOWN_LOC);
    let lazy = prim::op::if_then_else(&condition, int32(21), int32(22)).unwrap();
    let eager = prim::Select::new(&condition, int32(31), int32(32), UNKNOWN_LOC).unwrap();
    let root = Array::new(vec![lazy, eager.into()]);
    let mut seen = Leaves::default();
    visit(&root, &mut seen).unwrap();
    assert_eq!(
        seen.0.iter().map(|(n, facts)| (*n, facts.len())).collect::<Vec<_>>(),
        [(21, 1), (22, 1), (31, 0), (32, 0)]
    );
    let positive = seen.0[0].1[0].downcast_ref::<PredicateConstraintObj>().unwrap();
    assert!(positive.condition.same_as(&condition));
    let negative = seen.0[1].1[0].downcast_ref::<PredicateConstraintObj>().unwrap();
    assert!(negative.condition.downcast_ref::<prim::NotObj>().unwrap().a.same_as(&condition));
}

#[test]
fn native_assume_and_thread_launch_hooks_preserve_fact_timing() {
    let _serial = serial();
    let condition = Var::new("assumption", PrimType::bool(), UNKNOWN_LOC);
    let assume = Call::new(
        None,
        Op::get("tirx.assume_always").unwrap(),
        [&condition],
        None,
        Vec::<Type>::new(),
        UNKNOWN_LOC,
    )
    .unwrap();
    let thread = variable("thread_index");
    let launch = RegionStmt::new(
        Op::get("tirx.launch_thread").unwrap(),
        [Expr::from(StringImm::new("threadIdx.x", UNKNOWN_LOC)), int32(8)],
        [&thread],
        DictAttrs::new(Map::new()),
        eval(42),
        Vec::<Var>::new(),
        UNKNOWN_LOC,
    )
    .unwrap();
    let root = SeqStmt::new(
        [eval(40), Evaluate::new(assume, UNKNOWN_LOC).into(), eval(41), launch.into(), eval(43)],
        UNKNOWN_LOC,
    );
    let mut seen = Observe::default();
    visit(&root, &mut seen).unwrap();
    assert_eq!(fact_counts(&seen), [(40, 0), (41, 1), (42, 2), (43, 1)]);
    let predicate = seen.facts[1].1[0].downcast_ref::<PredicateConstraintObj>().unwrap();
    assert!(predicate.assume_always);
    assert!(predicate.condition.same_as(&condition));
    let range = seen.facts[2].1[1].downcast_ref::<RangeConstraintObj>().unwrap();
    assert!(range.var.same_as(&thread));
    assert!(seen
        .definitions
        .iter()
        .any(|(name, kind)| name == "thread_index" && *kind == DefRegionKind::Simple));
}
