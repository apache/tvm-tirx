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

//! Typed constraint-aware structural walk callbacks and analyzer scopes.

use tvm_tirx::ir::*;
use tvm_tirx::sym::{
    Analyzer, BindingConstraintObj, ConstraintContext, PredicateConstraintObj, RangeConstraintObj,
};
use tvm_tirx::tirx::structural_walk_with_constraint_context::structural_walk_with_constraint_context as walk;
use tvm_tirx::tvm_ffi::{
    AnyView, DefRegionKind, Error, ObjectRefCore, Result, WalkOrder, WalkResult, RUNTIME_ERROR,
};
use super::int32;
use crate::common::serial;

fn eval(value: i64) -> Stmt {
    Evaluate::new(int32(value), UNKNOWN_LOC).into()
}
fn variable(name: &str) -> Var {
    Var::new(name, PrimType::int(32), UNKNOWN_LOC)
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
    let body =
        For::new(&i, int32(0), int32(4), int32(1), ForKind::Default, eval(11), None, UNKNOWN_LOC)
            .unwrap();
    let root = SeqStmt::new([Stmt::from(body), eval(12)], UNKNOWN_LOC);
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
                WalkResult::InterruptWith(19_i64.into())
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
