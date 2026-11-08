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

//! Ordered mapping, result-derived facts and mutation permissions.

use tvm_tirx::ir::*;
use tvm_tirx::sym::{
    BindingConstraintObj, ConstraintContext, PredicateConstraintObj, RangeConstraintObj,
};
use tvm_tirx::tirx::structural_map_with_constraint_context::structural_map_with_constraint_context as map;
use tvm_tirx::tvm_ffi::{
    Any, AnyView, DefRegionKind, Error, ObjectRefCore, Result, Unchanged, WalkOrder, RUNTIME_ERROR,
};

use super::{int32, op};
use crate::common::serial;

fn eval(value: i64) -> Stmt {
    Evaluate::new(int32(value), UNKNOWN_LOC).into()
}

#[test]
fn ordered_mapping_collects_result_facts_and_restores_scopes() {
    let _serial = serial();
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        for moved in [false, true] {
            let x = PrimVar::new("map_x", PrimType::int(32), UNKNOWN_LOC);
            let i = PrimVar::new("map_i", PrimType::int(32), UNKNOWN_LOC);
            let body = For::new(
                &i,
                int32(0),
                int32(4),
                int32(1),
                ForKind::Default,
                eval(11),
                None,
                UNKNOWN_LOC,
            )
            .unwrap();
            let branch =
                If::new(op("prim._OpLT", &x, int32(4)), body, Some(&eval(12)), UNKNOWN_LOC)
                    .unwrap();
            let root = SeqStmt::new(
                [
                    Stmt::from(Bind::new(&x, int32(4), UNKNOWN_LOC).unwrap()),
                    branch.into(),
                    eval(13),
                ],
                UNKNOWN_LOC,
            );
            let pointer = &*root as *const SeqStmtObj;
            let retained = (!moved).then(|| root.clone());
            let mut facts = Vec::new();
            let mut definitions = Vec::new();
            let mut loop_contexts = Vec::new();
            let result = map(
                root,
                (
                    |node: &IntImmObj| -> Any {
                        if node.value.to_i64() == Some(4) {
                            int32(8).into()
                        } else {
                            Unchanged.into()
                        }
                    },
                    |_: &VarObj, _: &ConstraintContext, kind: DefRegionKind| {
                        definitions.push(kind);
                        Unchanged
                    },
                    (
                        |node: &EvaluateObj, context: &ConstraintContext| {
                            facts.push((
                                node.value
                                    .downcast_ref::<IntImmObj>()
                                    .unwrap()
                                    .value
                                    .to_i64()
                                    .unwrap(),
                                context.to_array().iter().cloned().collect::<Vec<_>>(),
                            ));
                            Unchanged
                        },
                        |_: &ForObj, context: &ConstraintContext| {
                            loop_contexts.push(context.to_array().len());
                            Unchanged
                        },
                    ),
                ),
                order,
            )
            .unwrap();
            let result = SeqStmt::try_from(result).unwrap();
            assert_eq!(std::ptr::eq(pointer, &*result), moved);
            assert_eq!(
                facts.iter().map(|(marker, facts)| (*marker, facts.len())).collect::<Vec<_>>(),
                [(11, 5), (12, 2), (13, 1)]
            );
            assert_eq!(loop_contexts, [2]);
            assert!(definitions.contains(&DefRegionKind::Simple));
            let binding = result.seq[0].downcast_ref::<BindObj>().unwrap();
            let branch = result.seq[1].downcast_ref::<IfObj>().unwrap();
            let body = branch.then_case.seq[0].downcast_ref::<ForObj>().unwrap();
            let fact = facts[0].1[0].downcast_ref::<BindingConstraintObj>().unwrap();
            assert!(fact.var.same_as(&binding.var));
            assert!(fact.value.same_as(&binding.value));
            assert_eq!(fact.value.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(8));
            let fact = facts[0].1[1].downcast_ref::<PredicateConstraintObj>().unwrap();
            assert!(fact.condition.same_as(&branch.condition));
            let fact = facts[0].1[2].downcast_ref::<RangeConstraintObj>().unwrap();
            assert!(fact.var.same_as(&body.var));
            assert_eq!(
                fact.range.begin.downcast_ref::<IntImmObj>().unwrap().value.to_i64(),
                Some(0)
            );
            assert_eq!(
                fact.range.extent.downcast_ref::<IntImmObj>().unwrap().value.to_i64(),
                Some(8)
            );
            assert!(facts[0].1[0].same_as(&facts[2].1[0]));
            if let Some(retained) = retained {
                let binding = retained.seq[0].downcast_ref::<BindObj>().unwrap();
                assert_eq!(
                    binding.value.downcast_ref::<IntImmObj>().unwrap().value.to_i64(),
                    Some(4)
                );
            }
        }
    }
}

#[test]
fn replacement_order_identity_aliases_and_errors() {
    let _serial = serial();
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        let mut seen = Vec::new();
        let result = map(
            eval(11),
            (
                |_: &EvaluateObj| eval(21),
                |node: &IntImmObj| {
                    let value = node.value.to_i64().unwrap();
                    seen.push(value);
                    int32(value + 1)
                },
            ),
            order,
        )
        .unwrap();
        let result = result.downcast_ref::<EvaluateObj>().unwrap();
        let expected = match order {
            WalkOrder::PreOrder => 22,
            WalkOrder::PostOrder => 21,
        };
        assert_eq!(
            result.value.downcast_ref::<IntImmObj>().unwrap().value.to_i64(),
            Some(expected)
        );
        assert_eq!(seen, [if matches!(order, WalkOrder::PreOrder) { 21 } else { 11 }]);

        let mut calls = 0;
        let error = map(
            SeqStmt::new([eval(11), eval(12)], UNKNOWN_LOC),
            |_: &EvaluateObj| -> Result<Any> {
                calls += 1;
                Err(Error::new(RUNTIME_ERROR, "map stop"))
            },
            order,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("map stop"));
        assert_eq!(calls, 1);
    }
    // Returning the same owning handle must not prevent reuse; retaining that
    // handle separately must force copying even after callback entry.
    for order in [WalkOrder::PreOrder, WalkOrder::PostOrder] {
        for retain_alias in [false, true] {
            let i = PrimVar::new("map_alias", PrimType::int(32), UNKNOWN_LOC);
            let root = For::new(
                &i,
                int32(0),
                int32(4),
                int32(1),
                ForKind::Default,
                eval(11),
                None,
                UNKNOWN_LOC,
            )
            .unwrap();
            let pointer = &*root as *const ForObj;
            let mut saved: Option<Any> = None;
            let result = map(
                root,
                (
                    |node: &ForObj| -> Any {
                        let value: Any = AnyView::from(node).into();
                        if retain_alias {
                            saved = Some(value.clone());
                        }
                        value
                    },
                    |node: &IntImmObj| -> Any {
                        if node.value.to_i64() == Some(4) {
                            int32(8).into()
                        } else {
                            Unchanged.into()
                        }
                    },
                ),
                order,
            )
            .unwrap();
            let result = result.downcast_ref::<ForObj>().unwrap();
            let post = matches!(order, WalkOrder::PostOrder);
            assert_eq!(std::ptr::eq(pointer, result), !retain_alias || post);
            assert_eq!(result.end.downcast_ref::<IntImmObj>().unwrap().value.to_i64(), Some(8));
            if let Some(saved) = saved {
                let saved = saved.downcast_ref::<ForObj>().unwrap();
                assert_eq!(
                    saved.end.downcast_ref::<IntImmObj>().unwrap().value.to_i64(),
                    Some(if post { 8 } else { 4 })
                );
            }
        }
    }
}
