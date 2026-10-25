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

//! Tests of `sym::analyzer`: the analyzer and its results.

use tvm_tirx::ir::{Expr, ExprObj};
use tvm_tirx::sym::*;
use tvm_tirx::tvm_ffi::Object;

use super::{context, fixture, int32, op, predicate_constraint, range, range_constraint, var, Fixture};
use crate::common::{assert_type_contracts, global};

#[test]
fn generated_types_match_the_registry() {
    assert_type_contracts!(true;
        ConstIntBoundObj: Object, ModularSetObj: Object, RewriteSimplifierStatsObj: Object,
    );
    // The analyzer holds its sub-analyzers unreflected; the expression hierarchy registers no
    // layout of its own, so the canonical expressions are opaque too.
    assert_type_contracts!(false;
        AnalyzerObj: Object, CanonicalExprObj: ExprObj, SplitExprObj: CanonicalExprObj,
        SumExprObj: CanonicalExprObj,
    );
}

#[test]
fn analyzer_results_are_read_as_fields() {
    let Fixture { analyzer, x } = fixture();
    let bound: ConstIntBound = global("sym.AnalyzerConstIntBound")
        .call_tuple((analyzer.clone(), op("prim._OpAdd", &x, int32(1))))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!((bound.min_value, bound.max_value), (1, 16));

    let modular: ModularSet = global("sym.AnalyzerModularSet")
        .call_tuple((
            analyzer.clone(),
            op("prim._OpAdd", op("prim._OpMul", &x, int32(4)), int32(2)),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!((modular.coeff, modular.base_), (4, 2));

    let stats: RewriteSimplifierStats = global("sym.AnalyzerGetRewriteSimplifyStats")
        .call_tuple((analyzer,))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(stats.nodes_visited >= 0);
}

#[test]
fn rust_allocated_bounds_reach_the_native_analyzer() {
    let Fixture { analyzer, x } = fixture();
    // Rust-allocated bound fields define the range premises consumed by the native analyzer.
    let bound = ConstIntBound::new(3, 7);
    let constraints = context([range_constraint(
        &x,
        range(bound.min_value, bound.max_value - bound.min_value + 1),
    )]);
    analyzer.sync_constraint_context(&constraints).unwrap();
    let read: ConstIntBound = global("sym.AnalyzerConstIntBound")
        .call_tuple((analyzer, Expr::from(x)))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!((read.min_value, read.max_value), (3, 7));
}

#[test]
fn bounds_and_simplification() {
    let analyzer = Analyzer::new();
    let x = var("x");
    let constraints = context([range_constraint(&x, range(0, 16))]);
    analyzer.sync_constraint_context(&constraints).unwrap();

    let bound = analyzer.const_int_bound(op("prim._OpAdd", &x, int32(1))).unwrap();
    assert_eq!((bound.min_value, bound.max_value), (1, 16));
    assert!(analyzer.const_int_bound_is_bound(&x).unwrap());

    let modular =
        analyzer.modular_set(op("prim._OpAdd", op("prim._OpMul", &x, int32(4)), int32(2))).unwrap();
    assert_eq!((modular.coeff, modular.base_), (4, 2));

    assert!(analyzer.can_prove(op("prim._OpLT", &x, int32(16)), ProofStrength::Default).unwrap());
    assert_eq!(analyzer.try_compare(&x, int32(16)).unwrap(), CompareResult::Lt);

    let simplified =
        analyzer.simplify(op("prim._OpSub", op("prim._OpAdd", &x, int32(3)), &x), 2).unwrap();
    assert!(analyzer.can_prove_equal(simplified, int32(3)).unwrap());
    assert!(analyzer.rewrite_simplify_stats().unwrap().nodes_visited >= 0);
    analyzer.reset_rewrite_simplify_stats().unwrap();
    let copy = analyzer.deep_clone().unwrap();
    assert!(copy.const_int_bound_is_bound(&x).unwrap());
}

#[test]
fn contexts_restore_parent_facts() {
    let analyzer = Analyzer::new();
    let x = var("x");
    let x_mod_3 = op("prim._OpFloorMod", &x, int32(3));
    let parent = context([]);
    let child = context([predicate_constraint(op("prim._OpEQ", &x_mod_3, int32(0)))]);
    analyzer.sync_constraint_context(&child).unwrap();
    assert_eq!(analyzer.modular_set(&x).unwrap().coeff, 3);
    analyzer.sync_constraint_context(&parent).unwrap();
    assert_eq!(analyzer.modular_set(&x).unwrap().coeff, 1);
    let _uint = allow_uint_as_index().unwrap();

    let constraints = context([range_constraint(&x, range(2, 4))]);
    analyzer.sync_constraint_context(&constraints).unwrap();
    assert_eq!(analyzer.const_int_bound(&x).unwrap().max_value, 5);
}
