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

//! Tests of `sym::iter_affine_map`: iterator maps.

use tvm_tirx::ir::{Expr, ExprObj, IntImm, Range, Var};
use tvm_tirx::sym::*;
use tvm_tirx::tvm_ffi::{Array, Map, Object, ObjectIdentity, ObjectRefCast};

use super::{fixture, int32, op, range, var, Fixture};
use crate::common::{assert_type_contracts, global};

#[test]
fn generated_types_match_the_registry() {
    assert_type_contracts!(true; IterMapResultObj: Object, IterMarkObj: Object);
    // The expression hierarchy registers no layout of its own, so the iterator expressions are
    // opaque.
    assert_type_contracts!(false;
        IterMapExprObj: ExprObj, IterSplitExprObj: IterMapExprObj, IterSumExprObj: IterMapExprObj,
    );
}

fn mark_is(mark: &IterMark, var: &Var, extent: i64) -> bool {
    let source: Var = mark.source.clone().try_cast().unwrap();
    let extent_imm: IntImm = mark.extent.clone().try_cast().unwrap();
    ObjectIdentity::of(&source) == ObjectIdentity::of(var)
        && extent_imm.value.to_i64() == Some(extent)
}

#[test]
fn iter_maps_decompose_into_marks_and_splits() {
    let Fixture { analyzer, x } = fixture();
    let indices = Array::new(vec![
        op("prim._OpFloorDiv", &x, int32(4)),
        op("prim._OpFloorMod", &x, int32(4)),
    ]);
    let iters: Map<Var, Range> = [(x.clone(), range(0, 16))].into_iter().collect();
    let result: IterMapResult = global("sym.DetectIterMap")
        .call_tuple((indices, iters, int32(1), 0i32, analyzer, false))
        .unwrap()
        .try_into()
        .unwrap();
    assert!(
        result.errors.is_empty(),
        "{:?}",
        result.errors.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );
    assert_eq!(result.indices.len(), 2);
    let as_int = |expr: Expr| expr.try_cast::<IntImm>().unwrap().value.to_i64().unwrap();
    // Each index is one split of a mark of extent 4, whose source is itself a sum over the
    // mark of x on [0, 16): x // 4 takes lower factor 4 of it, x % 4 lower factor 1.
    let only_split = |sum: &IterSumExpr| -> IterSplitExpr {
        let splits = sum.args().unwrap();
        assert_eq!(splits.len(), 1);
        splits.get(0).unwrap()
    };
    let mut lower_factors = Vec::new();
    for sum in result.indices.iter() {
        assert_eq!(as_int(sum.base_().unwrap()), 0);
        let outer = only_split(&sum);
        let mark: IterMark = outer.source().unwrap();
        assert_eq!(as_int(mark.extent.clone()), 4);
        let inner_sum: IterSumExpr = mark.source.clone().try_cast().unwrap();
        let inner = only_split(&inner_sum);
        let x_mark: IterMark = inner.source().unwrap();
        assert!(mark_is(&x_mark, &x, 16));
        assert_eq!(as_int(inner.extent().unwrap()), 4);
        lower_factors.push(as_int(inner.lower_factor().unwrap()));
    }
    assert_eq!(lower_factors, vec![4, 1]);
}

#[test]
fn iter_maps() {
    let analyzer = Analyzer::new();
    let (x, y) = (var("x"), var("y"));
    let dom = Map::from_iter([(x.clone(), range(0, 4)), (y.clone(), range(0, 4))]);
    let index = op("prim._OpAdd", op("prim._OpMul", &x, int32(4)), &y);
    let predicate = Expr::from(true);

    let detected =
        detect_iter_map([index.clone()], &dom, &predicate, IterMapLevel::Bijective, None, true)
            .unwrap();
    assert!(detected.errors.is_empty());
    assert_eq!(detected.indices.len(), 1);
    let detected_index = detected.indices.get(0).unwrap();
    let normalized = normalize_iter_map_to_expr(detected_index.clone()).unwrap();
    assert!(analyzer.can_prove_equal(normalized, index.clone()).unwrap());

    let sum = normalize_to_iter_sum(index.clone(), &dom, Some(&analyzer)).unwrap();
    assert_eq!(sum.args().unwrap().len(), 2);

    let simplified =
        iter_map_simplify([index.clone()], &dom, &predicate, IterMapLevel::default(), None, true)
            .unwrap();
    assert_eq!(simplified.len(), 1);

    let divided = subspace_divide(
        [index.clone()],
        &dom,
        [y.clone()],
        &predicate,
        IterMapLevel::Surjective,
        None,
        true,
    )
    .unwrap();
    assert_eq!(divided.len(), 2);

    let output = var("o");
    let inverse = inverse_affine_iter_map([detected_index], [output]).unwrap();
    assert_eq!(inverse.len(), 2);

    let mark = IterMark::new(x.clone(), int32(4));
    let split = IterSplitExpr::new(mark, int32(1), int32(4), int32(1)).unwrap();
    assert!(analyzer.can_prove_equal(split.scale().unwrap(), int32(1)).unwrap());
    let sum = IterSumExpr::new([split], int32(0));
    assert_eq!(sum.args().unwrap().len(), 1);
}
