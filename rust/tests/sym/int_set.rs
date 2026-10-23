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

//! Tests of `sym::int_set`: integer sets and region bounds.

use tvm_tirx::ir::{Expr, IntImm, Location, Range, Var};
use tvm_tirx::sym::*;
use tvm_tirx::tvm_ffi::{Map, Object, ObjectRefCast};

use super::{fixture, int32, op, range, var, Fixture};
use crate::common::{assert_type_contracts, global};

#[test]
fn generated_types_match_the_registry() {
    // `sym.IntSet` registers no layout of its own, so its subclasses are opaque too.
    assert_type_contracts!(false; IntSetObj: Object, IntervalSetObj: IntSetObj);
}

#[test]
fn interval_sets_are_read_through_accessors() {
    let Fixture { analyzer, x } = fixture();
    let y = var("y");
    // With an explicit domain map the evaluation ignores the analyzer's own bindings.
    let interval = |min: i64, max: i64| -> IntSet {
        global("sym.IntervalSet").call_tuple((int32(min), int32(max))).unwrap().try_into().unwrap()
    };
    let dom_map: Map<Var, IntSet> =
        [(x.clone(), interval(0, 15)), (y.clone(), interval(2, 5))].into_iter().collect();
    let set: IntSet = global("sym.AnalyzerIntSet")
        .call_tuple((analyzer, op("prim._OpAdd", &x, &y), dom_map))
        .unwrap()
        .try_into()
        .unwrap();
    let interval: IntervalSet = set.try_cast().unwrap();
    let as_int = |expr: Expr| expr.try_cast::<IntImm>().unwrap().value.to_i64().unwrap();
    assert_eq!(as_int(interval.min_value().unwrap()), 2);
    assert_eq!(as_int(interval.max_value().unwrap()), 20);
}

#[test]
fn int_sets() {
    let analyzer = Analyzer::new();
    let interval = IntervalSet::new(int32(0), int32(10));
    assert!(analyzer.can_prove_equal(interval.min_value().unwrap(), int32(0)).unwrap());
    let set: IntSet = interval.into();
    assert!(!set.is_nothing().unwrap());
    assert!(!set.is_everything().unwrap());

    let everything: IntSet = IntervalSet::new(neg_inf().unwrap(), pos_inf().unwrap()).into();
    assert!(everything.is_everything().unwrap());

    let x = var("x");
    assert!(!IntSet::single_point(&x).unwrap().is_nothing().unwrap());
    assert!(!IntSet::vector(&x).unwrap().is_everything().unwrap());

    let union = union_lower_bound([
        IntervalSet::new(int32(0), int32(3)),
        IntervalSet::new(int32(4), int32(7)),
    ])
    .unwrap();
    assert!(!union.is_nothing().unwrap());

    // x + 1 < 10 holds for x <= 8.
    let empty = Map::<Var, IntSet>::new();
    let bound = deduce_bound(
        &x,
        op("prim._OpLT", op("prim._OpAdd", &x, int32(1)), int32(10)),
        &empty,
        &empty,
    )
    .unwrap();
    let bound = analyzer.int_set(&x, Some(&Map::from_iter([(x.clone(), bound)]))).unwrap();
    assert!(!bound.is_everything().unwrap());
}

#[test]
fn region_bounds() {
    let x = var("x");
    let var_dom = Map::from_iter([(x.clone(), range(0, 16))]);
    let region = [Range::from_fields(x.clone(), int32(1), Location::unknown())];
    let predicate = Expr::from(true);

    let lower = estimate_region_lower_bound(region.clone(), &var_dom, &predicate, None).unwrap();
    assert_eq!(lower.unwrap().len(), 1);
    let analyzer = Analyzer::new();
    let strict =
        estimate_region_strict_bound(region.clone(), &var_dom, &predicate, Some(&analyzer))
            .unwrap();
    assert_eq!(strict.unwrap().len(), 1);
    let upper = estimate_region_upper_bound(region, &var_dom, &predicate, None).unwrap();
    assert_eq!(upper.len(), 1);
}
