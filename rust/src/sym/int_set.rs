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
#![allow(dead_code, unused_imports)]

//! Sets of integers.
//!
//! The objects `python/tvm_tirx/sym/int_set.py` defines. [`IntSet`] is the base, and an
//! [`IntervalSet`] is the interval between its `min_value` and `max_value`. The functions of
//! `int_set.py` are here too: region bound estimation, bound deduction and the symbolic infinities.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): crate::ir::expr::Range
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::ir::support;
use crate::sym::analyzer::Analyzer;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IntSet
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IntSet"]
pub struct IntSetObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntSet {
    base: ObjectArc<IntSetObj>,
}

impl Deref for IntSet {
    type Target = IntSetObj;
    fn deref(&self) -> &IntSetObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

impl IntSet {
    /// Whether the set is empty (`sym.IntSetIsNothing`).
    pub fn is_nothing(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.IntSetIsNothing").call_tuple((self,))?.try_into()
    }

    /// Whether the set holds every integer (`sym.IntSetIsEverything`).
    pub fn is_everything(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.IntSetIsEverything").call_tuple((self,))?.try_into()
    }

    /// The set of the lanes of the vector `vec` (`sym.IntSetVector`).
    pub fn vector(vec: impl IntoExpr) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.IntSetVector")
            .call_tuple((&vec.into_expr()?,))?
            .try_into()
    }

    /// The set that holds only `point` (`sym.IntSetSinglePoint`).
    pub fn single_point(point: impl IntoExpr) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.IntSetSinglePoint")
            .call_tuple((&point.into_expr()?,))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/sym.IntervalSet
/// Opaque: parent 'sym.IntSet' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IntervalSet"]
#[type_final]
pub struct IntervalSetObj {
    base: IntSetObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntervalSet {
    base: ObjectArc<IntervalSetObj>,
}

impl Deref for IntervalSet {
    type Target = IntervalSetObj;
    fn deref(&self) -> &IntervalSetObj {
        &self.base
    }
}

impl Deref for IntervalSetObj {
    type Target = IntSetObj;
    fn deref(&self) -> &IntSetObj {
        &self.base
    }
}

impl IntervalSetObj {
    pub fn min_value(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "min_value")?.get(self)
    }

    pub fn max_value(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "max_value")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IntervalSet => IntSet);
// tvm-ffi-stubgen(end)

impl IntervalSet {
    /// The interval `[min_value, max_value]` (`sym.IntervalSet`).
    pub fn new(min_value: impl Into<Expr>, max_value: impl Into<Expr>) -> Self {
        support::infallible(
            tvm_ffi::cached_global_func!("sym.IntervalSet")
                .call_tuple((&min_value.into(), &max_value.into()))
                .and_then(|any| any.try_into()),
        )
    }
}

/// The region a set of affine accesses covers at least, given the domains of their variables and
/// a predicate; `None` when the accesses are not affine (`sym.EstimateRegionLowerBound`).
pub fn estimate_region_lower_bound(
    region: impl IntoIterator<Item = impl Into<Range>>,
    var_dom: &Map<Var, Range>,
    predicate: impl IntoExpr,
    analyzer: Option<&Analyzer>,
) -> Result<Option<Array<IntSet>>> {
    let region = region.into_iter().map(Into::into).collect::<Array<Range>>();
    tvm_ffi::cached_global_func!("sym.EstimateRegionLowerBound")
        .call_tuple((&region, var_dom, &predicate.into_expr()?, analyzer.cloned()))?
        .try_into()
}

/// The exact region a set of affine accesses covers, given the domains of their variables and a
/// predicate; `None` when the accesses are not affine (`sym.EstimateRegionStrictBound`).
pub fn estimate_region_strict_bound(
    region: impl IntoIterator<Item = impl Into<Range>>,
    var_dom: &Map<Var, Range>,
    predicate: impl IntoExpr,
    analyzer: Option<&Analyzer>,
) -> Result<Option<Array<IntSet>>> {
    let region = region.into_iter().map(Into::into).collect::<Array<Range>>();
    tvm_ffi::cached_global_func!("sym.EstimateRegionStrictBound")
        .call_tuple((&region, var_dom, &predicate.into_expr()?, analyzer.cloned()))?
        .try_into()
}

/// A region that holds what a set of accesses covers, given the domains of their variables and a
/// predicate (`sym.EstimateRegionUpperBound`).
pub fn estimate_region_upper_bound(
    region: impl IntoIterator<Item = impl Into<Range>>,
    var_dom: &Map<Var, Range>,
    predicate: impl IntoExpr,
    analyzer: Option<&Analyzer>,
) -> Result<Array<IntSet>> {
    let region = region.into_iter().map(Into::into).collect::<Array<Range>>();
    tvm_ffi::cached_global_func!("sym.EstimateRegionUpperBound")
        .call_tuple((&region, var_dom, &predicate.into_expr()?, analyzer.cloned()))?
        .try_into()
}

/// The symbolic positive infinity (`sym.PosInf`).
pub fn pos_inf() -> Result<Expr> {
    tvm_ffi::cached_global_func!("sym.PosInf").call_tuple(())?.try_into()
}

/// The symbolic negative infinity (`sym.NegInf`).
pub fn neg_inf() -> Result<Expr> {
    tvm_ffi::cached_global_func!("sym.NegInf").call_tuple(())?.try_into()
}

/// A subset of the union of `sets`, which may drop some of their segments
/// (`sym.UnionLowerBound`).
pub fn union_lower_bound(sets: impl IntoIterator<Item = impl Into<IntSet>>) -> Result<IntSet> {
    let sets = sets.into_iter().map(Into::into).collect::<Array<IntSet>>();
    tvm_ffi::cached_global_func!("sym.UnionLowerBound").call_tuple((&sets,))?.try_into()
}

/// The values of `var` for which `cond` holds, with the variables of `hint_map` over their sets
/// and those of `relax_map` relaxed to theirs (`sym.DeduceBound`).
pub fn deduce_bound(
    var: impl IntoExpr,
    cond: impl IntoExpr,
    hint_map: &Map<Var, IntSet>,
    relax_map: &Map<Var, IntSet>,
) -> Result<IntSet> {
    tvm_ffi::cached_global_func!("sym.DeduceBound")
        .call_tuple((&var.into_expr()?, &cond.into_expr()?, hint_map, relax_map))?
        .try_into()
}
