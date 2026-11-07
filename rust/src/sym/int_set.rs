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

//! Integer sets. An [`IntSet`], such as the interval [`IntervalSet`], bounds the values an
//! expression takes; [`deduce_bound`] solves a condition for a variable, and the
//! `estimate_region_*` functions bound the region a set of accesses covers.

// tvm-ffi-stubgen(missing-docs): allow
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): crate::ir::expr::Range
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// tvm-ffi-stubgen(import-object): tvm_ffi::IntoObjectRef
// An interval is bounded by `PrimExpr`s, as C++ `IntervalSetObj` holds them.
// tvm-ffi-stubgen(field): sym.IntervalSet.min_value -> PrimExpr
// tvm-ffi-stubgen(field): sym.IntervalSet.max_value -> PrimExpr
// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::ir::support;
use crate::sym::analyzer::Analyzer;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::IntoObjectRef;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::reflection::FieldGetter;
// tvm-ffi-stubgen(end)

/// The base of every set of integers, one dimension each, that bounds the values an expression
/// takes; [`Analyzer::int_set`] computes one.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::PrimOps;
/// use tvm_tirx::ir::{PrimType, PrimVar, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::sym::{self, Analyzer, IntSet, IntervalSet};
/// use tvm_tirx::tvm_ffi::Map;
///
/// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let dom_map = Map::from_iter([(Var::from(&i), IntSet::from(IntervalSet::new(0, 15)))]);
/// let set = Analyzer::new().int_set(&i * 2 + 1, dom_map)?;
/// assert_eq!(script(&set)?, "sym.IntervalSet(min_value=1, max_value=31)");
///
/// let none = Map::<Var, IntSet>::new();
/// let bound = sym::deduce_bound(&i, (&i * 4 + 3).less(64), &none, &none)?;
/// assert_eq!(script(&bound)?, "sym.IntervalSet(min_value=neg_inf, max_value=15)");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/sym.IntSet
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntSet {
    data_: ObjectArc<IntSetObj>,
}

/// The object behind [`IntSet`].
///
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IntSet"]
pub struct IntSetObj {
    base_: Object,
}
// tvm-ffi-stubgen(end)

impl IntSet {
    /// Return whether the set is empty (`sym.IntSetIsNothing`).
    pub fn is_nothing(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.IntSetIsNothing").call_tuple((self,))?.try_into()
    }

    /// Return whether the set holds every integer (`sym.IntSetIsEverything`).
    pub fn is_everything(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.IntSetIsEverything").call_tuple((self,))?.try_into()
    }

    /// Return the set of the lanes of the vector `vec`, or of its value for a scalar
    /// (`sym.IntSetVector`).
    pub fn vector(vec: impl IntoExpr<PrimExpr>) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.IntSetVector")
            .call_tuple((vec.into_expr()?.borrow(),))?
            .try_into()
    }

    /// Return the set that holds only `point` (`sym.IntSetSinglePoint`).
    pub fn single_point(point: impl IntoExpr<PrimExpr>) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.IntSetSinglePoint")
            .call_tuple((point.into_expr()?.borrow(),))?
            .try_into()
    }
}

/// The interval `[min_value, max_value]` of integers, whose bounds are symbolic expressions and
/// may be [`neg_inf`] or [`pos_inf`].
// tvm-ffi-stubgen(begin): object/sym.IntervalSet
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntervalSet {
    data_: ObjectArc<IntervalSetObj>,
}

/// The object behind [`IntervalSet`].
///
/// Opaque: parent 'sym.IntSet' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IntervalSet"]
#[type_final]
pub struct IntervalSetObj {
    base_: IntSetObj,
}

impl Deref for IntervalSetObj {
    type Target = IntSetObj;
    fn deref(&self) -> &IntSetObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl IntervalSetObj {
    pub fn min_value(&self) -> Result<PrimExpr> {
        FieldGetter::new(Self::type_index(), "min_value")?.get(self)
    }

    pub fn max_value(&self) -> Result<PrimExpr> {
        FieldGetter::new(Self::type_index(), "max_value")?.get(self)
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for IntervalSetObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(IntervalSet => IntSet);
// tvm-ffi-stubgen(end)

impl IntervalSet {
    /// Create the interval `[min_value, max_value]`, whose bounds may be [`neg_inf`] and
    /// [`pos_inf`] (`sym.IntervalSet`).
    pub fn new(
        min_value: impl IntoObjectRef<PrimExpr>,
        max_value: impl IntoObjectRef<PrimExpr>,
    ) -> Self {
        support::infallible(
            tvm_ffi::cached_global_func!("sym.IntervalSet")
                .call_tuple((
                    min_value.into_object_ref().borrow(),
                    max_value.into_object_ref().borrow(),
                ))
                .and_then(|any| any.try_into()),
        )
    }
}

/// Return the region a set of affine accesses covers at least, given the domains of their variables
/// and a predicate, or `None` when the accesses are not affine (`sym.EstimateRegionLowerBound`).
///
/// Without an `analyzer`, a fresh one proves.
pub fn estimate_region_lower_bound(
    region: impl IntoIterator<Item = impl IntoObjectRef<Range>>,
    var_dom: impl Into<Map<Var, Range>>,
    predicate: impl IntoExpr<PrimExpr>,
    analyzer: Option<&Analyzer>,
) -> Result<Option<Array<IntSet>>> {
    let region =
        region.into_iter().map(IntoObjectRef::into_owned_object_ref).collect::<Array<Range>>();
    tvm_ffi::cached_global_func!("sym.EstimateRegionLowerBound")
        .call_tuple((&region, &var_dom.into(), predicate.into_expr()?.borrow(), analyzer.cloned()))?
        .try_into()
}

/// Return the exact region a set of affine accesses covers, given the domains of their variables
/// and a predicate, or `None` when the accesses are not affine (`sym.EstimateRegionStrictBound`).
pub fn estimate_region_strict_bound(
    region: impl IntoIterator<Item = impl IntoObjectRef<Range>>,
    var_dom: impl Into<Map<Var, Range>>,
    predicate: impl IntoExpr<PrimExpr>,
    analyzer: Option<&Analyzer>,
) -> Result<Option<Array<IntSet>>> {
    let region =
        region.into_iter().map(IntoObjectRef::into_owned_object_ref).collect::<Array<Range>>();
    tvm_ffi::cached_global_func!("sym.EstimateRegionStrictBound")
        .call_tuple((&region, &var_dom.into(), predicate.into_expr()?.borrow(), analyzer.cloned()))?
        .try_into()
}

/// Return a region that holds what a set of accesses covers, given the domains of their variables
/// and a predicate, also when the accesses are not affine (`sym.EstimateRegionUpperBound`).
pub fn estimate_region_upper_bound(
    region: impl IntoIterator<Item = impl IntoObjectRef<Range>>,
    var_dom: impl Into<Map<Var, Range>>,
    predicate: impl IntoExpr<PrimExpr>,
    analyzer: Option<&Analyzer>,
) -> Result<Array<IntSet>> {
    let region =
        region.into_iter().map(IntoObjectRef::into_owned_object_ref).collect::<Array<Range>>();
    tvm_ffi::cached_global_func!("sym.EstimateRegionUpperBound")
        .call_tuple((&region, &var_dom.into(), predicate.into_expr()?.borrow(), analyzer.cloned()))?
        .try_into()
}

/// Return the symbolic positive infinity (`sym.PosInf`).
pub fn pos_inf() -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("sym.PosInf").call_tuple(())?.try_into()
}

/// Return the symbolic negative infinity (`sym.NegInf`).
pub fn neg_inf() -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("sym.NegInf").call_tuple(())?.try_into()
}

/// Return a subset of the union of `sets`, which may drop some of their segments
/// (`sym.UnionLowerBound`).
pub fn union_lower_bound(
    sets: impl IntoIterator<Item = impl IntoObjectRef<IntSet>>,
) -> Result<IntSet> {
    let sets =
        sets.into_iter().map(IntoObjectRef::into_owned_object_ref).collect::<Array<IntSet>>();
    tvm_ffi::cached_global_func!("sym.UnionLowerBound").call_tuple((&sets,))?.try_into()
}

/// Return the values of `var` for which `cond` holds, with the variables of `hint_map` over their
/// sets and those of `relax_map` relaxed to theirs, or the empty set when no bound can be deduced
/// (`sym.DeduceBound`).
pub fn deduce_bound(
    var: impl IntoExpr<PrimExpr>,
    cond: impl IntoExpr<PrimExpr>,
    hint_map: impl Into<Map<Var, IntSet>>,
    relax_map: impl Into<Map<Var, IntSet>>,
) -> Result<IntSet> {
    tvm_ffi::cached_global_func!("sym.DeduceBound")
        .call_tuple((
            var.into_expr()?.borrow(),
            cond.into_expr()?.borrow(),
            &hint_map.into(),
            &relax_map.into(),
        ))?
        .try_into()
}
