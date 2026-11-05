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

//! Iterator maps: index expressions as sums of splits of iterators.
//!
//! An [`IterSumExpr`] is a sum of [`IterSplitExpr`]s `(source // lower_factor) % extent * scale`
//! plus `base_`. [`detect_iter_map`] detects them, and its [`IterMapResult`] reports `errors`.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{PrimType, PrimVar, Range, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//! use tvm_tirx::sym::{self, IterMapLevel};
//! use tvm_tirx::tvm_ffi::Map;
//!
//! let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
//! let y = PrimVar::new("y", PrimType::int(32), UNKNOWN_LOC);
//! let four = Range::from_min_extent(0, 4, UNKNOWN_LOC)?;
//! let iters = Map::from_iter([(Var::from(&x), four.clone()), (Var::from(&y), four)]);
//!
//! let bijective = IterMapLevel::Bijective;
//! let map = sym::detect_iter_map([&x * 4 + &y], &iters, true, bijective, None, true)?;
//! assert!(map.errors.is_empty());
//!
//! let o = PrimVar::new("o", PrimType::int(32), UNKNOWN_LOC);
//! let inverse = sym::inverse_affine_iter_map(map.indices.iter(), [&o])?;
//! assert_eq!(script(&inverse.get(&Var::from(&x))?.unwrap())?, "o // 4");
//! assert_eq!(script(&inverse.get(&Var::from(&y))?.unwrap())?, "o % 4");
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): crate::ir::expr::Range
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// The iterator forms are primitive expressions, as C++ `IterMapExpr` derives from `PrimExpr`, and
// their marks and padding predicate are `PrimExpr`s.
// tvm-ffi-stubgen(upcast): sym.IterMapExpr -> PrimExpr
// tvm-ffi-stubgen(upcast): sym.IterSplitExpr -> PrimExpr
// tvm-ffi-stubgen(upcast): sym.IterSumExpr -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterMark.source -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterMark.extent -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterMapResult.padding_predicate -> PrimExpr
// tvm-ffi-stubgen(nullable): sym.IterMapResult.padding_predicate
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// tvm-ffi-stubgen(field): sym.IterSplitExpr.lower_factor -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterSplitExpr.extent -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterSplitExpr.scale -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterSumExpr.base -> PrimExpr
// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::ir::support;
use crate::sym::analyzer::Analyzer;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterMapExpr
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterMapExpr"]
pub struct IterMapExprObj {
    base: ExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMapExpr {
    base: ObjectArc<IterMapExprObj>,
}

impl Deref for IterMapExpr {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base
    }
}

impl Deref for IterMapExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

tvm_ffi::impl_object_upcast!(IterMapExpr => Expr, IterMapExpr => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterMapResult
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterMapResult"]
#[type_final]
pub struct IterMapResultObj {
    base: Object,
    pub indices: Array<IterSumExpr>,
    pub errors: Array<String>,
    pub padding_predicate: Option<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<IterMapResultObj>() == 48);
    assert!(::core::mem::align_of::<IterMapResultObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMapResult {
    base: ObjectArc<IterMapResultObj>,
}

impl Deref for IterMapResult {
    type Target = IterMapResultObj;
    fn deref(&self) -> &IterMapResultObj {
        &self.base
    }
}

impl IterMapResultObj {
    pub(crate) fn new(
        indices: Array<IterSumExpr>,
        errors: Array<String>,
        padding_predicate: Option<PrimExpr>,
    ) -> Self {
        let base = Object::new();
        Self { base, indices, errors, padding_predicate }
    }
}

impl IterMapResult {
    /// Lossless complete-field allocation.
    pub fn new(
        indices: impl Into<Array<IterSumExpr>>,
        errors: impl Into<Array<String>>,
        padding_predicate: impl Into<Option<PrimExpr>>,
    ) -> Self {
        let obj = IterMapResultObj::new(indices.into(), errors.into(), padding_predicate.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterMark
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterMark"]
#[type_final]
pub struct IterMarkObj {
    base: Object,
    pub source: PrimExpr,
    pub extent: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<IterMarkObj>() == 40);
    assert!(::core::mem::align_of::<IterMarkObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMark {
    base: ObjectArc<IterMarkObj>,
}

impl Deref for IterMark {
    type Target = IterMarkObj;
    fn deref(&self) -> &IterMarkObj {
        &self.base
    }
}

impl IterMarkObj {
    pub(crate) fn new(source: PrimExpr, extent: PrimExpr) -> Self {
        let base = Object::new();
        Self { base, source, extent }
    }
}

impl IterMark {
    /// Lossless complete-field allocation.
    pub fn new(source: impl Into<PrimExpr>, extent: impl Into<PrimExpr>) -> Self {
        let obj = IterMarkObj::new(source.into(), extent.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterSplitExpr
/// Opaque: parent 'sym.IterMapExpr' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterSplitExpr"]
#[type_final]
pub struct IterSplitExprObj {
    base: IterMapExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterSplitExpr {
    base: ObjectArc<IterSplitExprObj>,
}

impl Deref for IterSplitExpr {
    type Target = IterSplitExprObj;
    fn deref(&self) -> &IterSplitExprObj {
        &self.base
    }
}

impl Deref for IterSplitExprObj {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base
    }
}

impl IterSplitExprObj {
    pub fn source(&self) -> Result<IterMark> {
        FieldGetter::new(Self::type_index(), "source")?.get(self)
    }

    pub fn lower_factor(&self) -> Result<PrimExpr> {
        FieldGetter::new(Self::type_index(), "lower_factor")?.get(self)
    }

    pub fn extent(&self) -> Result<PrimExpr> {
        FieldGetter::new(Self::type_index(), "extent")?.get(self)
    }

    pub fn scale(&self) -> Result<PrimExpr> {
        FieldGetter::new(Self::type_index(), "scale")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterSplitExpr => Expr, IterSplitExpr => IterMapExpr, IterSplitExpr => PrimExpr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.IterSumExpr
/// Opaque: parent 'sym.IterMapExpr' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.IterSumExpr"]
#[type_final]
pub struct IterSumExprObj {
    base: IterMapExprObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterSumExpr {
    base: ObjectArc<IterSumExprObj>,
}

impl Deref for IterSumExpr {
    type Target = IterSumExprObj;
    fn deref(&self) -> &IterSumExprObj {
        &self.base
    }
}

impl Deref for IterSumExprObj {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base
    }
}

impl IterSumExprObj {
    pub fn args(&self) -> Result<Array<IterSplitExpr>> {
        FieldGetter::new(Self::type_index(), "args")?.get(self)
    }

    pub fn base_(&self) -> Result<PrimExpr> {
        FieldGetter::new(Self::type_index(), "base")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterSumExpr => Expr, IterSumExpr => IterMapExpr, IterSumExpr => PrimExpr);
// tvm-ffi-stubgen(end)

// `IterMark::new` is generated: the native `sym.IterMark` constructor stores the fields as given.

impl IterSplitExpr {
    /// Create the split `(source // lower_factor) % extent * scale` (`sym.IterSplitExpr`).
    pub fn new(
        source: impl Into<IterMark>,
        lower_factor: impl IntoExpr<PrimExpr>,
        extent: impl IntoExpr<PrimExpr>,
        scale: impl IntoExpr<PrimExpr>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.IterSplitExpr")
            .call_tuple((
                &source.into(),
                lower_factor.into_expr()?.borrow(),
                extent.into_expr()?.borrow(),
                scale.into_expr()?.borrow(),
            ))?
            .try_into()
    }
}

impl IterSumExpr {
    /// Create the sum of the splits `args` plus `base` (`sym.IterSumExpr`).
    pub fn new(
        args: impl IntoIterator<Item = impl Into<IterSplitExpr>>,
        base: impl Into<PrimExpr>,
    ) -> Self {
        let args = args.into_iter().map(Into::into).collect::<Array<IterSplitExpr>>();
        support::infallible(
            tvm_ffi::cached_global_func!("sym.IterSumExpr")
                .call_tuple((&args, &base.into()))
                .and_then(|any| any.try_into()),
        )
    }
}

/// How strictly an iterator map is checked, the C++ `sym::IterMapLevel`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum IterMapLevel {
    /// The map must be bijective.
    Bijective = 0,
    /// The map must be surjective.
    #[default]
    Surjective = 1,
    /// The map is not checked.
    NoCheck = 3,
}

/// Detect `indices` as iterator maps of the iterators `input_iters` under `predicate`
/// (`sym.DetectIterMap`).
///
/// When they are not, the `indices` of the result are empty and its `errors` say why.
pub fn detect_iter_map(
    indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
    input_iters: &Map<Var, Range>,
    predicate: impl IntoExpr<PrimExpr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<IterMapResult> {
    let indices =
        indices.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    let (predicate, analyzer) = (predicate.into_owned_expr()?, analyzer.cloned());
    tvm_ffi::cached_global_func!("sym.DetectIterMap")
        .call_packed(&[
            AnyView::from(&indices),
            AnyView::from(input_iters),
            AnyView::from(&predicate),
            AnyView::from(&(check_level as i32)),
            AnyView::from(&analyzer),
            AnyView::from(&simplify_trivial_iterators),
        ])?
        .try_into()
}

/// Return `index` as a sum of splits of the iterators `input_iters`, with what is not detected in
/// its `base` (`sym.NormalizeToIterSum`).
pub fn normalize_to_iter_sum(
    index: impl IntoExpr<PrimExpr>,
    input_iters: &Map<Var, Range>,
    analyzer: Option<&Analyzer>,
) -> Result<IterSumExpr> {
    tvm_ffi::cached_global_func!("sym.NormalizeToIterSum")
        .call_tuple((index.into_expr()?.borrow(), input_iters, analyzer.cloned()))?
        .try_into()
}

/// Simplify `indices` through iterator map detection, or return them unchanged when they are not
/// iterator maps (`sym.IterMapSimplify`).
pub fn iter_map_simplify(
    indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
    input_iters: &Map<Var, Range>,
    predicate: impl IntoExpr<PrimExpr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<Array<PrimExpr>> {
    let indices =
        indices.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    let (predicate, analyzer) = (predicate.into_owned_expr()?, analyzer.cloned());
    tvm_ffi::cached_global_func!("sym.IterMapSimplify")
        .call_packed(&[
            AnyView::from(&indices),
            AnyView::from(input_iters),
            AnyView::from(&predicate),
            AnyView::from(&(check_level as i32)),
            AnyView::from(&analyzer),
            AnyView::from(&simplify_trivial_iterators),
        ])?
        .try_into()
}

/// Return the plain expression of the iterator map `expr` (`sym.NormalizeIterMapToExpr`).
pub fn normalize_iter_map_to_expr(expr: impl IntoExpr<PrimExpr>) -> Result<PrimExpr> {
    tvm_ffi::cached_global_func!("sym.NormalizeIterMapToExpr")
        .call_tuple((expr.into_expr()?.borrow(),))?
        .try_into()
}

/// Divide each binding into an outer map of the iterators not in `sub_iters` and an inner map of
/// those in it (`sym.SubspaceDivide`).
///
/// The result is one `[outer, inner]` pair of marks per binding, then a pair whose `extent`s are
/// the outer and inner predicates, or empty when the bindings cannot be divided.
pub fn subspace_divide(
    bindings: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
    input_iters: &Map<Var, Range>,
    sub_iters: impl IntoIterator<Item = impl Into<Var>>,
    predicate: impl IntoExpr<PrimExpr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<Array<Array<IterMark>>> {
    let bindings =
        bindings.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    let sub_iters = sub_iters.into_iter().map(Into::into).collect::<Array<Var>>();
    let (predicate, analyzer) = (predicate.into_owned_expr()?, analyzer.cloned());
    tvm_ffi::cached_global_func!("sym.SubspaceDivide")
        .call_packed(&[
            AnyView::from(&bindings),
            AnyView::from(input_iters),
            AnyView::from(&sub_iters),
            AnyView::from(&predicate),
            AnyView::from(&(check_level as i32)),
            AnyView::from(&analyzer),
            AnyView::from(&simplify_trivial_iterators),
        ])?
        .try_into()
}

/// Return the inputs of the bijective iterator map `iter_map` as expressions of its `outputs`
/// (`sym.InverseAffineIterMap`).
pub fn inverse_affine_iter_map(
    iter_map: impl IntoIterator<Item = impl Into<IterSumExpr>>,
    outputs: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
) -> Result<Map<Var, PrimExpr>> {
    let iter_map = iter_map.into_iter().map(Into::into).collect::<Array<IterSumExpr>>();
    let outputs =
        outputs.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    tvm_ffi::cached_global_func!("sym.InverseAffineIterMap")
        .call_tuple((&iter_map, &outputs))?
        .try_into()
}
