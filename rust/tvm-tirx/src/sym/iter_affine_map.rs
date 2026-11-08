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

//! Iterator maps. An iterator map writes index expressions as [`IterSumExpr`]s, sums of
//! [`IterSplitExpr`]s of [`IterMark`]ed iterators; [`detect_iter_map`] finds one, and the other
//! functions simplify, divide and invert such maps.

// tvm-ffi-stubgen(missing-docs): allow
// The iterator forms are hand-written: C++ constructs them, so they have no field-wise
// allocator.
// tvm-ffi-stubgen(ty-map): sym.IterMapExpr -> crate::sym::iter_affine_map::IterMapExpr
// tvm-ffi-stubgen(ty-map): sym.IterSplitExpr -> crate::sym::iter_affine_map::IterSplitExpr
// tvm-ffi-stubgen(ty-map): sym.IterSumExpr -> crate::sym::iter_affine_map::IterSumExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::expr::ExprObj
// tvm-ffi-stubgen(import-object): std::ops::Deref
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::Range
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimVar
// The marks and the padding predicate are `PrimExpr`s, as in C++.
// tvm-ffi-stubgen(field): sym.IterMark.source -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterMark.extent -> PrimExpr
// tvm-ffi-stubgen(field): sym.IterMapResult.padding_predicate -> PrimExpr
// tvm-ffi-stubgen(nullable): sym.IterMapResult.padding_predicate
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// tvm-ffi-stubgen(import-object): tvm_ffi::IntoArg
// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::PrimVar;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::ir::support;
use crate::sym::analyzer::Analyzer;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::IntoArg;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

/// The base of the iterator map forms, [`IterSplitExpr`] and [`IterSumExpr`]. They hold the result
/// of detection and do not appear in a well-formed function.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMapExpr {
    data_: ObjectArc<IterMapExprObj>,
}

/// The object behind [`IterMapExpr`].
///
/// Complete: reflected fields fill [40, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.IterMapExpr", immutable, rs_parent)]
pub struct IterMapExprObj {
    base_: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<IterMapExprObj>() == 40);
    assert!(::core::mem::align_of::<IterMapExprObj>() == 8);
};

impl Deref for IterMapExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

tvm_ffi::impl_object_ref_upcast!(IterMapExpr => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe IterMapExpr => PrimExpr);

/// The result of [`detect_iter_map`]: the detected `indices`, or empty `indices` and the `errors`
/// that say why detection failed.
///
/// `padding_predicate`, over the input iterators, holds for the indices that the returned
/// iterators cover beyond the bounds of the input ones; it is `None` if detection failed before
/// computing it.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{PrimType, PrimVar, Range, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::sym::{self, IterMapLevel};
/// use tvm_tirx::tvm_ffi::Map;
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let y = PrimVar::new("y", PrimType::int(32), UNKNOWN_LOC);
/// let four = Range::from_min_extent(0, 4, UNKNOWN_LOC)?;
/// let iters = Map::from_iter([(x.clone(), four.clone()), (y.clone(), four)]);
///
/// let bijective = IterMapLevel::Bijective;
/// let map = sym::detect_iter_map([&x * 4 + &y], iters, true, bijective, None, true)?;
/// assert!(map.errors.is_empty());
///
/// let o = PrimVar::new("o", PrimType::int(32), UNKNOWN_LOC);
/// let inverse = sym::inverse_affine_iter_map(map.indices.iter(), [&o])?;
/// assert_eq!(script(&inverse.get(&Var::from(&x))?.unwrap())?, "o // 4");
/// assert_eq!(script(&inverse.get(&Var::from(&y))?.unwrap())?, "o % 4");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/sym.IterMapResult
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMapResult {
    data_: ObjectArc<IterMapResultObj>,
}

/// The object behind [`IterMapResult`].
///
/// Complete: reflected fields fill [24, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.IterMapResult", final)]
pub struct IterMapResultObj {
    base_: Object,
    pub indices: Array<IterSumExpr>,
    pub errors: Array<String>,
    pub padding_predicate: Option<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<IterMapResultObj>() == 48);
    assert!(::core::mem::align_of::<IterMapResultObj>() == 8);
};

#[allow(missing_docs)]
impl IterMapResultObj {
    pub(crate) fn new(
        indices: Array<IterSumExpr>,
        errors: Array<String>,
        padding_predicate: Option<PrimExpr>,
    ) -> Self {
        let base_ = Object::new();
        Self { base_, indices, errors, padding_predicate }
    }
}

#[allow(missing_docs)]
impl IterMapResult {
    /// Lossless complete-field allocation.
    pub fn new(
        indices: impl Into<Array<IterSumExpr>>,
        errors: impl Into<Array<String>>,
        padding_predicate: impl Into<Option<PrimExpr>>,
    ) -> Self {
        let obj = IterMapResultObj::new(indices.into(), errors.into(), padding_predicate.into());
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

/// The mark of `source` as an iterator over `[0, extent)`.
// tvm-ffi-stubgen(begin): object/sym.IterMark
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterMark {
    data_: ObjectArc<IterMarkObj>,
}

/// The object behind [`IterMark`].
///
/// Complete: reflected fields fill [24, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.IterMark", final, immutable)]
pub struct IterMarkObj {
    base_: Object,
    pub source: PrimExpr,
    pub extent: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<IterMarkObj>() == 40);
    assert!(::core::mem::align_of::<IterMarkObj>() == 8);
};

#[allow(missing_docs)]
impl IterMarkObj {
    pub(crate) fn new(source: PrimExpr, extent: PrimExpr) -> Self {
        let base_ = Object::new();
        Self { base_, source, extent }
    }
}

#[allow(missing_docs)]
impl IterMark {
    /// Lossless complete-field allocation.
    pub fn new(source: impl IntoArg<PrimExpr>, extent: impl IntoArg<PrimExpr>) -> Self {
        let obj = IterMarkObj::new(source.into_owned(), extent.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

/// The split `(source // lower_factor) % extent * scale` of the marked iterator `source`.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterSplitExpr {
    data_: ObjectArc<IterSplitExprObj>,
}

/// The object behind [`IterSplitExpr`].
///
/// Complete: reflected fields fill [40, 72) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.IterSplitExpr", final, immutable)]
pub struct IterSplitExprObj {
    base_: IterMapExprObj,
    pub source: IterMark,
    pub lower_factor: PrimExpr,
    pub extent: PrimExpr,
    pub scale: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<IterSplitExprObj>() == 72);
    assert!(::core::mem::align_of::<IterSplitExprObj>() == 8);
};

impl Deref for IterSplitExprObj {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base_
    }
}

tvm_ffi::impl_object_ref_upcast!(IterSplitExpr => Expr, IterSplitExpr => IterMapExpr);
tvm_ffi::impl_object_ref_upcast!(unsafe IterSplitExpr => PrimExpr);

/// The sum of the splits `args` plus `base`.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IterSumExpr {
    data_: ObjectArc<IterSumExprObj>,
}

/// The object behind [`IterSumExpr`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.IterSumExpr", final, immutable)]
pub struct IterSumExprObj {
    base_: IterMapExprObj,
    pub args: Array<IterSplitExpr>,
    pub base: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<IterSumExprObj>() == 56);
    assert!(::core::mem::align_of::<IterSumExprObj>() == 8);
};

impl Deref for IterSumExprObj {
    type Target = IterMapExprObj;
    fn deref(&self) -> &IterMapExprObj {
        &self.base_
    }
}

tvm_ffi::impl_object_ref_upcast!(IterSumExpr => Expr, IterSumExpr => IterMapExpr);
tvm_ffi::impl_object_ref_upcast!(unsafe IterSumExpr => PrimExpr);

// `IterMark::new` is generated: the native `sym.IterMark` constructor stores the fields as given.

impl IterSplitExpr {
    /// Create the split `(source // lower_factor) % extent * scale` (`sym.IterSplitExpr`).
    pub fn new(
        source: impl IntoArg<IterMark>,
        lower_factor: impl IntoExpr<PrimExpr>,
        extent: impl IntoExpr<PrimExpr>,
        scale: impl IntoExpr<PrimExpr>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.IterSplitExpr")
            .call_tuple((
                source.into_arg().borrow(),
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
        args: impl IntoIterator<Item = impl IntoArg<IterSplitExpr>>,
        base: impl IntoArg<PrimExpr>,
    ) -> Self {
        let args = args.into_iter().map(IntoArg::into_owned).collect::<Array<IterSplitExpr>>();
        support::infallible(
            tvm_ffi::cached_global_func!("sym.IterSumExpr")
                .call_tuple((&args, base.into_arg().borrow()))
                .and_then(|any| any.try_into()),
        )
    }
}

/// How strictly an iterator map is checked.
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
    input_iters: impl Into<Map<PrimVar, Range>>,
    predicate: impl IntoExpr<PrimExpr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<IterMapResult> {
    let indices =
        indices.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    let predicate = predicate.into_expr()?;
    tvm_ffi::cached_global_func!("sym.DetectIterMap")
        .call_packed(&[
            AnyView::from(&indices),
            AnyView::from(&input_iters.into()),
            AnyView::from(predicate.borrow()),
            AnyView::from(&(check_level as i32)),
            analyzer.map_or_else(AnyView::new, AnyView::from),
            AnyView::from(&simplify_trivial_iterators),
        ])?
        .try_into()
}

/// Return `index` as a sum of splits of the iterators `input_iters`, with what is not detected in
/// its `base` (`sym.NormalizeToIterSum`).
pub fn normalize_to_iter_sum(
    index: impl IntoExpr<PrimExpr>,
    input_iters: impl Into<Map<PrimVar, Range>>,
    analyzer: Option<&Analyzer>,
) -> Result<IterSumExpr> {
    tvm_ffi::cached_global_func!("sym.NormalizeToIterSum")
        .call_tuple((index.into_expr()?.borrow(), &input_iters.into(), analyzer.cloned()))?
        .try_into()
}

/// Simplify `indices` through iterator map detection, or return them unchanged when they are not
/// iterator maps (`sym.IterMapSimplify`).
pub fn iter_map_simplify(
    indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
    input_iters: impl Into<Map<PrimVar, Range>>,
    predicate: impl IntoExpr<PrimExpr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<Array<PrimExpr>> {
    let indices =
        indices.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    let predicate = predicate.into_expr()?;
    tvm_ffi::cached_global_func!("sym.IterMapSimplify")
        .call_packed(&[
            AnyView::from(&indices),
            AnyView::from(&input_iters.into()),
            AnyView::from(predicate.borrow()),
            AnyView::from(&(check_level as i32)),
            analyzer.map_or_else(AnyView::new, AnyView::from),
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
    input_iters: impl Into<Map<PrimVar, Range>>,
    sub_iters: impl IntoIterator<Item = impl IntoArg<PrimVar>>,
    predicate: impl IntoExpr<PrimExpr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<Array<Array<IterMark>>> {
    let bindings =
        bindings.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    let sub_iters = sub_iters.into_iter().map(IntoArg::into_owned).collect::<Array<PrimVar>>();
    let predicate = predicate.into_expr()?;
    tvm_ffi::cached_global_func!("sym.SubspaceDivide")
        .call_packed(&[
            AnyView::from(&bindings),
            AnyView::from(&input_iters.into()),
            AnyView::from(&sub_iters),
            AnyView::from(predicate.borrow()),
            AnyView::from(&(check_level as i32)),
            analyzer.map_or_else(AnyView::new, AnyView::from),
            AnyView::from(&simplify_trivial_iterators),
        ])?
        .try_into()
}

/// Return the inputs of the bijective iterator map `iter_map` as expressions of its `outputs`
/// (`sym.InverseAffineIterMap`).
pub fn inverse_affine_iter_map(
    iter_map: impl IntoIterator<Item = impl IntoArg<IterSumExpr>>,
    outputs: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
) -> Result<Map<Var, PrimExpr>> {
    let iter_map = iter_map.into_iter().map(IntoArg::into_owned).collect::<Array<IterSumExpr>>();
    let outputs =
        outputs.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<PrimExpr>>>()?;
    tvm_ffi::cached_global_func!("sym.InverseAffineIterMap")
        .call_tuple((&iter_map, &outputs))?
        .try_into()
}
