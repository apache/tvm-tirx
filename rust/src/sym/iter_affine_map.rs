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

//! Iterator maps: index expressions decomposed into splits of iterators.
//!
//! The objects `python/tvm_tirx/sym/iter_affine_map.py` defines. An [`IterMark`] is an iterator
//! `source` with its `extent`; an [`IterSplitExpr`] is `(source // lower_factor) % extent * scale`;
//! an [`IterSumExpr`] is a sum of splits plus `base_`. An [`IterMapResult`] holds the detected
//! `indices`, the `errors` and the `padding_predicate`. The functions of `iter_affine_map.py`,
//! which detect, simplify, divide and invert iterator maps, are here too.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): crate::ir::expr::Range
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::ir::support;
use crate::sym::analyzer::Analyzer;
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

tvm_ffi::impl_object_upcast!(IterMapExpr => Expr);
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
    pub padding_predicate: Option<Expr>,
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
        padding_predicate: Option<Expr>,
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
        padding_predicate: impl Into<Option<Expr>>,
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
    pub source: Expr,
    pub extent: Expr,
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
    pub(crate) fn new(source: Expr, extent: Expr) -> Self {
        let base = Object::new();
        Self { base, source, extent }
    }
}

impl IterMark {
    /// Lossless complete-field allocation.
    pub fn new(source: impl Into<Expr>, extent: impl Into<Expr>) -> Self {
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

    pub fn lower_factor(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "lower_factor")?.get(self)
    }

    pub fn extent(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "extent")?.get(self)
    }

    pub fn scale(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "scale")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterSplitExpr => Expr, IterSplitExpr => IterMapExpr);
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

    pub fn base_(&self) -> Result<Expr> {
        FieldGetter::new(Self::type_index(), "base")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(IterSumExpr => Expr, IterSumExpr => IterMapExpr);
// tvm-ffi-stubgen(end)

// `IterMark::new` is generated: the native `sym.IterMark` constructor stores the fields as given.

impl IterSplitExpr {
    /// The split `(source // lower_factor) % extent * scale` (`sym.IterSplitExpr`).
    pub fn new(
        source: impl Into<IterMark>,
        lower_factor: impl Into<Expr>,
        extent: impl Into<Expr>,
        scale: impl Into<Expr>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.IterSplitExpr")
            .call_tuple((&source.into(), &lower_factor.into(), &extent.into(), &scale.into()))?
            .try_into()
    }
}

impl IterSumExpr {
    /// The sum of the splits `args` plus `base` (`sym.IterSumExpr`).
    pub fn new(
        args: impl IntoIterator<Item = impl Into<IterSplitExpr>>,
        base: impl Into<Expr>,
    ) -> Self {
        let args = args.into_iter().map(Into::into).collect::<Array<IterSplitExpr>>();
        support::infallible(
            tvm_ffi::cached_global_func!("sym.IterSumExpr")
                .call_tuple((&args, &base.into()))
                .and_then(|any| any.try_into()),
        )
    }
}

/// How strictly an iterator map is checked: `sym::IterMapLevel`.
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

/// Detect `indices` as iterator maps of the iterators `input_iters` under `predicate`; the
/// `indices` of the result are empty when they are not (`sym.DetectIterMap`).
///
/// With `simplify_trivial_iterators`, the iterators of extent 1 become constants.
pub fn detect_iter_map(
    indices: impl IntoIterator<Item = impl Into<Expr>>,
    input_iters: &Map<Var, Range>,
    predicate: impl Into<Expr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<IterMapResult> {
    let indices = indices.into_iter().map(Into::into).collect::<Array<Expr>>();
    let (predicate, analyzer) = (predicate.into(), analyzer.cloned());
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

/// `index` as a sum of splits of the iterators `input_iters`, with what is not detected in its
/// `base` (`sym.NormalizeToIterSum`).
pub fn normalize_to_iter_sum(
    index: impl Into<Expr>,
    input_iters: &Map<Var, Range>,
    analyzer: Option<&Analyzer>,
) -> Result<IterSumExpr> {
    tvm_ffi::cached_global_func!("sym.NormalizeToIterSum")
        .call_tuple((&index.into(), input_iters, analyzer.cloned()))?
        .try_into()
}

/// Simplify `indices` through iterator map detection; they are returned unchanged when they are
/// not iterator maps (`sym.IterMapSimplify`).
pub fn iter_map_simplify(
    indices: impl IntoIterator<Item = impl Into<Expr>>,
    input_iters: &Map<Var, Range>,
    predicate: impl Into<Expr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<Array<Expr>> {
    let indices = indices.into_iter().map(Into::into).collect::<Array<Expr>>();
    let (predicate, analyzer) = (predicate.into(), analyzer.cloned());
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

/// The plain expression of the iterator map `expr` (`sym.NormalizeIterMapToExpr`).
pub fn normalize_iter_map_to_expr(expr: impl Into<Expr>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("sym.NormalizeIterMapToExpr")
        .call_tuple((&expr.into(),))?
        .try_into()
}

/// Divide each binding into an outer map of the iterators not in `sub_iters` and an inner map of
/// those in it: one `[outer, inner]` pair of marks per binding, then a pair whose `extent`s are
/// the outer and inner predicates, or empty when the bindings cannot be divided
/// (`sym.SubspaceDivide`).
pub fn subspace_divide(
    bindings: impl IntoIterator<Item = impl Into<Expr>>,
    input_iters: &Map<Var, Range>,
    sub_iters: impl IntoIterator<Item = impl Into<Var>>,
    predicate: impl Into<Expr>,
    check_level: IterMapLevel,
    analyzer: Option<&Analyzer>,
    simplify_trivial_iterators: bool,
) -> Result<Array<Array<IterMark>>> {
    let bindings = bindings.into_iter().map(Into::into).collect::<Array<Expr>>();
    let sub_iters = sub_iters.into_iter().map(Into::into).collect::<Array<Var>>();
    let (predicate, analyzer) = (predicate.into(), analyzer.cloned());
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

/// The inputs of the bijective iterator map `iter_map` as expressions of its `outputs`
/// (`sym.InverseAffineIterMap`).
pub fn inverse_affine_iter_map(
    iter_map: impl IntoIterator<Item = impl Into<IterSumExpr>>,
    outputs: impl IntoIterator<Item = impl Into<Expr>>,
) -> Result<Map<Var, Expr>> {
    let iter_map = iter_map.into_iter().map(Into::into).collect::<Array<IterSumExpr>>();
    let outputs = outputs.into_iter().map(Into::into).collect::<Array<Expr>>();
    tvm_ffi::cached_global_func!("sym.InverseAffineIterMap")
        .call_tuple((&iter_map, &outputs))?
        .try_into()
}
