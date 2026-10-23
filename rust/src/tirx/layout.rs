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

//! Layouts: how the indices of a tensor map to memory and to threads.
//!
//! The objects `python/tvm_tirx/tirx/layout.py` defines. [`Layout`] is the base. A [`TileLayout`]
//! maps indices through the [`Iter`]s of its `shard` (distinct elements) and `replica` (copies),
//! each an extent and a stride over an [`Axis`]. A [`ComposeLayout`] applies an XOR swizzle to the
//! memory coordinate of a tile layout. The methods wrap the native `tirx.Layout*`, `tirx.Axis*` and
//! `tirx.TileLayout*` functions; a [`Layout`] method takes a subclass through `Layout::from(&x)`.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): crate::tirx::thread_scope::ThreadScope
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::support;
use crate::sym::analyzer::Analyzer;
use crate::tirx::thread_scope::ThreadScope;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::FieldGetter;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;
use tvm_ffi::String;

/// Header-only stand-in for the builtin `ffi.Enum`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.Enum"]
struct FfiEnumObj {
    base: Object,
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Axis
/// Opaque: parent 'ffi.Enum' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Axis"]
#[type_final]
pub struct AxisObj {
    base: FfiEnumObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Axis {
    base: ObjectArc<AxisObj>,
}

impl Deref for Axis {
    type Target = AxisObj;
    fn deref(&self) -> &AxisObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

impl Axis {
    /// The axis named `name`, registered on first use (`tirx.AxisGet`).
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.AxisGet").call_tuple((String::from(name),))?.try_into()
    }

    /// Whether the axis indexes threads (`tirx.AxisIsThreadAxis`).
    pub fn is_thread(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.AxisIsThreadAxis").call_tuple((self,))?.try_into()
    }

    /// Whether the axis indexes memory (`tirx.AxisIsMemoryAxis`).
    pub fn is_memory(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.AxisIsMemoryAxis").call_tuple((self,))?.try_into()
    }

    /// The backend that registered the axis, if any (`tirx.AxisGetBackend`).
    pub fn backend(&self) -> Result<Option<String>> {
        tvm_ffi::cached_global_func!("tirx.AxisGetBackend").call_tuple((self,))?.try_into()
    }

    /// The thread scope a thread axis spans, if any (`tirx.AxisGetScope`).
    pub fn get_scope(&self) -> Result<Option<ThreadScope>> {
        tvm_ffi::cached_global_func!("tirx.AxisGetScope").call_tuple((self,))?.try_into()
    }

    /// The thread scope a thread axis indexes within its scope, if any (`tirx.AxisGetSubscope`).
    pub fn get_subscope(&self) -> Result<Option<ThreadScope>> {
        tvm_ffi::cached_global_func!("tirx.AxisGetSubscope").call_tuple((self,))?.try_into()
    }
}

// tvm-ffi-stubgen(begin): object/tirx.Iter
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Iter"]
#[type_final]
pub struct IterObj {
    base: Object,
    pub extent: Expr,
    pub stride: Expr,
    pub axis: Axis,
}

const _: () = {
    assert!(::core::mem::size_of::<IterObj>() == 48);
    assert!(::core::mem::align_of::<IterObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Iter {
    base: ObjectArc<IterObj>,
}

impl Deref for Iter {
    type Target = IterObj;
    fn deref(&self) -> &IterObj {
        &self.base
    }
}

impl IterObj {
    pub(crate) fn new(extent: Expr, stride: Expr, axis: Axis) -> Self {
        let base = Object::new();
        Self { base, extent, stride, axis }
    }
}

impl Iter {
    /// Lossless complete-field allocation.
    pub fn new(extent: impl Into<Expr>, stride: impl Into<Expr>, axis: impl Into<Axis>) -> Self {
        let obj = IterObj::new(extent.into(), stride.into(), axis.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// The native `tirx.Iter` stores its fields as given, as the generated `Iter::new` does.

// tvm-ffi-stubgen(begin): object/tirx.Layout
/// Opaque: no metadata of its own: total_size is unknown. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Layout"]
pub struct LayoutObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Layout {
    base: ObjectArc<LayoutObj>,
}

impl Deref for Layout {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}
// tvm-ffi-stubgen(end)

impl Layout {
    /// Check the invariants of the layout, raising an error if one does not hold
    /// (`tirx.LayoutCheckWellFormed`).
    pub fn check_well_formed(&self) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.LayoutCheckWellFormed").call_tuple((self,))?;
        Ok(())
    }

    /// The number of elements the layout maps, along `axis_name` if given (`tirx.LayoutGetSize`).
    pub fn size(&self, axis_name: Option<&str>) -> Result<Expr> {
        tvm_ffi::cached_global_func!("tirx.LayoutGetSize")
            .call_tuple((self, axis_name.map(String::from)))?
            .try_into()
    }

    /// The extent of the coordinates the layout reaches, along `axis_name` if given
    /// (`tirx.LayoutGetSpan`).
    pub fn span(&self, axis_name: Option<&str>) -> Result<Expr> {
        tvm_ffi::cached_global_func!("tirx.LayoutGetSpan")
            .call_tuple((self, axis_name.map(String::from)))?
            .try_into()
    }

    /// The coordinate on each axis of the element at `coord` (`tirx.LayoutApply`).
    pub fn apply(
        &self,
        coord: impl IntoIterator<Item = impl Into<Expr>>,
    ) -> Result<Map<String, Expr>> {
        let coord = coord.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("tirx.LayoutApply").call_tuple((self, &coord))?.try_into()
    }

    /// The coordinate on each axis of the element at the linear index `coord`
    /// (`tirx.LayoutApplyLinear`).
    pub fn apply_linear(&self, coord: impl Into<Expr>) -> Result<Map<String, Expr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApplyLinear")
            .call_tuple((self, &coord.into()))?
            .try_into()
    }

    /// The coordinate on each axis of the element at `coord` of a tensor of `shape`
    /// (`tirx.LayoutApplyWithShape`).
    pub fn apply_with_shape(
        &self,
        coord: impl IntoIterator<Item = impl Into<Expr>>,
        shape: impl IntoIterator<Item = impl Into<Expr>>,
    ) -> Result<Map<String, Expr>> {
        let coord = coord.into_iter().map(Into::into).collect::<Array<Expr>>();
        let shape = shape.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("tirx.LayoutApplyWithShape")
            .call_tuple((self, &coord, &shape))?
            .try_into()
    }

    /// The layout in canonical form (`tirx.LayoutCanonicalize`).
    pub fn canonicalize(&self) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutCanonicalize").call_tuple((self,))?.try_into()
    }

    /// This layout, of `inner_shape`, tiled by `outer`, of `outer_shape` (`tirx.LayoutTile`).
    pub fn tile(
        &self,
        outer: impl Into<TileLayout>,
        outer_shape: impl IntoIterator<Item = impl Into<Expr>>,
        inner_shape: impl IntoIterator<Item = impl Into<Expr>>,
    ) -> Result<Layout> {
        let outer = outer.into();
        let outer_shape = outer_shape.into_iter().map(Into::into).collect::<Array<Expr>>();
        let inner_shape = inner_shape.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("tirx.LayoutTile")
            .call_tuple((self, &outer, &outer_shape, &inner_shape))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/tirx.ComposeLayout
/// Opaque: parent 'tirx.Layout' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ComposeLayout"]
#[type_final]
pub struct ComposeLayoutObj {
    base: LayoutObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ComposeLayout {
    base: ObjectArc<ComposeLayoutObj>,
}

impl Deref for ComposeLayout {
    type Target = ComposeLayoutObj;
    fn deref(&self) -> &ComposeLayoutObj {
        &self.base
    }
}

impl Deref for ComposeLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}

impl ComposeLayoutObj {
    pub fn per_element(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "per_element")?.get(self)
    }

    pub fn swizzle_len(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "swizzle_len")?.get(self)
    }

    pub fn atom_len(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "atom_len")?.get(self)
    }

    pub fn swizzle_inner(&self) -> Result<bool> {
        FieldGetter::new(Self::type_index(), "swizzle_inner")?.get(self)
    }

    pub fn inner_mask(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "inner_mask")?.get(self)
    }

    pub fn outer_mask(&self) -> Result<i64> {
        FieldGetter::new(Self::type_index(), "outer_mask")?.get(self)
    }

    pub fn tile_layout(&self) -> Result<TileLayout> {
        FieldGetter::new(Self::type_index(), "tile_layout")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(ComposeLayout => Layout);
// tvm-ffi-stubgen(end)

impl ComposeLayout {
    /// `tile_layout` with its memory coordinate swizzled: above the low `per_element` bits, the
    /// `swizzle_len` bits `atom_len` bits apart are XORed, into the lower group if
    /// `swizzle_inner` (`tirx.ComposeLayout`).
    pub fn new(
        per_element: i32,
        swizzle_len: i32,
        atom_len: i32,
        tile_layout: impl Into<TileLayout>,
        swizzle_inner: bool,
    ) -> Result<Self> {
        let tile_layout = tile_layout.into();
        tvm_ffi::cached_global_func!("tirx.ComposeLayout")
            .call_tuple((per_element, swizzle_len, atom_len, &tile_layout, swizzle_inner))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/tirx.TileLayout
/// Opaque: parent 'tirx.Layout' is opaque (layout-unknown). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TileLayout"]
#[type_final]
pub struct TileLayoutObj {
    base: LayoutObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TileLayout {
    base: ObjectArc<TileLayoutObj>,
}

impl Deref for TileLayout {
    type Target = TileLayoutObj;
    fn deref(&self) -> &TileLayoutObj {
        &self.base
    }
}

impl Deref for TileLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base
    }
}

impl TileLayoutObj {
    pub fn shard(&self) -> Result<Array<Iter>> {
        FieldGetter::new(Self::type_index(), "shard")?.get(self)
    }

    pub fn replica(&self) -> Result<Array<Iter>> {
        FieldGetter::new(Self::type_index(), "replica")?.get(self)
    }

    pub fn offset(&self) -> Result<Map<Axis, Expr>> {
        FieldGetter::new(Self::type_index(), "offset")?.get(self)
    }
}

tvm_ffi::impl_object_upcast!(TileLayout => Layout);
// tvm-ffi-stubgen(end)

impl TileLayout {
    /// The tile layout of the `shard` and `replica` iterators, shifted by `offset`
    /// (`tirx.TileLayout`).
    pub fn new(
        shard: impl IntoIterator<Item = impl Into<Iter>>,
        replica: impl IntoIterator<Item = impl Into<Iter>>,
        offset: &Map<Axis, Expr>,
    ) -> Self {
        let shard = shard.into_iter().map(Into::into).collect::<Array<Iter>>();
        let replica = replica.into_iter().map(Into::into).collect::<Array<Iter>>();
        support::infallible(
            tvm_ffi::cached_global_func!("tirx.TileLayout")
                .call_tuple((&shard, &replica, offset))
                .and_then(|any| any.try_into()),
        )
    }

    /// Whether the layout describes contiguous row-major memory
    /// (`tirx.TileLayoutIsContiguous`).
    pub fn is_contiguous(&self, analyzer: Option<&Analyzer>) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.TileLayoutIsContiguous")
            .call_tuple((self, analyzer.cloned()))?
            .try_into()
    }

    /// Whether the layout is the identity mapping onto memory (`tirx.TileLayoutIsTrivial`).
    pub fn is_trivial(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.TileLayoutIsTrivial").call_tuple((self,))?.try_into()
    }

    /// The layout regrouped by the dimensions of `shape`, and the separators of the groups
    /// (`tirx.TileLayoutGroup`).
    pub fn group(
        &self,
        shape: impl IntoIterator<Item = impl Into<Expr>>,
    ) -> Result<(TileLayout, Array<i64>)> {
        let shape = shape.into_iter().map(Into::into).collect::<Array<Expr>>();
        let pair: Array<Any> = tvm_ffi::cached_global_func!("tirx.TileLayoutGroup")
            .call_tuple((self, &shape))?
            .try_into()?;
        Ok((pair.get(0)?.try_into()?, pair.get(1)?.try_into()?))
    }

    /// The innermost and outermost thread scopes of the thread axes, if the layout has any
    /// (`tirx.TileLayoutGetScope`).
    pub fn get_scope(&self) -> Result<Option<(ThreadScope, ThreadScope)>> {
        let pair: Option<Array<Any>> = tvm_ffi::cached_global_func!("tirx.TileLayoutGetScope")
            .call_tuple((self,))?
            .try_into()?;
        pair.map(|pair| Ok((pair.get(0)?.try_into()?, pair.get(1)?.try_into()?))).transpose()
    }
}
