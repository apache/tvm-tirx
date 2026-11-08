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

//! Layouts. A [`Layout`] maps the logical indices of a tensor to coordinates on named [`Axis`]es,
//! a memory offset and the threads that hold each element; a [`TileLayout`] builds the map from
//! strided [`Iter`] factors, and a [`ComposeLayout`] swizzles the memory coordinate of one.

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// The layouts are hand-written: C++ constructs them, so they have no field-wise allocator.
// tvm-ffi-stubgen(object): tirx.Layout(ty_map = crate::tirx::layout::Layout)
// tvm-ffi-stubgen(object): tirx.TileLayout(ty_map = crate::tirx::layout::TileLayout)
// tvm-ffi-stubgen(object): tirx.ComposeLayout(ty_map = crate::tirx::layout::ComposeLayout)
// tvm-ffi-stubgen(import): std::ops::Deref
// tvm-ffi-stubgen(import): tvm_ffi::Array
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import): crate::tirx::thread_scope::ThreadScope
// tvm-ffi-stubgen(import): tvm_ffi::Any
// tvm-ffi-stubgen(import): tvm_ffi::String
// tvm-ffi-stubgen(import): tvm_ffi::Result
// tvm-ffi-stubgen(import): crate::ir::support
// tvm-ffi-stubgen(import): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import): std::borrow::Borrow
// tvm-ffi-stubgen(import): crate::ir::expr::PrimExpr
// An iterator's extent and stride are `PrimExpr`s, as C++ `IterObj` holds them.
// tvm-ffi-stubgen(field): tirx.Iter.extent(type = PrimExpr)
// tvm-ffi-stubgen(field): tirx.Iter.stride(type = PrimExpr)
// tvm-ffi-stubgen(import): tvm_ffi::Map
// tvm-ffi-stubgen(import): tvm_ffi::IntoArg

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::support;
use crate::sym::analyzer::Analyzer;
use crate::tirx::thread_scope::ThreadScope;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::IntoArg;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::enum_ref::EnumObj;
// tvm-ffi-stubgen(end)

/// The base of every layout: a map from the logical indices of a tensor to a coordinate on each
/// of its [`Axis`]es, such as the memory axis `m` or a thread axis.
///
/// [`apply`](Layout::apply) maps one index, [`size`](Layout::size) is the number of elements the
/// layout maps and [`span`](Layout::span) the extent of the coordinates it reaches. Each kind of
/// layout, such as [`TileLayout`] or [`ComposeLayout`], is a subtype.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{PrimType, PrimVar, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::sym::Analyzer;
/// use tvm_tirx::tirx::{Axis, Iter, Layout, TileLayout};
/// use tvm_tirx::tvm_ffi::Map;
///
/// let m = Axis::get_or_register("m");
/// let shard = [Iter::new(8, 8, m.clone()), Iter::new(8, 1, m)];
/// let layout = Layout::from(&TileLayout::new(shard, Vec::<Iter>::new(), Map::new()));
/// assert_eq!(script(&layout.size(None)?)?, "64");
///
/// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let j = PrimVar::new("j", PrimType::int(32), UNKNOWN_LOC);
/// let offset = layout.apply([&i, &j], &Analyzer::new())?.get("m")?.unwrap();
/// assert_eq!(script(&offset)?, "i * 8 + j");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Layout {
    data_: ObjectArc<LayoutObj>,
}

/// The object behind [`Layout`].
///
/// Complete: reflected fields fill [24, 24) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.Layout", immutable, rs_parent)]
pub struct LayoutObj {
    base_: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<LayoutObj>() == 24);
    assert!(::core::mem::align_of::<LayoutObj>() == 8);
};

impl Layout {
    /// Check the invariants of the layout (`tirx.LayoutCheckWellFormed`).
    ///
    /// # Errors
    ///
    /// Returns an error if an invariant does not hold, such as thread axes whose scopes are not
    /// connected, or a swizzle with negative lengths.
    pub fn check_well_formed(&self) -> Result<()> {
        tvm_ffi::cached_global_func!("tirx.LayoutCheckWellFormed").call_tuple((self,))?;
        Ok(())
    }

    /// Return the number of elements the layout maps, along `axis_name` if given
    /// (`tirx.LayoutGetSize`).
    pub fn size(&self, axis_name: Option<&str>) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("tirx.LayoutGetSize")
            .call_tuple((self, axis_name.map(String::from)))?
            .try_into()
    }

    /// Return the extent of the coordinates the layout reaches, along `axis_name` if given
    /// (`tirx.LayoutGetSpan`).
    pub fn span(&self, axis_name: Option<&str>) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("tirx.LayoutGetSpan")
            .call_tuple((self, axis_name.map(String::from)))?
            .try_into()
    }

    /// Return the coordinate on each axis of the element at `coord` (`tirx.LayoutApply`).
    ///
    /// # Errors
    ///
    /// Returns an error if `coord` does not have one index for each iterator of the shard.
    pub fn apply(
        &self,
        coord: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        analyzer: &Analyzer,
    ) -> Result<Map<String, PrimExpr>> {
        let coord = coord
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        tvm_ffi::cached_global_func!("tirx.LayoutApply")
            .call_tuple((self, &coord, analyzer))?
            .try_into()
    }

    /// Return the coordinate on each axis of the element at the linear index `coord`
    /// (`tirx.LayoutApplyLinear`).
    pub fn apply_linear(
        &self,
        coord: impl IntoExpr<PrimExpr>,
        analyzer: &Analyzer,
    ) -> Result<Map<String, PrimExpr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApplyLinear")
            .call_tuple((self, coord.into_expr()?.borrow(), analyzer))?
            .try_into()
    }

    /// Return the coordinate on each axis of the element at `coord` of a tensor of `shape`
    /// (`tirx.LayoutApplyWithShape`).
    pub fn apply_with_shape(
        &self,
        coord: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        shape: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        analyzer: &Analyzer,
    ) -> Result<Map<String, PrimExpr>> {
        let coord = coord
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let shape = shape
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        tvm_ffi::cached_global_func!("tirx.LayoutApplyWithShape")
            .call_tuple((self, &coord, &shape, analyzer))?
            .try_into()
    }

    /// Return the layout in canonical form (`tirx.LayoutCanonicalize`).
    pub fn canonicalize(&self) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutCanonicalize").call_tuple((self,))?.try_into()
    }

    /// Return this layout, of `inner_shape`, tiled by `outer`, of `outer_shape`
    /// (`tirx.LayoutTile`).
    ///
    /// # Errors
    ///
    /// Returns an error if `outer_shape` and `inner_shape` differ in length, or a layout cannot be
    /// grouped by its shape.
    pub fn tile(
        &self,
        outer: impl IntoArg<TileLayout>,
        outer_shape: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        inner_shape: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
    ) -> Result<Layout> {
        let outer_shape = outer_shape
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let inner_shape = inner_shape
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        tvm_ffi::cached_global_func!("tirx.LayoutTile")
            .call_tuple((self, outer.into_arg().borrow(), &outer_shape, &inner_shape))?
            .try_into()
    }
}

/// The layout of strided factors: each [`Iter`] of `shard` spans distinct elements and each of
/// `replica` their copies, adding its stride per position to the coordinate on its axis, from the
/// origin `offset` gives on each axis.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TileLayout {
    data_: ObjectArc<TileLayoutObj>,
}

/// The object behind [`TileLayout`].
///
/// Complete: reflected fields fill [24, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.TileLayout", final, immutable)]
pub struct TileLayoutObj {
    base_: LayoutObj,
    pub shard: Array<Iter>,
    pub replica: Array<Iter>,
    pub offset: Map<Axis, PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<TileLayoutObj>() == 48);
    assert!(::core::mem::align_of::<TileLayoutObj>() == 8);
};

impl Deref for TileLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base_
    }
}

tvm_ffi::impl_object_ref_upcast!(TileLayout => Layout);

impl TileLayout {
    /// Create the tile layout of the `shard` and `replica` iterators, shifted by `offset`
    /// (`tirx.TileLayout`).
    pub fn new(
        shard: impl IntoIterator<Item = impl IntoArg<Iter>>,
        replica: impl IntoIterator<Item = impl IntoArg<Iter>>,
        offset: impl Into<Map<Axis, PrimExpr>>,
    ) -> Self {
        let shard = shard.into_iter().map(IntoArg::into_owned).collect::<Array<Iter>>();
        let replica = replica.into_iter().map(IntoArg::into_owned).collect::<Array<Iter>>();
        support::infallible(
            tvm_ffi::cached_global_func!("tirx.TileLayout")
                .call_tuple((&shard, &replica, &offset.into()))
                .and_then(|any| any.try_into()),
        )
    }

    /// Return whether the layout describes contiguous row-major memory
    /// (`tirx.TileLayoutIsContiguous`).
    pub fn is_contiguous(&self, analyzer: Option<&Analyzer>) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.TileLayoutIsContiguous")
            .call_tuple((self, analyzer.cloned()))?
            .try_into()
    }

    /// Return whether the layout is the identity mapping onto memory (`tirx.TileLayoutIsTrivial`).
    pub fn is_trivial(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.TileLayoutIsTrivial").call_tuple((self,))?.try_into()
    }

    /// Return the layout regrouped by the dimensions of `shape`, and the separators of the groups
    /// (`tirx.TileLayoutGroup`).
    ///
    /// # Errors
    ///
    /// Returns an error if the iterators cannot be split into groups of the extents of `shape`.
    pub fn group(
        &self,
        shape: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
    ) -> Result<(TileLayout, Array<i64>)> {
        let shape = shape
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let pair: Array<Any> = tvm_ffi::cached_global_func!("tirx.TileLayoutGroup")
            .call_tuple((self, &shape))?
            .try_into()?;
        Ok((pair[0].clone().try_into()?, pair[1].clone().try_into()?))
    }

    /// Return the innermost and outermost thread scopes of the thread axes, if the layout has any
    /// (`tirx.TileLayoutGetScope`).
    ///
    /// # Errors
    ///
    /// Returns an error if the thread axes are of different backends, or their scopes do not form
    /// one chain.
    pub fn get_scope(&self) -> Result<Option<(ThreadScope, ThreadScope)>> {
        let pair: Option<Array<Any>> = tvm_ffi::cached_global_func!("tirx.TileLayoutGetScope")
            .call_tuple((self,))?
            .try_into()?;
        pair.map(|pair| Ok((pair[0].clone().try_into()?, pair[1].clone().try_into()?))).transpose()
    }
}

/// One factor of a [`TileLayout`]: `extent` positions, each adding `stride` to the coordinate on
/// `axis`.
// tvm-ffi-stubgen(begin): object/tirx.Iter
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Iter {
    data_: ObjectArc<IterObj>,
}

/// The object behind [`Iter`].
// Complete: reflected fields fill [24, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.Iter", final, immutable)]
pub struct IterObj {
    base_: Object,
    pub extent: PrimExpr,
    pub stride: PrimExpr,
    pub axis: Axis,
}

const _: () = {
    assert!(::core::mem::size_of::<IterObj>() == 48);
    assert!(::core::mem::align_of::<IterObj>() == 8);
};

#[allow(missing_docs)]
impl IterObj {
    pub(crate) fn new(extent: PrimExpr, stride: PrimExpr, axis: Axis) -> Self {
        let base_ = Object::new();
        Self { base_, extent, stride, axis }
    }
}

#[allow(missing_docs)]
impl Iter {
    /// Create a `Iter` from its field values.
    pub fn new(
        extent: impl IntoArg<PrimExpr>,
        stride: impl IntoArg<PrimExpr>,
        axis: impl IntoArg<Axis>,
    ) -> Self {
        let obj = IterObj::new(extent.into_owned(), stride.into_owned(), axis.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

// The native `tirx.Iter` stores its fields as given, as the generated `Iter::new` does.

/// A named coordinate a layout maps to: the memory axis `m`, or a thread axis, such as `laneid`,
/// that indexes the threads of a scope. Each name is one registered axis, which
/// [`Axis::get_or_register`] looks up, registering it on first use.
// tvm-ffi-stubgen(begin): object/tirx.Axis
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Axis {
    data_: ObjectArc<AxisObj>,
}

/// The object behind [`Axis`].
// Complete: reflected fields fill [48, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.Axis", final, immutable)]
pub struct AxisObj {
    base_: EnumObj,
}

const _: () = {
    assert!(::core::mem::size_of::<AxisObj>() == 48);
    assert!(::core::mem::align_of::<AxisObj>() == 8);
};

impl tvm_ffi::EnumRef for Axis {}
// tvm-ffi-stubgen(end)

impl Axis {
    /// Look up the axis named `name`, such as the memory axis `m`, registering it on first use
    /// (`tirx.AxisGet`). [`EnumRef::get`](tvm_ffi::EnumRef::get) only looks up.
    pub fn get_or_register(name: &str) -> Self {
        support::infallible(
            tvm_ffi::cached_global_func!("tirx.AxisGet")
                .call_tuple((String::from(name),))
                .and_then(|any| any.try_into()),
        )
    }

    /// Return whether the axis indexes threads (`tirx.AxisIsThreadAxis`).
    pub fn is_thread(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.AxisIsThreadAxis").call_tuple((self,))?.try_into()
    }

    /// Return whether the axis indexes memory (`tirx.AxisIsMemoryAxis`).
    pub fn is_memory(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.AxisIsMemoryAxis").call_tuple((self,))?.try_into()
    }

    /// Return the backend that registered the axis, if any (`tirx.AxisGetBackend`).
    pub fn backend(&self) -> Result<Option<String>> {
        tvm_ffi::cached_global_func!("tirx.AxisGetBackend").call_tuple((self,))?.try_into()
    }

    /// Return the thread scope a thread axis spans, if any (`tirx.AxisGetScope`).
    pub fn get_scope(&self) -> Result<Option<ThreadScope>> {
        tvm_ffi::cached_global_func!("tirx.AxisGetScope").call_tuple((self,))?.try_into()
    }

    /// Return the thread scope a thread axis indexes within its scope, if any
    /// (`tirx.AxisGetSubscope`).
    pub fn get_subscope(&self) -> Result<Option<ThreadScope>> {
        tvm_ffi::cached_global_func!("tirx.AxisGetSubscope").call_tuple((self,))?.try_into()
    }
}

/// The layout `tile_layout` with its memory coordinate XOR-swizzled: above the low `per_element`
/// bits, the group of `swizzle_len` bits `atom_len` bits up is XORed into the group below if
/// `swizzle_inner`, and the lower group into the upper one otherwise.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ComposeLayout {
    data_: ObjectArc<ComposeLayoutObj>,
}

/// The object behind [`ComposeLayout`].
///
/// Complete: reflected fields fill [24, 56) exactly (alignment padding [37, 40)).
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.ComposeLayout", final, immutable)]
pub struct ComposeLayoutObj {
    base_: LayoutObj,
    pub per_element: i32,
    pub swizzle_len: i32,
    pub atom_len: i32,
    pub swizzle_inner: bool,
    pub tile_layout: TileLayout,
    pub inner_mask: i32,
    pub outer_mask: i32,
}

const _: () = {
    assert!(::core::mem::size_of::<ComposeLayoutObj>() == 56);
    assert!(::core::mem::align_of::<ComposeLayoutObj>() == 8);
};

impl Deref for ComposeLayoutObj {
    type Target = LayoutObj;
    fn deref(&self) -> &LayoutObj {
        &self.base_
    }
}

tvm_ffi::impl_object_ref_upcast!(ComposeLayout => Layout);

impl ComposeLayout {
    /// Create `tile_layout` with its memory coordinate swizzled: above the low `per_element` bits,
    /// the `swizzle_len` bits `atom_len` bits apart are XORed, into the lower group if
    /// `swizzle_inner` (`tirx.ComposeLayout`).
    ///
    /// # Errors
    ///
    /// Returns an error if `per_element` or `swizzle_len` is negative, `atom_len` is less than
    /// `swizzle_len`, or `tile_layout` is not well formed.
    pub fn new(
        per_element: i32,
        swizzle_len: i32,
        atom_len: i32,
        tile_layout: impl IntoArg<TileLayout>,
        swizzle_inner: bool,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.ComposeLayout")
            .call_tuple((
                per_element,
                swizzle_len,
                atom_len,
                tile_layout.into_arg().borrow(),
                swizzle_inner,
            ))?
            .try_into()
    }
}
