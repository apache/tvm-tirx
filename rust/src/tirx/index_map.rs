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

//! Index maps.
//!
//! An [`IndexMap`] maps its `initial_indices` to its `final_indices`, as `(i, j) -> (j, i)` does.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{PrimExpr, PrimType, PrimVar, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//! use tvm_tirx::tirx::IndexMap;
//! use tvm_tirx::tvm_ffi::{Array, Result};
//!
//! let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
//! let j = PrimVar::new("j", PrimType::int(32), UNKNOWN_LOC);
//! let transpose = IndexMap::new([&i, &j], [&j, &i], None)?;
//! assert_eq!(script(&transpose)?, "T.index_map(lambda i, j: (j, i))");
//!
//! let text =
//!     |exprs: Array<PrimExpr>| exprs.iter().map(script).collect::<Result<Vec<_>>>();
//! assert_eq!(text(transpose.map_indices([1, 2], None)?)?, ["2", "1"]);
//!
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::expr::Range
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::Tensor
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// `custom-new` reserves `new` for the C++ constructor, which takes the inverse as an `IndexMap`; the
// generated `from_fields` stores it as the `ObjectRef` the field holds.
// tvm-ffi-stubgen(custom-new): tirx.IndexMap
// The indices a map computes are `PrimExpr`s, as C++ `IndexMapObj` holds them.
// tvm-ffi-stubgen(field): tirx.IndexMap.final_indices -> Array<PrimExpr>
// Its indices are `PrimVar`s, as C++ `IndexMapObj` holds them.
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimVar
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(field): tirx.IndexMap.initial_indices -> Array<PrimVar>
// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::PrimVar;
use crate::ir::expr::Range;
use crate::sym::analyzer::Analyzer;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::Tensor;
use tvm_ffi::object::ObjectRef;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.IndexMap
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.IndexMap"]
#[type_final]
pub struct IndexMapObj {
    base: Object,
    pub initial_indices: Array<PrimVar>,
    pub final_indices: Array<PrimExpr>,
    pub inverse_index_map: Option<ObjectRef>,
}

const _: () = {
    assert!(::core::mem::size_of::<IndexMapObj>() == 48);
    assert!(::core::mem::align_of::<IndexMapObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IndexMap {
    base: ObjectArc<IndexMapObj>,
}

impl IndexMapObj {
    pub(crate) fn new(
        initial_indices: Array<PrimVar>,
        final_indices: Array<PrimExpr>,
        inverse_index_map: Option<ObjectRef>,
    ) -> Self {
        let base = Object::new();
        Self { base, initial_indices, final_indices, inverse_index_map }
    }
}

impl IndexMap {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        initial_indices: impl Into<Array<PrimVar>>,
        final_indices: impl Into<Array<PrimExpr>>,
        inverse_index_map: impl Into<Option<ObjectRef>>,
    ) -> Self {
        let obj = IndexMapObj::new(
            initial_indices.into(),
            final_indices.into(),
            inverse_index_map.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for IndexMapObj {}
// tvm-ffi-stubgen(end)

impl IndexMap {
    /// Create the map from `initial_indices` to `final_indices`, with the inverse
    /// `inverse_index_map` if it is known (C++ `IndexMap(initial_indices, final_indices,
    /// inverse_index_map)`).
    pub fn new(
        initial_indices: impl IntoIterator<Item = impl Into<PrimVar>>,
        final_indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        inverse_index_map: Option<&IndexMap>,
    ) -> Result<Self> {
        let initial_indices =
            initial_indices.into_iter().map(Into::into).collect::<Array<PrimVar>>();
        let final_indices = final_indices
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let inverse_index_map = inverse_index_map.map(|map| ObjectRef::from(map.clone()));
        Ok(Self::from_fields(initial_indices, final_indices, inverse_index_map))
    }

    /// Return the final indices that `indices` map to (`tirx.IndexMapMapIndices`).
    ///
    /// # Errors
    ///
    /// Returns an error if the number of indices differs from that of `initial_indices`.
    pub fn map_indices(
        &self,
        indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        analyzer: Option<&Analyzer>,
    ) -> Result<Array<PrimExpr>> {
        let indices = indices
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        tvm_ffi::cached_global_func!("tirx.IndexMapMapIndices")
            .call_tuple((self, &indices, analyzer.cloned()))?
            .try_into()
    }

    /// Return the shape a buffer of `shape` has after the mapping (`tirx.IndexMapMapShape`).
    ///
    /// # Errors
    ///
    /// Returns an error if the number of extents differs from that of `initial_indices`.
    pub fn map_shape(
        &self,
        shape: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        analyzer: Option<&Analyzer>,
    ) -> Result<Array<PrimExpr>> {
        let shape = shape
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        tvm_ffi::cached_global_func!("tirx.IndexMapMapShape")
            .call_tuple((self, &shape, analyzer.cloned()))?
            .try_into()
    }

    /// Return a copy of `tensor` with its elements moved to the mapped indices
    /// (`tirx.IndexMapMapTensor`).
    ///
    /// # Errors
    ///
    /// Returns an error if the rank of `tensor` differs from the number of `initial_indices`.
    pub fn map_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        tvm_ffi::cached_global_func!("tirx.IndexMapMapTensor")
            .call_tuple((self, tensor))?
            .try_into()
    }

    /// Return the inverse map, which must be bijective over `ranges` (`tirx.IndexMapInverse`).
    ///
    /// # Errors
    ///
    /// Returns an error if the final indices are not affine in the initial ones, or the map is not
    /// provably bijective over `ranges`.
    pub fn inverse(
        &self,
        ranges: impl IntoIterator<Item = impl Into<Range>>,
        analyzer: Option<&Analyzer>,
    ) -> Result<IndexMap> {
        let ranges = ranges.into_iter().map(Into::into).collect::<Array<Range>>();
        tvm_ffi::cached_global_func!("tirx.IndexMapInverse")
            .call_tuple((self, &ranges, analyzer.cloned()))?
            .try_into()
    }

    /// Return the inverse of a map that may pad `ranges`, and the predicate under which the inverse
    /// maps to an index in `ranges` (`tirx.IndexMapNonSurjectiveInverse`).
    pub fn non_surjective_inverse(
        &self,
        ranges: impl IntoIterator<Item = impl Into<Range>>,
        analyzer: Option<&Analyzer>,
    ) -> Result<(IndexMap, PrimExpr)> {
        let ranges = ranges.into_iter().map(Into::into).collect::<Array<Range>>();
        let pair: Array<Any> = tvm_ffi::cached_global_func!("tirx.IndexMapNonSurjectiveInverse")
            .call_tuple((self, &ranges, analyzer.cloned()))?
            .try_into()?;
        Ok((pair.get(0)?.try_into()?, pair.get(1)?.try_into()?))
    }
}
