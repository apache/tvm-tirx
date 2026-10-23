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
//! The objects `python/tvm_tirx/tirx/index_map.py` defines. An [`IndexMap`] maps its
//! `initial_indices` to its `final_indices`, as `(i, j) -> (j, i)` does, and may hold its inverse.
//! Its methods apply it to indices, shapes and tensors, and invert it.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::expr::Range
// tvm-ffi-stubgen(import-object): crate::sym::analyzer::Analyzer
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::Tensor

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use crate::sym::analyzer::Analyzer;
use std::ops::Deref;
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
    pub initial_indices: Array<Var>,
    pub final_indices: Array<Expr>,
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

impl Deref for IndexMap {
    type Target = IndexMapObj;
    fn deref(&self) -> &IndexMapObj {
        &self.base
    }
}

impl IndexMapObj {
    pub(crate) fn new(
        initial_indices: Array<Var>,
        final_indices: Array<Expr>,
        inverse_index_map: Option<ObjectRef>,
    ) -> Self {
        let base = Object::new();
        Self { base, initial_indices, final_indices, inverse_index_map }
    }
}

impl IndexMap {
    /// Lossless complete-field allocation.
    pub fn new(
        initial_indices: impl Into<Array<Var>>,
        final_indices: impl Into<Array<Expr>>,
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
// tvm-ffi-stubgen(end)

// The native `tirx.IndexMap` stores its fields as given, as the generated `IndexMap::new` does.

impl IndexMap {
    /// The final indices `indices` map to (`tirx.IndexMapMapIndices`).
    pub fn map_indices(
        &self,
        indices: impl IntoIterator<Item = impl Into<Expr>>,
        analyzer: Option<&Analyzer>,
    ) -> Result<Array<Expr>> {
        let indices = indices.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("tirx.IndexMapMapIndices")
            .call_tuple((self, &indices, analyzer.cloned()))?
            .try_into()
    }

    /// The shape a buffer of `shape` has after the mapping (`tirx.IndexMapMapShape`).
    pub fn map_shape(
        &self,
        shape: impl IntoIterator<Item = impl Into<Expr>>,
        analyzer: Option<&Analyzer>,
    ) -> Result<Array<Expr>> {
        let shape = shape.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("tirx.IndexMapMapShape")
            .call_tuple((self, &shape, analyzer.cloned()))?
            .try_into()
    }

    /// A copy of `tensor` with its elements moved to the mapped indices
    /// (`tirx.IndexMapMapTensor`).
    pub fn map_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        tvm_ffi::cached_global_func!("tirx.IndexMapMapTensor")
            .call_tuple((self, tensor))?
            .try_into()
    }

    /// The inverse map, which must be bijective over `ranges` (`tirx.IndexMapInverse`).
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

    /// The inverse of a map that may pad `ranges`, and the predicate under which the inverse maps
    /// to an index in `ranges` (`tirx.IndexMapNonSurjectiveInverse`).
    pub fn non_surjective_inverse(
        &self,
        ranges: impl IntoIterator<Item = impl Into<Range>>,
        analyzer: Option<&Analyzer>,
    ) -> Result<(IndexMap, Expr)> {
        let ranges = ranges.into_iter().map(Into::into).collect::<Array<Range>>();
        let pair: Array<Any> = tvm_ffi::cached_global_func!("tirx.IndexMapNonSurjectiveInverse")
            .call_tuple((self, &ranges, analyzer.cloned()))?
            .try_into()?;
        Ok((pair.get(0)?.try_into()?, pair.get(1)?.try_into()?))
    }
}
