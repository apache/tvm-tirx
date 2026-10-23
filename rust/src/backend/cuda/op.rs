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
// The reflected field names `M`, `N`, and `K` are kept as they are.
#![allow(dead_code, unused_imports, non_snake_case)]

//! The attributes of CUDA instructions.
//!
//! The objects `python/tvm_tirx/backend/cuda/op.py` defines. [`TCGen05InstrDescriptorAttrs`] and
//! [`TCGen05InstrDescriptorBlockScaledAttrs`] are the options of a `tcgen05` instruction
//! descriptor: data types, the `M`/`N`/`K` shape, transposes and negations.
//! [`TensorMapEncodeTiledAttr`] is the element data type, rank, interleave, swizzle and
//! out-of-bound fill of a tiled tensor-map encoding.

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::attrs::AttrsObj;
use std::ops::Deref;
use tvm_ffi::DLDataType;
use tvm_ffi::ObjectArc;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.cuda.TCGen05InstrDescriptorAttrs
/// Complete: reflected fields fill [24, 120) exactly (alignment padding [98, 104), [116, 120)).
#[allow(non_snake_case)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.cuda.TCGen05InstrDescriptorAttrs"]
#[type_final]
pub struct TCGen05InstrDescriptorAttrsObj {
    base: AttrsObj,
    pub d_dtype: String,
    pub a_dtype: String,
    pub b_dtype: String,
    pub M: i64,
    pub N: i64,
    pub K: i64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub n_cta_groups: i64,
    pub neg_a: bool,
    pub neg_b: bool,
    pub sat_d: bool,
    pub is_sparse: bool,
}

const _: () = {
    assert!(::core::mem::size_of::<TCGen05InstrDescriptorAttrsObj>() == 120);
    assert!(::core::mem::align_of::<TCGen05InstrDescriptorAttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TCGen05InstrDescriptorAttrs {
    base: ObjectArc<TCGen05InstrDescriptorAttrsObj>,
}

impl Deref for TCGen05InstrDescriptorAttrs {
    type Target = TCGen05InstrDescriptorAttrsObj;
    fn deref(&self) -> &TCGen05InstrDescriptorAttrsObj {
        &self.base
    }
}

impl Deref for TCGen05InstrDescriptorAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl TCGen05InstrDescriptorAttrsObj {
    #[allow(non_snake_case)]
    pub(crate) fn new(
        d_dtype: String,
        a_dtype: String,
        b_dtype: String,
        M: i64,
        N: i64,
        K: i64,
        trans_a: bool,
        trans_b: bool,
        n_cta_groups: i64,
        neg_a: bool,
        neg_b: bool,
        sat_d: bool,
        is_sparse: bool,
    ) -> Self {
        let base = AttrsObj::new();
        Self {
            base,
            d_dtype,
            a_dtype,
            b_dtype,
            M,
            N,
            K,
            trans_a,
            trans_b,
            n_cta_groups,
            neg_a,
            neg_b,
            sat_d,
            is_sparse,
        }
    }
}

impl TCGen05InstrDescriptorAttrs {
    /// Lossless complete-field allocation.
    #[allow(non_snake_case)]
    pub fn new(
        d_dtype: impl Into<String>,
        a_dtype: impl Into<String>,
        b_dtype: impl Into<String>,
        M: i64,
        N: i64,
        K: i64,
        trans_a: bool,
        trans_b: bool,
        n_cta_groups: i64,
        neg_a: bool,
        neg_b: bool,
        sat_d: bool,
        is_sparse: bool,
    ) -> Self {
        let obj = TCGen05InstrDescriptorAttrsObj::new(
            d_dtype.into(),
            a_dtype.into(),
            b_dtype.into(),
            M,
            N,
            K,
            trans_a,
            trans_b,
            n_cta_groups,
            neg_a,
            neg_b,
            sat_d,
            is_sparse,
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TCGen05InstrDescriptorAttrs => Attrs);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.cuda.TCGen05InstrDescriptorBlockScaledAttrs
/// Complete: reflected fields fill [24, 152) exactly (alignment padding [130, 136), [147, 152)).
#[allow(non_snake_case)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.cuda.TCGen05InstrDescriptorBlockScaledAttrs"]
#[type_final]
pub struct TCGen05InstrDescriptorBlockScaledAttrsObj {
    base: AttrsObj,
    pub d_dtype: String,
    pub a_dtype: String,
    pub b_dtype: String,
    pub sfa_dtype: String,
    pub sfb_dtype: String,
    pub M: i64,
    pub N: i64,
    pub K: i64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub n_cta_groups: i64,
    pub neg_a: bool,
    pub neg_b: bool,
    pub is_sparse: bool,
}

const _: () = {
    assert!(::core::mem::size_of::<TCGen05InstrDescriptorBlockScaledAttrsObj>() == 152);
    assert!(::core::mem::align_of::<TCGen05InstrDescriptorBlockScaledAttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TCGen05InstrDescriptorBlockScaledAttrs {
    base: ObjectArc<TCGen05InstrDescriptorBlockScaledAttrsObj>,
}

impl Deref for TCGen05InstrDescriptorBlockScaledAttrs {
    type Target = TCGen05InstrDescriptorBlockScaledAttrsObj;
    fn deref(&self) -> &TCGen05InstrDescriptorBlockScaledAttrsObj {
        &self.base
    }
}

impl Deref for TCGen05InstrDescriptorBlockScaledAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl TCGen05InstrDescriptorBlockScaledAttrsObj {
    #[allow(non_snake_case)]
    pub(crate) fn new(
        d_dtype: String,
        a_dtype: String,
        b_dtype: String,
        sfa_dtype: String,
        sfb_dtype: String,
        M: i64,
        N: i64,
        K: i64,
        trans_a: bool,
        trans_b: bool,
        n_cta_groups: i64,
        neg_a: bool,
        neg_b: bool,
        is_sparse: bool,
    ) -> Self {
        let base = AttrsObj::new();
        Self {
            base,
            d_dtype,
            a_dtype,
            b_dtype,
            sfa_dtype,
            sfb_dtype,
            M,
            N,
            K,
            trans_a,
            trans_b,
            n_cta_groups,
            neg_a,
            neg_b,
            is_sparse,
        }
    }
}

impl TCGen05InstrDescriptorBlockScaledAttrs {
    /// Lossless complete-field allocation.
    #[allow(non_snake_case)]
    pub fn new(
        d_dtype: impl Into<String>,
        a_dtype: impl Into<String>,
        b_dtype: impl Into<String>,
        sfa_dtype: impl Into<String>,
        sfb_dtype: impl Into<String>,
        M: i64,
        N: i64,
        K: i64,
        trans_a: bool,
        trans_b: bool,
        n_cta_groups: i64,
        neg_a: bool,
        neg_b: bool,
        is_sparse: bool,
    ) -> Self {
        let obj = TCGen05InstrDescriptorBlockScaledAttrsObj::new(
            d_dtype.into(),
            a_dtype.into(),
            b_dtype.into(),
            sfa_dtype.into(),
            sfb_dtype.into(),
            M,
            N,
            K,
            trans_a,
            trans_b,
            n_cta_groups,
            neg_a,
            neg_b,
            is_sparse,
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TCGen05InstrDescriptorBlockScaledAttrs => Attrs);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.cuda.TensorMapEncodeTiledAttr
/// Complete: reflected fields fill [24, 80) exactly (alignment padding [28, 32)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.cuda.TensorMapEncodeTiledAttr"]
#[type_final]
pub struct TensorMapEncodeTiledAttrObj {
    base: AttrsObj,
    pub descriptor_dtype: DLDataType,
    pub rank: i64,
    pub interleave: i64,
    pub swizzle: i64,
    pub l2_promotion: i64,
    pub oob_fill: i64,
    pub force_cu_dtype: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorMapEncodeTiledAttrObj>() == 80);
    assert!(::core::mem::align_of::<TensorMapEncodeTiledAttrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorMapEncodeTiledAttr {
    base: ObjectArc<TensorMapEncodeTiledAttrObj>,
}

impl Deref for TensorMapEncodeTiledAttr {
    type Target = TensorMapEncodeTiledAttrObj;
    fn deref(&self) -> &TensorMapEncodeTiledAttrObj {
        &self.base
    }
}

impl Deref for TensorMapEncodeTiledAttrObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl TensorMapEncodeTiledAttrObj {
    pub(crate) fn new(
        descriptor_dtype: DLDataType,
        rank: i64,
        interleave: i64,
        swizzle: i64,
        l2_promotion: i64,
        oob_fill: i64,
        force_cu_dtype: i64,
    ) -> Self {
        let base = AttrsObj::new();
        Self {
            base,
            descriptor_dtype,
            rank,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
            force_cu_dtype,
        }
    }
}

impl TensorMapEncodeTiledAttr {
    /// Lossless complete-field allocation.
    pub fn new(
        descriptor_dtype: impl Into<DLDataType>,
        rank: i64,
        interleave: i64,
        swizzle: i64,
        l2_promotion: i64,
        oob_fill: i64,
        force_cu_dtype: i64,
    ) -> Self {
        let obj = TensorMapEncodeTiledAttrObj::new(
            descriptor_dtype.into(),
            rank,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
            force_cu_dtype,
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorMapEncodeTiledAttr => Attrs);
// tvm-ffi-stubgen(end)
