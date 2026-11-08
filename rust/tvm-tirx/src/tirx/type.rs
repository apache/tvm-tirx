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

//! Types of the dialect beyond tensors. A [`TensorMapType`] types the tensor-map descriptors that
//! backend memory operations take; tensors are typed by [`TensorType`](super::TensorType).

// tvm-ffi-stubgen(missing-docs): allow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::location::Location;
use crate::ir::r#type::Type;
use crate::ir::r#type::TypeObj;
use std::ops::Deref;
use tvm_ffi::IntoObjectRef;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

/// The type of a tensor-map descriptor, which backend memory operations take.
///
/// A pointer to it points at the descriptor storage, not at tensor data; the operation that
/// creates a descriptor defines its shape and layout.
// tvm-ffi-stubgen(begin): object/tirx.TensorMapType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorMapType {
    data_: ObjectArc<TensorMapTypeObj>,
}

/// The object behind [`TensorMapType`].
///
/// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TensorMapType"]
#[type_final]
#[type_immutable]
pub struct TensorMapTypeObj {
    base_: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorMapTypeObj>() == 32);
    assert!(::core::mem::align_of::<TensorMapTypeObj>() == 8);
};

impl Deref for TensorMapTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl TensorMapTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl TensorMapType {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl IntoObjectRef<Location>) -> Self {
        let obj = TensorMapTypeObj::new(loc.into_owned_object_ref());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(TensorMapType => Type);
// tvm-ffi-stubgen(end)

// The native `tirx.TensorMapType` stores the location as given, as the generated
// `TensorMapType::new` does.
