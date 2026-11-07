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

//! Types of TIRx operations.
//!
//! [`TensorMapType`] is the type of a tensor map.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::UNKNOWN_LOC;
//! use tvm_tirx::script::script;
//! use tvm_tirx::tirx::TensorMapType;
//!
//! let ty = TensorMapType::new(UNKNOWN_LOC);
//! assert_eq!(script(&ty)?, "T.TensorMap()");
//!
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// tvm-ffi-stubgen(missing-docs): allow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::location::Location;
use crate::ir::r#type::Type;
use crate::ir::r#type::TypeObj;
use std::ops::Deref;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.TensorMapType
/// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.TensorMapType"]
#[type_final]
pub struct TensorMapTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorMapTypeObj>() == 32);
    assert!(::core::mem::align_of::<TensorMapTypeObj>() == 8);
};

#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorMapType {
    base: ObjectArc<TensorMapTypeObj>,
}

impl Deref for TensorMapTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

#[allow(missing_docs)]
impl TensorMapTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = TypeObj::new(loc);
        Self { base }
    }
}

#[allow(missing_docs)]
impl TensorMapType {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = TensorMapTypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for TensorMapTypeObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(TensorMapType => Type);
// tvm-ffi-stubgen(end)

// The native `tirx.TensorMapType` stores the location as given, as the generated
// `TensorMapType::new` does.
