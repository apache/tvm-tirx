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

//! Attributes attached to calls, functions and modules.
//!
//! The objects `python/tvm_tirx/ir/attrs.py` defines. [`Attrs`] is the base of every attribute
//! object, and [`DictAttrs`] holds named values in a map.

// tvm-ffi-stubgen(begin): import-section
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ShallowCopy;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Attrs
/// Complete: reflected fields fill [24, 24) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Attrs"]
pub struct AttrsObj {
    base: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<AttrsObj>() == 24);
    assert!(::core::mem::align_of::<AttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Attrs {
    base: ObjectArc<AttrsObj>,
}

impl Deref for Attrs {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl AttrsObj {
    pub fn new() -> Self {
        let base = Object::new();
        Self { base }
    }
}

impl Attrs {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = AttrsObj::new();
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for AttrsObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.DictAttrs
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.DictAttrs"]
#[type_final]
pub struct DictAttrsObj {
    base: AttrsObj,
    pub dict: Map<String, Any>,
}

const _: () = {
    assert!(::core::mem::size_of::<DictAttrsObj>() == 32);
    assert!(::core::mem::align_of::<DictAttrsObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DictAttrs {
    base: ObjectArc<DictAttrsObj>,
}

impl Deref for DictAttrs {
    type Target = DictAttrsObj;
    fn deref(&self) -> &DictAttrsObj {
        &self.base
    }
}

impl Deref for DictAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl DictAttrsObj {
    pub(crate) fn new(dict: Map<String, Any>) -> Self {
        let base = AttrsObj::new();
        Self { base, dict }
    }
}

impl DictAttrs {
    /// Lossless complete-field allocation.
    pub fn new(dict: impl Into<Map<String, Any>>) -> Self {
        let obj = DictAttrsObj::new(dict.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(DictAttrs => Attrs);
// tvm-ffi-stubgen(end)
