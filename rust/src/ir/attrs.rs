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
// The generated `new` of a field-less base needs no `Default`.
#![allow(clippy::new_without_default)]

//! Attributes. Calls, functions and modules carry metadata as [`Attrs`]: a typed attribute object
//! of an operator, or a [`DictAttrs`] map of named values.

// tvm-ffi-stubgen(missing-docs): allow

// tvm-ffi-stubgen(begin): import-section
use std::ops::Deref;
use std::ptr::addr_of_mut;
use tvm_ffi::Any;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::String;
use tvm_ffi::reflection::ShallowCopy;
use tvm_ffi::reflection::UnsafeInit;
// tvm-ffi-stubgen(end)

/// The base of every attribute object, the metadata attached to a call, a function or a module.
// tvm-ffi-stubgen(begin): object/ir.Attrs
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Attrs {
    data_: ObjectArc<AttrsObj>,
}

/// The object behind [`Attrs`].
///
/// Complete: reflected fields fill [24, 24) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Attrs"]
pub struct AttrsObj {
    base_: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<AttrsObj>() == 24);
    assert!(::core::mem::align_of::<AttrsObj>() == 8);
};

#[allow(missing_docs)]
impl AttrsObj {
    pub fn new() -> Self {
        let base_ = Object::new();
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl Attrs {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = AttrsObj::new();
        Self { data_: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for AttrsObj {
    fn shallow_copy(&self) -> Self {
        Self { base_: ShallowCopy::shallow_copy(&self.base_) }
    }
}

unsafe impl UnsafeInit for AttrsObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base_));
        }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for AttrsObj {}
// tvm-ffi-stubgen(end)

/// Attributes as a map `dict` from names to values, such as the attributes of a function or a
/// module.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{DictAttrs, IRModule};
/// use tvm_tirx::tvm_ffi::{Any, Map, String};
///
/// let dict: Map<String, Any> =
///     [("system_lib_prefix".into(), Any::from(String::from("my_lib_")))].into_iter().collect();
/// let module = IRModule::new(&Map::new(), DictAttrs::new(dict), &Map::new())?;
///
/// let prefix = module.attrs.dict.get(&"system_lib_prefix".into())?.unwrap();
/// assert_eq!(String::try_from(prefix)?, "my_lib_");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.DictAttrs
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DictAttrs {
    data_: ObjectArc<DictAttrsObj>,
}

/// The object behind [`DictAttrs`].
///
/// Complete: reflected fields fill [24, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.DictAttrs"]
#[type_final]
pub struct DictAttrsObj {
    base_: AttrsObj,
    pub dict: Map<String, Any>,
}

const _: () = {
    assert!(::core::mem::size_of::<DictAttrsObj>() == 32);
    assert!(::core::mem::align_of::<DictAttrsObj>() == 8);
};

impl Deref for DictAttrsObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl DictAttrsObj {
    pub(crate) fn new(dict: Map<String, Any>) -> Self {
        let base_ = AttrsObj::new();
        Self { base_, dict }
    }
}

#[allow(missing_docs)]
impl DictAttrs {
    /// Lossless complete-field allocation.
    pub fn new(dict: impl Into<Map<String, Any>>) -> Self {
        let obj = DictAttrsObj::new(dict.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for DictAttrsObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(DictAttrs => Attrs);
// tvm-ffi-stubgen(end)
