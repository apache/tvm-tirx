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

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// `no_new` generates no allocator for the abstract base: a value is one of its subtypes.
// tvm-ffi-stubgen(object): ir.Attrs(no_new)

// tvm-ffi-stubgen(begin): import-section
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::String;
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
// Complete: reflected fields fill [24, 24) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.Attrs", immutable, rs_parent)]
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
/// let module = IRModule::new(Map::new(), DictAttrs::new(dict), Map::new())?;
///
/// let prefix = module.attrs.dict.get("system_lib_prefix")?.unwrap();
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
// Complete: reflected fields fill [24, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.DictAttrs", final, immutable)]
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
    /// Create a `DictAttrs` from its field values.
    pub fn new(dict: impl Into<Map<String, Any>>) -> Self {
        let obj = DictAttrsObj::new(dict.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(DictAttrs => Attrs);
// tvm-ffi-stubgen(end)
