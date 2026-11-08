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

//! Registry functions held by name. An [`EnvFunc`] keeps a registered global function with its
//! name, so IR that refers to it serializes the name and resolves it again when loaded.

// tvm-ffi-stubgen(missing-docs): allow
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result

// tvm-ffi-stubgen(begin): import-section
use tvm_ffi::Function;
use tvm_ffi::IntoObjectRef;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

/// A registered global function `func` kept with its registry name `name`.
///
/// The name identifies it: it is serialized by name, loading looks the name up again, and
/// structural equality compares the name and ignores `func`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::EnvFunc;
/// use tvm_tirx::tvm_ffi::{Function, Result};
///
/// # let double = Function::from_typed(|x: i64| -> Result<i64> { Ok(x * 2) });
/// # Function::register_global("my_ext.double", double)?;
/// let double = EnvFunc::get("my_ext.double")?;
///
/// assert_eq!(i64::try_from(double.func.call_tuple((21_i64,))?)?, 42);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.EnvFunc
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EnvFunc {
    data_: ObjectArc<EnvFuncObj>,
}

/// The object behind [`EnvFunc`].
///
/// Complete: reflected fields fill [24, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.EnvFunc"]
#[type_final]
#[type_immutable]
pub struct EnvFuncObj {
    base_: Object,
    pub name: String,
    pub func: Function,
}

const _: () = {
    assert!(::core::mem::size_of::<EnvFuncObj>() == 48);
    assert!(::core::mem::align_of::<EnvFuncObj>() == 8);
};

#[allow(missing_docs)]
impl EnvFuncObj {
    pub(crate) fn new(name: String, func: Function) -> Self {
        let base_ = Object::new();
        Self { base_, name, func }
    }
}

#[allow(missing_docs)]
impl EnvFunc {
    /// Lossless complete-field allocation.
    pub fn new(name: impl Into<String>, func: impl IntoObjectRef<Function>) -> Self {
        let obj = EnvFuncObj::new(name.into(), func.into_owned_object_ref());
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl EnvFunc {
    /// Return the global function `name` kept with its name (`ir.EnvFuncGet`).
    ///
    /// # Errors
    ///
    /// Returns an error if no global function is registered as `name`.
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.EnvFuncGet").call_tuple((String::from(name),))?.try_into()
    }
}
