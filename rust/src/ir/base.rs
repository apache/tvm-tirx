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

//! Functions of the global registry referred to by name.
//!
//! The objects `python/tvm_tirx/ir/base.py` defines. An [`EnvFunc`] is a registered global
//! function kept with its name, so it is serialized by name.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::EnvFunc;
//! use tvm_tirx::tvm_ffi::{Function, Result};
//!
//! # let double = Function::from_typed(|x: i64| -> Result<i64> { Ok(x * 2) });
//! # Function::register_global("my_ext.double", double)?;
//! let double = EnvFunc::get("my_ext.double")?;
//!
//! assert_eq!(i64::try_from(double.func.call_tuple((21_i64,))?)?, 42);
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result

// tvm-ffi-stubgen(begin): import-section
use tvm_ffi::Function;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.EnvFunc
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.EnvFunc"]
#[type_final]
pub struct EnvFuncObj {
    base: Object,
    pub name: String,
    pub func: Function,
}

const _: () = {
    assert!(::core::mem::size_of::<EnvFuncObj>() == 48);
    assert!(::core::mem::align_of::<EnvFuncObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EnvFunc {
    base: ObjectArc<EnvFuncObj>,
}

impl EnvFuncObj {
    pub(crate) fn new(name: String, func: Function) -> Self {
        let base = Object::new();
        Self { base, name, func }
    }
}

impl EnvFunc {
    /// Lossless complete-field allocation.
    pub fn new(name: impl Into<String>, func: impl Into<Function>) -> Self {
        let obj = EnvFuncObj::new(name.into(), func.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for EnvFuncObj {}
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
