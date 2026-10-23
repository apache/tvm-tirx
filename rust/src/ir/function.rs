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

//! The base of every function.
//!
//! The objects `python/tvm_tirx/ir/function.py` defines. [`BaseFunc`] carries the type, location
//! and attributes of a function; a dialect defines the function itself, such as
//! [`Function`](crate::tirx::Function). Its attribute methods wrap the native `ir.BaseFunc*`
//! functions.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): tvm_ffi::RValueRef
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::location::Location;
use crate::ir::r#type::Type;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::Map;
use tvm_ffi::ObjectArc;
use tvm_ffi::RValueRef;
use tvm_ffi::Result;
use tvm_ffi::ShallowCopy;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.BaseFunc
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.BaseFunc"]
pub struct BaseFuncObj {
    base: ExprObj,
    pub attrs: DictAttrs,
}

const _: () = {
    assert!(::core::mem::size_of::<BaseFuncObj>() == 48);
    assert!(::core::mem::align_of::<BaseFuncObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BaseFunc {
    base: ObjectArc<BaseFuncObj>,
}

impl Deref for BaseFunc {
    type Target = BaseFuncObj;
    fn deref(&self) -> &BaseFuncObj {
        &self.base
    }
}

impl Deref for BaseFuncObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BaseFuncObj {
    pub fn new(loc: Location, ty: Type, attrs: DictAttrs) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, attrs }
    }
}

impl BaseFunc {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>, attrs: impl Into<DictAttrs>) -> Self {
        let obj = BaseFuncObj::new(loc.into(), ty.into(), attrs.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for BaseFuncObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base), attrs: Clone::clone(&self.attrs) }
    }
}

tvm_ffi::impl_object_upcast!(BaseFunc => Expr);
// tvm-ffi-stubgen(end)

impl BaseFunc {
    /// A new handle to the same function, as the native side returns it (`ir.BaseFuncCopy`).
    pub fn copy(&self) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncCopy").call_tuple((self,))?.try_into()
    }

    /// The function with the attribute `key` set to `value` (`ir.BaseFuncWithAttr`).
    ///
    /// Consumes `self`: the function is updated in place when this was its only handle, and copied
    /// otherwise.
    pub fn with_attr(self, key: &str, value: impl Into<Any>) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncWithAttr")
            .call_tuple((RValueRef::new(self), String::from(key), value.into()))?
            .try_into()
    }

    /// The function with the attributes of `attrs` set (`ir.BaseFuncWithAttrs`).
    ///
    /// Consumes `self`, as [`BaseFunc::with_attr`] does.
    pub fn with_attrs(self, attrs: &Map<String, Any>) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncWithAttrs")
            .call_tuple((RValueRef::new(self), attrs))?
            .try_into()
    }

    /// The function without the attribute `key` (`ir.BaseFuncWithoutAttr`).
    ///
    /// Consumes `self`, as [`BaseFunc::with_attr`] does.
    pub fn without_attr(self, key: &str) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncWithoutAttr")
            .call_tuple((RValueRef::new(self), String::from(key)))?
            .try_into()
    }
}
