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

//! Functions. Every function is a [`BaseFunc`], an expression that carries the function's
//! attributes; a dialect defines the function itself, such as
//! [`Function`](crate::tirx::Function), and an [`IRModule`](super::IRModule) binds it to a name.

// tvm-ffi-stubgen(missing-docs): allow
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
use std::ptr::addr_of_mut;
use tvm_ffi::Any;
use tvm_ffi::IntoObjectRef;
use tvm_ffi::Map;
use tvm_ffi::ObjectArc;
use tvm_ffi::RValueRef;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::reflection::ShallowCopy;
use tvm_ffi::reflection::UnsafeInit;
use tvm_ffi::reflection::drop_if_defined;
// tvm-ffi-stubgen(end)

/// The base of every function: an expression, typed by the function's type, with the function
/// attributes `attrs`, such as its `global_symbol`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{BaseFunc, DictAttrs, PrimType, Return, SeqStmt, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::tirx::Function;
/// use tvm_tirx::tvm_ffi::{Map, String};
///
/// let body = SeqStmt::new([Return::new(0, UNKNOWN_LOC)], UNKNOWN_LOC);
/// let attrs = DictAttrs::new(Map::new());
/// let func = Function::new(Vec::<Var>::new(), &body, PrimType::int(32), attrs, UNKNOWN_LOC);
/// let func = BaseFunc::from(func);
/// assert!(script(&func)?.contains("@T.function(private=True)"));
///
/// let exported = func.with_attr("global_symbol", String::from("main"))?;
/// assert!(script(&exported)?.contains("@T.function\ndef main()"));
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.BaseFunc
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BaseFunc {
    data_: ObjectArc<BaseFuncObj>,
}

/// The object behind [`BaseFunc`].
///
/// Complete: reflected fields fill [40, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.BaseFunc"]
pub struct BaseFuncObj {
    base_: ExprObj,
    pub attrs: DictAttrs,
}

const _: () = {
    assert!(::core::mem::size_of::<BaseFuncObj>() == 48);
    assert!(::core::mem::align_of::<BaseFuncObj>() == 8);
};

impl Deref for BaseFuncObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BaseFuncObj {
    pub fn new(loc: Location, ty: Type, attrs: DictAttrs) -> Self {
        let base_ = ExprObj::new(loc, ty);
        Self { base_, attrs }
    }
}

#[allow(missing_docs)]
impl BaseFunc {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl IntoObjectRef<Location>,
        ty: impl IntoObjectRef<Type>,
        attrs: impl IntoObjectRef<DictAttrs>,
    ) -> Self {
        let obj = BaseFuncObj::new(
            loc.into_owned_object_ref(),
            ty.into_owned_object_ref(),
            attrs.into_owned_object_ref(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for BaseFuncObj {
    fn shallow_copy(&self) -> Self {
        Self { base_: ShallowCopy::shallow_copy(&self.base_), attrs: Clone::clone(&self.attrs) }
    }
}

unsafe impl UnsafeInit for BaseFuncObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base_));
            drop_if_defined(addr_of_mut!((*this).attrs));
        }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BaseFuncObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(BaseFunc => Expr);
// tvm-ffi-stubgen(end)

impl BaseFunc {
    /// Return a new handle to the same function, as the native side returns it (`ir.BaseFuncCopy`).
    pub fn copy(&self) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncCopy").call_tuple((self,))?.try_into()
    }

    /// Return the function with the attribute `key` set to `value` (`ir.BaseFuncWithAttr`).
    ///
    /// Consumes `self`: the function is updated in place when this was its only handle, and copied
    /// otherwise.
    pub fn with_attr(self, key: &str, value: impl Into<Any>) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncWithAttr")
            .call_tuple((RValueRef::new(self), String::from(key), value.into()))?
            .try_into()
    }

    /// Return the function with the attributes of `attrs` set (`ir.BaseFuncWithAttrs`).
    ///
    /// Consumes `self`, as [`with_attr`](Self::with_attr) does.
    pub fn with_attrs(self, attrs: impl Into<Map<String, Any>>) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncWithAttrs")
            .call_tuple((RValueRef::new(self), &attrs.into()))?
            .try_into()
    }

    /// Return the function without the attribute `key` (`ir.BaseFuncWithoutAttr`).
    ///
    /// Consumes `self`, as [`with_attr`](Self::with_attr) does.
    pub fn without_attr(self, key: &str) -> Result<BaseFunc> {
        tvm_ffi::cached_global_func!("ir.BaseFuncWithoutAttr")
            .call_tuple((RValueRef::new(self), String::from(key)))?
            .try_into()
    }
}
