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

//! Tensor functions.
//!
//! The objects `python/tvm_tirx/tirx/function.py` defines. A [`Function`] is a
//! [`BaseFunc`] with parameters, a return type and an optional body, built with [`Function::new`];
//! [`renew_def`] gives its variables fresh definitions.

// `custom-new` reserves `new` for a constructor with the C++ one's semantics (`ty` derived from the
// signature); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): tirx.Function

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::r#type::FuncType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::MissingTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::TupleType
// tvm-ffi-stubgen(import-object): crate::ir::location::UnknownLoc

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Expr;
use crate::ir::expr::Var;
use crate::ir::function::BaseFunc;
use crate::ir::function::BaseFuncObj;
use crate::ir::location::Location;
use crate::ir::location::UnknownLoc;
use crate::ir::r#type::FuncType;
use crate::ir::r#type::MissingTypeObj;
use crate::ir::r#type::TupleType;
use crate::ir::r#type::Type;
use crate::ir::stmt::SeqStmt;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.Function
/// Complete: reflected fields fill [48, 72) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.Function"]
#[type_final]
pub struct FunctionObj {
    base: BaseFuncObj,
    pub params: Array<Var>,
    pub ret_type: Type,
    pub body: Option<SeqStmt>,
}

const _: () = {
    assert!(::core::mem::size_of::<FunctionObj>() == 72);
    assert!(::core::mem::align_of::<FunctionObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Function {
    base: ObjectArc<FunctionObj>,
}

impl Deref for Function {
    type Target = FunctionObj;
    fn deref(&self) -> &FunctionObj {
        &self.base
    }
}

impl Deref for FunctionObj {
    type Target = BaseFuncObj;
    fn deref(&self) -> &BaseFuncObj {
        &self.base
    }
}

impl FunctionObj {
    pub(crate) fn new(
        loc: Location,
        ty: Type,
        attrs: DictAttrs,
        params: Array<Var>,
        ret_type: Type,
        body: Option<SeqStmt>,
    ) -> Self {
        let base = BaseFuncObj::new(loc, ty, attrs);
        Self { base, params, ret_type, body }
    }
}

impl Function {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        attrs: impl Into<DictAttrs>,
        params: impl Into<Array<Var>>,
        ret_type: impl Into<Type>,
        body: impl Into<Option<SeqStmt>>,
    ) -> Self {
        let obj = FunctionObj::new(
            loc.into(),
            ty.into(),
            attrs.into(),
            params.into(),
            ret_type.into(),
            body.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Function => Expr, Function => BaseFunc);
// tvm-ffi-stubgen(end)

impl Function {
    /// The function of `params`, `body`, `ret_type` and `attrs`, of the function type from the
    /// types of `params` to `ret_type`; a missing return type becomes void, the empty tuple type
    /// (C++ `Function(params, body, ret_type, attrs, loc)`).
    pub fn new(
        params: impl IntoIterator<Item = impl Into<Var>>,
        body: Option<&SeqStmt>,
        ret_type: impl Into<Type>,
        attrs: impl Into<DictAttrs>,
        loc: impl Into<Location>,
    ) -> Self {
        let params = params.into_iter().map(Into::into).collect::<Array<Var>>();
        let ret_type: Type = ret_type.into();
        let ret_type = match ret_type.as_node::<MissingTypeObj>() {
            Some(_) => TupleType::new(Vec::<Type>::new(), UnknownLoc::new()).into(),
            None => ret_type,
        };
        let param_types = params.iter().map(|param| param.ty.clone()).collect::<Vec<Type>>();
        let ty = FuncType::new(param_types, ret_type.clone());
        Self::from_fields(loc.into(), ty, attrs.into(), params, ret_type, body.cloned())
    }
}

/// `func` with fresh variable definitions, their uses remapped (`tirx.RenewDef`).
pub fn renew_def(func: impl Into<Function>) -> Result<Function> {
    tvm_ffi::cached_global_func!("tirx.RenewDef").call_tuple((&func.into(),))?.try_into()
}
