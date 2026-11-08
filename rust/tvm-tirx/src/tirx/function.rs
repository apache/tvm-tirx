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

//! Functions of the dialect. A [`Function`] is a [`BaseFunc`] whose type is derived from its
//! parameters and return type, with a [`SeqStmt`] body; [`renew_def`] copies one with fresh
//! variables.

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// `custom_new` reserves `new` for a hand-written constructor (`ty` derived from the signature); the
// generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(object): tirx.Function(custom_new)

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import): tvm_ffi::Result
// tvm-ffi-stubgen(import): crate::ir::r#type::FuncType
// tvm-ffi-stubgen(import): crate::ir::r#type::MissingTypeObj
// tvm-ffi-stubgen(import): crate::ir::r#type::TupleType
// tvm-ffi-stubgen(import): crate::ir::location::UNKNOWN_LOC
// tvm-ffi-stubgen(import): tvm_ffi::IntoArg
// tvm-ffi-stubgen(import): std::borrow::Borrow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::DictAttrs;
use crate::ir::expr::Expr;
use crate::ir::expr::Var;
use crate::ir::function::BaseFunc;
use crate::ir::function::BaseFuncObj;
use crate::ir::location::Location;
use crate::ir::location::UNKNOWN_LOC;
use crate::ir::r#type::FuncType;
use crate::ir::r#type::MissingTypeObj;
use crate::ir::r#type::TupleType;
use crate::ir::r#type::Type;
use crate::ir::stmt::SeqStmt;
use std::borrow::Borrow;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::IntoArg;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

/// A function: `params`, a `body` and a `ret_type`, typed by the [`FuncType`] of its parameter
/// and return types. A declaration has an empty body and the `declaration` attribute set.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{DictAttrs, PrimType, PrimVar, Return, SeqStmt, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::tirx::{renew_def, Function};
/// use tvm_tirx::tvm_ffi::{Map, ObjectRefCore};
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let body = SeqStmt::new([Return::new(&x, UNKNOWN_LOC)], UNKNOWN_LOC);
/// let attrs = DictAttrs::new(Map::new());
/// let identity = Function::new([&x], &body, PrimType::int(32), attrs, UNKNOWN_LOC);
/// assert!(script(&identity)?.as_str().contains("def main(x: T.int32) -> T.int32:"));
///
/// let copy = renew_def(&identity)?;
/// assert!(!copy.params[0].same_as(&x));
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/tirx.Function
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Function {
    data_: ObjectArc<FunctionObj>,
}

/// The object behind [`Function`].
// Complete: reflected fields fill [48, 72) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.Function", final, immutable)]
pub struct FunctionObj {
    base_: BaseFuncObj,
    pub params: Array<Var>,
    pub ret_type: Type,
    pub body: SeqStmt,
}

const _: () = {
    assert!(::core::mem::size_of::<FunctionObj>() == 72);
    assert!(::core::mem::align_of::<FunctionObj>() == 8);
};

impl Deref for FunctionObj {
    type Target = BaseFuncObj;
    fn deref(&self) -> &BaseFuncObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl FunctionObj {
    pub(crate) fn new(
        loc: Location,
        ty: Type,
        attrs: DictAttrs,
        params: Array<Var>,
        ret_type: Type,
        body: SeqStmt,
    ) -> Self {
        let base_ = BaseFuncObj::new(loc, ty, attrs);
        Self { base_, params, ret_type, body }
    }
}

#[allow(missing_docs)]
impl Function {
    /// Create a `Function` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<Type>,
        attrs: impl IntoArg<DictAttrs>,
        params: impl Into<Array<Var>>,
        ret_type: impl IntoArg<Type>,
        body: impl IntoArg<SeqStmt>,
    ) -> Self {
        let obj = FunctionObj::new(
            loc.into_owned(),
            ty.into_owned(),
            attrs.into_owned(),
            params.into(),
            ret_type.into_owned(),
            body.into_owned(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Function => Expr, Function => BaseFunc);
// tvm-ffi-stubgen(end)

impl Function {
    /// Create the function of `params`, `body`, `ret_type` and `attrs`, typed by the function
    /// type of its signature; a missing return type becomes void, the empty tuple type.
    pub fn new(
        params: impl IntoIterator<Item = impl IntoArg<Var>>,
        body: impl IntoArg<SeqStmt>,
        ret_type: impl IntoArg<Type>,
        attrs: impl IntoArg<DictAttrs>,
        loc: impl IntoArg<Location>,
    ) -> Self {
        let params = params.into_iter().map(IntoArg::into_owned).collect::<Array<Var>>();
        let ret_type = ret_type.into_owned();
        let ret_type = match ret_type.downcast_ref::<MissingTypeObj>() {
            Some(_) => TupleType::new(Vec::<Type>::new(), UNKNOWN_LOC).into(),
            None => ret_type,
        };
        let param_types = params.iter().map(|param| param.ty.clone()).collect::<Vec<Type>>();
        let ty = FuncType::new(param_types, ret_type.clone());
        Self::from_fields(
            loc.into_owned(),
            ty,
            attrs.into_owned(),
            params,
            ret_type,
            body.into_owned(),
        )
    }
}

/// Return `func` with fresh variable definitions, their uses remapped (`tirx.RenewDef`).
pub fn renew_def(func: impl IntoArg<Function>) -> Result<Function> {
    tvm_ffi::cached_global_func!("tirx.RenewDef")
        .call_tuple((func.into_arg().borrow(),))?
        .try_into()
}
