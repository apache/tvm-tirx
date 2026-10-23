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

//! Expressions: variables, constants, calls, tuples and tensor accesses.
//!
//! The objects `python/tvm_tirx/ir/expr.py` defines. [`Expr`] is the base of every expression and
//! carries its type and location. The primitive arithmetic and vector expressions are in
//! [`prim`](super::prim). The constructors are `new`, or `from_dtype` and the like where `new`
//! allocates the fields as given; they are written in Rust after the C++ constructors, with their
//! checks, defaults and errors. The constants of the common types are [`IntImm::int32`],
//! [`IntImm::int64`], [`IntImm::bool`], [`FloatImm::float32`] and [`FloatImm::float64`], and an
//! `i32`, `i64`, `bool`, `f32` or `f64` converts into an [`Expr`] as C++ converts it implicitly.

// `custom-new` reserves `new` for a constructor with the C++ one's semantics (defaults, implicit
// conversions, a derived type); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): ir.GlobalVar
// tvm-ffi-stubgen(custom-new): ir.Tuple
// tvm-ffi-stubgen(custom-new): ir.TensorLoad
// tvm-ffi-stubgen(custom-new): ir.StringImm
// tvm-ffi-stubgen(custom-new): ir.Call
// tvm-ffi-stubgen(custom-new): ir.Var
// tvm-ffi-stubgen(custom-new): ir.LambdaExpr
// tvm-ffi-stubgen(custom-new): ir.Range
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex
// tvm-ffi-stubgen(import-object): crate::ir::location::UnknownLoc
// tvm-ffi-stubgen(import-object): crate::ir::op::OpObj
// tvm-ffi-stubgen(import-object): crate::ir::prim::AddObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::AnyType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::FuncType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::FuncTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::MissingType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::MissingTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::StringType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::TupleType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::TupleTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::dtype

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::dtype;
use crate::ir::location::Location;
use crate::ir::location::UnknownLoc;
use crate::ir::op::OpObj;
use crate::ir::prim::AddObj;
use crate::ir::r#type::AnyType;
use crate::ir::r#type::FuncType;
use crate::ir::r#type::FuncTypeObj;
use crate::ir::r#type::MissingType;
use crate::ir::r#type::MissingTypeObj;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::StringType;
use crate::ir::r#type::TupleType;
use crate::ir::r#type::TupleTypeObj;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::BigInt;
use tvm_ffi::DLDataType;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::ShallowCopy;
use tvm_ffi::String;
use tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Expr
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: Object,
    pub loc: Location,
    pub ty: Type,
}

const _: () = {
    assert!(::core::mem::size_of::<ExprObj>() == 40);
    assert!(::core::mem::align_of::<ExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Expr {
    base: ObjectArc<ExprObj>,
}

impl Deref for Expr {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ExprObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = Object::new();
        Self { base, loc, ty }
    }
}

impl Expr {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = ExprObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for ExprObj {
    fn shallow_copy(&self) -> Self {
        Self {
            base: ShallowCopy::shallow_copy(&self.base),
            loc: Clone::clone(&self.loc),
            ty: Clone::clone(&self.ty),
        }
    }
}
// tvm-ffi-stubgen(end)

/// One index of [`Expr::subscript`]: an expression, or a `start:stop:step` slice whose missing
/// bounds are `None`.
#[derive(Clone)]
pub enum Subscript {
    Index(Expr),
    Slice(Option<Expr>, Option<Expr>, Option<Expr>),
}

impl From<&Subscript> for Any {
    fn from(subscript: &Subscript) -> Self {
        match subscript {
            Subscript::Index(index) => Any::from(index.clone()),
            Subscript::Slice(start, stop, step) => Any::from(
                [start, stop, step]
                    .into_iter()
                    .map(|bound| Any::from(bound.clone()))
                    .collect::<Array<Any>>(),
            ),
        }
    }
}

impl Expr {
    /// `self[indices]`, as the subscript rule of its type realizes it (`ir.SubscriptExprRealize`).
    pub fn subscript(
        &self,
        indices: impl IntoIterator<Item = Subscript>,
        loc: impl Into<Location>,
    ) -> Result<Any> {
        let indices = indices.into_iter().map(|index| Any::from(&index)).collect::<Array<Any>>();
        tvm_ffi::cached_global_func!("ir.SubscriptExprRealize").call_tuple((
            self,
            &indices,
            &loc.into(),
        ))
    }
}

// tvm-ffi-stubgen(begin): object/ir.StagingExpr
/// Complete: reflected fields fill [40, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.StagingExpr"]
pub struct StagingExprObj {
    base: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<StagingExprObj>() == 40);
    assert!(::core::mem::align_of::<StagingExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StagingExpr {
    base: ObjectArc<StagingExprObj>,
}

impl Deref for StagingExpr {
    type Target = StagingExprObj;
    fn deref(&self) -> &StagingExprObj {
        &self.base
    }
}

impl Deref for StagingExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl StagingExprObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl StagingExpr {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = StagingExprObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for StagingExprObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}

tvm_ffi::impl_object_upcast!(StagingExpr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.OpaqueExpr
/// Complete: reflected fields fill [40, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.OpaqueExpr"]
pub struct OpaqueExprObj {
    base: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<OpaqueExprObj>() == 40);
    assert!(::core::mem::align_of::<OpaqueExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct OpaqueExpr {
    base: ObjectArc<OpaqueExprObj>,
}

impl Deref for OpaqueExpr {
    type Target = OpaqueExprObj;
    fn deref(&self) -> &OpaqueExprObj {
        &self.base
    }
}

impl Deref for OpaqueExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OpaqueExprObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl OpaqueExpr {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = OpaqueExprObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for OpaqueExprObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}

tvm_ffi::impl_object_upcast!(OpaqueExpr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.GlobalVar
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GlobalVar"]
#[type_final]
pub struct GlobalVarObj {
    base: ExprObj,
    pub name_hint: String,
}

const _: () = {
    assert!(::core::mem::size_of::<GlobalVarObj>() == 56);
    assert!(::core::mem::align_of::<GlobalVarObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GlobalVar {
    base: ObjectArc<GlobalVarObj>,
}

impl Deref for GlobalVar {
    type Target = GlobalVarObj;
    fn deref(&self) -> &GlobalVarObj {
        &self.base
    }
}

impl Deref for GlobalVarObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GlobalVarObj {
    pub(crate) fn new(loc: Location, ty: Type, name_hint: String) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, name_hint }
    }
}

impl GlobalVar {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        name_hint: impl Into<String>,
    ) -> Self {
        let obj = GlobalVarObj::new(loc.into(), ty.into(), name_hint.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GlobalVar => Expr);
// tvm-ffi-stubgen(end)

impl GlobalVar {
    /// The global variable `name_hint`, of a missing type (C++ `GlobalVar(name_hint, loc)`).
    pub fn new(name_hint: &str, loc: impl Into<Location>) -> Self {
        let ty = MissingType::new();
        Self::from_fields(loc.into(), ty, String::from(name_hint))
    }
}

// tvm-ffi-stubgen(begin): object/ir.Tuple
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
    pub fields: Array<Expr>,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleObj>() == 48);
    assert!(::core::mem::align_of::<TupleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Tuple {
    base: ObjectArc<TupleObj>,
}

impl Deref for Tuple {
    type Target = TupleObj;
    fn deref(&self) -> &TupleObj {
        &self.base
    }
}

impl Deref for TupleObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TupleObj {
    pub(crate) fn new(loc: Location, ty: Type, fields: Array<Expr>) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, fields }
    }
}

impl Tuple {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        fields: impl Into<Array<Expr>>,
    ) -> Self {
        let obj = TupleObj::new(loc.into(), ty.into(), fields.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Tuple => Expr);
// tvm-ffi-stubgen(end)

impl Tuple {
    /// The tuple of `fields`, typed by the types of its fields when they are all known
    /// (C++ `Tuple(fields, loc)`).
    pub fn new(
        fields: impl IntoIterator<Item = impl Into<Expr>>,
        loc: impl Into<Location>,
    ) -> Self {
        let fields = fields.into_iter().map(Into::into).collect::<Array<Expr>>();
        let ty: Type =
            match fields.iter().any(|field| field.ty.as_node::<MissingTypeObj>().is_some()) {
                true => MissingType::new().into(),
                false => {
                    let field_types = fields.iter().map(|field| field.ty.clone());
                    TupleType::new(field_types, UnknownLoc::new()).into()
                }
            };
        Self::from_fields(loc.into(), ty, fields)
    }
}

// tvm-ffi-stubgen(begin): object/ir.TupleGetItem
/// Complete: reflected fields fill [40, 56) exactly (alignment padding [52, 56)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TupleGetItem"]
#[type_final]
pub struct TupleGetItemObj {
    base: ExprObj,
    pub tuple_value: Expr,
    pub index: i32,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleGetItemObj>() == 56);
    assert!(::core::mem::align_of::<TupleGetItemObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TupleGetItem {
    base: ObjectArc<TupleGetItemObj>,
}

impl Deref for TupleGetItem {
    type Target = TupleGetItemObj;
    fn deref(&self) -> &TupleGetItemObj {
        &self.base
    }
}

impl Deref for TupleGetItemObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TupleGetItemObj {
    pub(crate) fn new(loc: Location, ty: Type, tuple_value: Expr, index: i32) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, tuple_value, index }
    }
}

impl TupleGetItem {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        tuple_value: impl Into<Expr>,
        index: i32,
    ) -> Self {
        let obj = TupleGetItemObj::new(loc.into(), ty.into(), tuple_value.into(), index);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TupleGetItem => Expr);
// tvm-ffi-stubgen(end)

impl TupleGetItem {
    /// Field `index` of `tuple_value`, typed by the field type and checked against the tuple type
    /// when it is known (C++ `TupleGetItem(tuple, index, loc)`).
    pub fn from_tuple(
        tuple_value: impl Into<Expr>,
        index: i32,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let tuple_value = tuple_value.into();
        if index < 0 {
            let message = format!(
                "Index out of bounds: Tuple {} cannot be accessed with negative index {index}",
                support::repr(&tuple_value)
            );
            return Err(support::compare_failed(
                tvm_ffi::INDEX_ERROR,
                "index >= 0",
                index,
                0,
                &message,
            ));
        }
        let ty: Type = match tuple_value.ty.as_node::<TupleTypeObj>() {
            Some(tuple_type) => {
                let size = tuple_type.fields.len();
                if index as usize >= size {
                    let message = format!(
                        "Index out of bounds: Tuple {} is of size {size}, and cannot be accessed \
                         with index {index}",
                        support::repr(&tuple_value)
                    );
                    let condition = "index < tuple_type->fields.size()";
                    let error = support::compare_failed(
                        tvm_ffi::INDEX_ERROR,
                        condition,
                        index,
                        size,
                        &message,
                    );
                    return Err(error);
                }
                tuple_type.fields.get(index as usize)?
            }
            None => MissingType::new().into(),
        };
        Ok(Self::new(loc.into(), ty, tuple_value, index))
    }
}

// tvm-ffi-stubgen(begin): object/ir.TensorLoad
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorLoad"]
#[type_final]
pub struct TensorLoadObj {
    base: ExprObj,
    pub source: Expr,
    pub indices: Array<Expr>,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorLoadObj>() == 56);
    assert!(::core::mem::align_of::<TensorLoadObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorLoad {
    base: ObjectArc<TensorLoadObj>,
}

impl Deref for TensorLoad {
    type Target = TensorLoadObj;
    fn deref(&self) -> &TensorLoadObj {
        &self.base
    }
}

impl Deref for TensorLoadObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TensorLoadObj {
    pub(crate) fn new(loc: Location, ty: Type, source: Expr, indices: Array<Expr>) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, source, indices }
    }
}

impl TensorLoad {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        source: impl Into<Expr>,
        indices: impl Into<Array<Expr>>,
    ) -> Self {
        let obj = TensorLoadObj::new(loc.into(), ty.into(), source.into(), indices.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorLoad => Expr);
// tvm-ffi-stubgen(end)

impl TensorLoad {
    /// The element of `source` at `indices`, typed by the element type of `source`: the type the
    /// type of `source` infers (C++ `TensorLoad(source, indices, loc)`).
    pub fn new(
        source: impl Into<Expr>,
        indices: impl IntoIterator<Item = impl Into<Expr>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        static INFER_TYPE: support::TypeAttr =
            support::TypeAttr::new("__tvm_tensor_load_infer_type__");
        let (source, loc) = (source.into(), loc.into());
        let indices = indices.into_iter().map(Into::into).collect::<Array<Expr>>();
        for index in indices.iter() {
            support::prim_dtype(&index)?;
        }
        let ty_index = AnyView::from(&source.ty).type_index();
        let Some(infer) = INFER_TYPE.get(ty_index) else {
            let message =
                format!("Type {} does not support TensorLoad", support::type_key(ty_index));
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, "infer.has_value()", &message));
        };
        let load = Self::from_fields(loc, MissingType::new(), source, indices);
        let ty: PrimType = support::NativeHook::from_any(infer)?.call_as(&load)?;
        Ok(Self::from_fields(load.loc.clone(), ty, load.source.clone(), load.indices.clone()))
    }
}

// tvm-ffi-stubgen(begin): object/ir.Constant
/// Complete: reflected fields fill [40, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Constant"]
pub struct ConstantObj {
    base: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<ConstantObj>() == 40);
    assert!(::core::mem::align_of::<ConstantObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Constant {
    base: ObjectArc<ConstantObj>,
}

impl Deref for Constant {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl Deref for ConstantObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ConstantObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl Constant {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = ConstantObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for ConstantObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}

tvm_ffi::impl_object_upcast!(Constant => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.FloatImm
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.FloatImm"]
#[type_final]
pub struct FloatImmObj {
    base: ConstantObj,
    pub value: f64,
}

const _: () = {
    assert!(::core::mem::size_of::<FloatImmObj>() == 48);
    assert!(::core::mem::align_of::<FloatImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloatImm {
    base: ObjectArc<FloatImmObj>,
}

impl Deref for FloatImm {
    type Target = FloatImmObj;
    fn deref(&self) -> &FloatImmObj {
        &self.base
    }
}

impl Deref for FloatImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl FloatImmObj {
    pub(crate) fn new(loc: Location, ty: Type, value: f64) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl FloatImm {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>, value: f64) -> Self {
        let obj = FloatImmObj::new(loc.into(), ty.into(), value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloatImm => Expr, FloatImm => Constant);
// tvm-ffi-stubgen(end)

impl FloatImm {
    /// The floating-point constant `value` of type `dtype`, checked to fit it
    /// (C++ `FloatImm(PrimType(dtype), value, loc)`).
    pub fn from_dtype(dtype: DLDataType, value: f64, loc: impl Into<Location>) -> Result<Self> {
        let ty = PrimType::from_dtype(dtype)?;
        check_float_imm(dtype, value)?;
        Ok(Self::new(loc.into(), ty, value))
    }

    /// The `float32` constant `value` at the unknown location, which every `f32` is
    /// (C++ `FloatImm(PrimType::Float(32), value)`).
    pub fn float32(value: f32) -> Self {
        Self::builtin(dtype::float(32), value as f64)
    }

    /// The `float64` constant `value` at the unknown location, which every `f64` is
    /// (C++ `FloatImm(PrimType::Float(64), value)`).
    pub fn float64(value: f64) -> Self {
        Self::builtin(dtype::float(64), value)
    }

    /// [`FloatImm::from_dtype`] of a value its type holds, which fails only when the compiler
    /// library is not loaded.
    fn builtin(dtype: DLDataType, value: f64) -> Self {
        support::infallible(Self::from_dtype(dtype, value, UnknownLoc::new()))
    }
}

/// An `f32` is a `float32` constant, as C++ converts a `float` to a `PrimExpr`.
impl From<f32> for FloatImm {
    fn from(value: f32) -> Self {
        Self::float32(value)
    }
}

/// An `f64` is a `float64` constant. (C++ converts a `double` held by an `Any` to `float32`, as
/// [`prim::convert`](crate::ir::prim::convert) does.)
impl From<f64> for FloatImm {
    fn from(value: f64) -> Self {
        Self::float64(value)
    }
}

impl From<f32> for Expr {
    fn from(value: f32) -> Self {
        FloatImm::float32(value).into()
    }
}

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        FloatImm::float64(value).into()
    }
}

/// The largest finite value of each float8 type code, and whether the type is unsigned
/// (`support::kMax*` in `src/support/limits.h`).
fn float8_bound(code: u8) -> Option<(f64, bool)> {
    use tvm_ffi::DLDataTypeCode::*;
    let bound = match code {
        c if c == kDLFloat8_e3m4 as u8 => (31.0, false),
        c if c == kDLFloat8_e4m3 as u8 => (448.0, false),
        c if c == kDLFloat8_e4m3b11fnuz as u8 => (30.0, true),
        c if c == kDLFloat8_e4m3fn as u8 => (448.0, false),
        c if c == kDLFloat8_e4m3fnuz as u8 => (448.0, true),
        c if c == kDLFloat8_e5m2 as u8 => (57344.0, false),
        c if c == kDLFloat8_e5m2fnuz as u8 => (57344.0, true),
        // `support::kMaxE8M0FNU`, 2^128.
        c if c == kDLFloat8_e8m0fnu as u8 => (3.402_823_669_209_385e38, true),
        _ => return None,
    };
    Some(bound)
}

/// The largest finite value of each float6 type code.
fn float6_bound(code: u8) -> Option<f64> {
    use tvm_ffi::DLDataTypeCode::*;
    match code {
        c if c == kDLFloat6_e2m3fn as u8 => Some(7.5),
        c if c == kDLFloat6_e3m2fn as u8 => Some(28.0),
        _ => None,
    }
}

/// The checks of C++ `FloatImm(ty, value)`: a scalar floating-point type whose range holds a
/// finite `value`.
fn check_float_imm(dtype: DLDataType, value: f64) -> Result<()> {
    use support::{check_failed, compare_failed, CppDouble};
    use tvm_ffi::VALUE_ERROR;
    const FLOAT4_E2M1FN: u8 = tvm_ffi::DLDataTypeCode::kDLFloat4_e2m1fn as u8;
    /// `ffi::DLExtDataTypeCode::kDLExtCustomBegin`.
    const CUSTOM_BEGIN: u8 = 129;
    if support::is_scalable_vector(dtype) || support::is_fixed_length_vector(dtype) {
        let condition = "!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector()";
        return Err(check_failed(VALUE_ERROR, condition, "FloatImm can only take scalar."));
    }
    let (code, bits) = (dtype.code, dtype.bits);
    let float8 = float8_bound(code);
    let float6 = float6_bound(code);
    let supported = code == support::FLOAT
        || float8.is_some()
        || float6.is_some()
        || (code == support::BFLOAT && bits == 16)
        || (code == FLOAT4_E2M1FN && bits == 4)
        || code >= CUSTOM_BEGIN;
    let name = || support::dtype_name(dtype);
    if !supported {
        let condition = "value_ty.MatchesCode(DLDataTypeCode::kDLFloat, \
                         DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3, \
                         DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn, \
                         DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2, \
                         DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu, \
                         DLDataTypeCode::kDLFloat6_e2m3fn, DLDataTypeCode::kDLFloat6_e3m2fn) || \
                         value_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16) || \
                         value_ty.MatchesElementType(DLDataTypeCode::kDLFloat4_e2m1fn, 4) || \
                         static_cast<int>(code) >= \
                         static_cast<int>(ffi::DLExtDataTypeCode::kDLExtCustomBegin)";
        let message = format!("FloatImm supports only float, but {} was supplied.", name());
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    if value.is_infinite() || value.is_nan() {
        return Ok(());
    }
    // The bound checked from below and how its message says so, and the bound from above.
    let (low, high): ((&str, f64, &str), (&str, f64)) = if bits == 32 {
        (
            (
                "value >= std::numeric_limits<float>::lowest()",
                f32::MIN as f64,
                "exceeds minimum of",
            ),
            ("value <= std::numeric_limits<float>::max()", f32::MAX as f64),
        )
    } else if code == support::FLOAT && bits == 16 {
        (
            ("value >= -support::kMaxFloat16", -65504.0, "exceeds minimum of"),
            ("value <= support::kMaxFloat16", 65504.0),
        )
    } else if code == support::BFLOAT && bits == 16 {
        const MAX_BFLOAT16: f64 = 3.895313892515354759047080037148786688e38;
        (
            ("value >= -support::kMaxBFloat16", -MAX_BFLOAT16, "exceeds minimum of"),
            ("value <= support::kMaxBFloat16", MAX_BFLOAT16),
        )
    } else if let Some((bound, nonneg)) = float8 {
        let low = match nonneg {
            true => ("value >= 0", 0.0, "below zero for unsigned"),
            false => ("value >= -bound", -bound, "below minimum of"),
        };
        (low, ("value <= bound", bound))
    } else if let Some(bound) = float6 {
        (("value >= -bound", -bound, "below minimum of"), ("value <= bound", bound))
    } else if code == FLOAT4_E2M1FN {
        (("value >= -bound", -6.0, "below minimum of"), ("value <= bound", 6.0))
    } else {
        return Ok(());
    };
    let failed = |condition: &str, bound: f64, what: &str| {
        let message = format!("Literal value {} {what} {}", CppDouble(value), name());
        compare_failed(VALUE_ERROR, condition, CppDouble(value), CppDouble(bound), &message)
    };
    if value < low.1 {
        return Err(failed(low.0, low.1, low.2));
    }
    if value > high.1 {
        return Err(failed(high.0, high.1, "exceeds maximum of"));
    }
    Ok(())
}

// tvm-ffi-stubgen(begin): object/ir.IntImm
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.IntImm"]
#[type_final]
pub struct IntImmObj {
    base: ConstantObj,
    pub value: BigInt,
}

const _: () = {
    assert!(::core::mem::size_of::<IntImmObj>() == 56);
    assert!(::core::mem::align_of::<IntImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntImm {
    base: ObjectArc<IntImmObj>,
}

impl Deref for IntImm {
    type Target = IntImmObj;
    fn deref(&self) -> &IntImmObj {
        &self.base
    }
}

impl Deref for IntImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl IntImmObj {
    pub(crate) fn new(loc: Location, ty: Type, value: BigInt) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl IntImm {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>, value: impl Into<BigInt>) -> Self {
        let obj = IntImmObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(IntImm => Expr, IntImm => Constant);
// tvm-ffi-stubgen(end)

impl IntImm {
    /// The integer constant `value` of the scalar integer or boolean type `dtype`, checked to fit
    /// it (C++ `IntImm(PrimType(dtype), value, loc)`).
    pub fn from_dtype(
        dtype: DLDataType,
        value: impl Into<BigInt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (value, ty) = (value.into(), PrimType::from_dtype(dtype)?);
        check_int_imm(dtype, &value)?;
        Ok(Self::new(loc.into(), ty, value))
    }

    /// The `int32` constant `value` at the unknown location, which every `i32` is
    /// (C++ `IntImm::Int32(value)`).
    pub fn int32(value: i32) -> Self {
        Self::builtin(dtype::int(32), value)
    }

    /// The `int64` constant `value` at the unknown location, which every `i64` is
    /// (C++ `IntImm::Int64(value)`).
    pub fn int64(value: i64) -> Self {
        Self::builtin(dtype::int(64), value)
    }

    /// The boolean constant `value` at the unknown location (C++ `IntImm::Bool(value)`).
    pub fn bool(value: bool) -> Self {
        Self::builtin(dtype::bool(), value)
    }

    /// [`IntImm::from_dtype`] of a value its type holds, which fails only when the compiler
    /// library is not loaded.
    fn builtin(dtype: DLDataType, value: impl Into<BigInt>) -> Self {
        support::infallible(Self::from_dtype(dtype, value, UnknownLoc::new()))
    }
}

/// An `i32` is an `int32` constant, as C++ converts an `int` to a `PrimExpr`.
impl From<i32> for IntImm {
    fn from(value: i32) -> Self {
        Self::int32(value)
    }
}

/// An `i64` is an `int64` constant. (C++ converts an `int64_t` held by an `Any` to `int32` when
/// the value fits, as [`prim::convert`](crate::ir::prim::convert) does.)
impl From<i64> for IntImm {
    fn from(value: i64) -> Self {
        Self::int64(value)
    }
}

/// A `bool` is a boolean constant, as C++ converts a `bool` to a `PrimExpr`.
impl From<bool> for IntImm {
    fn from(value: bool) -> Self {
        Self::bool(value)
    }
}

impl From<i32> for Expr {
    fn from(value: i32) -> Self {
        IntImm::int32(value).into()
    }
}

impl From<i64> for Expr {
    fn from(value: i64) -> Self {
        IntImm::int64(value).into()
    }
}

impl From<bool> for Expr {
    fn from(value: bool) -> Self {
        IntImm::bool(value).into()
    }
}

/// The checks of C++ `IntImm(ty, value)`: a scalar integer or boolean type whose range holds
/// `value`. The messages are built only when a check fails.
fn check_int_imm(dtype: DLDataType, value: &BigInt) -> Result<()> {
    use support::{check_failed, compare_failed};
    use tvm_ffi::VALUE_ERROR;
    let name = || support::dtype_name(dtype);
    if support::is_scalable_vector(dtype) || support::is_fixed_length_vector(dtype) {
        let condition = "!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector()";
        let message = format!("IntImm can only take scalar, but {} was supplied.", name());
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    if !matches!(dtype.code, support::INT | support::UINT | support::BOOL) {
        let condition = "value_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, \
                         DLDataTypeCode::kDLBool)";
        let message =
            format!("IntImm supports only int or uint or bool type, but {} was supplied.", name());
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    let bits = dtype.bits as u32;
    if bits == 0 {
        let message = "IntImm requires a positive integer width";
        return Err(compare_failed(VALUE_ERROR, "bits > 0", bits, 0, message));
    }
    let zero = BigInt::from(0i64);
    if dtype.code == support::UINT {
        if *value < zero {
            let message =
                format!("Literal value {value} is negative for unsigned integer type {}", name());
            return Err(compare_failed(VALUE_ERROR, "value >= 0", value, 0, &message));
        }
        let message = || format!("Literal value {value} exceeds maximum of {}", name());
        if bits <= 64 {
            if !value.to_u64().is_some_and(|small| bits == 64 || small < 1u64 << bits) {
                let condition =
                    "small.has_value() && (bits == 64 || *small < (uint64_t{1} << bits))";
                return Err(check_failed(VALUE_ERROR, condition, &message()));
            }
        } else {
            let limit = BigInt::from(1i64).try_shl(bits as i64)?;
            if *value >= limit {
                let condition = "value < ffi::BigInt(1) << bits";
                return Err(compare_failed(VALUE_ERROR, condition, value, &limit, &message()));
            }
        }
    } else if bits == 1 || dtype.code == support::BOOL {
        if *value != zero && *value != BigInt::from(1i64) {
            let message = format!("{value} exceeds range of {}", name());
            return Err(check_failed(VALUE_ERROR, "value == 0 || value == 1", &message));
        }
    } else {
        let message = || format!("Literal value {value} exceeds range of {}", name());
        if bits <= 64 {
            let half = 1i128 << (bits - 1);
            let fits = value.to_i64().is_some_and(|small| {
                bits == 64 || ((small as i128) >= -half && (small as i128) < half)
            });
            if !fits {
                let condition = "small.has_value() && (bits == 64 || (*small >= -(int64_t{1} << \
                                 (bits - 1)) && *small < (int64_t{1} << (bits - 1))))";
                return Err(check_failed(VALUE_ERROR, condition, &message()));
            }
        } else {
            let limit = BigInt::from(1i64).try_shl(bits as i64 - 1)?;
            if *value < -limit.clone() || *value >= limit {
                let condition = "value >= -limit && value < limit";
                return Err(check_failed(VALUE_ERROR, condition, &message()));
            }
        }
    }
    Ok(())
}

// tvm-ffi-stubgen(begin): object/ir.GenericConst
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GenericConst"]
#[type_final]
pub struct GenericConstObj {
    base: ConstantObj,
    pub value: Any,
}

const _: () = {
    assert!(::core::mem::size_of::<GenericConstObj>() == 56);
    assert!(::core::mem::align_of::<GenericConstObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GenericConst {
    base: ObjectArc<GenericConstObj>,
}

impl Deref for GenericConst {
    type Target = GenericConstObj;
    fn deref(&self) -> &GenericConstObj {
        &self.base
    }
}

impl Deref for GenericConstObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl GenericConstObj {
    pub(crate) fn new(loc: Location, ty: Type, value: Any) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl GenericConst {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>, value: impl Into<Any>) -> Self {
        let obj = GenericConstObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GenericConst => Expr, GenericConst => Constant);
// tvm-ffi-stubgen(end)

impl GenericConst {
    /// The constant `value` of type `ty`, checked to have a type and not to be a primitive literal
    /// (C++ `GenericConst(ty, value, loc)`).
    pub fn checked(
        ty: impl Into<Type>,
        value: impl Into<Any>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        use TVMFFITypeIndex::*;
        let (ty, value) = (ty.into(), value.into());
        if ty.as_node::<MissingTypeObj>().is_some() {
            let condition = "!ty.as<MissingType>().has_value()";
            let message = "GenericConst requires an expression type";
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, message));
        }
        let primitive =
            [kTVMFFIInt, kTVMFFIBigInt, kTVMFFIBool, kTVMFFIFloat, kTVMFFISmallStr, kTVMFFIStr]
                .into_iter()
                .any(|index| value.type_index() == index as i32);
        if primitive {
            let condition =
                "!value.as<ffi::BigInt>() && !value.as<bool>() && !value.as<double>() && \
                             !value.as<ffi::String>()";
            let message = "Primitive literals use IntImm, FloatImm, or StringImm";
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, message));
        }
        Ok(Self::new(loc.into(), ty, value))
    }
}

// tvm-ffi-stubgen(begin): object/ir.DataTypeImm
/// Complete: reflected fields fill [40, 48) exactly (alignment padding [44, 48)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.DataTypeImm"]
#[type_final]
pub struct DataTypeImmObj {
    base: ConstantObj,
    pub value: DLDataType,
}

const _: () = {
    assert!(::core::mem::size_of::<DataTypeImmObj>() == 48);
    assert!(::core::mem::align_of::<DataTypeImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DataTypeImm {
    base: ObjectArc<DataTypeImmObj>,
}

impl Deref for DataTypeImm {
    type Target = DataTypeImmObj;
    fn deref(&self) -> &DataTypeImmObj {
        &self.base
    }
}

impl Deref for DataTypeImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl DataTypeImmObj {
    pub(crate) fn new(loc: Location, ty: Type, value: DLDataType) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl DataTypeImm {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        value: impl Into<DLDataType>,
    ) -> Self {
        let obj = DataTypeImmObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(DataTypeImm => Expr, DataTypeImm => Constant);
// tvm-ffi-stubgen(end)

impl DataTypeImm {
    /// The data type `value` as a constant of type [`AnyType`]
    /// (C++ `DataTypeImm(value, loc)`).
    pub fn from_dtype(value: DLDataType, loc: impl Into<Location>) -> Self {
        let ty = AnyType::new(UnknownLoc::new());
        Self::new(loc.into(), ty, value)
    }
}

// tvm-ffi-stubgen(begin): object/ir.StringImm
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ConstantObj,
    pub value: String,
}

const _: () = {
    assert!(::core::mem::size_of::<StringImmObj>() == 56);
    assert!(::core::mem::align_of::<StringImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StringImm {
    base: ObjectArc<StringImmObj>,
}

impl Deref for StringImm {
    type Target = StringImmObj;
    fn deref(&self) -> &StringImmObj {
        &self.base
    }
}

impl Deref for StringImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl StringImmObj {
    pub(crate) fn new(loc: Location, ty: Type, value: String) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl StringImm {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        value: impl Into<String>,
    ) -> Self {
        let obj = StringImmObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(StringImm => Expr, StringImm => Constant);
// tvm-ffi-stubgen(end)

impl StringImm {
    /// The string constant `value`, of type [`StringType`]
    /// (C++ `StringImm(value, loc)`).
    pub fn new(value: &str, loc: impl Into<Location>) -> Self {
        let ty = StringType::new(UnknownLoc::new());
        Self::from_fields(loc.into(), ty, String::from(value))
    }
}

// tvm-ffi-stubgen(begin): object/ir.Call
/// Complete: reflected fields fill [40, 72) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Call"]
#[type_final]
pub struct CallObj {
    base: ExprObj,
    pub op: Expr,
    pub args: Array<Expr>,
    pub attrs: Option<Attrs>,
    pub ty_args: Array<Type>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallObj>() == 72);
    assert!(::core::mem::align_of::<CallObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Call {
    base: ObjectArc<CallObj>,
}

impl Deref for Call {
    type Target = CallObj;
    fn deref(&self) -> &CallObj {
        &self.base
    }
}

impl Deref for CallObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl CallObj {
    pub(crate) fn new(
        loc: Location,
        ty: Type,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
    ) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, op, args, attrs, ty_args }
    }
}

impl Call {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        op: impl Into<Expr>,
        args: impl Into<Array<Expr>>,
        attrs: impl Into<Option<Attrs>>,
        ty_args: impl Into<Array<Type>>,
    ) -> Self {
        let obj = CallObj::new(
            loc.into(),
            ty.into(),
            op.into(),
            args.into(),
            attrs.into(),
            ty_args.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Call => Expr);
// tvm-ffi-stubgen(end)

impl Call {
    /// The call of `op` on `args`, of type `ty` or the type [`Call::reinfer_type`] gives it when
    /// `None` (C++ `Call(ty, op, args, attrs, ty_args, loc)`). The call is not validated.
    pub fn new(
        ty: Option<&Type>,
        op: impl Into<Expr>,
        args: impl IntoIterator<Item = impl Into<Expr>>,
        attrs: Option<&Attrs>,
        ty_args: impl IntoIterator<Item = impl Into<Type>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (op, attrs, loc) = (op.into(), attrs.cloned(), loc.into());
        let args = args.into_iter().map(Into::into).collect::<Array<Expr>>();
        let ty_args = ty_args.into_iter().map(Into::into).collect::<Array<Type>>();
        if let Some(ty) = ty {
            return Ok(Self::from_fields(loc, ty.clone(), op, args, attrs, ty_args));
        }
        let call = Self::from_fields(loc, MissingType::new(), op, args, attrs, ty_args);
        let ty = call.reinfer_type()?;
        Ok(Self::from_fields(
            call.loc.clone(),
            ty,
            call.op.clone(),
            call.args.clone(),
            call.attrs.clone(),
            call.ty_args.clone(),
        ))
    }

    /// Check the call against the contract its operator registered: the operator's validator, if
    /// it has one (C++ `Call::Validate`).
    pub fn validate(&self) -> Result<()> {
        let Some(op) = self.op.as_node::<OpObj>() else { return Ok(()) };
        if op.validator().type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(());
        }
        support::NativeHook::from_any(op.validator().clone())?.call_void(self)
    }

    /// The result type the operator infers from the current inputs, without changing the call: its
    /// `TFixedReturnType`, else what its `FInferType` infers; the return type of a callee of a
    /// function type; otherwise a missing type (C++ `Call::ReinferType`).
    pub fn reinfer_type(&self) -> Result<Type> {
        let Some(op) = self.op.as_node::<OpObj>() else {
            return match self.op.ty.as_node::<FuncTypeObj>() {
                Some(func_type) => Ok(func_type.ret_type.clone()),
                None => Ok(MissingType::new().into()),
            };
        };
        if let Some(fixed) = support::op_attr(op, "TFixedReturnType")? {
            return fixed.try_into();
        }
        match support::NativeHook::of_op(op, "FInferType")? {
            Some(infer) => infer.call_as(self),
            None => Ok(MissingType::new().into()),
        }
    }
}

// tvm-ffi-stubgen(begin): object/ir.TensorRegion
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorRegion"]
#[type_final]
pub struct TensorRegionObj {
    base: ExprObj,
    pub source: Expr,
    pub region: Array<Range>,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorRegionObj>() == 56);
    assert!(::core::mem::align_of::<TensorRegionObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorRegion {
    base: ObjectArc<TensorRegionObj>,
}

impl Deref for TensorRegion {
    type Target = TensorRegionObj;
    fn deref(&self) -> &TensorRegionObj {
        &self.base
    }
}

impl Deref for TensorRegionObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TensorRegionObj {
    pub(crate) fn new(loc: Location, ty: Type, source: Expr, region: Array<Range>) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, source, region }
    }
}

impl TensorRegion {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        source: impl Into<Expr>,
        region: impl Into<Array<Range>>,
    ) -> Self {
        let obj = TensorRegionObj::new(loc.into(), ty.into(), source.into(), region.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorRegion => Expr);
// tvm-ffi-stubgen(end)

// `ir.TensorRegion` stores its arguments as given: the generated `TensorRegion::new` is the same.

// tvm-ffi-stubgen(begin): object/ir.Var
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Var"]
pub struct VarObj {
    base: ExprObj,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<VarObj>() == 56);
    assert!(::core::mem::align_of::<VarObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Var {
    base: ObjectArc<VarObj>,
}

impl Deref for Var {
    type Target = VarObj;
    fn deref(&self) -> &VarObj {
        &self.base
    }
}

impl Deref for VarObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl VarObj {
    pub fn new(loc: Location, ty: Type, name: String) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, name }
    }
}

impl Var {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        name: impl Into<String>,
    ) -> Self {
        let obj = VarObj::new(loc.into(), ty.into(), name.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for VarObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base), name: Clone::clone(&self.name) }
    }
}

tvm_ffi::impl_object_upcast!(Var => Expr);
// tvm-ffi-stubgen(end)

impl Var {
    /// The variable `name` of type `ty` (C++ `Var(name, ty, loc)`); a variable whose type is not
    /// known yet takes [`MissingType::new`].
    pub fn new(name: &str, ty: impl Into<Type>, loc: impl Into<Location>) -> Self {
        Self::from_fields(loc.into(), ty.into(), String::from(name))
    }
}

// tvm-ffi-stubgen(begin): object/ir.LambdaExpr
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.LambdaExpr"]
#[type_final]
pub struct LambdaExprObj {
    base: StagingExprObj,
    pub vars: Array<Var>,
    pub body: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LambdaExprObj>() == 56);
    assert!(::core::mem::align_of::<LambdaExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LambdaExpr {
    base: ObjectArc<LambdaExprObj>,
}

impl Deref for LambdaExpr {
    type Target = LambdaExprObj;
    fn deref(&self) -> &LambdaExprObj {
        &self.base
    }
}

impl Deref for LambdaExprObj {
    type Target = StagingExprObj;
    fn deref(&self) -> &StagingExprObj {
        &self.base
    }
}

impl LambdaExprObj {
    pub(crate) fn new(loc: Location, ty: Type, vars: Array<Var>, body: Expr) -> Self {
        let base = StagingExprObj::new(loc, ty);
        Self { base, vars, body }
    }
}

impl LambdaExpr {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        vars: impl Into<Array<Var>>,
        body: impl Into<Expr>,
    ) -> Self {
        let obj = LambdaExprObj::new(loc.into(), ty.into(), vars.into(), body.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LambdaExpr => Expr, LambdaExpr => StagingExpr);
// tvm-ffi-stubgen(end)

impl LambdaExpr {
    /// The lambda of `vars` that evaluates to `body`, of the function type from the types of
    /// `vars` to the type of `body` (C++ `LambdaExpr(vars, body, loc)`).
    pub fn new(
        vars: impl IntoIterator<Item = impl Into<Var>>,
        body: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Self {
        let (vars, body) = (vars.into_iter().map(Into::into).collect::<Array<Var>>(), body.into());
        let arg_types = vars.iter().map(|var| var.ty.clone()).collect::<Vec<Type>>();
        let ty = FuncType::new(arg_types, body.ty.clone());
        Self::from_fields(loc.into(), ty, vars, body)
    }

    /// The body with `arguments` substituted for the variables (`ir.LambdaExprApply`).
    pub fn apply(&self, arguments: impl IntoIterator<Item = impl Into<Expr>>) -> Result<Expr> {
        let arguments = arguments.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("ir.LambdaExprApply")
            .call_tuple((self, &arguments))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.Range
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Range"]
#[type_final]
pub struct RangeObj {
    base: Object,
    pub begin: Expr,
    pub extent: Expr,
    pub loc: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<RangeObj>() == 48);
    assert!(::core::mem::align_of::<RangeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Range {
    base: ObjectArc<RangeObj>,
}

impl Deref for Range {
    type Target = RangeObj;
    fn deref(&self) -> &RangeObj {
        &self.base
    }
}

impl RangeObj {
    pub(crate) fn new(begin: Expr, extent: Expr, loc: Location) -> Self {
        let base = Object::new();
        Self { base, begin, extent, loc }
    }
}

impl Range {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        begin: impl Into<Expr>,
        extent: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Self {
        let obj = RangeObj::new(begin.into(), extent.into(), loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl Range {
    /// The range `[begin, end)` (C++ `Range(begin, end, loc)`);
    /// [`from_min_extent`](Self::from_min_extent) takes the extent instead of the end.
    pub fn new(
        begin: impl Into<Expr>,
        end: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (begin, end) = (begin.into(), end.into());
        let extent = Self::get_extent_from_begin_end(begin.clone(), end)?;
        Ok(Self::from_fields(begin, extent, loc.into()))
    }

    /// The range of `extent` values from `min_value` (C++ `Range::FromMinExtent`).
    pub fn from_min_extent(
        min_value: impl Into<Expr>,
        extent: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (min_value, extent) = (min_value.into(), extent.into());
        support::prim_dtype(&min_value)?;
        support::prim_dtype(&extent)?;
        Ok(Self::from_fields(min_value, extent, loc.into()))
    }

    /// The extent `end - begin` of the range `[begin, end)`: the other operand when `end` adds
    /// `begin` to it, `end` when `begin` is zero, else `end - begin` folded
    /// (C++ `Range::GetExtentFromBeginEnd`).
    pub fn get_extent_from_begin_end(begin: impl Into<Expr>, end: impl Into<Expr>) -> Result<Expr> {
        let (begin, end) = (begin.into(), end.into());
        support::prim_dtype(&begin)?;
        support::prim_dtype(&end)?;
        if let Some(add) = end.as_node::<AddObj>() {
            if crate::ir::prim::expr_deep_equal(add.a.clone(), begin.clone())? {
                return Ok(add.b.clone());
            }
            if crate::ir::prim::expr_deep_equal(add.b.clone(), begin.clone())? {
                return Ok(add.a.clone());
            }
        }
        let is_zero =
            begin.as_node::<IntImmObj>().is_some_and(|imm| imm.value == BigInt::from(0i64));
        match is_zero {
            true => Ok(end),
            false => crate::ir::prim::op::sub(end, begin, UnknownLoc::new()),
        }
    }
}
