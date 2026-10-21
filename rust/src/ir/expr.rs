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
//! [`prim`](super::prim). The constructors with the native semantics are `new`, or `from_dtype`
//! and the like where `new` allocates the fields as given.

// `custom-new` reserves `new` for a constructor with the native one's semantics (defaults, implicit
// conversions, a derived type); the generated `from_complete_fields` allocates the fields as given.
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
// tvm-ffi-stubgen(import-object): tvm_ffi::Result

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::location::Location;
use crate::ir::r#type::Type;
use std::ops::Deref;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::BigInt;
use tvm_ffi::DLDataType;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
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
    pub(crate) fn new(loc: Location, ty: Type) -> Self {
        let base = Object::new();
        Self { base, loc, ty }
    }
}

impl Expr {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location, ty: Type) -> Self {
        let obj = ExprObj::new(loc, ty);
        Self { base: ObjectArc::new(obj) }
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
    pub(crate) fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl StagingExpr {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location, ty: Type) -> Self {
        let obj = StagingExprObj::new(loc, ty);
        Self { base: ObjectArc::new(obj) }
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
    pub(crate) fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl OpaqueExpr {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location, ty: Type) -> Self {
        let obj = OpaqueExprObj::new(loc, ty);
        Self { base: ObjectArc::new(obj) }
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
    pub fn from_complete_fields(loc: Location, ty: Type, name_hint: String) -> Self {
        let obj = GlobalVarObj::new(loc, ty, name_hint);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GlobalVar => Expr);
// tvm-ffi-stubgen(end)

impl GlobalVar {
    /// The global variable `name_hint`, of a missing type (`ir.GlobalVar`).
    pub fn new(name_hint: &str, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.GlobalVar")
            .call_tuple((String::from(name_hint), &loc.into()))?
            .try_into()
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
    pub fn from_complete_fields(loc: Location, ty: Type, fields: Array<Expr>) -> Self {
        let obj = TupleObj::new(loc, ty, fields);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Tuple => Expr);
// tvm-ffi-stubgen(end)

impl Tuple {
    /// The tuple of `fields`, typed by the types of its fields when they are all known
    /// (`ir.Tuple`).
    pub fn new(
        fields: impl IntoIterator<Item = impl Into<Expr>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let fields = fields.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("ir.Tuple").call_tuple((&fields, &loc.into()))?.try_into()
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
    pub fn new(loc: Location, ty: Type, tuple_value: Expr, index: i32) -> Self {
        let obj = TupleGetItemObj::new(loc, ty, tuple_value, index);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TupleGetItem => Expr);
// tvm-ffi-stubgen(end)

impl TupleGetItem {
    /// Field `index` of `tuple_value`, typed by the field type and checked against the tuple type
    /// when it is known (`ir.TupleGetItem`).
    pub fn from_tuple(
        tuple_value: impl Into<Expr>,
        index: i32,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.TupleGetItem")
            .call_tuple((&tuple_value.into(), index, &loc.into()))?
            .try_into()
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
    pub fn from_complete_fields(
        loc: Location,
        ty: Type,
        source: Expr,
        indices: Array<Expr>,
    ) -> Self {
        let obj = TensorLoadObj::new(loc, ty, source, indices);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorLoad => Expr);
// tvm-ffi-stubgen(end)

impl TensorLoad {
    /// The element of `source` at `indices`, typed by the element type of `source`
    /// (`ir.TensorLoad`).
    pub fn new(
        source: impl Into<Expr>,
        indices: impl IntoIterator<Item = impl Into<Expr>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let indices = indices.into_iter().map(Into::into).collect::<Array<Expr>>();
        tvm_ffi::cached_global_func!("ir.TensorLoad")
            .call_tuple((&source.into(), &indices, &loc.into()))?
            .try_into()
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
    pub(crate) fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl Constant {
    /// Lossless complete-field allocation.
    pub fn new(loc: Location, ty: Type) -> Self {
        let obj = ConstantObj::new(loc, ty);
        Self { base: ObjectArc::new(obj) }
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
    pub fn new(loc: Location, ty: Type, value: f64) -> Self {
        let obj = FloatImmObj::new(loc, ty, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloatImm => Expr, FloatImm => Constant);
// tvm-ffi-stubgen(end)

impl FloatImm {
    /// The floating-point constant `value` of type `dtype`, checked to fit it (`ir.FloatImm`).
    pub fn from_dtype(dtype: DLDataType, value: f64, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.FloatImm")
            .call_tuple((dtype, value, &loc.into()))?
            .try_into()
    }
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
    pub fn new(loc: Location, ty: Type, value: BigInt) -> Self {
        let obj = IntImmObj::new(loc, ty, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(IntImm => Expr, IntImm => Constant);
// tvm-ffi-stubgen(end)

impl IntImm {
    /// The integer constant `value` of the scalar integer or boolean type `dtype`, checked to fit
    /// it (`ir.IntImm`).
    pub fn from_dtype(
        dtype: DLDataType,
        value: impl Into<BigInt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.IntImm")
            .call_tuple((dtype, value.into(), &loc.into()))?
            .try_into()
    }
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
    pub fn new(loc: Location, ty: Type, value: Any) -> Self {
        let obj = GenericConstObj::new(loc, ty, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GenericConst => Expr, GenericConst => Constant);
// tvm-ffi-stubgen(end)

impl GenericConst {
    /// The constant `value` of type `ty`, checked to have a type and not to be a primitive literal
    /// (`ir.GenericConst`).
    pub fn checked(
        ty: impl Into<Type>,
        value: impl Into<Any>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.GenericConst")
            .call_tuple((&ty.into(), value.into(), &loc.into()))?
            .try_into()
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
    pub fn new(loc: Location, ty: Type, value: DLDataType) -> Self {
        let obj = DataTypeImmObj::new(loc, ty, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(DataTypeImm => Expr, DataTypeImm => Constant);
// tvm-ffi-stubgen(end)

impl DataTypeImm {
    /// The data type `value` as a constant of type [`AnyType`](crate::ir::AnyType)
    /// (`ir.DataTypeImm`).
    pub fn from_dtype(value: DLDataType, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.DataTypeImm").call_tuple((value, &loc.into()))?.try_into()
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
    pub fn from_complete_fields(loc: Location, ty: Type, value: String) -> Self {
        let obj = StringImmObj::new(loc, ty, value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(StringImm => Expr, StringImm => Constant);
// tvm-ffi-stubgen(end)

impl StringImm {
    /// The string constant `value`, of type [`StringType`](crate::ir::StringType)
    /// (`ir.StringImm`).
    pub fn new(value: &str, loc: impl Into<Location>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.StringImm")
            .call_tuple((String::from(value), &loc.into()))?
            .try_into()
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
    pub fn from_complete_fields(
        loc: Location,
        ty: Type,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
    ) -> Self {
        let obj = CallObj::new(loc, ty, op, args, attrs, ty_args);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Call => Expr);
// tvm-ffi-stubgen(end)

impl Call {
    /// The call of `op` on `args`, of type `ty` or the type the operator infers when `None`
    /// (`ir.Call`).
    pub fn new(
        ty: Option<&Type>,
        op: impl Into<Expr>,
        args: impl IntoIterator<Item = impl Into<Expr>>,
        attrs: Option<&Attrs>,
        ty_args: impl IntoIterator<Item = impl Into<Type>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (ty, op, attrs, loc) = (ty.cloned(), op.into(), attrs.cloned(), loc.into());
        let args = args.into_iter().map(Into::into).collect::<Array<Expr>>();
        let ty_args = ty_args.into_iter().map(Into::into).collect::<Array<Type>>();
        tvm_ffi::cached_global_func!("ir.Call")
            .call_packed(&[
                AnyView::from(&ty),
                AnyView::from(&op),
                AnyView::from(&args),
                AnyView::from(&attrs),
                AnyView::from(&ty_args),
                AnyView::from(&loc),
            ])?
            .try_into()
    }

    /// Check the call against the contract its operator registered (`ir.CallValidate`).
    pub fn validate(&self) -> Result<()> {
        tvm_ffi::cached_global_func!("ir.CallValidate").call_tuple((self,))?;
        Ok(())
    }

    /// The result type the operator infers from the current inputs, without changing the call
    /// (`ir.CallReinferType`).
    pub fn reinfer_type(&self) -> Result<Type> {
        tvm_ffi::cached_global_func!("ir.CallReinferType").call_tuple((self,))?.try_into()
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
    pub fn new(loc: Location, ty: Type, source: Expr, region: Array<Range>) -> Self {
        let obj = TensorRegionObj::new(loc, ty, source, region);
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
    pub(crate) fn new(loc: Location, ty: Type, name: String) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, name }
    }
}

impl Var {
    /// Lossless complete-field allocation.
    pub fn from_complete_fields(loc: Location, ty: Type, name: String) -> Self {
        let obj = VarObj::new(loc, ty, name);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Var => Expr);
// tvm-ffi-stubgen(end)

impl Var {
    /// The variable `name`, of type `ty` or a missing type when `None` (`ir.Var`).
    pub fn new(name: &str, ty: Option<&Type>, loc: impl Into<Location>) -> Result<Self> {
        let ty = match ty {
            Some(ty) => ty.clone(),
            None => crate::ir::r#type::MissingType::new()?.into(),
        };
        tvm_ffi::cached_global_func!("ir.Var")
            .call_tuple((String::from(name), ty, &loc.into()))?
            .try_into()
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
    pub fn from_complete_fields(loc: Location, ty: Type, vars: Array<Var>, body: Expr) -> Self {
        let obj = LambdaExprObj::new(loc, ty, vars, body);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LambdaExpr => Expr, LambdaExpr => StagingExpr);
// tvm-ffi-stubgen(end)

impl LambdaExpr {
    /// The lambda of `vars` that evaluates to `body` (`ir.LambdaExpr`).
    pub fn new(
        vars: impl IntoIterator<Item = impl Into<Var>>,
        body: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let vars = vars.into_iter().map(Into::into).collect::<Array<Var>>();
        tvm_ffi::cached_global_func!("ir.LambdaExpr")
            .call_tuple((&vars, &body.into(), &loc.into()))?
            .try_into()
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
    pub fn from_complete_fields(begin: Expr, extent: Expr, loc: Location) -> Self {
        let obj = RangeObj::new(begin, extent, loc);
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl Range {
    /// The range `[begin, end)`, or `[0, begin)` when `end` is `None` (`ir.Range`).
    pub fn new(
        begin: impl Into<Expr>,
        end: Option<&Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.Range")
            .call_tuple((&begin.into(), end.cloned(), &loc.into()))?
            .try_into()
    }

    /// The range of `extent` values from `min_value` (`ir.RangeFromMinExtent`).
    pub fn from_min_extent(
        min_value: impl Into<Expr>,
        extent: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.RangeFromMinExtent")
            .call_tuple((&min_value.into(), &extent.into(), &loc.into()))?
            .try_into()
    }

    /// The extent `end - begin` of the range `[begin, end)`, simplified
    /// (`ir.RangeGetExtentFromBeginEnd`).
    pub fn get_extent_from_begin_end(begin: impl Into<Expr>, end: impl Into<Expr>) -> Result<Expr> {
        tvm_ffi::cached_global_func!("ir.RangeGetExtentFromBeginEnd")
            .call_tuple((&begin.into(), &end.into()))?
            .try_into()
    }
}
