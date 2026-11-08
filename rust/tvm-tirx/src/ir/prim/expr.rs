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
//! Primitive operations. Each node, such as [`Add`], [`LT`] or [`Select`], applies one operation
//! to [`PrimExpr`] operands, lane by lane for vectors; its `new` checks the operand types and
//! derives the node's type, without the promotion and folding of [`op`](super::op).

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// `custom_new` reserves `new` for a hand-written constructor (operand type checks and the derived
// result type); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(object): prim.Cast(custom_new)
// tvm-ffi-stubgen(object): prim.LShift(custom_new)
// tvm-ffi-stubgen(object): prim.RShift(custom_new)
// tvm-ffi-stubgen(object): prim.BitwiseAnd(custom_new)
// tvm-ffi-stubgen(object): prim.BitwiseOr(custom_new)
// tvm-ffi-stubgen(object): prim.BitwiseXor(custom_new)
// tvm-ffi-stubgen(object): prim.BitwiseNot(custom_new)
// tvm-ffi-stubgen(object): prim.Add(custom_new)
// tvm-ffi-stubgen(object): prim.Sub(custom_new)
// tvm-ffi-stubgen(object): prim.Mul(custom_new)
// tvm-ffi-stubgen(object): prim.Div(custom_new)
// tvm-ffi-stubgen(object): prim.Mod(custom_new)
// tvm-ffi-stubgen(object): prim.FloorDiv(custom_new)
// tvm-ffi-stubgen(object): prim.FloorMod(custom_new)
// tvm-ffi-stubgen(object): prim.Min(custom_new)
// tvm-ffi-stubgen(object): prim.Max(custom_new)
// tvm-ffi-stubgen(object): prim.EQ(custom_new)
// tvm-ffi-stubgen(object): prim.NE(custom_new)
// tvm-ffi-stubgen(object): prim.LT(custom_new)
// tvm-ffi-stubgen(object): prim.LE(custom_new)
// tvm-ffi-stubgen(object): prim.GT(custom_new)
// tvm-ffi-stubgen(object): prim.GE(custom_new)
// tvm-ffi-stubgen(object): prim.And(custom_new)
// tvm-ffi-stubgen(object): prim.Or(custom_new)
// tvm-ffi-stubgen(object): prim.Not(custom_new)
// tvm-ffi-stubgen(object): prim.Select(custom_new)
// tvm-ffi-stubgen(object): prim.Let(custom_new)
// Every node is typed by a `PrimType` and holds `PrimExpr` operands, as C++ `Add` derives from
// `PrimExpr` and `AddObj` holds `PrimExpr a, b`: its allocator takes a `PrimType`, and it converts
// into a `PrimExpr` without a check.
// tvm-ffi-stubgen(field): prim.Cast.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Cast.value(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Cast(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.LShift.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.LShift.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.LShift.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.LShift(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.RShift.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.RShift.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.RShift.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.RShift(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.BitwiseAnd.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.BitwiseAnd.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.BitwiseAnd.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.BitwiseAnd(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.BitwiseOr.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.BitwiseOr.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.BitwiseOr.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.BitwiseOr(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.BitwiseXor.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.BitwiseXor.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.BitwiseXor.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.BitwiseXor(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.BitwiseNot.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.BitwiseNot.a(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.BitwiseNot(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Add.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Add.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Add.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Add(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Sub.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Sub.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Sub.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Sub(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Mul.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Mul.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Mul.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Mul(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Div.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Div.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Div.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Div(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Mod.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Mod.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Mod.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Mod(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.FloorDiv.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.FloorDiv.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.FloorDiv.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.FloorDiv(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.FloorMod.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.FloorMod.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.FloorMod.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.FloorMod(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Min.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Min.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Min.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Min(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Max.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Max.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Max.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Max(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.EQ.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.EQ.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.EQ.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.EQ(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.NE.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.NE.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.NE.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.NE(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.LT.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.LT.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.LT.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.LT(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.LE.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.LE.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.LE.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.LE(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.GT.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.GT.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.GT.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.GT(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.GE.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.GE.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.GE.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.GE(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.And.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.And.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.And.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.And(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Or.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Or.a(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Or.b(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Or(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Not.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Not.a(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Not(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Select.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Select.condition(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Select.true_value(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Select.false_value(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Select(upcast = PrimExpr)
// tvm-ffi-stubgen(field): prim.Let.ty(type = PrimType)
// tvm-ffi-stubgen(field): prim.Let.value(type = PrimExpr)
// tvm-ffi-stubgen(field): prim.Let.body(type = PrimExpr)
// tvm-ffi-stubgen(object): prim.Let(upcast = PrimExpr)
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import): tvm_ffi::AnyView
// tvm-ffi-stubgen(import): tvm_ffi::DLDataTypeCode
// tvm-ffi-stubgen(import): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import): tvm_ffi::Result
// tvm-ffi-stubgen(import): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import): crate::ir::r#type::PrimTypeObj
// tvm-ffi-stubgen(import): crate::ir::support
// tvm-ffi-stubgen(import): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import): tvm_ffi::IntoArg

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::Var;
use crate::ir::location::Location;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::PrimTypeObj;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::AnyView;
use tvm_ffi::DLDataTypeCode;
use tvm_ffi::IntoArg;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

/// `impl $node { pub fn new(a, b, loc) }` for each binary node, which checks the operand types.
macro_rules! binary_new {
    ($($node:ident: $operands:expr, $doc:literal;)+) => {
        $(
            impl $node {
                #[doc = $doc]
                ///
                /// # Errors
                ///
                /// Returns a `TypeError` if the operand types differ, or do not suit the node, such
                /// as a float for a bitwise node.
                pub fn new(
                    a: impl IntoExpr<PrimExpr>,
                    b: impl IntoExpr<PrimExpr>,
                    loc: impl IntoArg<Location>,
                ) -> Result<Self> {
                    let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
                    let ty = binary_type(stringify!($node), $operands, &a, &b)?;
                    Ok(Self::from_fields(loc.into_owned(), ty, a, b))
                }
            }
        )+
    };
}

// ---------------------------------------------------------------------------------------------
// Casts

/// The conversion of `value` to the node's type, lane by lane: both have as many lanes.
// tvm-ffi-stubgen(begin): object/prim.Cast
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Cast {
    data_: ObjectArc<CastObj>,
}

/// The object behind [`Cast`].
// Complete: reflected fields fill [40, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Cast", final, immutable)]
pub struct CastObj {
    base_: ExprObj,
    pub value: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<CastObj>() == 48);
    assert!(::core::mem::align_of::<CastObj>() == 8);
};

impl Deref for CastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl CastObj {
    pub(crate) fn new(loc: Location, ty: PrimType, value: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, value }
    }
}

#[allow(missing_docs)]
impl Cast {
    /// Create a `Cast` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        value: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = CastObj::new(loc.into_owned(), ty.into_owned(), value.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Cast => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Cast => PrimExpr);
// tvm-ffi-stubgen(end)

impl Cast {
    /// Create the node converting `value` to `ty`, which has as many lanes as `value`.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `ty` and the type of `value` have different numbers of lanes.
    pub fn new(
        ty: impl IntoArg<PrimType>,
        value: impl IntoExpr<PrimExpr>,
        loc: impl IntoArg<Location>,
    ) -> Result<Self> {
        let (ty, value) = (ty.into_owned(), value.into_owned_expr()?);
        let value_dtype = value.prim_ty().dtype;
        if ty.dtype.lanes != value_dtype.lanes {
            let condition = "ty->dtype.lanes == value_expr_ty->dtype.lanes";
            let (lhs, rhs) = (ty.dtype.lanes, value_dtype.lanes);
            return Err(support::compare_failed(tvm_ffi::INTERNAL_ERROR, condition, lhs, rhs, ""));
        }
        Ok(Self::from_fields(loc.into_owned(), ty, value))
    }
}

// ---------------------------------------------------------------------------------------------
// Bitwise operations

/// The left shift `a << b` of integer operands of one type.
// tvm-ffi-stubgen(begin): object/prim.LShift
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LShift {
    data_: ObjectArc<LShiftObj>,
}

/// The object behind [`LShift`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.LShift", final, immutable)]
pub struct LShiftObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LShiftObj>() == 56);
    assert!(::core::mem::align_of::<LShiftObj>() == 8);
};

impl Deref for LShiftObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl LShiftObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl LShift {
    /// Create a `LShift` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = LShiftObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(LShift => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe LShift => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    LShift: Operands::Bitwise { allow_bool: false },
        "Create the node `a << b` of integer operands of one type.";
}

/// The right shift `a >> b` of integer operands of one type.
// tvm-ffi-stubgen(begin): object/prim.RShift
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RShift {
    data_: ObjectArc<RShiftObj>,
}

/// The object behind [`RShift`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.RShift", final, immutable)]
pub struct RShiftObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<RShiftObj>() == 56);
    assert!(::core::mem::align_of::<RShiftObj>() == 8);
};

impl Deref for RShiftObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl RShiftObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl RShift {
    /// Create a `RShift` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = RShiftObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(RShift => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe RShift => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    RShift: Operands::Bitwise { allow_bool: false },
        "Create the node `a >> b` of integer operands of one type.";
}

/// The bitwise and `a & b` of integer or boolean operands of one type.
// tvm-ffi-stubgen(begin): object/prim.BitwiseAnd
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseAnd {
    data_: ObjectArc<BitwiseAndObj>,
}

/// The object behind [`BitwiseAnd`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.BitwiseAnd", final, immutable)]
pub struct BitwiseAndObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseAndObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseAndObj>() == 8);
};

impl Deref for BitwiseAndObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BitwiseAndObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl BitwiseAnd {
    /// Create a `BitwiseAnd` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj =
            BitwiseAndObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(BitwiseAnd => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe BitwiseAnd => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    BitwiseAnd: Operands::Bitwise { allow_bool: true },
        "Create the node `a & b` of integer or boolean operands of one type.";
}

/// The bitwise or `a | b` of integer or boolean operands of one type.
// tvm-ffi-stubgen(begin): object/prim.BitwiseOr
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseOr {
    data_: ObjectArc<BitwiseOrObj>,
}

/// The object behind [`BitwiseOr`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.BitwiseOr", final, immutable)]
pub struct BitwiseOrObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseOrObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseOrObj>() == 8);
};

impl Deref for BitwiseOrObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BitwiseOrObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl BitwiseOr {
    /// Create a `BitwiseOr` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj =
            BitwiseOrObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(BitwiseOr => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe BitwiseOr => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    BitwiseOr: Operands::Bitwise { allow_bool: true },
        "Create the node `a | b` of integer or boolean operands of one type.";
}

/// The bitwise exclusive or `a ^ b` of integer or boolean operands of one type.
// tvm-ffi-stubgen(begin): object/prim.BitwiseXor
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseXor {
    data_: ObjectArc<BitwiseXorObj>,
}

/// The object behind [`BitwiseXor`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.BitwiseXor", final, immutable)]
pub struct BitwiseXorObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseXorObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseXorObj>() == 8);
};

impl Deref for BitwiseXorObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BitwiseXorObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl BitwiseXor {
    /// Create a `BitwiseXor` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj =
            BitwiseXorObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(BitwiseXor => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe BitwiseXor => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    BitwiseXor: Operands::Bitwise { allow_bool: true },
        "Create the node `a ^ b` of integer or boolean operands of one type.";
}

/// The bitwise complement `~a` of an integer or boolean operand.
// tvm-ffi-stubgen(begin): object/prim.BitwiseNot
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseNot {
    data_: ObjectArc<BitwiseNotObj>,
}

/// The object behind [`BitwiseNot`].
// Complete: reflected fields fill [40, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.BitwiseNot", final, immutable)]
pub struct BitwiseNotObj {
    base_: ExprObj,
    pub a: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseNotObj>() == 48);
    assert!(::core::mem::align_of::<BitwiseNotObj>() == 8);
};

impl Deref for BitwiseNotObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BitwiseNotObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a }
    }
}

#[allow(missing_docs)]
impl BitwiseNot {
    /// Create a `BitwiseNot` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = BitwiseNotObj::new(loc.into_owned(), ty.into_owned(), a.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(BitwiseNot => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe BitwiseNot => PrimExpr);
// tvm-ffi-stubgen(end)

impl BitwiseNot {
    /// Create the node `~a` over an integer or boolean operand.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `a` is not of an integer or boolean type.
    pub fn new(a: impl IntoExpr<PrimExpr>, loc: impl IntoArg<Location>) -> Result<Self> {
        let a = a.into_owned_expr()?;
        let integer_or_bool =
            [DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLBool];
        if !a.prim_ty().matches_code(&integer_or_bool) {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) || \
                             a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            let message = "BitwiseNot requires an integer or boolean operand";
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, message));
        }
        Ok(Self::from_fields(loc.into_owned(), a.prim_ty().clone(), a))
    }
}

// ---------------------------------------------------------------------------------------------
// Arithmetic

/// The sum `a + b` of operands of one type, elementwise for vectors.
///
/// Like every node, [`Add::new`] checks the operand types and neither promotes them nor folds
/// constants; [`op::add`](super::op::add) and the `+` operator do both.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::Add;
/// use tvm_tirx::ir::{PrimType, PrimVar, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// assert_eq!(script(&Add::new(&x, 1, UNKNOWN_LOC)?)?, "x + 1");
///
/// let f = PrimVar::new("f", PrimType::float(32), UNKNOWN_LOC);
/// assert!(Add::new(&x, &f, UNKNOWN_LOC).is_err());
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/prim.Add
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Add {
    data_: ObjectArc<AddObj>,
}

/// The object behind [`Add`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Add", final, immutable)]
pub struct AddObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<AddObj>() == 56);
    assert!(::core::mem::align_of::<AddObj>() == 8);
};

impl Deref for AddObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl AddObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Add {
    /// Create a `Add` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = AddObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Add => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Add => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Add: Operands::Arith,
        "Create the node `a + b` of operands of one type.";
}

/// The difference `a - b` of operands of one type, elementwise for vectors.
// tvm-ffi-stubgen(begin): object/prim.Sub
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Sub {
    data_: ObjectArc<SubObj>,
}

/// The object behind [`Sub`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Sub", final, immutable)]
pub struct SubObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<SubObj>() == 56);
    assert!(::core::mem::align_of::<SubObj>() == 8);
};

impl Deref for SubObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl SubObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Sub {
    /// Create a `Sub` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = SubObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Sub => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Sub => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Sub: Operands::Arith,
        "Create the node `a - b` of operands of one type.";
}

/// The product `a * b` of operands of one type, elementwise for vectors.
// tvm-ffi-stubgen(begin): object/prim.Mul
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mul {
    data_: ObjectArc<MulObj>,
}

/// The object behind [`Mul`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Mul", final, immutable)]
pub struct MulObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<MulObj>() == 56);
    assert!(::core::mem::align_of::<MulObj>() == 8);
};

impl Deref for MulObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl MulObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Mul {
    /// Create a `Mul` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = MulObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Mul => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Mul => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Mul: Operands::Arith,
        "Create the node `a * b` of operands of one type.";
}

/// The quotient `a / b` of operands of one type; integer division truncates toward zero.
// tvm-ffi-stubgen(begin): object/prim.Div
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Div {
    data_: ObjectArc<DivObj>,
}

/// The object behind [`Div`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Div", final, immutable)]
pub struct DivObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<DivObj>() == 56);
    assert!(::core::mem::align_of::<DivObj>() == 8);
};

impl Deref for DivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl DivObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Div {
    /// Create a `Div` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = DivObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Div => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Div => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Div: Operands::Arith,
        "Create the node `a / b` of operands of one type.";
}

/// The remainder `a % b` of division truncated toward zero, of operands of one type.
// tvm-ffi-stubgen(begin): object/prim.Mod
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mod {
    data_: ObjectArc<ModObj>,
}

/// The object behind [`Mod`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Mod", final, immutable)]
pub struct ModObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<ModObj>() == 56);
    assert!(::core::mem::align_of::<ModObj>() == 8);
};

impl Deref for ModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl ModObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Mod {
    /// Create a `Mod` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = ModObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Mod => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Mod => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Mod: Operands::Arith,
        "Create the truncated node `a % b` of operands of one type.";
}

/// The quotient `floor(a / b)` of operands of one type, rounded toward negative infinity.
// tvm-ffi-stubgen(begin): object/prim.FloorDiv
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorDiv {
    data_: ObjectArc<FloorDivObj>,
}

/// The object behind [`FloorDiv`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.FloorDiv", final, immutable)]
pub struct FloorDivObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorDivObj>() == 56);
    assert!(::core::mem::align_of::<FloorDivObj>() == 8);
};

impl Deref for FloorDivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl FloorDivObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl FloorDiv {
    /// Create a `FloorDiv` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj =
            FloorDivObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(FloorDiv => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe FloorDiv => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    FloorDiv: Operands::Arith,
        "Create the node `floor(a / b)` of operands of one type.";
}

/// The remainder `a - floor(a / b) * b` of floor division, of operands of one type.
// tvm-ffi-stubgen(begin): object/prim.FloorMod
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorMod {
    data_: ObjectArc<FloorModObj>,
}

/// The object behind [`FloorMod`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.FloorMod", final, immutable)]
pub struct FloorModObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorModObj>() == 56);
    assert!(::core::mem::align_of::<FloorModObj>() == 8);
};

impl Deref for FloorModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl FloorModObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl FloorMod {
    /// Create a `FloorMod` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj =
            FloorModObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(FloorMod => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe FloorMod => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    FloorMod: Operands::Arith,
        "Create the node `a - floor(a / b) * b` of operands of one type.";
}

/// The smaller of `a` and `b`, of one type, elementwise for vectors.
// tvm-ffi-stubgen(begin): object/prim.Min
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Min {
    data_: ObjectArc<MinObj>,
}

/// The object behind [`Min`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Min", final, immutable)]
pub struct MinObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<MinObj>() == 56);
    assert!(::core::mem::align_of::<MinObj>() == 8);
};

impl Deref for MinObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl MinObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Min {
    /// Create a `Min` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = MinObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Min => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Min => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Min: Operands::Arith,
        "Create the node `min(a, b)` of operands of one type.";
}

/// The larger of `a` and `b`, of one type, elementwise for vectors.
// tvm-ffi-stubgen(begin): object/prim.Max
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Max {
    data_: ObjectArc<MaxObj>,
}

/// The object behind [`Max`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Max", final, immutable)]
pub struct MaxObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<MaxObj>() == 56);
    assert!(::core::mem::align_of::<MaxObj>() == 8);
};

impl Deref for MaxObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl MaxObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Max {
    /// Create a `Max` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = MaxObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Max => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Max => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Max: Operands::Arith,
        "Create the node `max(a, b)` of operands of one type.";
}

// ---------------------------------------------------------------------------------------------
// Comparisons

/// The comparison `a == b` of operands of one type: a boolean with their lanes.
// tvm-ffi-stubgen(begin): object/prim.EQ
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EQ {
    data_: ObjectArc<EQObj>,
}

/// The object behind [`EQ`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.EQ", final, immutable)]
pub struct EQObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<EQObj>() == 56);
    assert!(::core::mem::align_of::<EQObj>() == 8);
};

impl Deref for EQObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl EQObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl EQ {
    /// Create a `EQ` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = EQObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(EQ => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe EQ => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    EQ: Operands::Compare,
        "Create the boolean node `a == b` of operands of one type.";
}

/// The comparison `a != b` of operands of one type: a boolean with their lanes.
// tvm-ffi-stubgen(begin): object/prim.NE
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct NE {
    data_: ObjectArc<NEObj>,
}

/// The object behind [`NE`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.NE", final, immutable)]
pub struct NEObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<NEObj>() == 56);
    assert!(::core::mem::align_of::<NEObj>() == 8);
};

impl Deref for NEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl NEObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl NE {
    /// Create a `NE` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = NEObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(NE => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe NE => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    NE: Operands::Compare,
        "Create the boolean node `a != b` of operands of one type.";
}

/// The comparison `a < b` of operands of one type: a boolean with their lanes.
// tvm-ffi-stubgen(begin): object/prim.LT
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LT {
    data_: ObjectArc<LTObj>,
}

/// The object behind [`LT`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.LT", final, immutable)]
pub struct LTObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LTObj>() == 56);
    assert!(::core::mem::align_of::<LTObj>() == 8);
};

impl Deref for LTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl LTObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl LT {
    /// Create a `LT` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = LTObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(LT => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe LT => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    LT: Operands::Compare,
        "Create the boolean node `a < b` of operands of one type.";
}

/// The comparison `a <= b` of operands of one type: a boolean with their lanes.
// tvm-ffi-stubgen(begin): object/prim.LE
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LE {
    data_: ObjectArc<LEObj>,
}

/// The object behind [`LE`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.LE", final, immutable)]
pub struct LEObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LEObj>() == 56);
    assert!(::core::mem::align_of::<LEObj>() == 8);
};

impl Deref for LEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl LEObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl LE {
    /// Create a `LE` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = LEObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(LE => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe LE => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    LE: Operands::Compare,
        "Create the boolean node `a <= b` of operands of one type.";
}

/// The comparison `a > b` of operands of one type: a boolean with their lanes.
// tvm-ffi-stubgen(begin): object/prim.GT
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GT {
    data_: ObjectArc<GTObj>,
}

/// The object behind [`GT`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.GT", final, immutable)]
pub struct GTObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<GTObj>() == 56);
    assert!(::core::mem::align_of::<GTObj>() == 8);
};

impl Deref for GTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl GTObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl GT {
    /// Create a `GT` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = GTObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(GT => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe GT => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    GT: Operands::Compare,
        "Create the boolean node `a > b` of operands of one type.";
}

/// The comparison `a >= b` of operands of one type: a boolean with their lanes.
// tvm-ffi-stubgen(begin): object/prim.GE
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GE {
    data_: ObjectArc<GEObj>,
}

/// The object behind [`GE`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.GE", final, immutable)]
pub struct GEObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<GEObj>() == 56);
    assert!(::core::mem::align_of::<GEObj>() == 8);
};

impl Deref for GEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl GEObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl GE {
    /// Create a `GE` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = GEObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(GE => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe GE => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    GE: Operands::Compare,
        "Create the boolean node `a >= b` of operands of one type.";
}

// ---------------------------------------------------------------------------------------------
// Logical operations

/// The conjunction `a && b` of boolean operands of one type.
// tvm-ffi-stubgen(begin): object/prim.And
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct And {
    data_: ObjectArc<AndObj>,
}

/// The object behind [`And`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.And", final, immutable)]
pub struct AndObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<AndObj>() == 56);
    assert!(::core::mem::align_of::<AndObj>() == 8);
};

impl Deref for AndObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl AndObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl And {
    /// Create a `And` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = AndObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(And => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe And => PrimExpr);
// tvm-ffi-stubgen(end)

impl And {
    /// Create the boolean node `a && b` over boolean operands.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not boolean, or a `TypeError` if the operand
    /// types differ.
    pub fn new(
        a: impl IntoExpr<PrimExpr>,
        b: impl IntoExpr<PrimExpr>,
        loc: impl IntoArg<Location>,
    ) -> Result<Self> {
        let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
        let ty = logical_type(&a, &b)?;
        Ok(Self::from_fields(loc.into_owned(), ty, a, b))
    }
}

/// The disjunction `a || b` of boolean operands of one type.
// tvm-ffi-stubgen(begin): object/prim.Or
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Or {
    data_: ObjectArc<OrObj>,
}

/// The object behind [`Or`].
// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Or", final, immutable)]
pub struct OrObj {
    base_: ExprObj,
    pub a: PrimExpr,
    pub b: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<OrObj>() == 56);
    assert!(::core::mem::align_of::<OrObj>() == 8);
};

impl Deref for OrObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl OrObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr, b: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a, b }
    }
}

#[allow(missing_docs)]
impl Or {
    /// Create a `Or` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
        b: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = OrObj::new(loc.into_owned(), ty.into_owned(), a.into_owned(), b.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Or => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Or => PrimExpr);
// tvm-ffi-stubgen(end)

impl Or {
    /// Create the boolean node `a || b` over boolean operands.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not boolean, or a `TypeError` if the operand
    /// types differ.
    pub fn new(
        a: impl IntoExpr<PrimExpr>,
        b: impl IntoExpr<PrimExpr>,
        loc: impl IntoArg<Location>,
    ) -> Result<Self> {
        let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
        let ty = logical_type(&a, &b)?;
        Ok(Self::from_fields(loc.into_owned(), ty, a, b))
    }
}

/// The negation `!a` of a boolean operand.
// tvm-ffi-stubgen(begin): object/prim.Not
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Not {
    data_: ObjectArc<NotObj>,
}

/// The object behind [`Not`].
// Complete: reflected fields fill [40, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Not", final, immutable)]
pub struct NotObj {
    base_: ExprObj,
    pub a: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<NotObj>() == 48);
    assert!(::core::mem::align_of::<NotObj>() == 8);
};

impl Deref for NotObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl NotObj {
    pub(crate) fn new(loc: Location, ty: PrimType, a: PrimExpr) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, a }
    }
}

#[allow(missing_docs)]
impl Not {
    /// Create a `Not` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        a: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = NotObj::new(loc.into_owned(), ty.into_owned(), a.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Not => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Not => PrimExpr);
// tvm-ffi-stubgen(end)

impl Not {
    /// Create the boolean node `!a` over a boolean operand.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `a` is not boolean.
    pub fn new(a: impl IntoExpr<PrimExpr>, loc: impl IntoArg<Location>) -> Result<Self> {
        let a = a.into_owned_expr()?;
        if !a.prim_ty().matches_code(&[DLDataTypeCode::kDLBool]) {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_fields(loc.into_owned(), bool_type(a.prim_ty())?, a))
    }
}

// ---------------------------------------------------------------------------------------------
// `Select` and `Let`

/// The choice of `true_value` where `condition` holds and `false_value` elsewhere, lane by lane
/// for a vector condition.
///
/// Both values may be evaluated; [`op::if_then_else`](super::op::if_then_else) evaluates only the
/// chosen one.
// tvm-ffi-stubgen(begin): object/prim.Select
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Select {
    data_: ObjectArc<SelectObj>,
}

/// The object behind [`Select`].
// Complete: reflected fields fill [40, 64) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Select", final, immutable)]
pub struct SelectObj {
    base_: ExprObj,
    pub condition: PrimExpr,
    pub true_value: PrimExpr,
    pub false_value: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<SelectObj>() == 64);
    assert!(::core::mem::align_of::<SelectObj>() == 8);
};

impl Deref for SelectObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl SelectObj {
    pub(crate) fn new(
        loc: Location,
        ty: PrimType,
        condition: PrimExpr,
        true_value: PrimExpr,
        false_value: PrimExpr,
    ) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, condition, true_value, false_value }
    }
}

#[allow(missing_docs)]
impl Select {
    /// Create a `Select` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        condition: impl IntoArg<PrimExpr>,
        true_value: impl IntoArg<PrimExpr>,
        false_value: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = SelectObj::new(
            loc.into_owned(),
            ty.into_owned(),
            condition.into_owned(),
            true_value.into_owned(),
            false_value.into_owned(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Select => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Select => PrimExpr);
// tvm-ffi-stubgen(end)

impl Select {
    /// Create the node choosing `true_value` where `condition` holds and `false_value` elsewhere,
    /// evaluating both, unlike [`op::if_then_else`](super::op::if_then_else).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `condition` is not boolean or has other lanes than the values,
    /// or a `TypeError` if the values are of different types.
    pub fn new(
        condition: impl IntoExpr<PrimExpr>,
        true_value: impl IntoExpr<PrimExpr>,
        false_value: impl IntoExpr<PrimExpr>,
        loc: impl IntoArg<Location>,
    ) -> Result<Self> {
        let (condition, true_value, false_value) = (
            condition.into_owned_expr()?,
            true_value.into_owned_expr()?,
            false_value.into_owned_expr()?,
        );
        let (condition_ty, true_ty) = (condition.prim_ty(), true_value.prim_ty());
        let (true_dtype, false_dtype) = (true_ty.dtype, false_value.prim_ty().dtype);
        if !condition_ty.matches_code(&[DLDataTypeCode::kDLBool]) {
            let condition = "condition_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        // C++ `GetLanesOrVScaleFactor(ty)`.
        let lanes_or_vscale_factor = |ty: &PrimType| match ty.vscale_factor() {
            Some(factor) => Ok(i32::from(factor)),
            None => support::lanes(ty),
        };
        if lanes_or_vscale_factor(condition_ty)? != lanes_or_vscale_factor(true_ty)?
            && !condition_ty.is_scalar()
        {
            let condition = "GetLanesOrVScaleFactor(condition_ty) == \
                             GetLanesOrVScaleFactor(true_ty) || condition_ty.IsScalar()";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        if false_dtype != true_dtype {
            let message = format!(
                "mismatched types. False type: {}; True type: {}",
                support::dtype_name(false_dtype),
                support::dtype_name(true_dtype)
            );
            return Err(support::check_failed(
                tvm_ffi::TYPE_ERROR,
                "false_ty == true_ty",
                &message,
            ));
        }
        let ty = true_ty.clone();
        Ok(Self::from_fields(loc.into_owned(), ty, condition, true_value, false_value))
    }
}

/// The expression `body` with `var` bound to `value`, of the type of `body`.
// tvm-ffi-stubgen(begin): object/prim.Let
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Let {
    data_: ObjectArc<LetObj>,
}

/// The object behind [`Let`].
// Complete: reflected fields fill [40, 64) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "prim.Let", final, immutable)]
pub struct LetObj {
    base_: ExprObj,
    pub var: Var,
    pub value: PrimExpr,
    pub body: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<LetObj>() == 64);
    assert!(::core::mem::align_of::<LetObj>() == 8);
};

impl Deref for LetObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl LetObj {
    pub(crate) fn new(
        loc: Location,
        ty: PrimType,
        var: Var,
        value: PrimExpr,
        body: PrimExpr,
    ) -> Self {
        let base_ = ExprObj::new(loc, ty.into());
        Self { base_, var, value, body }
    }
}

#[allow(missing_docs)]
impl Let {
    /// Create a `Let` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        ty: impl IntoArg<PrimType>,
        var: impl IntoArg<Var>,
        value: impl IntoArg<PrimExpr>,
        body: impl IntoArg<PrimExpr>,
    ) -> Self {
        let obj = LetObj::new(
            loc.into_owned(),
            ty.into_owned(),
            var.into_owned(),
            value.into_owned(),
            body.into_owned(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(Let => Expr);
tvm_ffi::impl_object_ref_upcast!(unsafe Let => PrimExpr);
// tvm-ffi-stubgen(end)

impl Let {
    /// Create the node binding `var` to `value` in `body`, of the type of `body`.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `value` is not of the type of `var`, or a `TypeError` if the
    /// type of `var` is not a `PrimType`.
    pub fn new(
        var: impl IntoArg<Var>,
        value: impl IntoExpr<PrimExpr>,
        body: impl IntoExpr<PrimExpr>,
        loc: impl IntoArg<Location>,
    ) -> Result<Self> {
        let (var, value, body) =
            (var.into_owned(), value.into_owned_expr()?, body.into_owned_expr()?);
        let value_dtype = value.prim_ty().dtype;
        let Some(var_ty) = var.ty.downcast_ref::<PrimTypeObj>() else {
            let message = format!(
                "Cannot treat type `{}` as type `ir.PrimType`",
                tvm_ffi::reflection::type_key(AnyView::from(&var.ty).type_index())
                    .unwrap_or_default()
            );
            return Err(tvm_ffi::Error::new(tvm_ffi::TYPE_ERROR, &message));
        };
        if value_dtype != var_ty.dtype {
            let condition = "value.ty() == var->ty.as_or_throw<PrimType>()";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        let ty = body.prim_ty().clone();
        Ok(Self::from_fields(loc.into_owned(), ty, var, value, body))
    }
}

// ---------------------------------------------------------------------------------------------
// What the node constructors check

/// What the constructor of a binary node checks of its operands, and the type it gives the node.
#[derive(Clone, Copy)]
enum Operands {
    /// `Add`, `Min`, ...: operands of one type, which is the node's.
    Arith,
    /// `LShift`, `BitwiseAnd`, ...: as `Arith`, of an integer type, or boolean if allowed.
    Bitwise { allow_bool: bool },
    /// `EQ`, `LT`, ...: operands of one type; the node is boolean.
    Compare,
}

/// The type of the binary node `node` of `a` and `b`, after checking the operands.
fn binary_type(node: &str, operands: Operands, a: &PrimExpr, b: &PrimExpr) -> Result<PrimType> {
    let (a_ty, b_dtype) = (a.prim_ty(), b.prim_ty().dtype);
    let a_dtype = a_ty.dtype;
    if a_dtype != b_dtype {
        let message = format!(
            "mismatched types. {} vs. {}\n",
            support::dtype_name(a_dtype),
            support::dtype_name(b_dtype)
        );
        let condition = "a_ty->dtype == b_ty->dtype";
        return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, &message));
    }
    match operands {
        Operands::Arith => Ok(a_ty.clone()),
        Operands::Bitwise { allow_bool } => {
            if a_ty.matches_code(&[DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt])
                || (allow_bool && a_ty.matches_code(&[DLDataTypeCode::kDLBool]))
            {
                return Ok(a_ty.clone());
            }
            let condition = format!(
                "a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) || \
                 ({allow_bool} && a.ty().MatchesCode(DLDataTypeCode::kDLBool))"
            );
            let operands = if allow_bool { "integer or boolean" } else { "integer" };
            let message = format!("{node} requires {operands} operands");
            Err(support::check_failed(tvm_ffi::TYPE_ERROR, &condition, &message))
        }
        Operands::Compare => bool_type(a_ty),
    }
}

/// The boolean type with the lanes of `operand`.
fn bool_type(operand: &PrimType) -> Result<PrimType> {
    use crate::ir::dtype;
    PrimType::try_from(dtype::with_lanes(dtype::bool(), operand.dtype.lanes))
}

/// The type of the logical node of `a` and `b`, after checking that the operands are booleans of
/// one type.
fn logical_type(a: &PrimExpr, b: &PrimExpr) -> Result<PrimType> {
    for (ty, operand) in [(a.prim_ty(), "a_ty"), (b.prim_ty(), "b_ty")] {
        if !ty.matches_code(&[DLDataTypeCode::kDLBool]) {
            let condition = format!("{operand}.MatchesCode(DLDataTypeCode::kDLBool)");
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, &condition, ""));
        }
    }
    if a.prim_ty().dtype != b.prim_ty().dtype {
        return Err(support::check_failed(tvm_ffi::TYPE_ERROR, "a_ty == b_ty", "mismatched types"));
    }
    bool_type(a.prim_ty())
}
