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
//! Scalar expressions: casts, arithmetic, bitwise and logical operations, comparisons, `Select`
//! and `Let`.
//!
//! The objects `python/tvm_tirx/ir/prim/expr.py` defines. Their `new` checks the operand types,
//! and never promotes them as the operators of [`super::op`] do.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::prim::{Add, Select, LT};
//! use tvm_tirx::ir::{PrimType, PrimVar, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//!
//! let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
//! let next = Add::new(&x, 1, UNKNOWN_LOC)?;
//! let capped = Select::new(LT::new(&x, 8, UNKNOWN_LOC)?, next, 8, UNKNOWN_LOC)?;
//! assert_eq!(script(&capped)?, "T.Select(x < 8, x + 1, 8)");
//!
//! let f = PrimVar::new("f", PrimType::float(32), UNKNOWN_LOC);
//! assert!(Add::new(&x, &f, UNKNOWN_LOC).is_err());
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// tvm-ffi-stubgen(missing-docs): allow
// `custom-new` reserves `new` for a constructor with the C++ one's semantics (operand type checks
// and the derived result type); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): prim.Cast
// tvm-ffi-stubgen(custom-new): prim.LShift
// tvm-ffi-stubgen(custom-new): prim.RShift
// tvm-ffi-stubgen(custom-new): prim.BitwiseAnd
// tvm-ffi-stubgen(custom-new): prim.BitwiseOr
// tvm-ffi-stubgen(custom-new): prim.BitwiseXor
// tvm-ffi-stubgen(custom-new): prim.BitwiseNot
// tvm-ffi-stubgen(custom-new): prim.Add
// tvm-ffi-stubgen(custom-new): prim.Sub
// tvm-ffi-stubgen(custom-new): prim.Mul
// tvm-ffi-stubgen(custom-new): prim.Div
// tvm-ffi-stubgen(custom-new): prim.Mod
// tvm-ffi-stubgen(custom-new): prim.FloorDiv
// tvm-ffi-stubgen(custom-new): prim.FloorMod
// tvm-ffi-stubgen(custom-new): prim.Min
// tvm-ffi-stubgen(custom-new): prim.Max
// tvm-ffi-stubgen(custom-new): prim.EQ
// tvm-ffi-stubgen(custom-new): prim.NE
// tvm-ffi-stubgen(custom-new): prim.LT
// tvm-ffi-stubgen(custom-new): prim.LE
// tvm-ffi-stubgen(custom-new): prim.GT
// tvm-ffi-stubgen(custom-new): prim.GE
// tvm-ffi-stubgen(custom-new): prim.And
// tvm-ffi-stubgen(custom-new): prim.Or
// tvm-ffi-stubgen(custom-new): prim.Not
// tvm-ffi-stubgen(custom-new): prim.Select
// tvm-ffi-stubgen(custom-new): prim.Let
// Every node is typed by a `PrimType` and holds `PrimExpr` operands, as C++ `Add` derives from
// `PrimExpr` and `AddObj` holds `PrimExpr a, b`: its allocator takes a `PrimType`, and it converts
// into a `PrimExpr` without a check.
// tvm-ffi-stubgen(field): prim.Cast.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Cast.value -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Cast -> PrimExpr
// tvm-ffi-stubgen(field): prim.LShift.ty -> PrimType
// tvm-ffi-stubgen(field): prim.LShift.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.LShift.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.LShift -> PrimExpr
// tvm-ffi-stubgen(field): prim.RShift.ty -> PrimType
// tvm-ffi-stubgen(field): prim.RShift.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.RShift.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.RShift -> PrimExpr
// tvm-ffi-stubgen(field): prim.BitwiseAnd.ty -> PrimType
// tvm-ffi-stubgen(field): prim.BitwiseAnd.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.BitwiseAnd.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.BitwiseAnd -> PrimExpr
// tvm-ffi-stubgen(field): prim.BitwiseOr.ty -> PrimType
// tvm-ffi-stubgen(field): prim.BitwiseOr.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.BitwiseOr.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.BitwiseOr -> PrimExpr
// tvm-ffi-stubgen(field): prim.BitwiseXor.ty -> PrimType
// tvm-ffi-stubgen(field): prim.BitwiseXor.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.BitwiseXor.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.BitwiseXor -> PrimExpr
// tvm-ffi-stubgen(field): prim.BitwiseNot.ty -> PrimType
// tvm-ffi-stubgen(field): prim.BitwiseNot.a -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.BitwiseNot -> PrimExpr
// tvm-ffi-stubgen(field): prim.Add.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Add.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Add.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Add -> PrimExpr
// tvm-ffi-stubgen(field): prim.Sub.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Sub.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Sub.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Sub -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mul.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Mul.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mul.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Mul -> PrimExpr
// tvm-ffi-stubgen(field): prim.Div.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Div.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Div.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Div -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mod.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Mod.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Mod.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Mod -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorDiv.ty -> PrimType
// tvm-ffi-stubgen(field): prim.FloorDiv.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorDiv.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.FloorDiv -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorMod.ty -> PrimType
// tvm-ffi-stubgen(field): prim.FloorMod.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.FloorMod.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.FloorMod -> PrimExpr
// tvm-ffi-stubgen(field): prim.Min.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Min.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Min.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Min -> PrimExpr
// tvm-ffi-stubgen(field): prim.Max.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Max.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Max.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Max -> PrimExpr
// tvm-ffi-stubgen(field): prim.EQ.ty -> PrimType
// tvm-ffi-stubgen(field): prim.EQ.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.EQ.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.EQ -> PrimExpr
// tvm-ffi-stubgen(field): prim.NE.ty -> PrimType
// tvm-ffi-stubgen(field): prim.NE.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.NE.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.NE -> PrimExpr
// tvm-ffi-stubgen(field): prim.LT.ty -> PrimType
// tvm-ffi-stubgen(field): prim.LT.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.LT.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.LT -> PrimExpr
// tvm-ffi-stubgen(field): prim.LE.ty -> PrimType
// tvm-ffi-stubgen(field): prim.LE.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.LE.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.LE -> PrimExpr
// tvm-ffi-stubgen(field): prim.GT.ty -> PrimType
// tvm-ffi-stubgen(field): prim.GT.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.GT.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.GT -> PrimExpr
// tvm-ffi-stubgen(field): prim.GE.ty -> PrimType
// tvm-ffi-stubgen(field): prim.GE.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.GE.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.GE -> PrimExpr
// tvm-ffi-stubgen(field): prim.And.ty -> PrimType
// tvm-ffi-stubgen(field): prim.And.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.And.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.And -> PrimExpr
// tvm-ffi-stubgen(field): prim.Or.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Or.a -> PrimExpr
// tvm-ffi-stubgen(field): prim.Or.b -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Or -> PrimExpr
// tvm-ffi-stubgen(field): prim.Not.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Not.a -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Not -> PrimExpr
// tvm-ffi-stubgen(field): prim.Select.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Select.condition -> PrimExpr
// tvm-ffi-stubgen(field): prim.Select.true_value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Select.false_value -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Select -> PrimExpr
// tvm-ffi-stubgen(field): prim.Let.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Let.value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Let.body -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Let -> PrimExpr
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataTypeCode
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr

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
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

/// `impl $node { pub fn new(a, b, loc) }` for each binary node, with the checks of its C++
/// constructor.
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
                    loc: impl Into<Location>,
                ) -> Result<Self> {
                    let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
                    let ty = binary_type(stringify!($node), $operands, &a, &b)?;
                    Ok(Self::from_fields(loc.into(), ty, a, b))
                }
            }
        )+
    };
}

// ---------------------------------------------------------------------------------------------
// Casts

// tvm-ffi-stubgen(begin): object/prim.Cast
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Cast {
    data_: ObjectArc<CastObj>,
}

/// The object behind [`Cast`].
///
/// Complete: reflected fields fill [40, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Cast"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        value: impl Into<PrimExpr>,
    ) -> Self {
        let obj = CastObj::new(loc.into(), ty.into(), value.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for CastObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Cast => Expr, Cast => PrimExpr);
// tvm-ffi-stubgen(end)

impl Cast {
    /// Create the node converting `value` to `ty`, which has as many lanes as `value`
    /// (C++ `Cast(ty, value, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `ty` and the type of `value` have different numbers of lanes.
    pub fn new(
        ty: impl Into<PrimType>,
        value: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (ty, value) = (ty.into(), value.into_owned_expr()?);
        let value_dtype = value.prim_ty().dtype;
        if ty.dtype.lanes != value_dtype.lanes {
            let condition = "ty->dtype.lanes == value_expr_ty->dtype.lanes";
            let (lhs, rhs) = (ty.dtype.lanes, value_dtype.lanes);
            return Err(support::compare_failed(tvm_ffi::INTERNAL_ERROR, condition, lhs, rhs, ""));
        }
        Ok(Self::from_fields(loc.into(), ty, value))
    }
}

// ---------------------------------------------------------------------------------------------
// Bitwise operations

// tvm-ffi-stubgen(begin): object/prim.LShift
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LShift {
    data_: ObjectArc<LShiftObj>,
}

/// The object behind [`LShift`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LShift"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = LShiftObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for LShiftObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(LShift => Expr, LShift => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    LShift: Operands::Bitwise { allow_bool: false },
        "Create the node `a << b` of integer operands of one type, as C++ \
         `LShift(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.RShift
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RShift {
    data_: ObjectArc<RShiftObj>,
}

/// The object behind [`RShift`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.RShift"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = RShiftObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for RShiftObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(RShift => Expr, RShift => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    RShift: Operands::Bitwise { allow_bool: false },
        "Create the node `a >> b` of integer operands of one type, as C++ \
         `RShift(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.BitwiseAnd
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseAnd {
    data_: ObjectArc<BitwiseAndObj>,
}

/// The object behind [`BitwiseAnd`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseAnd"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = BitwiseAndObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BitwiseAndObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(BitwiseAnd => Expr, BitwiseAnd => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    BitwiseAnd: Operands::Bitwise { allow_bool: true },
        "Create the node `a & b` of integer or boolean operands of one type, as C++ \
         `BitwiseAnd(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.BitwiseOr
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseOr {
    data_: ObjectArc<BitwiseOrObj>,
}

/// The object behind [`BitwiseOr`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseOr"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = BitwiseOrObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BitwiseOrObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(BitwiseOr => Expr, BitwiseOr => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    BitwiseOr: Operands::Bitwise { allow_bool: true },
        "Create the node `a | b` of integer or boolean operands of one type, as C++ \
         `BitwiseOr(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.BitwiseXor
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseXor {
    data_: ObjectArc<BitwiseXorObj>,
}

/// The object behind [`BitwiseXor`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseXor"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = BitwiseXorObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BitwiseXorObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(BitwiseXor => Expr, BitwiseXor => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    BitwiseXor: Operands::Bitwise { allow_bool: true },
        "Create the node `a ^ b` of integer or boolean operands of one type, as C++ \
         `BitwiseXor(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.BitwiseNot
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseNot {
    data_: ObjectArc<BitwiseNotObj>,
}

/// The object behind [`BitwiseNot`].
///
/// Complete: reflected fields fill [40, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseNot"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
    ) -> Self {
        let obj = BitwiseNotObj::new(loc.into(), ty.into(), a.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BitwiseNotObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(BitwiseNot => Expr, BitwiseNot => PrimExpr);
// tvm-ffi-stubgen(end)

impl BitwiseNot {
    /// Create the node `~a` over an integer or boolean operand (C++ `BitwiseNot(a, loc)`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `a` is not of an integer or boolean type.
    pub fn new(a: impl IntoExpr<PrimExpr>, loc: impl Into<Location>) -> Result<Self> {
        let a = a.into_owned_expr()?;
        let integer_or_bool =
            [DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLBool];
        if !a.prim_ty().matches_code(&integer_or_bool) {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) || \
                             a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            let message = "BitwiseNot requires an integer or boolean operand";
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, message));
        }
        Ok(Self::from_fields(loc.into(), a.prim_ty().clone(), a))
    }
}

// ---------------------------------------------------------------------------------------------
// Arithmetic

// tvm-ffi-stubgen(begin): object/prim.Add
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Add {
    data_: ObjectArc<AddObj>,
}

/// The object behind [`Add`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Add"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = AddObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for AddObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Add => Expr, Add => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Add: Operands::Arith,
        "Create the node `a + b` of operands of one type, as C++ `Add(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.Sub
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Sub {
    data_: ObjectArc<SubObj>,
}

/// The object behind [`Sub`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Sub"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = SubObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for SubObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Sub => Expr, Sub => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Sub: Operands::Arith,
        "Create the node `a - b` of operands of one type, as C++ `Sub(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.Mul
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mul {
    data_: ObjectArc<MulObj>,
}

/// The object behind [`Mul`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mul"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = MulObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for MulObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Mul => Expr, Mul => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Mul: Operands::Arith,
        "Create the node `a * b` of operands of one type, as C++ `Mul(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.Div
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Div {
    data_: ObjectArc<DivObj>,
}

/// The object behind [`Div`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Div"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = DivObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for DivObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Div => Expr, Div => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Div: Operands::Arith,
        "Create the node `a / b` of operands of one type, as C++ `Div(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.Mod
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mod {
    data_: ObjectArc<ModObj>,
}

/// The object behind [`Mod`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mod"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = ModObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ModObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Mod => Expr, Mod => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Mod: Operands::Arith,
        "Create the truncated node `a % b` of operands of one type, as C++ `Mod(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.FloorDiv
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorDiv {
    data_: ObjectArc<FloorDivObj>,
}

/// The object behind [`FloorDiv`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorDiv"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = FloorDivObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for FloorDivObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(FloorDiv => Expr, FloorDiv => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    FloorDiv: Operands::Arith,
        "Create the node `floor(a / b)` of operands of one type, as C++ `FloorDiv(a, b, loc)` \
         does.";
}

// tvm-ffi-stubgen(begin): object/prim.FloorMod
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorMod {
    data_: ObjectArc<FloorModObj>,
}

/// The object behind [`FloorMod`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorMod"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = FloorModObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for FloorModObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(FloorMod => Expr, FloorMod => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    FloorMod: Operands::Arith,
        "Create the node `a - floor(a / b) * b` of operands of one type, as C++ \
         `FloorMod(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.Min
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Min {
    data_: ObjectArc<MinObj>,
}

/// The object behind [`Min`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Min"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = MinObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for MinObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Min => Expr, Min => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Min: Operands::Arith,
        "Create the node `min(a, b)` of operands of one type, as C++ `Min(a, b, loc)` does.";
}

// tvm-ffi-stubgen(begin): object/prim.Max
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Max {
    data_: ObjectArc<MaxObj>,
}

/// The object behind [`Max`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Max"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = MaxObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for MaxObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Max => Expr, Max => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    Max: Operands::Arith,
        "Create the node `max(a, b)` of operands of one type, as C++ `Max(a, b, loc)` does.";
}

// ---------------------------------------------------------------------------------------------
// Comparisons

// tvm-ffi-stubgen(begin): object/prim.EQ
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EQ {
    data_: ObjectArc<EQObj>,
}

/// The object behind [`EQ`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.EQ"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = EQObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for EQObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(EQ => Expr, EQ => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    EQ: Operands::Compare,
        "Create the boolean node `a == b` of operands of one type, as C++ `EQ(a, b, loc)` \
         does.";
}

// tvm-ffi-stubgen(begin): object/prim.NE
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct NE {
    data_: ObjectArc<NEObj>,
}

/// The object behind [`NE`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.NE"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = NEObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for NEObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(NE => Expr, NE => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    NE: Operands::Compare,
        "Create the boolean node `a != b` of operands of one type, as C++ `NE(a, b, loc)` \
         does.";
}

// tvm-ffi-stubgen(begin): object/prim.LT
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LT {
    data_: ObjectArc<LTObj>,
}

/// The object behind [`LT`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LT"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = LTObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for LTObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(LT => Expr, LT => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    LT: Operands::Compare,
        "Create the boolean node `a < b` of operands of one type, as C++ `LT(a, b, loc)` \
         does.";
}

// tvm-ffi-stubgen(begin): object/prim.LE
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LE {
    data_: ObjectArc<LEObj>,
}

/// The object behind [`LE`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LE"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = LEObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for LEObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(LE => Expr, LE => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    LE: Operands::Compare,
        "Create the boolean node `a <= b` of operands of one type, as C++ `LE(a, b, loc)` \
         does.";
}

// tvm-ffi-stubgen(begin): object/prim.GT
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GT {
    data_: ObjectArc<GTObj>,
}

/// The object behind [`GT`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GT"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = GTObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for GTObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(GT => Expr, GT => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    GT: Operands::Compare,
        "Create the boolean node `a > b` of operands of one type, as C++ `GT(a, b, loc)` \
         does.";
}

// tvm-ffi-stubgen(begin): object/prim.GE
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GE {
    data_: ObjectArc<GEObj>,
}

/// The object behind [`GE`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GE"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = GEObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for GEObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(GE => Expr, GE => PrimExpr);
// tvm-ffi-stubgen(end)

binary_new! {
    GE: Operands::Compare,
        "Create the boolean node `a >= b` of operands of one type, as C++ `GE(a, b, loc)` \
         does.";
}

// ---------------------------------------------------------------------------------------------
// Logical operations

// tvm-ffi-stubgen(begin): object/prim.And
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct And {
    data_: ObjectArc<AndObj>,
}

/// The object behind [`And`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.And"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = AndObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for AndObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(And => Expr, And => PrimExpr);
// tvm-ffi-stubgen(end)

impl And {
    /// Create the boolean node `a && b` over boolean operands (C++ `And(a, b, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not boolean, or a `TypeError` if the operand
    /// types differ.
    pub fn new(
        a: impl IntoExpr<PrimExpr>,
        b: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
        let ty = logical_type(&a, &b)?;
        Ok(Self::from_fields(loc.into(), ty, a, b))
    }
}

// tvm-ffi-stubgen(begin): object/prim.Or
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Or {
    data_: ObjectArc<OrObj>,
}

/// The object behind [`Or`].
///
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Or"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
        b: impl Into<PrimExpr>,
    ) -> Self {
        let obj = OrObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for OrObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Or => Expr, Or => PrimExpr);
// tvm-ffi-stubgen(end)

impl Or {
    /// Create the boolean node `a || b` over boolean operands (C++ `Or(a, b, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not boolean, or a `TypeError` if the operand
    /// types differ.
    pub fn new(
        a: impl IntoExpr<PrimExpr>,
        b: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
        let ty = logical_type(&a, &b)?;
        Ok(Self::from_fields(loc.into(), ty, a, b))
    }
}

// tvm-ffi-stubgen(begin): object/prim.Not
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Not {
    data_: ObjectArc<NotObj>,
}

/// The object behind [`Not`].
///
/// Complete: reflected fields fill [40, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Not"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        a: impl Into<PrimExpr>,
    ) -> Self {
        let obj = NotObj::new(loc.into(), ty.into(), a.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for NotObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Not => Expr, Not => PrimExpr);
// tvm-ffi-stubgen(end)

impl Not {
    /// Create the boolean node `!a` over a boolean operand (C++ `Not(a, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `a` is not boolean.
    pub fn new(a: impl IntoExpr<PrimExpr>, loc: impl Into<Location>) -> Result<Self> {
        let a = a.into_owned_expr()?;
        if !a.prim_ty().matches_code(&[DLDataTypeCode::kDLBool]) {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_fields(loc.into(), bool_type(a.prim_ty())?, a))
    }
}

// ---------------------------------------------------------------------------------------------
// `Select` and `Let`

// tvm-ffi-stubgen(begin): object/prim.Select
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Select {
    data_: ObjectArc<SelectObj>,
}

/// The object behind [`Select`].
///
/// Complete: reflected fields fill [40, 64) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Select"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        condition: impl Into<PrimExpr>,
        true_value: impl Into<PrimExpr>,
        false_value: impl Into<PrimExpr>,
    ) -> Self {
        let obj = SelectObj::new(
            loc.into(),
            ty.into(),
            condition.into(),
            true_value.into(),
            false_value.into(),
        );
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for SelectObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Select => Expr, Select => PrimExpr);
// tvm-ffi-stubgen(end)

impl Select {
    /// Create the node choosing `true_value` where `condition` holds and `false_value` elsewhere,
    /// evaluating both, unlike [`op::if_then_else`](super::op::if_then_else) (C++ `Select`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `condition` is not boolean or has other lanes than the values,
    /// or a `TypeError` if the values are of different types.
    pub fn new(
        condition: impl IntoExpr<PrimExpr>,
        true_value: impl IntoExpr<PrimExpr>,
        false_value: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
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
        Ok(Self::from_fields(loc.into(), ty, condition, true_value, false_value))
    }
}

// tvm-ffi-stubgen(begin): object/prim.Let
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Let {
    data_: ObjectArc<LetObj>,
}

/// The object behind [`Let`].
///
/// Complete: reflected fields fill [40, 64) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Let"]
#[type_final]
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
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        var: impl Into<Var>,
        value: impl Into<PrimExpr>,
        body: impl Into<PrimExpr>,
    ) -> Self {
        let obj = LetObj::new(loc.into(), ty.into(), var.into(), value.into(), body.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for LetObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Let => Expr, Let => PrimExpr);
// tvm-ffi-stubgen(end)

impl Let {
    /// Create the node binding `var` to `value` in `body`, of the type of `body`
    /// (C++ `Let(var, value, body, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `value` is not of the type of `var`, or a `TypeError` if the
    /// type of `var` is not a `PrimType`.
    pub fn new(
        var: impl Into<Var>,
        value: impl IntoExpr<PrimExpr>,
        body: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (var, value, body) = (var.into(), value.into_owned_expr()?, body.into_owned_expr()?);
        let value_dtype = value.prim_ty().dtype;
        let Some(var_ty) = var.ty.downcast_ref::<PrimTypeObj>() else {
            let message = format!(
                "Cannot treat type `{}` as type `ir.PrimType`",
                support::type_key(AnyView::from(&var.ty).type_index())
            );
            return Err(tvm_ffi::Error::new(tvm_ffi::TYPE_ERROR, &message, ""));
        };
        if value_dtype != var_ty.dtype {
            let condition = "value.ty() == var->ty.as_or_throw<PrimType>()";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        let ty = body.prim_ty().clone();
        Ok(Self::from_fields(loc.into(), ty, var, value, body))
    }
}

// ---------------------------------------------------------------------------------------------
// What the node constructors check

/// What the constructor of a binary node checks of its operands, and the type it gives the node
/// (`TVM_DEFINE_BINOP_CONSTRUCTOR` and its siblings in `src/ir/prim/expr.cc`).
#[derive(Clone, Copy)]
enum Operands {
    /// `Add`, `Min`, ...: operands of one type, which is the node's.
    Arith,
    /// `LShift`, `BitwiseAnd`, ...: as `Arith`, of an integer type, or boolean if allowed.
    Bitwise { allow_bool: bool },
    /// `EQ`, `LT`, ...: operands of one type; the node is boolean.
    Compare,
}

/// The type of the binary node `node` of `a` and `b`, after the checks of its C++ constructor.
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

/// The boolean type with the lanes of `operand` (C++ `PrimType(DLDataType{kDLBool, 8, lanes})`).
fn bool_type(operand: &PrimType) -> Result<PrimType> {
    use crate::ir::dtype;
    PrimType::try_from(dtype::with_lanes(dtype::bool(), operand.dtype.lanes))
}

/// The type of the logical node of `a` and `b`, after the checks of the C++ `And` and `Or`
/// constructors: boolean operands of one type.
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
