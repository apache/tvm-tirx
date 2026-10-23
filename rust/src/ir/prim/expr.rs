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

//! Scalar expressions: casts, arithmetic, bitwise and logical operations, comparisons, `Select`
//! and `Let`.
//!
//! The objects `python/tvm_tirx/ir/prim/expr.py` defines. A binary operation such as [`Add`]
//! holds its operands `a` and `b`. Their `new` is written in Rust after the C++ constructor: it
//! checks the operand types and derives the result type. The type-promoting operators are in
//! [`super::op`].

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
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataType
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::Var;
use crate::ir::location::Location;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::PrimTypeObj;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::AnyView;
use tvm_ffi::DLDataType;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Cast
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Cast"]
#[type_final]
pub struct CastObj {
    base: ExprObj,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<CastObj>() == 48);
    assert!(::core::mem::align_of::<CastObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Cast {
    base: ObjectArc<CastObj>,
}

impl Deref for Cast {
    type Target = CastObj;
    fn deref(&self) -> &CastObj {
        &self.base
    }
}

impl Deref for CastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl CastObj {
    pub(crate) fn new(loc: Location, ty: Type, value: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, value }
    }
}

impl Cast {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        value: impl Into<Expr>,
    ) -> Self {
        let obj = CastObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Cast => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LShift
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LShift"]
#[type_final]
pub struct LShiftObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LShiftObj>() == 56);
    assert!(::core::mem::align_of::<LShiftObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LShift {
    base: ObjectArc<LShiftObj>,
}

impl Deref for LShift {
    type Target = LShiftObj;
    fn deref(&self) -> &LShiftObj {
        &self.base
    }
}

impl Deref for LShiftObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LShiftObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl LShift {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = LShiftObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LShift => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.RShift
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.RShift"]
#[type_final]
pub struct RShiftObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<RShiftObj>() == 56);
    assert!(::core::mem::align_of::<RShiftObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RShift {
    base: ObjectArc<RShiftObj>,
}

impl Deref for RShift {
    type Target = RShiftObj;
    fn deref(&self) -> &RShiftObj {
        &self.base
    }
}

impl Deref for RShiftObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl RShiftObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl RShift {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = RShiftObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(RShift => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseAnd
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseAnd"]
#[type_final]
pub struct BitwiseAndObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseAndObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseAndObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseAnd {
    base: ObjectArc<BitwiseAndObj>,
}

impl Deref for BitwiseAnd {
    type Target = BitwiseAndObj;
    fn deref(&self) -> &BitwiseAndObj {
        &self.base
    }
}

impl Deref for BitwiseAndObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseAndObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl BitwiseAnd {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = BitwiseAndObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseAnd => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseOr
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseOr"]
#[type_final]
pub struct BitwiseOrObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseOrObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseOrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseOr {
    base: ObjectArc<BitwiseOrObj>,
}

impl Deref for BitwiseOr {
    type Target = BitwiseOrObj;
    fn deref(&self) -> &BitwiseOrObj {
        &self.base
    }
}

impl Deref for BitwiseOrObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseOrObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl BitwiseOr {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = BitwiseOrObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseOr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseXor
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseXor"]
#[type_final]
pub struct BitwiseXorObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseXorObj>() == 56);
    assert!(::core::mem::align_of::<BitwiseXorObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseXor {
    base: ObjectArc<BitwiseXorObj>,
}

impl Deref for BitwiseXor {
    type Target = BitwiseXorObj;
    fn deref(&self) -> &BitwiseXorObj {
        &self.base
    }
}

impl Deref for BitwiseXorObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseXorObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl BitwiseXor {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = BitwiseXorObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseXor => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.BitwiseNot
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.BitwiseNot"]
#[type_final]
pub struct BitwiseNotObj {
    base: ExprObj,
    pub a: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BitwiseNotObj>() == 48);
    assert!(::core::mem::align_of::<BitwiseNotObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BitwiseNot {
    base: ObjectArc<BitwiseNotObj>,
}

impl Deref for BitwiseNot {
    type Target = BitwiseNotObj;
    fn deref(&self) -> &BitwiseNotObj {
        &self.base
    }
}

impl Deref for BitwiseNotObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BitwiseNotObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a }
    }
}

impl BitwiseNot {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, ty: impl Into<Type>, a: impl Into<Expr>) -> Self {
        let obj = BitwiseNotObj::new(loc.into(), ty.into(), a.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BitwiseNot => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Add
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<AddObj>() == 56);
    assert!(::core::mem::align_of::<AddObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Add {
    base: ObjectArc<AddObj>,
}

impl Deref for Add {
    type Target = AddObj;
    fn deref(&self) -> &AddObj {
        &self.base
    }
}

impl Deref for AddObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl AddObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Add {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = AddObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Add => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Sub
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Sub"]
#[type_final]
pub struct SubObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<SubObj>() == 56);
    assert!(::core::mem::align_of::<SubObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Sub {
    base: ObjectArc<SubObj>,
}

impl Deref for Sub {
    type Target = SubObj;
    fn deref(&self) -> &SubObj {
        &self.base
    }
}

impl Deref for SubObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl SubObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Sub {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = SubObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Sub => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Mul
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mul"]
#[type_final]
pub struct MulObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<MulObj>() == 56);
    assert!(::core::mem::align_of::<MulObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mul {
    base: ObjectArc<MulObj>,
}

impl Deref for Mul {
    type Target = MulObj;
    fn deref(&self) -> &MulObj {
        &self.base
    }
}

impl Deref for MulObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MulObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Mul {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = MulObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Mul => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Div
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Div"]
#[type_final]
pub struct DivObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<DivObj>() == 56);
    assert!(::core::mem::align_of::<DivObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Div {
    base: ObjectArc<DivObj>,
}

impl Deref for Div {
    type Target = DivObj;
    fn deref(&self) -> &DivObj {
        &self.base
    }
}

impl Deref for DivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl DivObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Div {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = DivObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Div => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Mod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Mod"]
#[type_final]
pub struct ModObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<ModObj>() == 56);
    assert!(::core::mem::align_of::<ModObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Mod {
    base: ObjectArc<ModObj>,
}

impl Deref for Mod {
    type Target = ModObj;
    fn deref(&self) -> &ModObj {
        &self.base
    }
}

impl Deref for ModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ModObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Mod {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = ModObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Mod => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.FloorDiv
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorDiv"]
#[type_final]
pub struct FloorDivObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorDivObj>() == 56);
    assert!(::core::mem::align_of::<FloorDivObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorDiv {
    base: ObjectArc<FloorDivObj>,
}

impl Deref for FloorDiv {
    type Target = FloorDivObj;
    fn deref(&self) -> &FloorDivObj {
        &self.base
    }
}

impl Deref for FloorDivObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl FloorDivObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl FloorDiv {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = FloorDivObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloorDiv => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.FloorMod
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.FloorMod"]
#[type_final]
pub struct FloorModObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<FloorModObj>() == 56);
    assert!(::core::mem::align_of::<FloorModObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloorMod {
    base: ObjectArc<FloorModObj>,
}

impl Deref for FloorMod {
    type Target = FloorModObj;
    fn deref(&self) -> &FloorModObj {
        &self.base
    }
}

impl Deref for FloorModObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl FloorModObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl FloorMod {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = FloorModObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloorMod => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Min
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Min"]
#[type_final]
pub struct MinObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<MinObj>() == 56);
    assert!(::core::mem::align_of::<MinObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Min {
    base: ObjectArc<MinObj>,
}

impl Deref for Min {
    type Target = MinObj;
    fn deref(&self) -> &MinObj {
        &self.base
    }
}

impl Deref for MinObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MinObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Min {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = MinObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Min => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Max
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Max"]
#[type_final]
pub struct MaxObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<MaxObj>() == 56);
    assert!(::core::mem::align_of::<MaxObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Max {
    base: ObjectArc<MaxObj>,
}

impl Deref for Max {
    type Target = MaxObj;
    fn deref(&self) -> &MaxObj {
        &self.base
    }
}

impl Deref for MaxObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl MaxObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Max {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = MaxObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Max => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.EQ
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.EQ"]
#[type_final]
pub struct EQObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<EQObj>() == 56);
    assert!(::core::mem::align_of::<EQObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct EQ {
    base: ObjectArc<EQObj>,
}

impl Deref for EQ {
    type Target = EQObj;
    fn deref(&self) -> &EQObj {
        &self.base
    }
}

impl Deref for EQObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl EQObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl EQ {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = EQObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(EQ => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.NE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.NE"]
#[type_final]
pub struct NEObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<NEObj>() == 56);
    assert!(::core::mem::align_of::<NEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct NE {
    base: ObjectArc<NEObj>,
}

impl Deref for NE {
    type Target = NEObj;
    fn deref(&self) -> &NEObj {
        &self.base
    }
}

impl Deref for NEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl NEObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl NE {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = NEObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(NE => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LT"]
#[type_final]
pub struct LTObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LTObj>() == 56);
    assert!(::core::mem::align_of::<LTObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LT {
    base: ObjectArc<LTObj>,
}

impl Deref for LT {
    type Target = LTObj;
    fn deref(&self) -> &LTObj {
        &self.base
    }
}

impl Deref for LTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LTObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl LT {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = LTObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LT => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.LE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.LE"]
#[type_final]
pub struct LEObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LEObj>() == 56);
    assert!(::core::mem::align_of::<LEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LE {
    base: ObjectArc<LEObj>,
}

impl Deref for LE {
    type Target = LEObj;
    fn deref(&self) -> &LEObj {
        &self.base
    }
}

impl Deref for LEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LEObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl LE {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = LEObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LE => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.GT
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GT"]
#[type_final]
pub struct GTObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<GTObj>() == 56);
    assert!(::core::mem::align_of::<GTObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GT {
    base: ObjectArc<GTObj>,
}

impl Deref for GT {
    type Target = GTObj;
    fn deref(&self) -> &GTObj {
        &self.base
    }
}

impl Deref for GTObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GTObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl GT {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = GTObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GT => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.GE
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.GE"]
#[type_final]
pub struct GEObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<GEObj>() == 56);
    assert!(::core::mem::align_of::<GEObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GE {
    base: ObjectArc<GEObj>,
}

impl Deref for GE {
    type Target = GEObj;
    fn deref(&self) -> &GEObj {
        &self.base
    }
}

impl Deref for GEObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GEObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl GE {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = GEObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GE => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.And
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.And"]
#[type_final]
pub struct AndObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<AndObj>() == 56);
    assert!(::core::mem::align_of::<AndObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct And {
    base: ObjectArc<AndObj>,
}

impl Deref for And {
    type Target = AndObj;
    fn deref(&self) -> &AndObj {
        &self.base
    }
}

impl Deref for AndObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl AndObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl And {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = AndObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(And => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Or
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Or"]
#[type_final]
pub struct OrObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<OrObj>() == 56);
    assert!(::core::mem::align_of::<OrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Or {
    base: ObjectArc<OrObj>,
}

impl Deref for Or {
    type Target = OrObj;
    fn deref(&self) -> &OrObj {
        &self.base
    }
}

impl Deref for OrObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OrObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr, b: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a, b }
    }
}

impl Or {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        a: impl Into<Expr>,
        b: impl Into<Expr>,
    ) -> Self {
        let obj = OrObj::new(loc.into(), ty.into(), a.into(), b.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Or => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Not
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Not"]
#[type_final]
pub struct NotObj {
    base: ExprObj,
    pub a: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<NotObj>() == 48);
    assert!(::core::mem::align_of::<NotObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Not {
    base: ObjectArc<NotObj>,
}

impl Deref for Not {
    type Target = NotObj;
    fn deref(&self) -> &NotObj {
        &self.base
    }
}

impl Deref for NotObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl NotObj {
    pub(crate) fn new(loc: Location, ty: Type, a: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, a }
    }
}

impl Not {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, ty: impl Into<Type>, a: impl Into<Expr>) -> Self {
        let obj = NotObj::new(loc.into(), ty.into(), a.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Not => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Select
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Select"]
#[type_final]
pub struct SelectObj {
    base: ExprObj,
    pub condition: Expr,
    pub true_value: Expr,
    pub false_value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<SelectObj>() == 64);
    assert!(::core::mem::align_of::<SelectObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Select {
    base: ObjectArc<SelectObj>,
}

impl Deref for Select {
    type Target = SelectObj;
    fn deref(&self) -> &SelectObj {
        &self.base
    }
}

impl Deref for SelectObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl SelectObj {
    pub(crate) fn new(
        loc: Location,
        ty: Type,
        condition: Expr,
        true_value: Expr,
        false_value: Expr,
    ) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, condition, true_value, false_value }
    }
}

impl Select {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        condition: impl Into<Expr>,
        true_value: impl Into<Expr>,
        false_value: impl Into<Expr>,
    ) -> Self {
        let obj = SelectObj::new(
            loc.into(),
            ty.into(),
            condition.into(),
            true_value.into(),
            false_value.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Select => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Let
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Let"]
#[type_final]
pub struct LetObj {
    base: ExprObj,
    pub var: Var,
    pub value: Expr,
    pub body: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LetObj>() == 64);
    assert!(::core::mem::align_of::<LetObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Let {
    base: ObjectArc<LetObj>,
}

impl Deref for Let {
    type Target = LetObj;
    fn deref(&self) -> &LetObj {
        &self.base
    }
}

impl Deref for LetObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl LetObj {
    pub(crate) fn new(loc: Location, ty: Type, var: Var, value: Expr, body: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, var, value, body }
    }
}

impl Let {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        var: impl Into<Var>,
        value: impl Into<Expr>,
        body: impl Into<Expr>,
    ) -> Self {
        let obj = LetObj::new(loc.into(), ty.into(), var.into(), value.into(), body.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Let => Expr);
// tvm-ffi-stubgen(end)

impl Cast {
    /// The node converting `value` to `ty`, which has as many lanes as `value`
    /// (C++ `Cast(ty, value, loc)`).
    pub fn new(
        ty: impl Into<PrimType>,
        value: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (ty, value) = (ty.into(), value.into());
        let value_dtype = support::prim_dtype(&value)?;
        if ty.dtype.lanes != value_dtype.lanes {
            let condition = "ty->dtype.lanes == value_expr_ty->dtype.lanes";
            let (lhs, rhs) = (ty.dtype.lanes, value_dtype.lanes);
            return Err(support::compare_failed(tvm_ffi::INTERNAL_ERROR, condition, lhs, rhs, ""));
        }
        Ok(Self::from_fields(loc.into(), ty, value))
    }
}

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
fn binary_type(node: &str, operands: Operands, a: &Expr, b: &Expr) -> Result<Type> {
    let (a_dtype, b_dtype) = (support::prim_dtype(a)?, support::prim_dtype(b)?);
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
        Operands::Arith => Ok(a.ty.clone()),
        Operands::Bitwise { allow_bool } => {
            let code = a_dtype.code;
            if matches!(code, support::INT | support::UINT) || (allow_bool && code == support::BOOL)
            {
                return Ok(a.ty.clone());
            }
            let condition = format!(
                "a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) || \
                 ({allow_bool} && a.ty().MatchesCode(DLDataTypeCode::kDLBool))"
            );
            let operands = if allow_bool { "integer or boolean" } else { "integer" };
            let message = format!("{node} requires {operands} operands");
            Err(support::check_failed(tvm_ffi::TYPE_ERROR, &condition, &message))
        }
        Operands::Compare => Ok(bool_type(a_dtype)?.into()),
    }
}

/// The boolean type with the lanes of `dtype` (C++ `PrimType(DLDataType{kDLBool, 8, lanes})`).
fn bool_type(operand: DLDataType) -> Result<PrimType> {
    use crate::ir::dtype;
    PrimType::from_dtype(dtype::with_lanes(dtype::bool(), operand.lanes))
}

/// `impl $node { pub fn new(a, b, loc) }` for each binary node, with the checks of its C++
/// constructor.
macro_rules! binary_new {
    ($($node:ident: $operands:expr, $doc:literal;)+) => {
        $(
            impl $node {
                #[doc = $doc]
                pub fn new(
                    a: impl Into<Expr>,
                    b: impl Into<Expr>,
                    loc: impl Into<Location>,
                ) -> Result<Self> {
                    let (a, b) = (a.into(), b.into());
                    let ty = binary_type(stringify!($node), $operands, &a, &b)?;
                    Ok(Self::from_fields(loc.into(), ty, a, b))
                }
            }
        )+
    };
}

binary_new! {
    LShift: Operands::Bitwise { allow_bool: false },
        "The node `a << b` over integer operands of the same type (C++ `LShift(a, b, loc)`).";
    RShift: Operands::Bitwise { allow_bool: false },
        "The node `a >> b` over integer operands of the same type (C++ `RShift(a, b, loc)`).";
    BitwiseAnd: Operands::Bitwise { allow_bool: true },
        "The node `a & b` over integer or boolean operands of the same type \
         (C++ `BitwiseAnd(a, b, loc)`).";
    BitwiseOr: Operands::Bitwise { allow_bool: true },
        "The node `a | b` over integer or boolean operands of the same type \
         (C++ `BitwiseOr(a, b, loc)`).";
    BitwiseXor: Operands::Bitwise { allow_bool: true },
        "The node `a ^ b` over integer or boolean operands of the same type \
         (C++ `BitwiseXor(a, b, loc)`).";
    Add: Operands::Arith, "The node `a + b` over operands of the same type (C++ `Add(a, b, loc)`).";
    Sub: Operands::Arith, "The node `a - b` over operands of the same type (C++ `Sub(a, b, loc)`).";
    Mul: Operands::Arith, "The node `a * b` over operands of the same type (C++ `Mul(a, b, loc)`).";
    Div: Operands::Arith, "The node `a / b` over operands of the same type (C++ `Div(a, b, loc)`).";
    Mod: Operands::Arith,
        "The node `a % b`, truncated, over operands of the same type (C++ `Mod(a, b, loc)`).";
    FloorDiv: Operands::Arith,
        "The node `floor(a / b)` over operands of the same type (C++ `FloorDiv(a, b, loc)`).";
    FloorMod: Operands::Arith,
        "The node `a - floor(a / b) * b` over operands of the same type \
         (C++ `FloorMod(a, b, loc)`).";
    Min: Operands::Arith,
        "The node `min(a, b)` over operands of the same type (C++ `Min(a, b, loc)`).";
    Max: Operands::Arith,
        "The node `max(a, b)` over operands of the same type (C++ `Max(a, b, loc)`).";
    EQ: Operands::Compare,
        "The node the boolean `a == b` over operands of the same type (C++ `EQ(a, b, loc)`).";
    NE: Operands::Compare,
        "The node the boolean `a != b` over operands of the same type (C++ `NE(a, b, loc)`).";
    LT: Operands::Compare,
        "The node the boolean `a < b` over operands of the same type (C++ `LT(a, b, loc)`).";
    LE: Operands::Compare,
        "The node the boolean `a <= b` over operands of the same type (C++ `LE(a, b, loc)`).";
    GT: Operands::Compare,
        "The node the boolean `a > b` over operands of the same type (C++ `GT(a, b, loc)`).";
    GE: Operands::Compare,
        "The node the boolean `a >= b` over operands of the same type (C++ `GE(a, b, loc)`).";
}

/// The type of the logical node of `a` and `b`, after the checks of the C++ `And` and `Or`
/// constructors: boolean operands of one type.
fn logical_type(a: &Expr, b: &Expr) -> Result<Type> {
    let (a_dtype, b_dtype) = (support::prim_dtype(a)?, support::prim_dtype(b)?);
    for (dtype, operand) in [(a_dtype, "a_ty"), (b_dtype, "b_ty")] {
        if dtype.code != support::BOOL {
            let condition = format!("{operand}.MatchesCode(DLDataTypeCode::kDLBool)");
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, &condition, ""));
        }
    }
    if a_dtype != b_dtype {
        return Err(support::check_failed(tvm_ffi::TYPE_ERROR, "a_ty == b_ty", "mismatched types"));
    }
    Ok(bool_type(a_dtype)?.into())
}

impl And {
    /// The node the boolean `a && b` over boolean operands (C++ `And(a, b, loc)`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        let (a, b) = (a.into(), b.into());
        let ty = logical_type(&a, &b)?;
        Ok(Self::from_fields(loc.into(), ty, a, b))
    }
}

impl Or {
    /// The node the boolean `a || b` over boolean operands (C++ `Or(a, b, loc)`).
    pub fn new(a: impl Into<Expr>, b: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        let (a, b) = (a.into(), b.into());
        let ty = logical_type(&a, &b)?;
        Ok(Self::from_fields(loc.into(), ty, a, b))
    }
}

impl BitwiseNot {
    /// The node `~a` over an integer or boolean operand (C++ `BitwiseNot(a, loc)`).
    pub fn new(a: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        let a = a.into();
        if !matches!(support::prim_dtype(&a)?.code, support::INT | support::UINT | support::BOOL) {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) || \
                             a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            let message = "BitwiseNot requires an integer or boolean operand";
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, message));
        }
        Ok(Self::from_fields(loc.into(), a.ty.clone(), a))
    }
}

impl Not {
    /// The node the boolean `!a` over a boolean operand (C++ `Not(a, loc)`).
    pub fn new(a: impl Into<Expr>, loc: impl Into<Location>) -> Result<Self> {
        let a = a.into();
        let a_dtype = support::prim_dtype(&a)?;
        if a_dtype.code != support::BOOL {
            let condition = "a_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        Ok(Self::from_fields(loc.into(), bool_type(a_dtype)?, a))
    }
}

impl Select {
    /// The node choosing `true_value` where `condition` holds and `false_value` elsewhere, which
    /// evaluates both (C++ `Select(condition, true_value, false_value, loc)`).
    pub fn new(
        condition: impl Into<Expr>,
        true_value: impl Into<Expr>,
        false_value: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (condition, true_value, false_value) =
            (condition.into(), true_value.into(), false_value.into());
        let condition_dtype = support::prim_dtype(&condition)?;
        let true_dtype = support::prim_dtype(&true_value)?;
        let false_dtype = support::prim_dtype(&false_value)?;
        if condition_dtype.code != support::BOOL {
            let condition = "condition_ty.MatchesCode(DLDataTypeCode::kDLBool)";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
        }
        // C++ `GetLanesOrVScaleFactor(ty)`.
        let lanes_or_vscale_factor = |dtype: DLDataType| -> Result<i32> {
            match support::is_scalable_vector(dtype) {
                true => Ok(-(support::encoded_lanes(dtype) as i32)),
                false => support::lanes(dtype),
            }
        };
        if lanes_or_vscale_factor(condition_dtype)? != lanes_or_vscale_factor(true_dtype)?
            && !support::is_scalar(condition_dtype)
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
        let ty = true_value.ty.clone();
        Ok(Self::from_fields(loc.into(), ty, condition, true_value, false_value))
    }
}

impl Let {
    /// The node binding `var` to `value` in `body`, of the type of `body`
    /// (C++ `Let(var, value, body, loc)`).
    pub fn new(
        var: impl Into<Var>,
        value: impl Into<Expr>,
        body: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (var, value, body) = (var.into(), value.into(), body.into());
        let value_dtype = support::prim_dtype(&value)?;
        support::prim_dtype(&body)?;
        let Some(var_ty) = var.ty.as_node::<PrimTypeObj>() else {
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
        let ty = body.ty.clone();
        Ok(Self::from_fields(loc.into(), ty, var, value, body))
    }
}
