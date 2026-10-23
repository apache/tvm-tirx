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

//! Vector expressions: [`Ramp`], [`Broadcast`] and [`Shuffle`].
//!
//! The objects `python/tvm_tirx/ir/prim/vector_expr.py` defines. A [`Ramp`] is `lanes` values
//! from `base_` by `stride`, a [`Broadcast`] repeats a value over `lanes`, and a [`Shuffle`]
//! selects lanes from its vectors. Their `new` is written in Rust after the C++ constructor: it
//! checks the operands and derives the vector type.

// `custom-new` reserves `new` for a constructor with the C++ one's semantics (operand checks and the
// derived vector type); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): prim.Ramp
// tvm-ffi-stubgen(custom-new): prim.Broadcast
// tvm-ffi-stubgen(custom-new): prim.Shuffle
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::BigInt
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataType
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::Call
// tvm-ffi-stubgen(import-object): crate::ir::expr::CallObj
// tvm-ffi-stubgen(import-object): crate::ir::expr::FloatImmObj
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImm
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImmObj
// tvm-ffi-stubgen(import-object): crate::ir::expr::FloatImm
// tvm-ffi-stubgen(import-object): crate::ir::location::UnknownLoc
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Cast
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Mul
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::MulObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::Type
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::dtype::with_lanes

// tvm-ffi-stubgen(begin): import-section
use crate::ir::dtype::with_lanes;
use crate::ir::expr::Call;
use crate::ir::expr::CallObj;
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::FloatImm;
use crate::ir::expr::FloatImmObj;
use crate::ir::expr::IntImm;
use crate::ir::expr::IntImmObj;
use crate::ir::location::Location;
use crate::ir::location::UnknownLoc;
use crate::ir::prim::expr::Cast;
use crate::ir::prim::expr::Mul;
use crate::ir::prim::expr::MulObj;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::BigInt;
use tvm_ffi::DLDataType;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Ramp
/// Complete: reflected fields fill [40, 64) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Ramp"]
#[type_final]
pub struct RampObj {
    base: ExprObj,
    pub base_: Expr,
    pub stride: Expr,
    pub lanes: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<RampObj>() == 64);
    assert!(::core::mem::align_of::<RampObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Ramp {
    base: ObjectArc<RampObj>,
}

impl Deref for Ramp {
    type Target = RampObj;
    fn deref(&self) -> &RampObj {
        &self.base
    }
}

impl Deref for RampObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl RampObj {
    pub(crate) fn new(loc: Location, ty: Type, base_: Expr, stride: Expr, lanes: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, base_, stride, lanes }
    }
}

impl Ramp {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        base_: impl Into<Expr>,
        stride: impl Into<Expr>,
        lanes: impl Into<Expr>,
    ) -> Self {
        let obj = RampObj::new(loc.into(), ty.into(), base_.into(), stride.into(), lanes.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Ramp => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Broadcast
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Broadcast"]
#[type_final]
pub struct BroadcastObj {
    base: ExprObj,
    pub value: Expr,
    pub lanes: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BroadcastObj>() == 56);
    assert!(::core::mem::align_of::<BroadcastObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Broadcast {
    base: ObjectArc<BroadcastObj>,
}

impl Deref for Broadcast {
    type Target = BroadcastObj;
    fn deref(&self) -> &BroadcastObj {
        &self.base
    }
}

impl Deref for BroadcastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl BroadcastObj {
    pub(crate) fn new(loc: Location, ty: Type, value: Expr, lanes: Expr) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, value, lanes }
    }
}

impl Broadcast {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        value: impl Into<Expr>,
        lanes: impl Into<Expr>,
    ) -> Self {
        let obj = BroadcastObj::new(loc.into(), ty.into(), value.into(), lanes.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Broadcast => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Shuffle
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Shuffle"]
#[type_final]
pub struct ShuffleObj {
    base: ExprObj,
    pub vectors: Array<Expr>,
    pub indices: Array<Expr>,
}

const _: () = {
    assert!(::core::mem::size_of::<ShuffleObj>() == 56);
    assert!(::core::mem::align_of::<ShuffleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Shuffle {
    base: ObjectArc<ShuffleObj>,
}

impl Deref for Shuffle {
    type Target = ShuffleObj;
    fn deref(&self) -> &ShuffleObj {
        &self.base
    }
}

impl Deref for ShuffleObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ShuffleObj {
    pub(crate) fn new(loc: Location, ty: Type, vectors: Array<Expr>, indices: Array<Expr>) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, vectors, indices }
    }
}

impl Shuffle {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        vectors: impl Into<Array<Expr>>,
        indices: impl Into<Array<Expr>>,
    ) -> Self {
        let obj = ShuffleObj::new(loc.into(), ty.into(), vectors.into(), indices.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Shuffle => Expr);
// tvm-ffi-stubgen(end)

impl Ramp {
    /// The vector of `lanes` values from `base` by `stride`, with `stride` cast to the type of
    /// `base` (C++ `Ramp(base, stride, lanes, loc)`).
    pub fn new(
        base: impl Into<Expr>,
        stride: impl Into<Expr>,
        lanes: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (base, stride, lanes) = (base.into(), stride.into(), lanes.into());
        let base_dtype = support::prim_dtype(&base)?;
        let stride_dtype = support::prim_dtype(&stride)?;
        support::prim_dtype(&lanes)?;
        if !support::is_scalar(base_dtype) {
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, "base_ty.IsScalar()", ""));
        }
        if !support::is_scalar(stride_dtype) {
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, "stride_ty.IsScalar()", ""));
        }
        let stride = match stride_dtype == base_dtype {
            true => stride,
            false => cast_scalar(stride, base_dtype)?,
        };
        let (dtype, lanes) = vector_lanes(base_dtype, lanes)?;
        let ty = PrimType::from_dtype(dtype)?;
        Ok(Self::from_fields(loc.into(), ty, base, stride, lanes))
    }
}

impl Broadcast {
    /// The vector of `lanes` copies of the scalar `value` (C++ `Broadcast(value, lanes, loc)`).
    pub fn new(
        value: impl Into<Expr>,
        lanes: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (value, lanes) = (value.into(), lanes.into());
        let value_dtype = support::prim_dtype(&value)?;
        support::prim_dtype(&lanes)?;
        if !support::is_scalar(value_dtype) {
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, "value_ty.IsScalar()", ""));
        }
        let (dtype, lanes) = vector_lanes(value_dtype, lanes)?;
        let ty = PrimType::from_dtype(dtype)?;
        Ok(Self::from_fields(loc.into(), ty, value, lanes))
    }
}

impl Shuffle {
    /// The vector of the lanes at `indices` of the concatenated `vectors`
    /// (C++ `Shuffle(vectors, indices, loc)`).
    pub fn new(
        vectors: impl IntoIterator<Item = impl Into<Expr>>,
        indices: impl IntoIterator<Item = impl Into<Expr>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let vectors = vectors.into_iter().map(Into::into).collect::<Array<Expr>>();
        let indices = indices.into_iter().map(Into::into).collect::<Array<Expr>>();
        let dtypes = vectors
            .iter()
            .map(|vector| support::prim_dtype(&vector))
            .collect::<Result<Vec<_>>>()?;
        for index in indices.iter() {
            support::prim_dtype(&index)?;
        }
        let empty =
            |condition: &str| support::compare_failed(tvm_ffi::INTERNAL_ERROR, condition, 0, 0, "");
        let Some(first) = dtypes.first() else { return Err(empty("vectors.size() != 0U")) };
        if indices.is_empty() {
            return Err(empty("indices.size() != 0U"));
        }
        let base_dtype = with_lanes(*first, 1);
        let mut total_lanes = 0;
        for dtype in &dtypes {
            if with_lanes(*dtype, 1) != base_dtype {
                let condition = "val_ty.WithLanes(1)->dtype == base_type->dtype";
                return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
            }
            total_lanes += support::lanes(*dtype)?;
        }
        if indices.len() > total_lanes as usize {
            let condition = "indices.size() <= static_cast<size_t>(total_lanes)";
            let (count, total) = (indices.len(), total_lanes);
            return Err(support::compare_failed(
                tvm_ffi::INTERNAL_ERROR,
                condition,
                count,
                total,
                "",
            ));
        }
        let ty = PrimType::from_dtype(with_lanes(base_dtype, indices.len() as u16))?;
        Ok(Self::from_fields(loc.into(), ty, vectors, indices))
    }
}

/// The vector type of `element` elements over `lanes`, and the lanes a `Ramp` or `Broadcast` keeps:
/// a constant count as an `int32` constant, or a scalable count as `vscale * factor`.
fn vector_lanes(element: DLDataType, lanes: Expr) -> Result<(DLDataType, Expr)> {
    if let Some(imm) = lanes.as_node::<IntImmObj>() {
        let Some(count) = imm.value.to_i64().and_then(|count| i32::try_from(count).ok()) else {
            return Err(tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, "bad optional access", ""));
        };
        if count <= 1 {
            return Err(support::compare_failed(
                tvm_ffi::INTERNAL_ERROR,
                "lanes > 1",
                count,
                1,
                "",
            ));
        }
        return Ok((with_lanes(element, count as u16), IntImm::int32(count).into()));
    }
    let Some(factor) = vscale_factor(&lanes)? else {
        let message = format!("Invalid expression for scalable lanes {}", support::repr(&lanes));
        return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, "vscale_factor", &message));
    };
    Ok((support::scalable_vector(element, factor)?, vscale_lanes(factor)?))
}

/// The lane count of a scalable vector of `factor` times `vscale` lanes, as C++ spells it:
/// `vscale() * factor`, both `int32`.
pub(crate) fn vscale_lanes(factor: i32) -> Result<Expr> {
    let vscale = support::cached_op("prim.vscale")?;
    let vscale = Call::new(
        Some(&PrimType::int(32).into()),
        vscale,
        Vec::<Expr>::new(),
        None,
        Vec::<Type>::new(),
        UnknownLoc::new(),
    )?;
    Ok(Mul::new(vscale, IntImm::int32(factor), UnknownLoc::new())?.into())
}

/// `factor` when `lanes` is `factor * vscale` or `vscale * factor` (C++ `ExtractVscaleFactor`).
fn vscale_factor(lanes: &Expr) -> Result<Option<i32>> {
    let Some(mul) = lanes.as_node::<MulObj>() else { return Ok(None) };
    let vscale = support::cached_op("prim.vscale")?;
    let is_vscale =
        |expr: &Expr| expr.as_node::<CallObj>().is_some_and(|call| call.op.same_as(&vscale));
    let factor = |imm: &IntImmObj| imm.value.to_i64().and_then(|value| i32::try_from(value).ok());
    if let Some(imm) = mul.a.as_node::<IntImmObj>() {
        if is_vscale(&mul.b) {
            return Ok(factor(imm));
        }
    }
    if let Some(imm) = mul.b.as_node::<IntImmObj>() {
        if is_vscale(&mul.a) {
            return Ok(factor(imm));
        }
    }
    Ok(None)
}

/// C++ `prim::cast(value, dtype)` to the scalar type `dtype`, which differs from the type of
/// `value`: a constant is converted (`MakeConst`), anything else becomes a `Cast` node.
fn cast_scalar(value: Expr, dtype: DLDataType) -> Result<Expr> {
    if let Some(imm) = value.as_node::<IntImmObj>() {
        return make_const_scalar(dtype, ConstValue::Int(imm.value.clone()), &imm.loc);
    }
    if let Some(imm) = value.as_node::<FloatImmObj>() {
        return make_const_scalar(dtype, ConstValue::Float(imm.value), &imm.loc);
    }
    Ok(Cast::new(PrimType::from_dtype(dtype)?, value, UnknownLoc::new())?.into())
}

/// The value of a constant [`make_const_scalar`] converts.
pub(crate) enum ConstValue {
    Int(BigInt),
    Float(f64),
}

/// C++ `prim::MakeConstScalar(dtype, value, loc)`: an `IntImm` for an integer or boolean type, a
/// `FloatImm` for a floating-point type. [`make_const`](crate::ir::prim::op::make_const)
/// broadcasts it over the lanes of a vector type.
pub(crate) fn make_const_scalar(
    dtype: DLDataType,
    value: ConstValue,
    loc: &Location,
) -> Result<Expr> {
    use tvm_ffi::DLDataTypeCode::*;
    let code = dtype.code;
    let float_codes = [
        kDLFloat,
        kDLFloat8_e3m4,
        kDLFloat8_e4m3,
        kDLFloat8_e4m3b11fnuz,
        kDLFloat8_e4m3fn,
        kDLFloat8_e4m3fnuz,
        kDLFloat8_e5m2,
        kDLFloat8_e5m2fnuz,
        kDLFloat8_e8m0fnu,
        kDLFloat6_e2m3fn,
        kDLFloat6_e3m2fn,
        kDLFloat4_e2m1fn,
    ];
    if matches!(code, support::INT | support::UINT | support::BOOL) {
        let value = match value {
            ConstValue::Int(value) => value,
            ConstValue::Float(value) => {
                if code == support::UINT && value < 0.0 {
                    let message = format!(
                        "cannot make uint from negative value {}",
                        support::CppDouble(value)
                    );
                    return Err(support::check_failed(
                        tvm_ffi::INTERNAL_ERROR,
                        "value >= 0",
                        &message,
                    ));
                }
                BigInt::try_from(value)?
            }
        };
        if code == support::UINT && value.is_negative() {
            let message = format!("cannot make uint from negative value {value}");
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, "value >= 0", &message));
        }
        return Ok(IntImm::from_dtype(dtype, value, loc.clone())?.into());
    }
    if float_codes.iter().any(|float| code == *float as u8)
        || (code == support::BFLOAT && dtype.bits == 16)
    {
        let value = match value {
            ConstValue::Int(value) => value.to_f64()?,
            ConstValue::Float(value) => value,
        };
        return Ok(FloatImm::from_dtype(dtype, value, loc.clone())?.into());
    }
    let ty = PrimType::from_dtype(dtype)?;
    let message = format!("cannot make const for type {}", support::repr(&ty));
    Err(tvm_ffi::Error::new(tvm_ffi::INTERNAL_ERROR, &message, ""))
}
