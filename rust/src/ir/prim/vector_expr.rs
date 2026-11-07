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

//! Vector expressions: [`Ramp`], [`Broadcast`] and [`Shuffle`].
//!
//! The objects `python/tvm_tirx/ir/prim/vector_expr.py` defines. A [`Ramp`] is `lanes` values
//! from `base_` by `stride`, a [`Broadcast`] repeats a value over `lanes`, and a [`Shuffle`]
//! selects lanes from its vectors.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::prim::{Broadcast, Ramp, Shuffle};
//! use tvm_tirx::ir::{PrimExpr, PrimType, PrimVar, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//!
//! let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
//! let index = Ramp::new(&i * 4, 1, 4, UNKNOWN_LOC)?;
//! assert_eq!(script(&index)?, "T.Ramp(i * 4, 1, 4)");
//!
//! let shifted = PrimExpr::try_from(&index + Broadcast::new(&i, 4, UNKNOWN_LOC)?)?;
//! assert_eq!(script(&shifted)?, "T.Ramp(i * 4, 1, 4) + T.Broadcast(i, 4)");
//!
//! let reversed = Shuffle::new([&index], [3, 2, 1, 0], UNKNOWN_LOC)?;
//! assert_eq!(reversed.indices.len(), 4);
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// tvm-ffi-stubgen(missing-docs): allow
// `custom-new` reserves `new` for a constructor with the C++ one's semantics (operand checks and the
// derived vector type); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): prim.Ramp
// tvm-ffi-stubgen(custom-new): prim.Broadcast
// tvm-ffi-stubgen(custom-new): prim.Shuffle
// Every node is typed by a `PrimType` and holds `PrimExpr` operands, as in C++: its allocator
// takes a `PrimType`, and it converts into a `PrimExpr` without a check.
// tvm-ffi-stubgen(field): prim.Ramp.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Ramp.base -> PrimExpr
// tvm-ffi-stubgen(field): prim.Ramp.stride -> PrimExpr
// tvm-ffi-stubgen(field): prim.Ramp.lanes -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Ramp -> PrimExpr
// tvm-ffi-stubgen(field): prim.Broadcast.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Broadcast.value -> PrimExpr
// tvm-ffi-stubgen(field): prim.Broadcast.lanes -> PrimExpr
// tvm-ffi-stubgen(upcast): prim.Broadcast -> PrimExpr
// tvm-ffi-stubgen(field): prim.Shuffle.ty -> PrimType
// tvm-ffi-stubgen(field): prim.Shuffle.vectors -> Array<PrimExpr>
// tvm-ffi-stubgen(field): prim.Shuffle.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(upcast): prim.Shuffle -> PrimExpr
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataTypeCode
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::Call
// tvm-ffi-stubgen(import-object): crate::ir::expr::CallObj
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImm
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImmObj
// tvm-ffi-stubgen(import-object): crate::ir::location::UNKNOWN_LOC
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Mul
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::MulObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(import-object): crate::ir::r#type::Type
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::cast
// tvm-ffi-stubgen(import-object): crate::ir::dtype::with_lanes
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr

// tvm-ffi-stubgen(begin): import-section
use crate::ir::dtype::with_lanes;
use crate::ir::expr::Call;
use crate::ir::expr::CallObj;
use crate::ir::expr::Expr;
use crate::ir::expr::ExprObj;
use crate::ir::expr::IntImm;
use crate::ir::expr::IntImmObj;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::location::Location;
use crate::ir::location::UNKNOWN_LOC;
use crate::ir::prim::expr::Mul;
use crate::ir::prim::expr::MulObj;
use crate::ir::prim::op::cast;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::DLDataTypeCode;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/prim.Ramp
/// Complete: reflected fields fill [40, 64) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Ramp"]
#[type_final]
pub struct RampObj {
    base: ExprObj,
    pub base_: PrimExpr,
    pub stride: PrimExpr,
    pub lanes: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<RampObj>() == 64);
    assert!(::core::mem::align_of::<RampObj>() == 8);
};

#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Ramp {
    base: ObjectArc<RampObj>,
}

impl Deref for RampObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

#[allow(missing_docs)]
impl RampObj {
    pub(crate) fn new(
        loc: Location,
        ty: PrimType,
        base_: PrimExpr,
        stride: PrimExpr,
        lanes: PrimExpr,
    ) -> Self {
        let base = ExprObj::new(loc, ty.into());
        Self { base, base_, stride, lanes }
    }
}

#[allow(missing_docs)]
impl Ramp {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        base_: impl Into<PrimExpr>,
        stride: impl Into<PrimExpr>,
        lanes: impl Into<PrimExpr>,
    ) -> Self {
        let obj = RampObj::new(loc.into(), ty.into(), base_.into(), stride.into(), lanes.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for RampObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Ramp => Expr, Ramp => PrimExpr);
// tvm-ffi-stubgen(end)

impl Ramp {
    /// Create the vector of `lanes` values from `base` by `stride`, with `stride` cast to the type
    /// of `base` (C++ `Ramp(base, stride, lanes, loc)`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `base` or `stride` is not a scalar `int32`, `int64`, `uint32` or
    /// `uint64`, or an `InternalError` if `lanes` is neither a constant greater than 1 nor
    /// `vscale() * factor`.
    pub fn new(
        base: impl IntoExpr<PrimExpr>,
        stride: impl IntoExpr<PrimExpr>,
        lanes: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (base, stride, lanes) =
            (base.into_owned_expr()?, stride.into_owned_expr()?, lanes.into_owned_expr()?);
        let (base_ty, stride_ty) = (base.prim_ty().clone(), stride.prim_ty().clone());
        // Only 32-bit and 64-bit integers step through the lanes of a ramp.
        let index = |ty: &PrimType| {
            ty.is_scalar()
                && ty.matches_code(&[DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt])
                && matches!(ty.bits(), 32 | 64)
        };
        for (field, ty) in [("base", &base_ty), ("stride", &stride_ty)] {
            if !index(ty) {
                let condition = format!(
                    "{field}_ty.IsScalar() && \
                     {field}_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) && \
                     ({field}_ty.bits() == 32 || {field}_ty.bits() == 64)"
                );
                let message = format!(
                    "Ramp {field} must be scalar int32, int64, uint32, or uint64, but got {}",
                    support::repr(ty)
                );
                return Err(support::check_failed(tvm_ffi::TYPE_ERROR, &condition, &message));
            }
        }
        let stride = match stride_ty.dtype == base_ty.dtype {
            true => stride,
            false => cast(stride, base_ty.clone())?,
        };
        let (ty, lanes) = vector_lanes(&base_ty, lanes)?;
        Ok(Self::from_fields(loc.into(), ty, base, stride, lanes))
    }
}

// tvm-ffi-stubgen(begin): object/prim.Broadcast
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Broadcast"]
#[type_final]
pub struct BroadcastObj {
    base: ExprObj,
    pub value: PrimExpr,
    pub lanes: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<BroadcastObj>() == 56);
    assert!(::core::mem::align_of::<BroadcastObj>() == 8);
};

#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Broadcast {
    base: ObjectArc<BroadcastObj>,
}

impl Deref for BroadcastObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

#[allow(missing_docs)]
impl BroadcastObj {
    pub(crate) fn new(loc: Location, ty: PrimType, value: PrimExpr, lanes: PrimExpr) -> Self {
        let base = ExprObj::new(loc, ty.into());
        Self { base, value, lanes }
    }
}

#[allow(missing_docs)]
impl Broadcast {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        value: impl Into<PrimExpr>,
        lanes: impl Into<PrimExpr>,
    ) -> Self {
        let obj = BroadcastObj::new(loc.into(), ty.into(), value.into(), lanes.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BroadcastObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Broadcast => Expr, Broadcast => PrimExpr);
// tvm-ffi-stubgen(end)

impl Broadcast {
    /// Create the vector of `lanes` copies of the scalar `value` (C++ `Broadcast`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `value` is a vector, or `lanes` is neither a constant greater
    /// than 1 nor `vscale() * factor`.
    pub fn new(
        value: impl IntoExpr<PrimExpr>,
        lanes: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (value, lanes) = (value.into_owned_expr()?, lanes.into_owned_expr()?);
        if !value.prim_ty().is_scalar() {
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, "value_ty.IsScalar()", ""));
        }
        let (ty, lanes) = vector_lanes(value.prim_ty(), lanes)?;
        Ok(Self::from_fields(loc.into(), ty, value, lanes))
    }
}

// tvm-ffi-stubgen(begin): object/prim.Shuffle
/// Complete: reflected fields fill [40, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "prim.Shuffle"]
#[type_final]
pub struct ShuffleObj {
    base: ExprObj,
    pub vectors: Array<PrimExpr>,
    pub indices: Array<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<ShuffleObj>() == 56);
    assert!(::core::mem::align_of::<ShuffleObj>() == 8);
};

#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Shuffle {
    base: ObjectArc<ShuffleObj>,
}

impl Deref for ShuffleObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

#[allow(missing_docs)]
impl ShuffleObj {
    pub(crate) fn new(
        loc: Location,
        ty: PrimType,
        vectors: Array<PrimExpr>,
        indices: Array<PrimExpr>,
    ) -> Self {
        let base = ExprObj::new(loc, ty.into());
        Self { base, vectors, indices }
    }
}

#[allow(missing_docs)]
impl Shuffle {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        vectors: impl Into<Array<PrimExpr>>,
        indices: impl Into<Array<PrimExpr>>,
    ) -> Self {
        let obj = ShuffleObj::new(loc.into(), ty.into(), vectors.into(), indices.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ShuffleObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(Shuffle => Expr, Shuffle => PrimExpr);
// tvm-ffi-stubgen(end)

impl Shuffle {
    /// Create the vector of the lanes at `indices` of the concatenated `vectors`
    /// (C++ `Shuffle(vectors, indices, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `vectors` or `indices` is empty, the vectors have different
    /// element types, or there are more indices than lanes.
    pub fn new(
        vectors: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let vectors = vectors
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let indices = indices
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<PrimExpr>>>()?;
        let types = vectors.iter().map(|vector| vector.prim_ty().clone()).collect::<Vec<_>>();
        let empty =
            |condition: &str| support::compare_failed(tvm_ffi::INTERNAL_ERROR, condition, 0, 0, "");
        let Some(first) = types.first() else { return Err(empty("vectors.size() != 0U")) };
        if indices.is_empty() {
            return Err(empty("indices.size() != 0U"));
        }
        let base_dtype = with_lanes(first.dtype, 1);
        let mut total_lanes = 0;
        for ty in &types {
            if with_lanes(ty.dtype, 1) != base_dtype {
                let condition = "val_ty.WithLanes(1)->dtype == base_type->dtype";
                return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, ""));
            }
            total_lanes += support::lanes(ty)?;
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
        let ty = PrimType::try_from(with_lanes(base_dtype, indices.len() as u16))?;
        Ok(Self::from_fields(loc.into(), ty, vectors, indices))
    }
}

// ---------------------------------------------------------------------------------------------
// Lanes the constructors share

/// The vector type of `element` elements over `lanes`, and the lanes a `Ramp` or `Broadcast` keeps:
/// a constant count as an `int32` constant, or a scalable count as `vscale * factor`.
fn vector_lanes(element: &PrimType, lanes: PrimExpr) -> Result<(PrimType, PrimExpr)> {
    if let Some(imm) = lanes.downcast_ref::<IntImmObj>() {
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
        let ty = PrimType::try_from(with_lanes(element.dtype, count as u16))?;
        return Ok((ty, IntImm::int32(count).into()));
    }
    let Some(factor) = vscale_factor(&lanes)? else {
        let message = format!("Invalid expression for scalable lanes {}", support::repr(&lanes));
        return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, "vscale_factor", &message));
    };
    let ty = PrimType::try_from(support::scalable_vector(element.dtype, factor)?)?;
    Ok((ty, vscale_lanes(factor)?))
}

/// The lane count of a scalable vector of `factor` times `vscale` lanes, as C++ spells it:
/// `vscale() * factor`, both `int32`.
pub(crate) fn vscale_lanes(factor: i32) -> Result<PrimExpr> {
    let vscale = support::cached_op("prim.vscale")?;
    let vscale: Expr = Call::new(
        Some(&PrimType::int(32).into()),
        vscale,
        Vec::<Expr>::new(),
        None,
        Vec::<Type>::new(),
        UNKNOWN_LOC,
    )?
    .into();
    Ok(Mul::new(PrimExpr::try_from(vscale)?, IntImm::int32(factor), UNKNOWN_LOC)?.into())
}

/// `factor` when `lanes` is `factor * vscale` or `vscale * factor` (C++ `ExtractVscaleFactor`).
fn vscale_factor(lanes: &PrimExpr) -> Result<Option<i32>> {
    let Some(mul) = lanes.downcast_ref::<MulObj>() else { return Ok(None) };
    let vscale = support::cached_op("prim.vscale")?;
    let is_vscale = |expr: &PrimExpr| {
        expr.downcast_ref::<CallObj>().is_some_and(|call| call.op.same_as(&vscale))
    };
    let factor = |imm: &IntImmObj| imm.value.to_i64().and_then(|value| i32::try_from(value).ok());
    if let Some(imm) = mul.a.downcast_ref::<IntImmObj>() {
        if is_vscale(&mul.b) {
            return Ok(factor(imm));
        }
    }
    if let Some(imm) = mul.b.downcast_ref::<IntImmObj>() {
        if is_vscale(&mul.a) {
            return Ok(factor(imm));
        }
    }
    Ok(None)
}
