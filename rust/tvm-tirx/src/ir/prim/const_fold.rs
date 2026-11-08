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

//! Constant folding. Each function folds one operation on operands already matched to one type, or
//! returns `None` when there is nothing to fold; a constant operand also folds identities such as
//! `x + 0`, and division by a constant zero is an error.

use tvm_ffi::{BigInt, DLDataTypeCode, ObjectRefCore, Result, INTERNAL_ERROR};

use crate::ir::expr::{FloatImm, FloatImmObj, IntImm, IntImmObj, PrimExpr};
use crate::ir::location::UNKNOWN_LOC;
use crate::ir::r#type::PrimType;
use crate::ir::support;

/// The folded expression, or `None` when there is nothing to fold.
pub(super) type Folded = Result<Option<PrimExpr>>;

// ---------------------------------------------------------------------------------------------
// Arithmetic

/// Fold `a + b`.
pub(super) fn add(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return exact(ty, x + y);
    }
    if pa.is_some_and(BigInt::is_zero) {
        return keep(b);
    }
    if pb.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match ty.bits() {
            32 => return float_imm_of(ty, single(x, y, |x, y| x + y)),
            64 => return float_imm(ty, x + y),
            _ => {}
        }
    }
    if fa == Some(0.0) {
        return keep(b);
    }
    if fb == Some(0.0) {
        return keep(a);
    }
    Ok(None)
}

/// Fold `a - b`.
pub(super) fn sub(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return exact(ty, x - y);
    }
    if pb.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match ty.bits() {
            32 => return float_imm_of(ty, single(x, y, |x, y| x - y)),
            64 => return float_imm(ty, x - y),
            _ => {}
        }
    }
    if fb == Some(0.0) {
        return keep(a);
    }
    Ok(None)
}

/// Fold `a * b`.
pub(super) fn mul(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return exact(ty, x * y);
    }
    if let Some(x) = pa {
        if is_one(x) {
            return keep(b);
        }
        if x.is_zero() {
            return keep(a);
        }
    }
    if let Some(y) = pb {
        if is_one(y) {
            return keep(a);
        }
        if y.is_zero() {
            return keep(b);
        }
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match ty.bits() {
            32 => return float_imm_of(ty, single(x, y, |x, y| x * y)),
            64 => return float_imm(ty, x * y),
            _ => {}
        }
    }
    if let Some(x) = fa {
        if x == 1.0 {
            return keep(b);
        }
        if x == 0.0 {
            return keep(a);
        }
    }
    if let Some(y) = fb {
        if y == 1.0 {
            return keep(a);
        }
        if y == 0.0 {
            return keep(b);
        }
    }
    Ok(None)
}

/// Fold `a / b`: integers divide truncating.
pub(super) fn div(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        check_int_divisor(y)?;
        return exact(ty, x.try_div(y)?);
    }
    if pa.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let Some(y) = pb {
        if is_one(y) {
            return keep(a);
        }
        check_int_divisor(y)?;
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        check_float_divisor(y)?;
        match ty.bits() {
            32 => return float_imm_of(ty, single(x, y, |x, y| x / y)),
            64 => return float_imm(ty, x / y),
            _ => {}
        }
    }
    if fa == Some(0.0) {
        return keep(a);
    }
    if let Some(y) = fb {
        if y == 1.0 {
            return keep(a);
        }
        check_float_divisor(y)?;
    }
    Ok(None)
}

/// Fold the truncating remainder `a % b`: index types fold with the identities, wider integers
/// fold only two constants.
pub(super) fn truncmod(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    let (pa, pb) = (int(a), int(b));
    if is_index_type(ty) && is_index_type(b.prim_ty()) {
        if let (Some(x), Some(y)) = (pa, pb) {
            check_int_divisor(y)?;
            return exact(ty, x.try_rem(y)?);
        }
        if pa.is_some_and(BigInt::is_zero) {
            return keep(a);
        }
        if let Some(y) = pb {
            if is_one(y) {
                return Ok(Some(IntImm::new(ty.clone(), BigInt::from(0i64), UNKNOWN_LOC)?.into()));
            }
            check_int_divisor(y)?;
        }
    }
    if let (Some(x), Some(y)) = (pa, pb)
        && ty.bits() >= 64
    {
        check_int_divisor(y)?;
        return exact(ty, x.try_rem(y)?);
    }
    Ok(None)
}

/// Fold `floor(a / b)`.
pub(super) fn floordiv(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        check_int_divisor(y)?;
        return exact(ty, x.floor_div(y)?);
    }
    if pa.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let Some(y) = pb {
        if is_one(y) {
            return keep(a);
        }
        check_int_divisor(y)?;
    }
    if let (Some(x), Some(y)) = (fa, fb)
        && y != 0.0
    {
        return match ty.bits() {
            32 => float_imm_of(ty, single(x, y, |x, y| x / y).map(f64::floor)),
            64 => float_imm(ty, (x / y).floor()),
            _ => Ok(None),
        };
    }
    if fa == Some(0.0) {
        return keep(a);
    }
    if let Some(y) = fb {
        if y == 1.0 {
            return keep(a);
        }
        check_float_divisor(y)?;
    }
    Ok(None)
}

/// Fold `a - floor(a / b) * b`: index types fold with the identities, wider or unsigned integers
/// fold only two constants.
pub(super) fn floormod(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    let (pa, pb) = (int(a), int(b));
    if is_index_type(ty) && is_index_type(b.prim_ty()) {
        if let (Some(x), Some(y)) = (pa, pb) {
            check_int_divisor(y)?;
            return exact(ty, x.floor_mod(y)?);
        }
        if pa.is_some_and(BigInt::is_zero) {
            return keep(a);
        }
        if let Some(y) = pb {
            if is_one(y) {
                return Ok(Some(IntImm::new(ty.clone(), BigInt::from(0i64), UNKNOWN_LOC)?.into()));
            }
            check_int_divisor(y)?;
        }
    }
    if let (Some(x), Some(y)) = (pa, pb)
        && (ty.bits() >= 64 || ty.matches_code(&[DLDataTypeCode::kDLUInt]))
    {
        check_int_divisor(y)?;
        return exact(ty, x.floor_mod(y)?);
    }
    Ok(None)
}

/// Fold `min(a, b)`: two constants, or an operand with itself. Of two floats, the first is kept
/// when it is NaN or less than the second, and the second otherwise: a NaN in either operand is
/// the result, and of two equal values (`0.0` and `-0.0`) the second is.
pub(super) fn min(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return Ok(Some(IntImm::new(ty.clone(), x.min(y).clone(), UNKNOWN_LOC)?.into()));
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return keep(if x.is_nan() || x < y { a } else { b });
    }
    if a.same_as(b) {
        return keep(a);
    }
    Ok(None)
}

/// Fold `max(a, b)`, as [`min`]: the first float is kept when it is NaN or greater.
pub(super) fn max(a: &PrimExpr, b: &PrimExpr) -> Folded {
    let ty = a.prim_ty();
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return Ok(Some(IntImm::new(ty.clone(), x.max(y).clone(), UNKNOWN_LOC)?.into()));
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return keep(if x.is_nan() || x > y { a } else { b });
    }
    if a.same_as(b) {
        return keep(a);
    }
    Ok(None)
}

// ---------------------------------------------------------------------------------------------
// Comparisons and logic

/// Fold a comparison: two constants compare to a boolean constant.
pub(super) fn compare(
    a: &PrimExpr,
    b: &PrimExpr,
    ints: fn(&BigInt, &BigInt) -> bool,
    floats: fn(f64, f64) -> bool,
) -> Folded {
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return boolean(ints(x, y));
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return boolean(floats(x, y));
    }
    Ok(None)
}

/// Fold `a && b`: a true constant leaves the other side, a false one is the result.
pub(super) fn and(a: &PrimExpr, b: &PrimExpr) -> Folded {
    if let Some(x) = int(a) {
        return keep(if x.is_zero() { a } else { b });
    }
    if let Some(y) = int(b) {
        return keep(if y.is_zero() { b } else { a });
    }
    Ok(None)
}

/// Fold `a || b`: a true constant is the result, a false one leaves the other side.
pub(super) fn or(a: &PrimExpr, b: &PrimExpr) -> Folded {
    if let Some(x) = int(a) {
        return keep(if x.is_zero() { b } else { a });
    }
    if let Some(y) = int(b) {
        return keep(if y.is_zero() { a } else { b });
    }
    Ok(None)
}

/// Fold `!a`.
pub(super) fn not(a: &PrimExpr) -> Folded {
    match int(a) {
        Some(x) => boolean(x.is_zero()),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers

/// Whether `ty` is an index type, a scalar `int32` or `int64`, which index arithmetic folds
/// eagerly.
pub(super) fn is_index_type(ty: &PrimType) -> bool {
    ty.is_scalar() && ty.matches_code(&[DLDataTypeCode::kDLInt]) && matches!(ty.bits(), 32 | 64)
}

/// Whether the exact integer `value` is a literal of the scalar type `ty`: an integer in its
/// range, or a boolean 0 or 1 (as is an `int1`).
pub(super) fn is_representable(value: &BigInt, ty: &PrimType) -> bool {
    if !ty.is_scalar() || ty.bits() == 0 {
        return false;
    }
    let (code, bits) = (ty.dtype.code, u32::from(ty.bits()));
    if let Some(small) = value.to_i64() {
        return match code {
            support::UINT => small >= 0 && (bits >= 63 || (small as u64) < (1u64 << bits)),
            support::BOOL => small == 0 || small == 1,
            support::INT if bits == 1 => small == 0 || small == 1,
            support::INT => {
                bits >= 64 || (-(1i64 << (bits - 1))..1i64 << (bits - 1)).contains(&small)
            }
            _ => false,
        };
    }
    match code {
        support::UINT => !value.is_negative() && *value < BigInt::from(1i64) << i64::from(bits),
        support::INT if bits > 1 => {
            let limit = BigInt::from(1i64) << i64::from(bits - 1);
            *value >= -limit.clone() && *value < limit
        }
        _ => false,
    }
}

/// Whether `value` is in the float32 range: it rounds to a finite float32, or is not finite
/// itself.
pub(super) fn is_in_float32_range(value: f64) -> bool {
    !value.is_finite() || (f64::from(f32::MIN)..=f64::from(f32::MAX)).contains(&value)
}

/// The value of an integer constant.
pub(super) fn int(expr: &PrimExpr) -> Option<&BigInt> {
    expr.downcast_ref::<IntImmObj>().map(|imm| &imm.value)
}

/// The value of a float constant.
fn float(expr: &PrimExpr) -> Option<f64> {
    expr.downcast_ref::<FloatImmObj>().map(|imm| imm.value)
}

/// The constant `value` of type `ty`, or nothing to fold when it is not a value of the type.
fn exact(ty: &PrimType, value: BigInt) -> Folded {
    match is_representable(&value, ty) {
        true => Ok(Some(IntImm::new(ty.clone(), value, UNKNOWN_LOC)?.into())),
        false => Ok(None),
    }
}

/// The float constant `value` of type `ty`.
fn float_imm(ty: &PrimType, value: f64) -> Folded {
    Ok(Some(FloatImm::new(ty.clone(), value, UNKNOWN_LOC)?.into()))
}

/// The boolean constant `value`.
fn boolean(value: bool) -> Folded {
    Ok(Some(IntImm::bool(value).into()))
}

/// A float32 operation on the operands rounded to float32, or `None` when the result is beyond
/// the float32 range: the result is checked in double precision, before rounding can turn a
/// finite overflow into infinity, and rounding it to float32 gives the single-precision result (a
/// double holds more than twice the digits of a float).
fn single(x: f64, y: f64, op: fn(f64, f64) -> f64) -> Option<f64> {
    let result = op(f64::from(x as f32), f64::from(y as f32));
    is_in_float32_range(result).then(|| f64::from(result as f32))
}

/// The float constant of a folded `value` of type `ty`, or nothing to fold.
fn float_imm_of(ty: &PrimType, value: Option<f64>) -> Folded {
    match value {
        Some(value) => float_imm(ty, value),
        None => Ok(None),
    }
}

/// The `Divide by zero` error of a zero integer divisor.
fn check_int_divisor(divisor: &BigInt) -> Result<()> {
    match divisor.is_zero() {
        true => Err(support::compare_failed(
            INTERNAL_ERROR,
            "pb->value != 0",
            divisor,
            0,
            "Divide by zero",
        )),
        false => Ok(()),
    }
}

/// The `Divide by zero` error of a zero float divisor.
fn check_float_divisor(divisor: f64) -> Result<()> {
    match divisor == 0.0 {
        true => Err(support::compare_failed(
            INTERNAL_ERROR,
            "fb->value != 0",
            support::CppDouble(divisor),
            0,
            "Divide by zero",
        )),
        false => Ok(()),
    }
}

fn is_one(value: &BigInt) -> bool {
    value.to_i64() == Some(1)
}

/// The operand itself.
fn keep(expr: &PrimExpr) -> Folded {
    Ok(Some(expr.clone()))
}
