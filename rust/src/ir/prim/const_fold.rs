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

//! Constant folding of the operators, after `TryConstFold<Op>`
//! (`include/tvm/ir/prim/const_fold.h`).
//!
//! Each function takes operands of one type, as `BinaryOpMatchTypes` leaves them, and returns the
//! folded expression, or `None` when there is nothing to fold. Integers fold exactly, and only
//! when the result is a value of their type (`IsRepresentable`); floats fold for 32 and 64 bits, a
//! float32 in single precision and only within the float32 range (`IsInFloat32Range`); a constant
//! operand also folds the identities (`x + 0`, `x * 1`, ...). Division by a constant zero is an
//! error, as in C++.

use tvm_ffi::{BigInt, DLDataType, ObjectRefCore, Result, INTERNAL_ERROR};

use crate::ir::expr::{Expr, FloatImm, FloatImmObj, IntImm, IntImmObj};
use crate::ir::location::UnknownLoc;
use crate::ir::support;

/// The folded expression, or `None` when there is nothing to fold.
pub(super) type Folded = Result<Option<Expr>>;

// ---------------------------------------------------------------------------------------------
// Helpers

/// C++ `IsIndexType`: a scalar `int32` or `int64`, which index arithmetic folds eagerly.
pub(super) fn is_index_type(dtype: DLDataType) -> bool {
    dtype.code == support::INT && (dtype.bits == 32 || dtype.bits == 64) && dtype.lanes == 1
}

/// C++ `IsRepresentable(value, dtype)`: whether the exact integer `value` is a literal of the
/// scalar type `dtype`, an integer in its range or a boolean 0 or 1 (as is an `int1`).
pub(super) fn is_representable(value: &BigInt, dtype: DLDataType) -> bool {
    if !support::is_scalar(dtype) || dtype.bits == 0 {
        return false;
    }
    let bits = u32::from(dtype.bits);
    if let Some(small) = value.to_i64() {
        return match dtype.code {
            support::UINT => small >= 0 && (bits >= 63 || (small as u64) < (1u64 << bits)),
            support::BOOL => small == 0 || small == 1,
            support::INT if bits == 1 => small == 0 || small == 1,
            support::INT => {
                bits >= 64 || (-(1i64 << (bits - 1))..1i64 << (bits - 1)).contains(&small)
            }
            _ => false,
        };
    }
    match dtype.code {
        support::UINT => !value.is_negative() && *value < BigInt::from(1i64) << i64::from(bits),
        support::INT if bits > 1 => {
            let limit = BigInt::from(1i64) << i64::from(bits - 1);
            *value >= -limit.clone() && *value < limit
        }
        _ => false,
    }
}

/// C++ `IsInFloat32Range(value)`: a value that rounds to a finite float32, or one that is not
/// finite.
pub(super) fn is_in_float32_range(value: f64) -> bool {
    !value.is_finite() || (f64::from(f32::MIN)..=f64::from(f32::MAX)).contains(&value)
}

/// The value of an integer constant.
pub(super) fn int(expr: &Expr) -> Option<&BigInt> {
    expr.downcast_ref::<IntImmObj>().map(|imm| &imm.value)
}

/// The value of a float constant.
fn float(expr: &Expr) -> Option<f64> {
    expr.downcast_ref::<FloatImmObj>().map(|imm| imm.value)
}

/// C++ `IntImm(dtype, value)`.
fn int_imm(dtype: DLDataType, value: BigInt) -> Folded {
    Ok(Some(IntImm::from_dtype(dtype, value, UnknownLoc::new())?.into()))
}

/// C++ `IntImm(result_ty, value)` of an exact result, or nothing to fold when it is not a value of
/// the type.
fn exact(dtype: DLDataType, value: BigInt) -> Folded {
    match is_representable(&value, dtype) {
        true => int_imm(dtype, value),
        false => Ok(None),
    }
}

/// C++ `FloatImm(dtype, value)`.
fn float_imm(dtype: DLDataType, value: f64) -> Folded {
    Ok(Some(FloatImm::from_dtype(dtype, value, UnknownLoc::new())?.into()))
}

/// C++ `IntImm::Bool(value)`.
fn boolean(value: bool) -> Folded {
    Ok(Some(IntImm::bool(value).into()))
}

/// A float32 operation on the operands rounded to float32, as C++ computes `float` arithmetic, or
/// `None` when the result is beyond the float32 range: C++ checks the result in double precision
/// before rounding can turn a finite overflow into infinity, and rounding that result to float32
/// gives the single-precision one (a double holds more than twice the digits of a float).
fn single(x: f64, y: f64, op: fn(f64, f64) -> f64) -> Option<f64> {
    let result = op(f64::from(x as f32), f64::from(y as f32));
    is_in_float32_range(result).then(|| f64::from(result as f32))
}

/// C++ `FloatImm(dtype, value)` of a folded value, or nothing to fold.
fn float_imm_of(dtype: DLDataType, value: Option<f64>) -> Folded {
    match value {
        Some(value) => float_imm(dtype, value),
        None => Ok(None),
    }
}

/// The error of `TVM_FFI_ICHECK_NE(pb->value, 0) << "Divide by zero"`.
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

/// The error of `TVM_FFI_ICHECK_NE(fb->value, 0) << "Divide by zero"`.
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
fn keep(expr: &Expr) -> Folded {
    Ok(Some(expr.clone()))
}

// ---------------------------------------------------------------------------------------------
// Arithmetic

/// `TryConstFold<Add>`.
pub(super) fn add(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return exact(dtype, x + y);
    }
    if pa.is_some_and(BigInt::is_zero) {
        return keep(b);
    }
    if pb.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match dtype.bits {
            32 => return float_imm_of(dtype, single(x, y, |x, y| x + y)),
            64 => return float_imm(dtype, x + y),
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

/// `TryConstFold<Sub>`.
pub(super) fn sub(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return exact(dtype, x - y);
    }
    if pb.is_some_and(BigInt::is_zero) {
        return keep(a);
    }
    if let (Some(x), Some(y)) = (fa, fb) {
        match dtype.bits {
            32 => return float_imm_of(dtype, single(x, y, |x, y| x - y)),
            64 => return float_imm(dtype, x - y),
            _ => {}
        }
    }
    if fb == Some(0.0) {
        return keep(a);
    }
    Ok(None)
}

/// `TryConstFold<Mul>`.
pub(super) fn mul(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        return exact(dtype, x * y);
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
        match dtype.bits {
            32 => return float_imm_of(dtype, single(x, y, |x, y| x * y)),
            64 => return float_imm(dtype, x * y),
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

/// `TryConstFold<Div>`: integers divide truncating.
pub(super) fn div(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        check_int_divisor(y)?;
        return exact(dtype, x.try_div(y)?);
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
        match dtype.bits {
            32 => return float_imm_of(dtype, single(x, y, |x, y| x / y)),
            64 => return float_imm(dtype, x / y),
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

/// `TryConstFold<Mod>`, the truncating remainder: index types fold with the identities, wider
/// integers fold only two constants.
pub(super) fn truncmod(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    let (pa, pb) = (int(a), int(b));
    if is_index_type(dtype) && is_index_type(support::prim_dtype(b)?) {
        if let (Some(x), Some(y)) = (pa, pb) {
            check_int_divisor(y)?;
            return exact(dtype, x.try_rem(y)?);
        }
        if pa.is_some_and(BigInt::is_zero) {
            return keep(a);
        }
        if let Some(y) = pb {
            if is_one(y) {
                return int_imm(dtype, BigInt::from(0i64));
            }
            check_int_divisor(y)?;
        }
    }
    if let (Some(x), Some(y)) = (pa, pb) {
        if dtype.bits >= 64 {
            check_int_divisor(y)?;
            return exact(dtype, x.try_rem(y)?);
        }
    }
    Ok(None)
}

/// `TryConstFold<FloorDiv>`.
pub(super) fn floordiv(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    let (pa, pb, fa, fb) = (int(a), int(b), float(a), float(b));
    if let (Some(x), Some(y)) = (pa, pb) {
        check_int_divisor(y)?;
        return exact(dtype, x.floor_div(y)?);
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
        if y != 0.0 {
            return match dtype.bits {
                32 => float_imm_of(dtype, single(x, y, |x, y| x / y).map(f64::floor)),
                64 => float_imm(dtype, (x / y).floor()),
                _ => Ok(None),
            };
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

/// `TryConstFold<FloorMod>`: index types fold with the identities, wider or unsigned integers
/// fold only two constants.
pub(super) fn floormod(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    let (pa, pb) = (int(a), int(b));
    if is_index_type(dtype) && is_index_type(support::prim_dtype(b)?) {
        if let (Some(x), Some(y)) = (pa, pb) {
            check_int_divisor(y)?;
            return exact(dtype, x.floor_mod(y)?);
        }
        if pa.is_some_and(BigInt::is_zero) {
            return keep(a);
        }
        if let Some(y) = pb {
            if is_one(y) {
                return int_imm(dtype, BigInt::from(0i64));
            }
            check_int_divisor(y)?;
        }
    }
    if let (Some(x), Some(y)) = (pa, pb) {
        if dtype.bits >= 64 || dtype.code == support::UINT {
            check_int_divisor(y)?;
            return exact(dtype, x.floor_mod(y)?);
        }
    }
    Ok(None)
}

/// `TryConstFold<Min>`: two constants, or an operand with itself. Of two floats, the first is kept
/// when it is NaN or less than the second, and the second otherwise: a NaN in either operand is
/// the result, and of two equal values (`0.0` and `-0.0`) the second is.
pub(super) fn min(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return int_imm(dtype, x.min(y).clone());
    }
    if let (Some(x), Some(y)) = (float(a), float(b)) {
        return keep(if x.is_nan() || x < y { a } else { b });
    }
    if a.same_as(b) {
        return keep(a);
    }
    Ok(None)
}

/// `TryConstFold<Max>`, as [`min`]: the first float is kept when it is NaN or greater.
pub(super) fn max(a: &Expr, b: &Expr) -> Folded {
    let dtype = support::prim_dtype(a)?;
    if let (Some(x), Some(y)) = (int(a), int(b)) {
        return int_imm(dtype, x.max(y).clone());
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

/// `TryConstFold` of a comparison: two constants compare to a boolean constant.
pub(super) fn compare(
    a: &Expr,
    b: &Expr,
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

/// `TryConstFold<And>`: a true constant leaves the other side, a false one is the result.
pub(super) fn and(a: &Expr, b: &Expr) -> Folded {
    if let Some(x) = int(a) {
        return keep(if x.is_zero() { a } else { b });
    }
    if let Some(y) = int(b) {
        return keep(if y.is_zero() { b } else { a });
    }
    Ok(None)
}

/// `TryConstFold<Or>`: a true constant is the result, a false one leaves the other side.
pub(super) fn or(a: &Expr, b: &Expr) -> Folded {
    if let Some(x) = int(a) {
        return keep(if x.is_zero() { b } else { a });
    }
    if let Some(y) = int(b) {
        return keep(if y.is_zero() { a } else { b });
    }
    Ok(None)
}

/// `TryConstFold<Not>`.
pub(super) fn not(a: &Expr) -> Folded {
    match int(a) {
        Some(x) => boolean(x.is_zero()),
        None => Ok(None),
    }
}
