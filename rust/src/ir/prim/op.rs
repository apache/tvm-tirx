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

//! The primitive operators: arithmetic, comparisons, logic, casts and math functions, written in
//! Rust after `src/ir/prim/op.cc`, `src/ir/prim/math.cc` and `include/tvm/ir/prim/op.h`.
//!
//! The functions `python/tvm_tirx/ir/prim/op.py` defines, with the operators that
//! `python/tvm_tirx/ir/_prim_expr_overload.py` gives expressions (`a + b` is [`add`]). Unlike the
//! node constructors such as [`Add::new`](super::Add::new), which take operands of one type and
//! never fold, these match their operand types (`BinaryOpMatchTypes`), fold constants
//! (`TryConstFold`) and may return any [`Expr`]: a node, a constant or an intrinsic call, with the
//! checks and errors of the C++ functions.
//!
//! Each function builds at the unknown location, as the C++ default argument does; the `_at`
//! variant takes a location. An operand is an expression, an expression under construction (an
//! [`Ex`], see below) or a literal, typed as C++ types it: an integer next to an expression is
//! `MakeConst(other.ty(), v)`, an `f32` a float32 constant, an `f64` a float64 one, and a literal
//! on its own the constant [`From`] makes of it (an `i32` is an `int32`):
//!
//! ```no_run
//! use tvm_tirx::ir::prim::op::{add, floordiv};
//! use tvm_tirx::ir::{PrimType, Var, UNKNOWN_LOC};
//!
//! let x = Var::new("x", PrimType::int(32), UNKNOWN_LOC);
//! let index = floordiv(&x + 1, 4)?; // C++ `floordiv(x + 1, 4)`
//! let shifted = add(&x, 1)?;
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```
//!
//! `&x + 1` is the operator syntax of [`Ex`]: the C++ operators on an expression, a `Var` or an
//! `Ex`, with literals typed the same way. An `Ex` holds the expression under construction or the
//! first error any step raised; it passes into any function or constructor that takes
//! `impl IntoExpr`, whose `?` then reports that error, and [`Ex::expr`] takes the expression out.
//! The operations Rust has no operator for are the [`PrimOps`] methods (`x.floordiv(4)`,
//! `x.less(y)`, `x.cast(ty)`).

// Names the hand-written code uses: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::BigInt
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataType
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataTypeCode
// tvm-ffi-stubgen(import-object): tvm_ffi::Error
// tvm-ffi-stubgen(import-object): tvm_ffi::INTERNAL_ERROR
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::TYPE_ERROR
// tvm-ffi-stubgen(import-object): tvm_ffi::VALUE_ERROR
// tvm-ffi-stubgen(import-object): crate::ir::dtype::void
// tvm-ffi-stubgen(import-object): crate::ir::dtype::with_lanes
// tvm-ffi-stubgen(import-object): crate::ir::expr::Call
// tvm-ffi-stubgen(import-object): crate::ir::expr::CallObj
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::expr::FloatImm
// tvm-ffi-stubgen(import-object): crate::ir::expr::FloatImmObj
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImm
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntImmObj
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::TensorLoad
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::ir::location::Location
// tvm-ffi-stubgen(import-object): crate::ir::location::UNKNOWN_LOC
// tvm-ffi-stubgen(import-object): crate::ir::location::UnknownLoc
// tvm-ffi-stubgen(import-object): crate::ir::prim::const_fold
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Add
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::And
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::BitwiseAnd
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::BitwiseNot
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::BitwiseOr
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::BitwiseXor
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Cast
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Div
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::EQ
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::FloorDiv
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::FloorMod
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::GE
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::GT
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::LE
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::LShift
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::LT
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Let
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Max
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Min
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Mod
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Mul
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::NE
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Not
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Or
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::RShift
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Select
// tvm-ffi-stubgen(import-object): crate::ir::prim::expr::Sub
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::Broadcast
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::BroadcastObj
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::Ramp
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::RampObj
// tvm-ffi-stubgen(import-object): crate::ir::prim::vector_expr::Shuffle
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::Type
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::sym

// tvm-ffi-stubgen(begin): import-section
use crate::ir::dtype::void;
use crate::ir::dtype::with_lanes;
use crate::ir::expr::Call;
use crate::ir::expr::CallObj;
use crate::ir::expr::Expr;
use crate::ir::expr::FloatImm;
use crate::ir::expr::FloatImmObj;
use crate::ir::expr::IntImm;
use crate::ir::expr::IntImmObj;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::TensorLoad;
use crate::ir::expr::Var;
use crate::ir::location::Location;
use crate::ir::location::UNKNOWN_LOC;
use crate::ir::location::UnknownLoc;
use crate::ir::prim::const_fold;
use crate::ir::prim::expr::Add;
use crate::ir::prim::expr::And;
use crate::ir::prim::expr::BitwiseAnd;
use crate::ir::prim::expr::BitwiseNot;
use crate::ir::prim::expr::BitwiseOr;
use crate::ir::prim::expr::BitwiseXor;
use crate::ir::prim::expr::Cast;
use crate::ir::prim::expr::Div;
use crate::ir::prim::expr::EQ;
use crate::ir::prim::expr::FloorDiv;
use crate::ir::prim::expr::FloorMod;
use crate::ir::prim::expr::GE;
use crate::ir::prim::expr::GT;
use crate::ir::prim::expr::LE;
use crate::ir::prim::expr::LShift;
use crate::ir::prim::expr::LT;
use crate::ir::prim::expr::Let;
use crate::ir::prim::expr::Max;
use crate::ir::prim::expr::Min;
use crate::ir::prim::expr::Mod;
use crate::ir::prim::expr::Mul;
use crate::ir::prim::expr::NE;
use crate::ir::prim::expr::Not;
use crate::ir::prim::expr::Or;
use crate::ir::prim::expr::RShift;
use crate::ir::prim::expr::Select;
use crate::ir::prim::expr::Sub;
use crate::ir::prim::vector_expr::Broadcast;
use crate::ir::prim::vector_expr::BroadcastObj;
use crate::ir::prim::vector_expr::Ramp;
use crate::ir::prim::vector_expr::RampObj;
use crate::ir::prim::vector_expr::Shuffle;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::Type;
use crate::ir::support;
use crate::sym;
use tvm_ffi::Any;
use tvm_ffi::BigInt;
use tvm_ffi::DLDataType;
use tvm_ffi::DLDataTypeCode;
use tvm_ffi::Error;
use tvm_ffi::INTERNAL_ERROR;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::TYPE_ERROR;
use tvm_ffi::VALUE_ERROR;
// tvm-ffi-stubgen(end)

// ---------------------------------------------------------------------------------------------
// Types

/// The `float4` code C++ `IsFloat4Type` matches.
const FLOAT4: DLDataTypeCode = DLDataTypeCode::kDLFloat4_e2m1fn;

/// C++ `IsFloat8Type`.
fn is_float8(code: u8) -> bool {
    support::float8_bound(code).is_some()
}

/// C++ `IsFloat6Type`.
fn is_float6(code: u8) -> bool {
    support::float6_bound(code).is_some()
}

/// The largest finite value of a float8 or float6 type, and whether the type is unsigned, its
/// minimum then 0 (C++ `max_value` and `min_value`).
fn small_float_bound(code: u8) -> Option<(f64, bool)> {
    support::float8_bound(code).or_else(|| support::float6_bound(code).map(|max| (max, false)))
}

fn is_float4(code: u8) -> bool {
    FLOAT4 as u8 == code
}

/// C++ `MatchesElementType(kDLBfloat, 16)`.
fn is_bfloat16(dtype: DLDataType) -> bool {
    dtype.code == support::BFLOAT && dtype.bits == 16
}

/// A vector of any kind: fixed length or scalable.
fn is_vector(dtype: DLDataType) -> bool {
    support::is_scalable_vector(dtype) || support::is_fixed_length_vector(dtype)
}

/// The vscale factor a scalable vector type encodes (C++ `VScaleFactor`).
fn vscale_factor(dtype: DLDataType) -> i32 {
    -i32::from(support::encoded_lanes(dtype))
}

/// The lanes of a vector type as an expression: the count, or `factor * vscale()` for a scalable
/// vector (C++ `BroadcastToMatchLanes` and `cast`).
fn lanes_of(dtype: DLDataType) -> Result<Expr> {
    match support::is_scalable_vector(dtype) {
        true => {
            let factor = IntImm::int32(vscale_factor(dtype));
            Ok(Mul::new(factor, vscale()?, UNKNOWN_LOC)?.into())
        }
        false => Ok(IntImm::int32(support::lanes(dtype)?).into()),
    }
}

/// `Call(ty, Op::Get(op), args, {}, {}, loc)`; without `ty` the constructor infers the type.
fn call(ty: Option<&Type>, op: &'static str, args: Vec<Expr>, loc: Location) -> Result<Expr> {
    let op = support::cached_op(op)?;
    Ok(Call::new(ty, op, args, None, Vec::<Type>::new(), loc)?.into())
}

/// Whether `expr` is a call of `prim.vscale`.
fn is_vscale_call(expr: &Expr) -> Result<bool> {
    let Some(call) = expr.as_node::<CallObj>() else { return Ok(false) };
    Ok(call.op.same_as(&support::cached_op("prim.vscale")?))
}

/// The `TVM_FFI_ICHECK` an operation makes of an operand's type code: `codes` is the C++ text
/// of the accepted codes, `message` what it reports.
fn check_code(
    operand: &Expr,
    side: &str,
    accepted: &[u8],
    codes: &str,
    message: impl FnOnce() -> String,
) -> Result<()> {
    if accepted.contains(&support::prim_dtype(operand)?.code) {
        return Ok(());
    }
    let condition = format!("{side}.ty().MatchesCode({codes})");
    Err(support::check_failed(INTERNAL_ERROR, &condition, &message()))
}

const INTEGER: [u8; 2] = [support::INT, support::UINT];
const INTEGER_CODES: &str = "DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt";
const INTEGER_OR_BOOL: [u8; 3] = [support::INT, support::UINT, support::BOOL];
const INTEGER_OR_BOOL_CODES: &str =
    "DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLBool";
const BOOL_CODES: &str = "DLDataTypeCode::kDLBool";

/// `TVM_FFI_ICHECK(x.ty().MatchesCode(kDLInt, kDLUInt)) << x` of the integer divisions, on both
/// operands.
fn check_integers(a: &Expr, b: &Expr) -> Result<()> {
    check_code(a, "a", &INTEGER, INTEGER_CODES, || support::repr(a))?;
    check_code(b, "b", &INTEGER, INTEGER_CODES, || support::repr(b))
}

/// The operand checks of an operator `symbol` (`what` names it) that takes `kind` arguments:
/// integers, integers or booleans, or booleans.
fn check_operands(
    a: &Expr,
    b: &Expr,
    symbol: &str,
    what: &str,
    kind: &str,
    (accepted, codes): (&[u8], &str),
) -> Result<()> {
    for (operand, side, which) in [(a, "a", "LHS"), (b, "b", "RHS")] {
        check_code(operand, side, accepted, codes, || {
            format!(
                "Expected {kind} argument as {which} of {symbol} operator ({what}), but received \
                 {} of type {}",
                support::repr(operand),
                support::repr(&operand.ty)
            )
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Operands

/// An operand of an operator: an expression, an expression under construction ([`Ex`]) or a
/// literal, which is typed after the expression it meets (C++
/// `TVM_DEFINE_BINOP_CONST_VAL_OVERLOAD` and friends): an `i32` or `i64` is
/// `MakeConst(other.ty(), v)`, an `f32` a float32 constant, an `f64` a float64 one. A literal
/// that meets no expression is the constant [`From`] makes of it.
pub trait Operand {
    #[doc(hidden)]
    fn classify(self) -> OperandKind;
}

/// What an [`Operand`] is.
#[doc(hidden)]
pub enum OperandKind {
    Expr(Ex),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
}

impl<T: Into<Ex>> Operand for T {
    fn classify(self) -> OperandKind {
        OperandKind::Expr(self.into())
    }
}

/// C++ `int`: `MakeConst(other.ty(), v)`. `i32` is what an unsuffixed literal falls back to.
impl Operand for i32 {
    fn classify(self) -> OperandKind {
        OperandKind::Int32(self)
    }
}

impl Operand for i64 {
    fn classify(self) -> OperandKind {
        OperandKind::Int64(self)
    }
}

/// C++ `float`: a float32 constant, whatever the other side.
impl Operand for f32 {
    fn classify(self) -> OperandKind {
        OperandKind::Float32(self)
    }
}

/// C++ `double`: a float64 constant, whatever the other side.
impl Operand for f64 {
    fn classify(self) -> OperandKind {
        OperandKind::Float64(self)
    }
}

/// An operand next to `peer`: a literal is typed after it.
fn typed_after(operand: OperandKind, peer: &Ex) -> Ex {
    let constant = |value: i64| {
        Ex(peer.dtype().and_then(|dtype| make_const(PrimType::from_dtype(dtype)?, value)))
    };
    match operand {
        OperandKind::Expr(ex) => ex,
        OperandKind::Int32(value) => constant(i64::from(value)),
        OperandKind::Int64(value) => constant(value),
        OperandKind::Float32(value) => Ex(Ok(FloatImm::float32(value).into())),
        OperandKind::Float64(value) => Ex(Ok(FloatImm::float64(value).into())),
    }
}

/// An operand on its own: a literal is the constant [`From`] makes of it.
fn on_its_own(operand: OperandKind) -> Ex {
    match operand {
        OperandKind::Expr(ex) => ex,
        OperandKind::Int32(value) => Ex(Ok(IntImm::int32(value).into())),
        OperandKind::Int64(value) => Ex(Ok(IntImm::int64(value).into())),
        OperandKind::Float32(value) => Ex(Ok(FloatImm::float32(value).into())),
        OperandKind::Float64(value) => Ex(Ok(FloatImm::float64(value).into())),
    }
}

/// The two operands of a binary operation as expressions: a literal takes the type of the other
/// side, two literals their own.
fn operands(a: impl Operand, b: impl Operand) -> Result<(Expr, Expr)> {
    let (a, b) = match (a.classify(), b.classify()) {
        (OperandKind::Expr(a), b) => {
            let b = typed_after(b, &a);
            (a, b)
        }
        (a, OperandKind::Expr(b)) => {
            let a = typed_after(a, &b);
            (a, b)
        }
        (a, b) => (on_its_own(a), on_its_own(b)),
    };
    Ok((a.expr()?, b.expr()?))
}

/// The functions without a location: each builds at the unknown location through its `_at`
/// variant, as the C++ default argument does.
macro_rules! unlocated {
    ($($(#[$doc:meta])* $name:ident($($arg:ident: $ty:ty),*) => $at:ident;)+) => {
        $(
            $(#[$doc])*
            pub fn $name($($arg: $ty),*) -> Result<Expr> {
                $at($($arg,)* UNKNOWN_LOC)
            }
        )+
    };
}

// ---------------------------------------------------------------------------------------------
// Constants (C++ `MakeConst`)

/// The value of a constant, as C++ `MakeConst` takes it: an integer or a double.
#[derive(Clone, Debug)]
pub enum ConstValue {
    Int(BigInt),
    Float(f64),
}

macro_rules! const_value_from_int {
    ($($int:ty),+) => {
        $(
            impl From<$int> for ConstValue {
                fn from(value: $int) -> Self {
                    ConstValue::Int(BigInt::from(value))
                }
            }
        )+
    };
}

const_value_from_int!(i8, i16, i32, i64, u8, u16, u32, u64, bool, BigInt);

impl From<f64> for ConstValue {
    fn from(value: f64) -> Self {
        ConstValue::Float(value)
    }
}

impl From<f32> for ConstValue {
    fn from(value: f32) -> Self {
        ConstValue::Float(f64::from(value))
    }
}

/// C++ `prim::MakeConstScalar(dtype, value, loc)`: an `IntImm` for an integer or boolean type, a
/// `FloatImm` for a floating-point type.
pub(crate) fn make_const_scalar(
    dtype: DLDataType,
    value: ConstValue,
    loc: &Location,
) -> Result<Expr> {
    let code = dtype.code;
    if matches!(code, support::INT | support::UINT | support::BOOL) {
        let value = match value {
            ConstValue::Int(value) => value,
            ConstValue::Float(value) => {
                if code == support::UINT && value < 0.0 {
                    let message = format!(
                        "cannot make uint from negative value {}",
                        support::CppDouble(value)
                    );
                    return Err(support::check_failed(INTERNAL_ERROR, "value >= 0", &message));
                }
                BigInt::try_from(value)?
            }
        };
        if code == support::UINT && value.is_negative() {
            let message = format!("cannot make uint from negative value {value}");
            return Err(support::check_failed(INTERNAL_ERROR, "value >= 0", &message));
        }
        return Ok(IntImm::from_dtype(dtype, value, loc.clone())?.into());
    }
    if code == support::FLOAT
        || is_float8(code)
        || is_float6(code)
        || is_float4(code)
        || is_bfloat16(dtype)
    {
        let value = match value {
            ConstValue::Int(value) => value.to_f64()?,
            ConstValue::Float(value) => value,
        };
        return Ok(FloatImm::from_dtype(dtype, value, loc.clone())?.into());
    }
    let ty = PrimType::from_dtype(dtype)?;
    let message = format!("cannot make const for type {}", support::repr(&ty));
    Err(Error::new(INTERNAL_ERROR, &message, ""))
}

unlocated! {
    /// The constant `value` of type `ty`: an [`IntImm`] of an integer or boolean type, a
    /// [`FloatImm`] of a floating-point type, and the scalar constant broadcast over the lanes of
    /// a vector type (C++ `prim::MakeConst(ty, value)`, which has no global function). An
    /// unsigned type takes no negative value. Prefer [`IntImm::from_dtype`] or
    /// [`FloatImm::from_dtype`] when the kind of the type is known.
    ///
    /// [`FloatImm`]: crate::ir::FloatImm
    /// [`FloatImm::from_dtype`]: crate::ir::FloatImm::from_dtype
    make_const(ty: impl Into<PrimType>, value: impl Into<ConstValue>) => make_const_at;
}

/// [`make_const`] at `loc`.
pub fn make_const_at(
    ty: impl Into<PrimType>,
    value: impl Into<ConstValue>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (dtype, value, loc) = (ty.into().dtype, value.into(), loc.into());
    if !is_vector(dtype) {
        return make_const_scalar(dtype, value, &loc);
    }
    let scalar = make_const_scalar(with_lanes(dtype, 1), value, &loc)?;
    let lanes = match support::is_fixed_length_vector(dtype) {
        true => IntImm::int32(support::lanes(dtype)?).into(),
        false => super::vector_expr::vscale_lanes(vscale_factor(dtype))?,
    };
    Ok(Broadcast::new(scalar, lanes, loc)?.into())
}

/// `value` as an expression: a Rust `bool`, integer or float becomes a constant of the default
/// type, and an expression is returned as is (`prim.convert`).
pub fn convert(value: impl Into<Any>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("prim.convert").call_tuple((value.into(),))?.try_into()
}

// ---------------------------------------------------------------------------------------------
// Casts and type matching (C++ `cast` and `BinaryOpMatchTypes`)

/// The exponent C++ `frexp` gives a finite `value`, `e` with `value = m * 2^e` and
/// `0.5 <= |m| < 1`, and 0 for zero.
fn frexp_exponent(value: f64) -> i32 {
    match ((value.to_bits() >> 52) & 0x7ff) as i32 {
        0 if value == 0.0 => 0,
        // A subnormal, scaled into the normal range.
        0 => frexp_exponent(value * pow2(64)) - 64,
        biased => biased - 1022,
    }
}

/// `2^exponent` for an exponent of a normal double.
fn pow2(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// Whether the finite `value` of a float16 or bfloat16 `FloatImm` is a value of its type: a
/// `FloatImm` holds an unrounded double, which stands for the value the type rounds it to.
fn is_rounded_to_16_bits(value: f64, bfloat: bool) -> bool {
    let (digits, min_step) = if bfloat { (8, -133) } else { (11, -24) };
    let step = (frexp_exponent(value) - digits).max(min_step);
    let scaled = value * pow2(-step);
    scaled.trunc() == scaled
}

/// A constant cast to the scalar type `to` as C++ `cast` folds it: only when the result is the
/// same value in `to`, an integer it can represent or a float within its range, and when a float
/// type that rounds the result does so itself; `None` leaves a [`Cast`].
fn fold_cast(value: &Expr, to: DLDataType) -> Result<Option<Expr>> {
    let integer = matches!(to.code, support::INT | support::UINT | support::BOOL);
    let float32 = to.code == support::FLOAT && to.bits == 32;
    let float64 = to.code == support::FLOAT && to.bits == 64;
    let limit_16 = match (to.code, to.bits) {
        (support::FLOAT, 16) => Some(support::MAX_FLOAT16),
        (support::BFLOAT, 16) => Some(support::MAX_BFLOAT16),
        _ => None,
    };
    let int = |value: BigInt, loc: &Location| -> Result<Option<Expr>> {
        Ok(Some(IntImm::from_dtype(to, value, loc.clone())?.into()))
    };
    let float = |value: f64, loc: &Location| -> Result<Option<Expr>> {
        Ok(Some(FloatImm::from_dtype(to, value, loc.clone())?.into()))
    };
    if let Some(imm) = value.as_node::<IntImmObj>() {
        if integer {
            if const_fold::is_representable(&imm.value, to) {
                return int(imm.value.clone(), &imm.loc);
            }
        } else if float32 {
            if let Some(small) = imm.value.to_i64() {
                return float(f64::from(small as f32), &imm.loc);
            }
        } else if float64 {
            let limit = BigInt::try_from(f64::MAX)?;
            if (-&limit..=limit).contains(&imm.value) {
                return float(imm.value.to_f64()?, &imm.loc);
            }
        } else if let Some(limit) = limit_16 {
            let source = imm.value.to_f64()?;
            // Keep the integer exact until the destination performs its rounding.
            if (-limit..=limit).contains(&source) && BigInt::try_from(source)? == imm.value {
                return float(source, &imm.loc);
            }
        }
    } else if let Some(imm) = value.as_node::<FloatImmObj>() {
        let from = support::prim_dtype(value)?;
        if imm.value.is_finite() && from.bits == 16 {
            let bfloat = match from.code {
                support::FLOAT => Some(false),
                support::BFLOAT => Some(true),
                _ => None,
            };
            // Using the unrounded double would skip the rounding of the source type.
            if bfloat.is_some_and(|bfloat| !is_rounded_to_16_bits(imm.value, bfloat)) {
                return Ok(None);
            }
        }
        let source = match from.code == support::FLOAT && from.bits == 32 {
            true => f64::from(imm.value as f32),
            false => imm.value,
        };
        if integer {
            // A cast to bool tests for nonzero rather than truncating a fraction.
            let fits_bool = to.code != support::BOOL || source == 0.0 || source == 1.0;
            if source.is_finite() && fits_bool {
                let result = BigInt::try_from(source)?;
                if const_fold::is_representable(&result, to) {
                    return int(result, &imm.loc);
                }
            }
        } else if float32 {
            if const_fold::is_in_float32_range(source) {
                return float(f64::from(source as f32), &imm.loc);
            }
        } else if float64
            && ((from.code == support::FLOAT && matches!(from.bits, 16 | 32)) || is_bfloat16(from))
        {
            // Source rounding was checked above; widening preserves its value exactly.
            return float(source, &imm.loc);
        } else if let Some(limit) = limit_16 {
            if !source.is_finite() || (-limit..=limit).contains(&source) {
                return float(source, &imm.loc);
            }
        }
    }
    Ok(None)
}

/// C++ `cast` to a scalar type: a constant folded to `ty` keeping its location (see
/// [`fold_cast`]); anything else a [`Cast`].
fn cast_scalar(value: Expr, ty: PrimType, loc: &Location) -> Result<Expr> {
    match fold_cast(&value, ty.dtype)? {
        Some(folded) => Ok(folded),
        None => Ok(Cast::new(ty, value, loc.clone())?.into()),
    }
}

unlocated! {
    /// `value` converted to `ty` (C++ `prim::cast(value, t)`, `prim._cast`): a value already of
    /// that type is itself, a constant is folded when the result is the same value in `ty`, a
    /// scalar cast to a vector type is broadcast, a broadcast is cast element by element, and so
    /// is a constant ramp cast to an integer type when every lane fits both types; anything else
    /// is a [`Cast`] node, which [`Cast::new`] builds in every case.
    cast(value: impl IntoExpr, ty: impl Into<PrimType>) => cast_at;
}

/// [`cast`] at `loc`.
pub fn cast_at(
    value: impl IntoExpr,
    ty: impl Into<PrimType>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (value, ty, loc) = (value.into_expr()?, ty.into(), loc.into());
    let (from, to) = (support::prim_dtype(&value)?, ty.dtype);
    if from == to {
        return Ok(value);
    }
    if from == void() {
        let message = "Cannot cast an expression with the void sentinel type";
        return Err(support::check_failed(TYPE_ERROR, "!value.ty().IsVoid()", message));
    }
    if support::is_scalar(to) {
        return cast_scalar(value, ty, &loc);
    }
    let element = PrimType::from_dtype(with_lanes(to, 1))?;
    if !is_vector(from) {
        let value = cast_at(value, element, loc.clone())?;
        return Ok(Broadcast::new(value, lanes_of(to)?, loc)?.into());
    }
    if support::is_scalable_vector(from) != support::is_scalable_vector(to) {
        let condition = "value.ty().IsScalableVector() == dtype.IsScalableVector()";
        return Err(support::check_failed(INTERNAL_ERROR, condition, ""));
    }
    let lanes_match = match support::is_scalable_vector(from) {
        true => vscale_factor(from) == vscale_factor(to),
        false => support::lanes(from)? == support::lanes(to)?,
    };
    if !lanes_match {
        return Err(support::check_failed(INTERNAL_ERROR, "lanes_match", ""));
    }
    if let Some(broadcast) = value.as_node::<BroadcastObj>() {
        let value = cast_at(broadcast.value.clone(), element, loc.clone())?;
        return Ok(Broadcast::new(value, broadcast.lanes.clone(), loc)?.into());
    }
    if let Some(ramp) = value.as_node::<RampObj>() {
        // Casting the base and the stride is valid only when every lane is the same integer in
        // both types: widening must not remove an overflow of the source, and a floating-point
        // cast need not be affine.
        let integer = matches!(to.code, support::INT | support::UINT) && matches!(to.bits, 32 | 64);
        if let (Some(base), Some(stride)) =
            (const_fold::int(&ramp.base_), const_fold::int(&ramp.stride))
        {
            if support::is_fixed_length_vector(to) && integer {
                let last = base + stride * i64::from(support::lanes(to)? - 1);
                let source = support::prim_dtype(&ramp.base_)?;
                let fits = |value: &BigInt| const_fold::is_representable(value, element.dtype);
                if fits(base)
                    && fits(stride)
                    && fits(&last)
                    && const_fold::is_representable(&last, source)
                {
                    let base = cast_at(ramp.base_.clone(), element.clone(), loc.clone())?;
                    let stride = cast_at(ramp.stride.clone(), element, loc.clone())?;
                    return Ok(Ramp::new(base, stride, ramp.lanes.clone(), loc)?.into());
                }
            }
        }
    }
    Ok(Cast::new(ty, value, loc)?.into())
}

/// C++ `BroadcastToMatchLanes`: `value`, broadcast to the lanes of `other` when it is a scalar
/// next to a vector.
fn broadcast_to_match_lanes(value: Expr, other: DLDataType) -> Result<Expr> {
    if is_vector(support::prim_dtype(&value)?) || !is_vector(other) {
        return Ok(value);
    }
    Ok(Broadcast::new(value, lanes_of(other)?, UNKNOWN_LOC)?.into())
}

/// C++ `PromoteBinaryOpType`: the type both operands of a binary operation take.
fn promote_binary_op_type(lhs: DLDataType, rhs: DLDataType) -> Result<DLDataType> {
    if lhs == rhs {
        return Ok(lhs);
    }
    let wider = || if lhs.bits < rhs.bits { rhs } else { lhs };
    let float = |dtype: DLDataType| dtype.code == support::FLOAT;
    let bfloat = |dtype: DLDataType| dtype.code == support::BFLOAT;
    let families: [fn(DLDataType) -> bool; 5] = [
        float,
        bfloat,
        |dtype| is_float8(dtype.code),
        |dtype| is_float6(dtype.code),
        |dtype| is_float4(dtype.code),
    ];
    if float(lhs) && float(rhs) {
        return Ok(wider());
    }
    // Each floating-point family takes precedence over every family after it.
    for family in families {
        match (family(lhs), family(rhs)) {
            (false, true) => return Ok(rhs),
            (true, false) => return Ok(lhs),
            _ => {}
        }
    }
    let integer = |dtype: DLDataType| matches!(dtype.code, support::INT | support::UINT);
    match (lhs.code, rhs.code) {
        (support::BOOL, _) if integer(rhs) => Ok(rhs),
        (_, support::BOOL) if integer(lhs) => Ok(lhs),
        (support::INT, support::INT) | (support::UINT, support::UINT) => Ok(wider()),
        (support::INT, support::UINT) | (support::UINT, support::INT) => match lhs.bits != rhs.bits
        {
            true => Ok(wider()),
            false => Ok(DLDataType { code: support::UINT, ..lhs }),
        },
        _ => {
            let (lhs, rhs) = (support::dtype_name(lhs), support::dtype_name(rhs));
            let message = format!("Cannot match type {lhs} vs {rhs}");
            Err(Error::new(INTERNAL_ERROR, &message, ""))
        }
    }
}

/// `a` and `b` brought to one type, as every binary operator does first
/// (C++ `BinaryOpMatchTypes(lhs, rhs)`): a scalar next to a vector is broadcast to its lanes,
/// and both sides are cast to the type `PromoteBinaryOpType` picks (the wider float, a float over
/// an integer, the wider integer, unsigned over signed of the same width).
pub fn match_types(a: impl IntoExpr, b: impl IntoExpr) -> Result<(Expr, Expr)> {
    match_types_at(a, b, UNKNOWN_LOC)
}

/// [`match_types`] with the casts at `loc`.
pub fn match_types_at(
    a: impl IntoExpr,
    b: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<(Expr, Expr)> {
    let (a, b, loc) = (a.into_expr()?, b.into_expr()?, loc.into());
    let (a_dtype, b_dtype) = (support::prim_dtype(&a)?, support::prim_dtype(&b)?);
    if a_dtype == b_dtype {
        return Ok((a, b));
    }
    let a = broadcast_to_match_lanes(a, b_dtype)?;
    let b = broadcast_to_match_lanes(b, support::prim_dtype(&a)?)?;
    let (a_dtype, b_dtype) = (support::prim_dtype(&a)?, support::prim_dtype(&b)?);
    if support::is_scalable_vector(a_dtype) != support::is_scalable_vector(b_dtype) {
        let condition = "lhs_ty.IsScalableVector() == rhs_ty.IsScalableVector()";
        let message = "Can't match scalable and fixed length vectors";
        return Err(support::check_failed(INTERNAL_ERROR, condition, message));
    }
    let lanes_match = match support::is_scalable_vector(a_dtype) {
        true => vscale_factor(a_dtype) == vscale_factor(b_dtype),
        false => support::lanes(a_dtype)? == support::lanes(b_dtype)?,
    };
    if !lanes_match {
        let (lhs, rhs) = (support::dtype_name(a_dtype), support::dtype_name(b_dtype));
        let message = format!("Cannot match type {lhs} vs {rhs}");
        return Err(support::check_failed(INTERNAL_ERROR, "lanes_match", &message));
    }
    let promoted = promote_binary_op_type(a_dtype, b_dtype)?;
    let a = match a_dtype == promoted {
        true => a,
        false => cast_at(a, PrimType::from_dtype(promoted)?, loc.clone())?,
    };
    let b = match b_dtype == promoted {
        true => b,
        false => cast_at(b, PrimType::from_dtype(promoted)?, loc)?,
    };
    Ok((a, b))
}

// ---------------------------------------------------------------------------------------------
// Arithmetic (`src/ir/prim/op.cc`)

/// The binary operators that match the operand types, fold constants and otherwise build the
/// raw node; each in a pair with its `_at` variant.
macro_rules! folded_binary {
    ($($(#[$doc:meta])* $name:ident / $at:ident: $fold:expr => $node:ident),+ $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(a: impl Operand, b: impl Operand) -> Result<Expr> {
                $at(a, b, UNKNOWN_LOC)
            }

            #[doc = concat!("[`", stringify!($name), "`] at `loc`.")]
            pub fn $at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
                let ((a, b), loc) = (operands(a, b)?, loc.into());
                let (a, b) = match_types_at(a, b, loc.clone())?;
                match $fold(&a, &b)? {
                    Some(folded) => Ok(folded),
                    None => Ok($node::new(a, b, loc)?.into()),
                }
            }
        )+
    };
}

folded_binary!(
    /// `a + b` (C++ `add`, `prim._OpAdd`).
    add / add_at: const_fold::add => Add,
    /// `a - b` (C++ `sub`, `prim._OpSub`).
    sub / sub_at: const_fold::sub => Sub,
    /// `a * b` (C++ `mul`, `prim._OpMul`).
    mul / mul_at: const_fold::mul => Mul,
    /// `a / b`; integer division rounds towards zero, as in C (C++ `div`, `prim._OpDiv`).
    div / div_at: const_fold::div => Div,
    /// The remainder of [`truncdiv`], with the sign of `a` (C++ `truncmod`, `prim._OpTruncMod`).
    truncmod / truncmod_at: const_fold::truncmod => Mod,
    /// The boolean `a > b` (C++ `greater`, `prim._OpGT`).
    gt / gt_at: fold_gt => GT,
    /// The boolean `a >= b` (C++ `greater_equal`, `prim._OpGE`).
    ge / ge_at: fold_ge => GE,
    /// The boolean `a < b` (C++ `less`, `prim._OpLT`).
    lt / lt_at: fold_lt => LT,
    /// The boolean `a <= b` (C++ `less_equal`, `prim._OpLE`).
    le / le_at: fold_le => LE,
    /// The boolean `a != b` (C++ `not_equal`, `prim._OpNE`).
    ne / ne_at: fold_ne => NE,
);

fn fold_gt(a: &Expr, b: &Expr) -> const_fold::Folded {
    const_fold::compare(a, b, |x, y| x > y, |x, y| x > y)
}

fn fold_ge(a: &Expr, b: &Expr) -> const_fold::Folded {
    const_fold::compare(a, b, |x, y| x >= y, |x, y| x >= y)
}

fn fold_lt(a: &Expr, b: &Expr) -> const_fold::Folded {
    const_fold::compare(a, b, |x, y| x < y, |x, y| x < y)
}

fn fold_le(a: &Expr, b: &Expr) -> const_fold::Folded {
    const_fold::compare(a, b, |x, y| x <= y, |x, y| x <= y)
}

fn fold_ne(a: &Expr, b: &Expr) -> const_fold::Folded {
    const_fold::compare(a, b, |x, y| x != y, |x, y| x != y)
}

unlocated! {
    /// The boolean `a == b`; two `vscale()` calls are equal (C++ `equal`, `prim._OpEQ`).
    eq(a: impl Operand, b: impl Operand) => eq_at;
    /// `-a`: a constant is negated in place when the result is a value of its type, anything
    /// else is `MakeConst(a.ty(), 0) - a` (C++ `neg`).
    neg(a: impl IntoExpr) => neg_at;
    /// `a / b`, rounding towards zero, of two integers (C++ `truncdiv`, `prim._OpTruncDiv`).
    truncdiv(a: impl Operand, b: impl Operand) => truncdiv_at;
    /// `a / b`, rounding down, of two integers (C++ `floordiv`, `prim._OpFloorDiv`).
    floordiv(a: impl Operand, b: impl Operand) => floordiv_at;
    /// The remainder of [`floordiv`], with the sign of `b` (C++ `floormod`, `prim._OpFloorMod`).
    floormod(a: impl Operand, b: impl Operand) => floormod_at;
    /// `a / b` over non-negative indices, which is [`floordiv`] (C++ `indexdiv`,
    /// `prim._OpIndexDiv`).
    indexdiv(a: impl Operand, b: impl Operand) => indexdiv_at;
    /// `a % b` over non-negative indices, which is [`floormod`] (C++ `indexmod`,
    /// `prim._OpIndexMod`).
    indexmod(a: impl Operand, b: impl Operand) => indexmod_at;
    /// `a / b`, rounding up, of two integers: `floordiv(a + b - 1, b)` (C++ `ceildiv`,
    /// `prim._OpCeilDiv`).
    ceildiv(a: impl Operand, b: impl Operand) => ceildiv_at;
    /// The smaller of `a` and `b`; a symbolic infinity on either side decides it (C++ `min`,
    /// `prim._OpMin`).
    min(a: impl Operand, b: impl Operand) => min_at;
    /// The larger of `a` and `b`; a symbolic infinity on either side decides it (C++ `max`,
    /// `prim._OpMax`).
    max(a: impl Operand, b: impl Operand) => max_at;
}

/// [`eq`] at `loc`.
pub fn eq_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    let (a, b) = match_types_at(a, b, loc.clone())?;
    if let Some(folded) = const_fold::compare(&a, &b, |x, y| x == y, |x, y| x == y)? {
        return Ok(folded);
    }
    if is_vscale_call(&a)? && is_vscale_call(&b)? {
        return Ok(IntImm::bool(true).into());
    }
    Ok(EQ::new(a, b, loc)?.into())
}

/// [`neg`] at `loc`.
pub fn neg_at(a: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let (a, loc) = (a.into_expr()?, loc.into());
    let dtype = support::prim_dtype(&a)?;
    if let Some(imm) = a.as_node::<IntImmObj>() {
        let value = -imm.value.clone();
        if const_fold::is_representable(&value, dtype) {
            return Ok(IntImm::from_dtype(dtype, value, loc)?.into());
        }
    }
    if let Some(imm) = a.as_node::<FloatImmObj>() {
        return Ok(FloatImm::from_dtype(dtype, -imm.value, loc)?.into());
    }
    sub(make_const_at(PrimType::from_dtype(dtype)?, 0, loc)?, a)
}

/// [`truncdiv`] at `loc`.
pub fn truncdiv_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let (a, b) = operands(a, b)?;
    check_integers(&a, &b)?;
    div_at(a, b, loc)
}

/// [`floordiv`] at `loc`.
pub fn floordiv_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    check_integers(&a, &b)?;
    let (a, b) = match_types_at(a, b, loc.clone())?;
    match const_fold::floordiv(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(FloorDiv::new(a, b, loc)?.into()),
    }
}

/// [`floormod`] at `loc`.
pub fn floormod_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    check_integers(&a, &b)?;
    let (a, b) = match_types_at(a, b, loc.clone())?;
    match const_fold::floormod(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(FloorMod::new(a, b, loc)?.into()),
    }
}

/// [`indexdiv`] at `loc`.
pub fn indexdiv_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    floordiv_at(a, b, loc)
}

/// [`indexmod`] at `loc`.
pub fn indexmod_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    floormod_at(a, b, loc)
}

/// [`ceildiv`] at `loc`.
pub fn ceildiv_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    check_integers(&a, &b)?;
    let (a, b) = match_types_at(a, b, loc.clone())?;
    let sum = add(a, b.clone())?;
    let one = make_const(support::prim_type(&sum)?, 1)?;
    let numerator = sub(sum, one)?;
    match const_fold::floordiv(&numerator, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(FloorDiv::new(numerator, b, loc)?.into()),
    }
}

/// `SymbolicLimits::pos_inf_` and `neg_inf_`, the opaque infinities of integer analysis, which
/// [`min`] and [`max`] recognize by identity; fetched once per thread.
fn symbolic_infinities() -> (Expr, Expr) {
    use std::cell::OnceCell;
    thread_local! {
        static INFINITIES: OnceCell<(Expr, Expr)> = const { OnceCell::new() };
    }
    support::shared(&INFINITIES, || Ok((sym::pos_inf()?, sym::neg_inf()?)))
}

/// [`min`] at `loc`.
pub fn min_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    let (pos_inf, neg_inf) = symbolic_infinities();
    if a.same_as(&pos_inf) {
        return Ok(b);
    }
    if a.same_as(&neg_inf) {
        return Ok(a);
    }
    if b.same_as(&pos_inf) {
        return Ok(a);
    }
    if b.same_as(&neg_inf) {
        return Ok(b);
    }
    let (a, b) = match_types_at(a, b, loc.clone())?;
    match const_fold::min(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(Min::new(a, b, loc)?.into()),
    }
}

/// [`max`] at `loc`.
pub fn max_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    let (pos_inf, neg_inf) = symbolic_infinities();
    if a.same_as(&pos_inf) {
        return Ok(a);
    }
    if a.same_as(&neg_inf) {
        return Ok(b);
    }
    if b.same_as(&pos_inf) {
        return Ok(b);
    }
    if b.same_as(&neg_inf) {
        return Ok(a);
    }
    let (a, b) = match_types_at(a, b, loc.clone())?;
    match const_fold::max(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(Max::new(a, b, loc)?.into()),
    }
}

// ---------------------------------------------------------------------------------------------
// Logic

unlocated! {
    /// `a && b` of two booleans, which are not matched; a constant side folds
    /// (C++ `logical_and`, `prim._OpAnd`).
    logical_and(a: impl IntoExpr, b: impl IntoExpr) => logical_and_at;
    /// `a || b` of two booleans, which are not matched; a constant side folds
    /// (C++ `logical_or`, `prim._OpOr`).
    logical_or(a: impl IntoExpr, b: impl IntoExpr) => logical_or_at;
    /// `!a` of a boolean; a constant folds (C++ `logical_not`).
    logical_not(a: impl IntoExpr) => logical_not_at;
    /// The conjunction of `conds`, which must not be empty (`prim.all`).
    all(conds: impl IntoIterator<Item = impl IntoExpr>) => all_at;
    /// The disjunction of `conds`, which must not be empty (`prim.any`).
    any(conds: impl IntoIterator<Item = impl IntoExpr>) => any_at;
    /// `true_value` where `cond` holds and `false_value` elsewhere, evaluating only the chosen
    /// one; a constant condition folds (C++ `if_then_else`, `prim._OpIfThenElse`).
    if_then_else(cond: impl IntoExpr, true_value: impl IntoExpr, false_value: impl IntoExpr) => if_then_else_at;
}

/// [`logical_and`] at `loc`.
pub fn logical_and_at(
    a: impl IntoExpr,
    b: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (a, b) = (a.into_expr()?, b.into_expr()?);
    check_operands(&a, &b, "&&", "logical AND", "boolean", (&[support::BOOL], BOOL_CODES))?;
    match const_fold::and(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(And::new(a, b, loc)?.into()),
    }
}

/// [`logical_or`] at `loc`.
pub fn logical_or_at(a: impl IntoExpr, b: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let (a, b) = (a.into_expr()?, b.into_expr()?);
    check_operands(&a, &b, "||", "logical OR", "boolean", (&[support::BOOL], BOOL_CODES))?;
    match const_fold::or(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(Or::new(a, b, loc)?.into()),
    }
}

/// [`logical_not`] at `loc`.
pub fn logical_not_at(a: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let a = a.into_expr()?;
    check_code(&a, "a", &[support::BOOL], BOOL_CODES, || {
        format!(
            "Expected boolean argument for ! operator (logical NOT), but received {} of type {}",
            support::repr(&a),
            support::repr(&a.ty)
        )
    })?;
    match const_fold::not(&a)? {
        Some(folded) => Ok(folded),
        None => Ok(Not::new(a, loc)?.into()),
    }
}

/// [`all`] at `loc`.
pub fn all_at(
    conds: impl IntoIterator<Item = impl IntoExpr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    fold_conditions(conds, loc.into(), "all", logical_and_at)
}

/// [`any`] at `loc`.
pub fn any_at(
    conds: impl IntoIterator<Item = impl IntoExpr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    fold_conditions(conds, loc.into(), "any", logical_or_at)
}

fn fold_conditions(
    conds: impl IntoIterator<Item = impl IntoExpr>,
    loc: Location,
    name: &str,
    op: fn(Expr, Expr, Location) -> Result<Expr>,
) -> Result<Expr> {
    let mut conds = conds.into_iter().map(IntoExpr::into_expr);
    let Some(first) = conds.next() else {
        let message = format!("{name} must take at least 1 argument");
        return Err(Error::new(VALUE_ERROR, &message, ""));
    };
    conds.try_fold(first?, |acc, cond| op(acc, cond?, loc.clone()))
}

/// [`if_then_else`] at `loc`.
pub fn if_then_else_at(
    cond: impl IntoExpr,
    true_value: impl IntoExpr,
    false_value: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (cond, loc) = (cond.into_expr()?, loc.into());
    if support::prim_dtype(&cond)?.code != support::BOOL {
        let condition = "cond.ty().MatchesCode(DLDataTypeCode::kDLBool)";
        let message = "if_then_else only accept the condition to be boolean type.";
        return Err(support::check_failed(INTERNAL_ERROR, condition, message));
    }
    let (true_value, false_value) = match_types_at(true_value, false_value, loc.clone())?;
    if let Some(imm) = cond.as_node::<IntImmObj>() {
        return Ok(if imm.value.is_zero() { false_value } else { true_value });
    }
    let ty = true_value.ty.clone();
    call(Some(&ty), "prim.if_then_else", vec![cond, true_value, false_value], loc)
}

// ---------------------------------------------------------------------------------------------
// Bitwise operators

/// The folding of a shift of index types: a constant amount is checked against the width, two
/// constants fold to the value `shift` gives them, if any, and a shift by zero is the operand
/// itself (`TVM_PRIM_INDEX_CONST_PROPAGATION`).
fn fold_shift(
    a: &Expr,
    b: &Expr,
    shift: impl FnOnce(&BigInt, &BigInt) -> Result<Option<BigInt>>,
    loc: &Location,
) -> const_fold::Folded {
    let dtype = support::prim_dtype(a)?;
    if !const_fold::is_index_type(dtype) || !const_fold::is_index_type(support::prim_dtype(b)?) {
        return Ok(None);
    }
    let (pa, pb) = (const_fold::int(a), const_fold::int(b));
    if let Some(y) = pb {
        if y.is_negative() || *y >= BigInt::from(i64::from(dtype.bits)) {
            let ty = support::prim_type(a)?;
            let condition = "pb->value >= 0 && pb->value < result_ty.bits()";
            let message = format!(
                "Shift amount must be non-negative and less than {} for type {}",
                dtype.bits,
                support::repr(&ty)
            );
            return Err(support::check_failed(INTERNAL_ERROR, condition, &message));
        }
    }
    if let (Some(x), Some(y)) = (pa, pb) {
        if let Some(value) = shift(x, y)? {
            return Ok(Some(IntImm::from_dtype(dtype, value, loc.clone())?.into()));
        }
    }
    if pb.is_some_and(BigInt::is_zero) {
        return Ok(Some(a.clone()));
    }
    Ok(None)
}

unlocated! {
    /// `a << b` of two integers; two constants fold when the result is a value of their type
    /// (C++ `left_shift`, `prim.left_shift`).
    shift_left(a: impl Operand, b: impl Operand) => shift_left_at;
    /// `a >> b` of two integers, sign-extending (C++ `right_shift`, `prim.right_shift`).
    shift_right(a: impl Operand, b: impl Operand) => shift_right_at;
    /// The node `~a` of an integer or boolean, which, unlike the other operators, neither
    /// promotes nor folds (C++ `bitwise_neg`, which checks the operand before
    /// [`BitwiseNot::new`] does).
    bitwise_not(a: impl IntoExpr) => bitwise_not_at;
}

/// [`shift_left`] at `loc`.
pub fn shift_left_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    check_operands(&a, &b, "<<", "left shift", "integer", (&INTEGER, INTEGER_CODES))?;
    let (a, b) = match_types_at(a, b, loc.clone())?;
    let dtype = support::prim_dtype(&a)?;
    let shift = |x: &BigInt, y: &BigInt| {
        let value = x.try_shl(y.clone())?;
        Ok(const_fold::is_representable(&value, dtype).then_some(value))
    };
    match fold_shift(&a, &b, shift, &loc)? {
        Some(folded) => Ok(folded),
        None => Ok(LShift::new(a, b, loc)?.into()),
    }
}

/// [`shift_right`] at `loc`.
pub fn shift_right_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    check_operands(&a, &b, ">>", "right shift", "integer", (&INTEGER, INTEGER_CODES))?;
    let (a, b) = match_types_at(a, b, loc.clone())?;
    match fold_shift(&a, &b, |x, y| x.try_shr(y.clone()).map(Some), &loc)? {
        Some(folded) => Ok(folded),
        None => Ok(RShift::new(a, b, loc)?.into()),
    }
}

/// The bitwise operators of integers or booleans, which fold two constants of index types.
macro_rules! bitwise {
    ($($(#[$doc:meta])* $name:ident / $at:ident: $symbol:literal, $what:literal, $op:tt => $node:ident),+ $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(a: impl Operand, b: impl Operand) -> Result<Expr> {
                $at(a, b, UNKNOWN_LOC)
            }

            #[doc = concat!("[`", stringify!($name), "`] at `loc`.")]
            pub fn $at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
                let ((a, b), loc) = (operands(a, b)?, loc.into());
                let accepted = (&INTEGER_OR_BOOL[..], INTEGER_OR_BOOL_CODES);
                check_operands(&a, &b, $symbol, $what, "integer", accepted)?;
                let (a, b) = match_types_at(a, b, loc.clone())?;
                let dtype = support::prim_dtype(&a)?;
                let index = const_fold::is_index_type(dtype)
                    && const_fold::is_index_type(support::prim_dtype(&b)?);
                if let (true, Some(x), Some(y)) =
                    (index, const_fold::int(&a), const_fold::int(&b))
                {
                    return Ok(IntImm::from_dtype(dtype, x $op y, loc)?.into());
                }
                Ok($node::new(a, b, loc)?.into())
            }
        )+
    };
}

bitwise!(
    /// `a & b` (C++ `bitwise_and`, `prim.bitwise_and`).
    bitwise_and / bitwise_and_at: "&", "bitwise AND", & => BitwiseAnd,
    /// `a | b` (C++ `bitwise_or`, `prim.bitwise_or`).
    bitwise_or / bitwise_or_at: "|", "bitwise OR", | => BitwiseOr,
    /// `a ^ b` (C++ `bitwise_xor`, `prim.bitwise_xor`).
    bitwise_xor / bitwise_xor_at: "^", "bitwise XOR", ^ => BitwiseXor,
);

/// [`bitwise_not`] at `loc`.
pub fn bitwise_not_at(a: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let a = a.into_expr()?;
    check_code(&a, "a", &INTEGER_OR_BOOL, INTEGER_OR_BOOL_CODES, || {
        format!(
            "Expected integer or boolean argument for ~ operator (bitwise NOT), but received {} \
             of type {}",
            support::repr(&a),
            support::repr(&a.ty)
        )
    })?;
    Ok(BitwiseNot::new(a, loc)?.into())
}

// ---------------------------------------------------------------------------------------------
// Math functions (`src/ir/prim/math.cc`)

/// `floor`, `ceil`, `round`, `nearbyint` and `trunc`: an integer or a boolean is itself, a float
/// constant is folded with `fold`, keeping its location, and anything else is a call.
fn rounding(x: Expr, op: &'static str, fold: fn(f64) -> f64, loc: Location) -> Result<Expr> {
    let dtype = support::prim_dtype(&x)?;
    if matches!(dtype.code, support::INT | support::UINT | support::BOOL) {
        return Ok(x);
    }
    if let Some(imm) = x.as_node::<FloatImmObj>() {
        return Ok(FloatImm::from_dtype(dtype, fold(imm.value), imm.loc.clone())?.into());
    }
    let ty = x.ty.clone();
    call(Some(&ty), op, vec![x], loc)
}

/// The unary math functions, each in a pair with its `_at` variant, implemented by a function
/// of the expression and the location.
macro_rules! unary {
    ($($(#[$doc:meta])* $name:ident / $at:ident => $imp:expr;)+) => {
        $(
            $(#[$doc])*
            pub fn $name(x: impl IntoExpr) -> Result<Expr> {
                $at(x, UNKNOWN_LOC)
            }

            #[doc = concat!("[`", stringify!($name), "`] at `loc`.")]
            pub fn $at(x: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
                $imp(x.into_expr()?, loc.into())
            }
        )+
    };
}

unary! {
    /// `x` rounded up (C++ `ceil`, `prim.ceil`).
    ceil / ceil_at => |x, loc| rounding(x, "prim.ceil", f64::ceil, loc);
    /// `x` rounded down (C++ `floor`, `prim.floor`).
    floor / floor_at => |x, loc| rounding(x, "prim.floor", f64::floor, loc);
    /// `x` rounded to the nearest integer; a constant folds a tie to even, as `std::nearbyint`
    /// does (C++ `round`, `prim.round`).
    round / round_at => |x, loc| rounding(x, "prim.round", f64::round_ties_even, loc);
    /// `x` rounded to the nearest integer under the current rounding mode, a tie to even
    /// (C++ `nearbyint`, `prim.nearbyint`).
    nearbyint / nearbyint_at => |x, loc| rounding(x, "prim.nearbyint", f64::round_ties_even, loc);
    /// `x` rounded towards zero (C++ `trunc`, `prim.trunc`).
    trunc / trunc_at => |x, loc| rounding(x, "prim.trunc", f64::trunc, loc);
    /// The absolute value of `x`: `select(x >= 0, x, -x)` for a signed integer, a call of
    /// `prim.fabs` for a float, and an unsigned integer itself; a constant folds (C++ `abs`,
    /// `prim.abs`).
    abs / abs_at => absolute;
    /// Whether `x` is NaN, a call typed by its `FInferType` (C++ `isnan`, `prim.isnan`).
    isnan / isnan_at => |x, loc| call(None, "prim.isnan", vec![x], loc);
    /// Whether `x` is infinite: false for an integer, `abs(x) == inf && !isnan(x)` for a float
    /// (C++ `isinf`, `prim.isinf`).
    isinf / isinf_at => infinite;
    /// Whether `x` is finite: `!isinf(x) && !isnan(x)` (C++ `isfinite`, `prim.isfinite`).
    isfinite / isfinite_at => finite;
    /// `cond`, marked as likely to hold; an integer constant is itself (C++ `likely`,
    /// `prim.likely`).
    likely / likely_at => likely_hint;
    /// A statement that `cond` holds, for simplifications (C++ `assume`, `prim.assume`).
    assume / assume_at => |cond, loc| {
        let ty: Type = PrimType::bool().into();
        call(Some(&ty), "prim.assume", vec![cond], loc)
    };
    /// The number of leading zero bits of `x`, a call typed as `x` (C++ `clz`, `prim.clz`).
    clz / clz_at => |x: Expr, loc| {
        let ty = x.ty.clone();
        call(Some(&ty), "prim.clz", vec![x], loc)
    };
}

fn absolute(x: Expr, loc: Location) -> Result<Expr> {
    let dtype = support::prim_dtype(&x)?;
    match dtype.code {
        support::INT => integer_abs(x, loc),
        support::FLOAT | support::BFLOAT => match x.as_node::<FloatImmObj>() {
            Some(imm) => Ok(FloatImm::from_dtype(dtype, imm.value.abs(), imm.loc.clone())?.into()),
            None => {
                let ty = x.ty.clone();
                call(Some(&ty), "prim.fabs", vec![x], loc)
            }
        },
        support::UINT => Ok(x),
        _ => {
            let message = format!(
                "Data type {} not supported for absolute op. Skipping absolute op...",
                support::repr(&x.ty)
            );
            Err(Error::new(INTERNAL_ERROR, &message, ""))
        }
    }
}

/// C++ `IntegerAbs(x, loc)` of a signed integer: a constant folds when its absolute value is a
/// value of its type; anything else is `select(x >= 0, x, -x)`.
fn integer_abs(x: Expr, loc: Location) -> Result<Expr> {
    let dtype = support::prim_dtype(&x)?;
    if let Some(imm) = x.as_node::<IntImmObj>() {
        let value = match imm.value.is_negative() {
            true => -imm.value.clone(),
            false => imm.value.clone(),
        };
        if const_fold::is_representable(&value, dtype) {
            return Ok(IntImm::from_dtype(dtype, value, imm.loc.clone())?.into());
        }
    }
    let zero = make_const(PrimType::from_dtype(dtype)?, 0)?;
    let nonnegative = ge(x.clone(), zero)?;
    let negated = neg(x.clone())?;
    Ok(Select::new(nonnegative, x, negated, loc)?.into())
}

fn infinite(x: Expr, loc: Location) -> Result<Expr> {
    let dtype = support::prim_dtype(&x)?;
    let result = PrimType::bool().with_lanes(support::lanes(dtype)? as u16)?;
    match dtype.code {
        support::INT | support::UINT => make_const_at(result, false, loc),
        support::FLOAT => {
            let infinity = infinity_at(support::prim_type(&x)?, loc.clone())?;
            let at_infinity = eq(abs_at(x.clone(), loc.clone())?, infinity)?;
            let not_nan = logical_not(isnan_at(x, loc)?)?;
            logical_and(at_infinity, not_nan)
        }
        _ => {
            let message = format!(
                "Data type {} not supported for finiteness ops. Skipping it...",
                support::repr(&x.ty)
            );
            Err(Error::new(INTERNAL_ERROR, &message, ""))
        }
    }
}

fn finite(x: Expr, loc: Location) -> Result<Expr> {
    let not_infinite = logical_not(isinf_at(x.clone(), loc.clone())?)?;
    let not_nan = logical_not(isnan_at(x, loc)?)?;
    logical_and(not_infinite, not_nan)
}

fn likely_hint(cond: Expr, loc: Location) -> Result<Expr> {
    if cond.as_node::<IntImmObj>().is_some() {
        return Ok(cond);
    }
    let ty = cond.ty.clone();
    call(Some(&ty), "prim.likely", vec![cond], loc)
}

/// The check `TVM_FFI_ICHECK_EQ(dtype.lanes(), 1)` of the limits.
fn check_scalar(dtype: DLDataType) -> Result<()> {
    let lanes = support::lanes(dtype)?;
    match lanes == 1 {
        true => Ok(()),
        false => Err(support::compare_failed(INTERNAL_ERROR, "dtype.lanes() == 1", lanes, 1, "")),
    }
}

unlocated! {
    /// The largest value of the scalar type `ty` (C++ `max_value`, `prim.max_value`).
    max_value(ty: impl Into<PrimType>) => max_value_at;
    /// The smallest value of the scalar type `ty` (C++ `min_value`, `prim.min_value`).
    min_value(ty: impl Into<PrimType>) => min_value_at;
    /// The positive infinity of the scalar float type `ty` (C++ `infinity`, `prim.infinity`).
    infinity(ty: impl Into<PrimType>) => infinity_at;
    /// The call of `prim.vscale`, the number of lanes a scalable vector has per `vscale`, an
    /// `int32` (C++ `Call(PrimType::Int(32), vscale_op(), {})`).
    vscale() => vscale_at;
}

/// [`max_value`] at `loc`.
pub fn max_value_at(ty: impl Into<PrimType>, loc: impl Into<Location>) -> Result<Expr> {
    let (ty, loc) = (ty.into(), loc.into());
    let dtype = ty.dtype;
    check_scalar(dtype)?;
    let (code, bits) = (dtype.code, dtype.bits);
    let int = |value: BigInt| Ok(IntImm::from_dtype(dtype, value, loc.clone())?.into());
    let float = |value: f64| Ok(FloatImm::from_dtype(dtype, value, loc.clone())?.into());
    match code {
        support::INT if bits == 64 => int(BigInt::from(i64::MAX)),
        support::INT if bits < 64 => int(BigInt::from((1i64 << (bits - 1)) - 1)),
        support::UINT if bits == 64 => int(BigInt::from(u64::MAX)),
        support::UINT if bits < 64 => int(BigInt::from((1u64 << bits) - 1)),
        support::FLOAT if bits == 64 => float(f64::MAX),
        support::FLOAT if bits == 32 => float(f64::from(f32::MAX)),
        support::FLOAT if bits == 16 => float(support::MAX_FLOAT16),
        support::BFLOAT => float(support::MAX_BFLOAT16),
        _ if is_float4(code) => float(6.0),
        _ => match small_float_bound(code) {
            Some((max, _)) => float(max),
            None => {
                let message = format!("Cannot decide max_value for type{}", support::repr(&ty));
                Err(Error::new(INTERNAL_ERROR, &message, ""))
            }
        },
    }
}

/// [`min_value`] at `loc`.
pub fn min_value_at(ty: impl Into<PrimType>, loc: impl Into<Location>) -> Result<Expr> {
    let (ty, loc) = (ty.into(), loc.into());
    let dtype = ty.dtype;
    check_scalar(dtype)?;
    let (code, bits) = (dtype.code, dtype.bits);
    let int = |value: BigInt| Ok(IntImm::from_dtype(dtype, value, loc.clone())?.into());
    let float = |value: f64| Ok(FloatImm::from_dtype(dtype, value, loc.clone())?.into());
    match code {
        support::INT if bits == 64 => int(BigInt::from(i64::MIN)),
        support::INT if bits < 64 => int(BigInt::from(-(1i64 << (bits - 1)))),
        support::UINT => int(BigInt::from(0i64)),
        support::FLOAT if bits == 64 => float(f64::MIN),
        support::FLOAT if bits == 32 => float(f64::from(f32::MIN)),
        support::FLOAT if bits == 16 => float(-support::MAX_FLOAT16),
        support::BFLOAT => float(-support::MAX_BFLOAT16),
        _ if is_float4(code) => float(-6.0),
        _ => match small_float_bound(code) {
            Some((_, true)) => float(0.0),
            Some((max, false)) => float(-max),
            None => {
                let message = format!("Cannot decide min_value for type{}", support::repr(&ty));
                Err(Error::new(INTERNAL_ERROR, &message, ""))
            }
        },
    }
}

/// [`infinity`] at `loc`.
pub fn infinity_at(ty: impl Into<PrimType>, loc: impl Into<Location>) -> Result<Expr> {
    let (ty, loc) = (ty.into(), loc.into());
    let dtype = ty.dtype;
    check_scalar(dtype)?;
    match (dtype.code, dtype.bits) {
        (support::FLOAT, 64) => Ok(FloatImm::from_dtype(dtype, f64::INFINITY, loc)?.into()),
        (support::FLOAT, 32 | 16) => {
            Ok(FloatImm::from_dtype(dtype, f64::from(f32::INFINITY), loc)?.into())
        }
        _ => {
            let message = format!("Cannot decide infinity for type {}", support::repr(&ty));
            Err(Error::new(INTERNAL_ERROR, &message, ""))
        }
    }
}

/// [`vscale`] at `loc`.
pub fn vscale_at(loc: impl Into<Location>) -> Result<Expr> {
    let ty: Type = PrimType::int(32).into();
    call(Some(&ty), "prim.vscale", vec![], loc.into())
}

/// A function of two matched floats, called with the type of the first (`pow` and `fmod`).
fn float_binary(x: Expr, y: Expr, op: &'static str, what: &str, loc: Location) -> Result<Expr> {
    let (x, y) = match_types_at(x, y, loc.clone())?;
    if support::prim_dtype(&x)?.code != support::FLOAT {
        let condition = "x.ty().MatchesCode(DLDataTypeCode::kDLFloat)";
        let message = format!("{what} only applies to float");
        return Err(support::check_failed(INTERNAL_ERROR, condition, &message));
    }
    let ty = x.ty.clone();
    call(Some(&ty), op, vec![x, y], loc)
}

unlocated! {
    /// `x` to the power `y`, of two matched floats (C++ `pow`, `prim._OpPow`). C++ also logs a
    /// warning for a constant exponent of 3 or more, which is not repeated here.
    pow(x: impl IntoExpr, y: impl IntoExpr) => pow_at;
    /// The floating-point remainder of `x / y`, of two matched floats (C++ `fmod`).
    fmod(x: impl IntoExpr, y: impl IntoExpr) => fmod_at;
    /// `x * y + z` in one rounding, a call typed as `x` (C++ `fma`, `prim.fma`).
    fma(x: impl IntoExpr, y: impl IntoExpr, z: impl IntoExpr) => fma_at;
}

/// [`pow`] at `loc`.
pub fn pow_at(x: impl IntoExpr, y: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    float_binary(x.into_expr()?, y.into_expr()?, "prim.pow", "power", loc.into())
}

/// [`fmod`] at `loc`.
pub fn fmod_at(x: impl IntoExpr, y: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    float_binary(x.into_expr()?, y.into_expr()?, "prim.fmod", "fmod", loc.into())
}

/// [`fma`] at `loc`.
pub fn fma_at(
    x: impl IntoExpr,
    y: impl IntoExpr,
    z: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let x = x.into_expr()?;
    let ty = x.ty.clone();
    call(Some(&ty), "prim.fma", vec![x, y.into_expr()?, z.into_expr()?], loc.into())
}

// ---------------------------------------------------------------------------------------------
// Intrinsics (`TVM_DECLARE_INTRIN_UNARY` and friends in `include/tvm/ir/prim/op.h`)

/// A unary intrinsic `op`, a call typed as its argument. A bfloat16 argument is computed in
/// float32 and cast back; with `float_only`, any argument but a float or a bfloat16 is a
/// `TypeError` (C++ `CheckMathUnaryOpInputDType`).
fn unary_intrinsic(op: &'static str, x: Expr, float_only: bool, loc: Location) -> Result<Expr> {
    let dtype = support::prim_dtype(&x)?;
    if float_only && dtype.code != support::FLOAT && !is_bfloat16(dtype) {
        let condition = "dtype.code() == DLDataTypeCode::kDLFloat || \
                         dtype.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)";
        let message =
            format!("{op} only supports floating-point inputs, but got {}", support::repr(&x.ty));
        return Err(support::check_failed(TYPE_ERROR, condition, &message));
    }
    if !is_bfloat16(dtype) {
        let ty = x.ty.clone();
        return call(Some(&ty), op, vec![x], loc);
    }
    let float32 = match support::is_scalable_vector(dtype) {
        true => support::scalable_vector(PrimType::float(32).dtype, vscale_factor(dtype))?,
        false => with_lanes(PrimType::float(32).dtype, dtype.lanes),
    };
    let float32 = PrimType::from_dtype(float32)?;
    let x_float32: Expr = Cast::new(float32.clone(), x, loc.clone())?.into();
    let result = call(Some(&float32.into()), op, vec![x_float32], loc.clone())?;
    Ok(Cast::new(PrimType::from_dtype(dtype)?, result, loc)?.into())
}

/// The unary intrinsics; `float_only` marks `TVM_DECLARE_FLOAT_INTRIN_UNARY`.
macro_rules! unary_intrinsics {
    ($($(#[$doc:meta])* $name:ident / $at:ident => $op:literal, $float_only:literal;)+) => {
        unary! {
            $(
                $(#[$doc])*
                $name / $at => |x, loc| unary_intrinsic($op, x, $float_only, loc);
            )+
        }
    };
}

unary_intrinsics! {
    /// `e` to the power `x` (C++ `exp`, `prim.exp`).
    exp / exp_at => "prim.exp", false;
    /// `2` to the power `x` (C++ `exp2`, `prim.exp2`).
    exp2 / exp2_at => "prim.exp2", false;
    /// `10` to the power `x` (C++ `exp10`, `prim.exp10`).
    exp10 / exp10_at => "prim.exp10", false;
    /// The error function of `x` (C++ `erf`, `prim.erf`).
    erf / erf_at => "prim.erf", false;
    /// The sigmoid of `x` (C++ `sigmoid`, `prim.sigmoid`).
    sigmoid / sigmoid_at => "prim.sigmoid", false;
    /// The square root of `x` (C++ `sqrt`, `prim.sqrt`).
    sqrt / sqrt_at => "prim.sqrt", false;
    /// The reciprocal square root of `x` (C++ `rsqrt`, `prim.rsqrt`).
    rsqrt / rsqrt_at => "prim.rsqrt", false;
    /// The natural logarithm of `x` (C++ `log`, `prim.log`).
    log / log_at => "prim.log", false;
    /// The base-2 logarithm of `x` (C++ `log2`, `prim.log2`).
    log2 / log2_at => "prim.log2", false;
    /// The base-10 logarithm of `x` (C++ `log10`, `prim.log10`).
    log10 / log10_at => "prim.log10", false;
    /// `log(1 + x)` (C++ `log1p`, `prim.log1p`).
    log1p / log1p_at => "prim.log1p", false;
    /// The number of set bits of `x` (C++ `popcount`, `prim.popcount`).
    popcount / popcount_at => "prim.popcount", false;
    /// The hyperbolic tangent of the float `x` (C++ `tanh`, `prim.tanh`).
    tanh / tanh_at => "prim.tanh", true;
    /// The tangent of the float `x` (C++ `tan`, `prim.tan`).
    tan / tan_at => "prim.tan", true;
    /// The cosine of the float `x` (C++ `cos`, `prim.cos`).
    cos / cos_at => "prim.cos", true;
    /// The hyperbolic cosine of the float `x` (C++ `cosh`, `prim.cosh`).
    cosh / cosh_at => "prim.cosh", true;
    /// The sine of the float `x` (C++ `sin`, `prim.sin`).
    sin / sin_at => "prim.sin", true;
    /// The hyperbolic sine of the float `x` (C++ `sinh`, `prim.sinh`).
    sinh / sinh_at => "prim.sinh", true;
    /// The arc sine of the float `x` (C++ `asin`, `prim.asin`).
    asin / asin_at => "prim.asin", true;
    /// The arc cosine of the float `x` (C++ `acos`, `prim.acos`).
    acos / acos_at => "prim.acos", true;
    /// The arc tangent of the float `x` (C++ `atan`, `prim.atan`).
    atan / atan_at => "prim.atan", true;
    /// The inverse hyperbolic cosine of the float `x` (C++ `acosh`, `prim.acosh`).
    acosh / acosh_at => "prim.acosh", true;
    /// The inverse hyperbolic sine of the float `x` (C++ `asinh`, `prim.asinh`).
    asinh / asinh_at => "prim.asinh", true;
    /// The inverse hyperbolic tangent of the float `x` (C++ `atanh`, `prim.atanh`).
    atanh / atanh_at => "prim.atanh", true;
}

/// The binary intrinsics, calls typed as their first argument, whose arguments are not matched
/// (`TVM_DECLARE_INTRIN_BINARY`).
macro_rules! binary_intrinsics {
    ($($(#[$doc:meta])* $name:ident / $at:ident => $op:literal;)+) => {
        $(
            $(#[$doc])*
            pub fn $name(x: impl IntoExpr, y: impl IntoExpr) -> Result<Expr> {
                $at(x, y, UNKNOWN_LOC)
            }

            #[doc = concat!("[`", stringify!($name), "`] at `loc`.")]
            pub fn $at(
                x: impl IntoExpr,
                y: impl IntoExpr,
                loc: impl Into<Location>,
            ) -> Result<Expr> {
                let x = x.into_expr()?;
                let ty = x.ty.clone();
                call(Some(&ty), $op, vec![x, y.into_expr()?], loc.into())
            }
        )+
    };
}

binary_intrinsics! {
    /// The arc tangent of `x / y`, by the signs of both (C++ `atan2`, `prim.atan2`).
    atan2 / atan2_at => "prim.atan2";
    /// The next representable value after `x` towards `y` (C++ `nextafter`, `prim.nextafter`).
    nextafter / nextafter_at => "prim.nextafter";
    /// `x` with the sign of `y` (C++ `copysign`, `prim.copysign`).
    copysign / copysign_at => "prim.copysign";
    /// `sqrt(x * x + y * y)` (C++ `hypot`, `prim.hypot`).
    hypot / hypot_at => "prim.hypot";
    /// `x * 2 ^ y` (C++ `ldexp`, `prim.ldexp`).
    ldexp / ldexp_at => "prim.ldexp";
}

// ---------------------------------------------------------------------------------------------
// `Ex`: expressions under construction

/// An expression being built by operators: either the expression, or the first error any step
/// raised. Later steps pass an error through untouched. An `Ex` is an [`IntoExpr`], so it passes
/// into any function or constructor that takes one, whose `?` then reports the error;
/// [`Ex::expr`] takes the expression out.
///
/// Any expression node converts into an `Ex`, as it converts into a C++ `PrimExpr`. The operators
/// `+`, `-`, `*`, `&`, `|`, `^`, `<<`, `>>`, unary `-` and `!` are the C++ ones, typed after C++:
/// an integer literal on either side is `MakeConst(other.ty(), v)`, an `f32` a float32 constant,
/// an `f64` a float64 one. `/` and `%` are not defined: name the division
/// ([`PrimOps::truncdiv`], [`PrimOps::floordiv`], [`PrimOps::div`], ...), as C++ requires of `%`
/// and of `/` next to a literal (`DivAmbiguityError`); C++ `/` between two expressions is `div`.
/// Comparisons, `&&` and `||` are [`PrimOps`] methods too, because Rust's comparison traits
/// return `bool`.
#[derive(Clone)]
#[must_use = "an `Ex` holds the result; pass it on, or take it with `.expr()`"]
pub struct Ex(Result<Expr>);

impl Ex {
    /// The built expression, or the first error.
    pub fn expr(self) -> Result<Expr> {
        self.0
    }

    /// The dtype a literal next to this expression takes (`MakeConst(this.ty(), v)`).
    fn dtype(&self) -> Result<DLDataType> {
        match &self.0 {
            Ok(expr) => support::prim_dtype(expr),
            Err(error) => Err(error.clone()),
        }
    }
}

impl IntoExpr for Ex {
    fn into_expr(self) -> Result<Expr> {
        self.0
    }
}

impl IntoExpr for &Ex {
    fn into_expr(self) -> Result<Expr> {
        self.0.clone()
    }
}

/// `value` as an [`Ex`], to start an expression from something that is not one yet.
pub fn ex(value: impl Into<Ex>) -> Ex {
    value.into()
}

impl<T: Into<Expr>> From<Result<T>> for Ex {
    fn from(value: Result<T>) -> Self {
        Ex(value.map(Into::into))
    }
}

impl From<&Ex> for Ex {
    fn from(value: &Ex) -> Self {
        value.clone()
    }
}

macro_rules! ex_from_nodes {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for Ex {
                fn from(value: $ty) -> Self {
                    Ex(Ok(value.into()))
                }
            }
            impl From<&$ty> for Ex {
                fn from(value: &$ty) -> Self {
                    Ex(Ok(value.clone().into()))
                }
            }
        )+
    };
}

ex_from_nodes!(
    Expr, Var, IntImm, FloatImm, Call, TensorLoad, Add, And, BitwiseAnd, BitwiseNot, BitwiseOr,
    BitwiseXor, Broadcast, Cast, Div, EQ, FloorDiv, FloorMod, GE, GT, LE, LShift, LT, Let, Max,
    Min, Mod, Mul, NE, Not, Or, RShift, Ramp, Select, Shuffle, Sub,
);

/// `lhs <op> rhs`; a literal `rhs` is typed after `lhs`.
fn binary(lhs: Ex, rhs: impl Operand, op: impl FnOnce(Expr, Expr) -> Result<Expr>) -> Ex {
    let rhs = typed_after(rhs.classify(), &lhs);
    match (lhs.0, rhs.0) {
        (Ok(a), Ok(b)) => Ex(op(a, b)),
        (Err(error), _) | (_, Err(error)) => Ex(Err(error)),
    }
}

/// `lhs <op> rhs` for a literal `lhs`, typed after `rhs`.
fn binary_literal(lhs: impl Operand, rhs: Ex, op: impl FnOnce(Expr, Expr) -> Result<Expr>) -> Ex {
    let lhs = typed_after(lhs.classify(), &rhs);
    binary(lhs, rhs, op)
}

mod operators {
    //! The operator traits of [`Ex`], and of the expressions and literals that start one.

    use std::ops::{
        Add, AddAssign, BitAnd, BitOr, BitXor, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
    };

    use super::{binary, binary_literal, Ex, Expr, Operand, Var};

    macro_rules! binary_operators {
        ($($Trait:ident::$method:ident => $op:ident),+ $(,)?) => {
            $(
                impl<R: Operand> $Trait<R> for Ex {
                    type Output = Ex;
                    fn $method(self, rhs: R) -> Ex {
                        binary(self, rhs, super::$op)
                    }
                }
                binary_operators!(@lhs $Trait::$method => $op; &Ex, Expr, &Expr, Var, &Var);
                binary_operators!(@literal $Trait::$method => $op; i32, i64, f32, f64);
            )+
        };
        (@lhs $Trait:ident::$method:ident => $op:ident; $($lhs:ty),+) => {
            $(
                impl<R: Operand> $Trait<R> for $lhs {
                    type Output = Ex;
                    fn $method(self, rhs: R) -> Ex {
                        binary(self.into(), rhs, super::$op)
                    }
                }
            )+
        };
        (@literal $Trait:ident::$method:ident => $op:ident; $($lit:ty),+) => {
            $(
                binary_operators!(@literal_rhs $Trait::$method => $op; $lit; Ex, &Ex, Expr, &Expr, Var, &Var);
            )+
        };
        (@literal_rhs $Trait:ident::$method:ident => $op:ident; $lit:ty; $($rhs:ty),+) => {
            $(
                impl $Trait<$rhs> for $lit {
                    type Output = Ex;
                    fn $method(self, rhs: $rhs) -> Ex {
                        binary_literal(self, rhs.into(), super::$op)
                    }
                }
            )+
        };
    }

    binary_operators!(
        Add::add => add,
        Sub::sub => sub,
        Mul::mul => mul,
        BitAnd::bitand => bitwise_and,
        BitOr::bitor => bitwise_or,
        BitXor::bitxor => bitwise_xor,
        Shl::shl => shift_left,
        Shr::shr => shift_right,
    );

    macro_rules! assign_operators {
        ($($Trait:ident::$method:ident => $op:ident),+ $(,)?) => {
            $(
                impl<R: Operand> $Trait<R> for Ex {
                    fn $method(&mut self, rhs: R) {
                        *self = binary(self.clone(), rhs, super::$op);
                    }
                }
            )+
        };
    }

    assign_operators!(AddAssign::add_assign => add, SubAssign::sub_assign => sub, MulAssign::mul_assign => mul);

    macro_rules! unary_operators {
        ($($ty:ty),+) => {
            $(
                /// C++ unary `-` (`neg`).
                impl Neg for $ty {
                    type Output = Ex;
                    fn neg(self) -> Ex {
                        Ex(super::neg(Ex::from(self)))
                    }
                }
                /// C++ `!` (`logical_not`).
                impl Not for $ty {
                    type Output = Ex;
                    fn not(self) -> Ex {
                        Ex(super::logical_not(Ex::from(self)))
                    }
                }
            )+
        };
    }

    unary_operators!(Ex, &Ex, Expr, &Expr, Var, &Var);
}

/// [`PrimOps`] methods that are a function of one expression.
macro_rules! unary_methods {
    ($($(#[$doc:meta])* $name:ident;)+) => {
        $(
            $(#[$doc])*
            fn $name(&self) -> Ex {
                Ex($name(self.to_ex()))
            }
        )+
    };
}

/// [`PrimOps`] methods that are a function of two expressions, the second typed after the first
/// when it is a literal.
macro_rules! binary_methods {
    ($($(#[$doc:meta])* $name:ident;)+) => {
        $(
            $(#[$doc])*
            fn $name(&self, rhs: impl Operand) -> Ex {
                binary(self.to_ex(), rhs, $name)
            }
        )+
    };
}

/// The C++ `PrimExpr` operations that are functions rather than Rust operators, as methods of
/// an [`Ex`], an [`Expr`] or a [`Var`]. Each takes an expression or a literal on the right, typed
/// as for the operators, and leaves `self` usable; the comparisons are named after the C++
/// functions (`less`, `equal`), since Rust's `eq` and `lt` return a `bool`.
pub trait PrimOps {
    /// This expression as an [`Ex`] (a reference-count increment).
    fn to_ex(&self) -> Ex;

    /// C++ `a < b` ([`lt`]).
    fn less(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, lt)
    }
    /// C++ `a <= b` ([`le`]).
    fn less_equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, le)
    }
    /// C++ `a > b` ([`gt`]).
    fn greater(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, gt)
    }
    /// C++ `a >= b` ([`ge`]).
    fn greater_equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, ge)
    }
    /// C++ `a == b` ([`eq`]).
    fn equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, eq)
    }
    /// C++ `a != b` ([`ne`]).
    fn not_equal(&self, rhs: impl Operand) -> Ex {
        binary(self.to_ex(), rhs, ne)
    }
    /// C++ `a && b` ([`logical_and`]).
    fn logical_and(&self, rhs: impl IntoExpr) -> Ex {
        Ex(logical_and(self.to_ex(), rhs))
    }
    /// C++ `a || b` ([`logical_or`]).
    fn logical_or(&self, rhs: impl IntoExpr) -> Ex {
        Ex(logical_or(self.to_ex(), rhs))
    }

    binary_methods! {
        /// C++ `div`: `a / b`, truncating for integers.
        div;
        /// C++ `truncdiv`.
        truncdiv;
        /// C++ `truncmod`.
        truncmod;
        /// C++ `floordiv`.
        floordiv;
        /// C++ `floormod`.
        floormod;
        /// C++ `indexdiv`.
        indexdiv;
        /// C++ `indexmod`.
        indexmod;
        /// C++ `ceildiv`.
        ceildiv;
        /// C++ `min`.
        min;
        /// C++ `max`.
        max;
    }

    /// C++ `pow(a, b)` ([`pow`]).
    fn pow(&self, rhs: impl IntoExpr) -> Ex {
        Ex(pow(self.to_ex(), rhs))
    }
    /// C++ `fmod(a, b)` ([`fmod`]).
    fn fmod(&self, rhs: impl IntoExpr) -> Ex {
        Ex(fmod(self.to_ex(), rhs))
    }
    /// C++ `atan2(a, b)` ([`atan2`]).
    fn atan2(&self, rhs: impl IntoExpr) -> Ex {
        Ex(atan2(self.to_ex(), rhs))
    }
    /// C++ `nextafter(a, b)` ([`nextafter`]).
    fn nextafter(&self, rhs: impl IntoExpr) -> Ex {
        Ex(nextafter(self.to_ex(), rhs))
    }
    /// C++ `copysign(a, b)` ([`copysign`]).
    fn copysign(&self, rhs: impl IntoExpr) -> Ex {
        Ex(copysign(self.to_ex(), rhs))
    }
    /// C++ `hypot(a, b)` ([`hypot`]).
    fn hypot(&self, rhs: impl IntoExpr) -> Ex {
        Ex(hypot(self.to_ex(), rhs))
    }
    /// C++ `ldexp(a, b)` ([`ldexp`]).
    fn ldexp(&self, rhs: impl IntoExpr) -> Ex {
        Ex(ldexp(self.to_ex(), rhs))
    }

    unary_methods! {
        /// C++ `~a` ([`bitwise_not`]).
        bitwise_not;
        /// C++ `floor`.
        floor;
        /// C++ `ceil`.
        ceil;
        /// C++ `round`.
        round;
        /// C++ `nearbyint`.
        nearbyint;
        /// C++ `trunc`.
        trunc;
        /// C++ `abs`.
        abs;
        /// C++ `isnan`.
        isnan;
        /// C++ `isinf`.
        isinf;
        /// C++ `isfinite`.
        isfinite;
        /// C++ `likely`.
        likely;
        /// C++ `assume`.
        assume;
        /// C++ `clz`.
        clz;
        /// C++ `exp`.
        exp;
        /// C++ `exp2`.
        exp2;
        /// C++ `exp10`.
        exp10;
        /// C++ `erf`.
        erf;
        /// C++ `sigmoid`.
        sigmoid;
        /// C++ `sqrt`.
        sqrt;
        /// C++ `rsqrt`.
        rsqrt;
        /// C++ `log`.
        log;
        /// C++ `log2`.
        log2;
        /// C++ `log10`.
        log10;
        /// C++ `log1p`.
        log1p;
        /// C++ `popcount`.
        popcount;
        /// C++ `tanh`.
        tanh;
        /// C++ `tan`.
        tan;
        /// C++ `cos`.
        cos;
        /// C++ `cosh`.
        cosh;
        /// C++ `sin`.
        sin;
        /// C++ `sinh`.
        sinh;
        /// C++ `asin`.
        asin;
        /// C++ `acos`.
        acos;
        /// C++ `atan`.
        atan;
        /// C++ `acosh`.
        acosh;
        /// C++ `asinh`.
        asinh;
        /// C++ `atanh`.
        atanh;
    }

    /// C++ `cast(a, t)` ([`cast`]).
    fn cast(&self, ty: impl Into<PrimType>) -> Ex {
        Ex(cast(self.to_ex(), ty))
    }

    /// C++ `if_then_else(self, t, f)` ([`if_then_else`]), with `self` the condition.
    fn if_then_else(&self, t: impl IntoExpr, f: impl IntoExpr) -> Ex {
        Ex(if_then_else(self.to_ex(), t, f))
    }

    /// The raw `Select(self, t, f)` ([`Select::new`]), with `self` the condition.
    fn select(&self, t: impl IntoExpr, f: impl IntoExpr) -> Ex {
        Ex(Select::new(self.to_ex(), t, f, UNKNOWN_LOC).map(Into::into))
    }
}

impl PrimOps for Ex {
    fn to_ex(&self) -> Ex {
        self.clone()
    }
}

impl PrimOps for Expr {
    fn to_ex(&self) -> Ex {
        self.into()
    }
}

impl PrimOps for Var {
    fn to_ex(&self) -> Ex {
        self.into()
    }
}
