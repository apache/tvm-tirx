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

//! The primitive operators: arithmetic, comparisons, logic, casts and math functions, after C++
//! `src/ir/prim/op.cc`.
//!
//! Unlike the node constructors such as [`Add::new`](super::Add::new), these match their operand
//! types and fold constants, as C++ does. Each builds at the unknown location, and its `_at`
//! variant takes one. Operators such as `&x + 1` build an [`ExprResult`], and [`PrimOps`] has the
//! operations Rust has no operator for.
//!
//! Every function returns a `TypeError` for an operand that is not a primitive expression, and the
//! trigonometric and hyperbolic ones also for an operand that is not a float.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::prim::op::{add, floordiv, floormod, mul};
//! use tvm_tirx::ir::{PrimType, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//! use tvm_tirx::tvm_ffi::ObjectRefCore;
//!
//! let i = Var::new("i", PrimType::int(32), UNKNOWN_LOC);
//! assert_eq!(script(&floordiv(&i, 16)?)?, "i // 16");
//! assert_eq!(script(&floormod(&i, 16)?)?, "i % 16");
//!
//! assert_eq!(script(&mul(4, 16)?)?, "64");
//! assert!(add(&i, 0)?.same_as(&i));
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

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
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
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
use crate::ir::expr::PrimExpr;
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
// `ExprResult`: expressions under construction

/// An expression being built by operators, or the first error any step raised.
///
/// It is an [`IntoExpr`], so it passes into any function that takes one, whose `?` reports the
/// error; `PrimExpr::try_from` takes the expression out. `/` and `%` are not defined: name the
/// division, such as [`PrimOps::floordiv`].
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::{ExprResult, PrimOps};
/// use tvm_tirx::ir::prim::Ramp;
/// use tvm_tirx::ir::{PrimExpr, PrimType, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
///
/// let i = Var::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let j = Var::new("j", PrimType::int(32), UNKNOWN_LOC);
/// let offset: ExprResult = &i * 16 + &j;
/// let lanes = Ramp::new(offset, 1, 4, UNKNOWN_LOC)?;
/// assert_eq!(script(&lanes)?, "T.Ramp(i * 16 + j, 1, 4)");
///
/// let f = Var::new("f", PrimType::float(32), UNKNOWN_LOC);
/// assert!(PrimExpr::try_from(f.truncdiv(2) + 1).is_err());
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
#[derive(Clone)]
#[must_use = "an `ExprResult` holds the result; pass it on, or convert it with `PrimExpr::try_from`"]
pub struct ExprResult(Result<PrimExpr>);

impl TryFrom<ExprResult> for Expr {
    type Error = Error;

    /// Take out the built expression, or the first error raised while building it.
    fn try_from(value: ExprResult) -> Result<Self> {
        value.0.map(Into::into)
    }
}

impl TryFrom<ExprResult> for PrimExpr {
    type Error = Error;

    /// Take out the built expression, which every step checks is a `PrimExpr`, or the first error
    /// raised while building it.
    fn try_from(value: ExprResult) -> Result<Self> {
        value.0
    }
}

impl ExprResult {
    /// The built expression, or the first error the chain raised.
    ///
    /// This is [`IntoExpr::into_expr`] with the `Expr` target, so a direct call needs no
    /// annotation; take `IntoExpr<PrimExpr>` for the `PrimExpr`.
    ///
    /// # Returns
    ///
    /// The expression.
    ///
    /// # Errors
    ///
    /// Returns the first error an operator in the chain raised.
    pub fn into_expr(self) -> Result<Expr> {
        self.0.map(Into::into)
    }

    /// The dtype a literal next to this expression takes (`MakeConst(this.ty(), v)`).
    fn dtype(&self) -> Result<DLDataType> {
        match &self.0 {
            Ok(expr) => Ok(expr.dtype()),
            Err(error) => Err(error.clone()),
        }
    }
}

impl IntoExpr for ExprResult {
    type Output = Expr;

    fn into_expr(self) -> Result<Expr> {
        self.0.map(Into::into)
    }
}

impl<'a> IntoExpr for &'a ExprResult {
    type Output = &'a Expr;

    fn into_expr(self) -> Result<&'a Expr> {
        match &self.0 {
            Ok(expr) => Ok(expr.as_ref()),
            Err(error) => Err(error.clone()),
        }
    }
}

impl IntoExpr<PrimExpr> for ExprResult {
    type Output = PrimExpr;

    fn into_expr(self) -> Result<PrimExpr> {
        self.0
    }
}

impl<'a> IntoExpr<PrimExpr> for &'a ExprResult {
    type Output = &'a PrimExpr;

    fn into_expr(self) -> Result<&'a PrimExpr> {
        match &self.0 {
            Ok(prim) => Ok(prim),
            Err(error) => Err(error.clone()),
        }
    }
}

impl<T: Into<Expr>> From<Result<T>> for ExprResult {
    fn from(value: Result<T>) -> Self {
        ExprResult::checked(value.map(Into::into))
    }
}

impl ExprResult {
    /// A step's result, checked to be a `PrimExpr`; an expression of another type is a TypeError,
    /// raised where it enters the chain.
    fn checked(value: Result<Expr>) -> Self {
        ExprResult(value.and_then(PrimExpr::try_from))
    }
}

impl From<&ExprResult> for ExprResult {
    fn from(value: &ExprResult) -> Self {
        value.clone()
    }
}

/// `lhs <op> rhs`; a literal `rhs` is typed after `lhs`.
fn binary(
    lhs: ExprResult,
    rhs: impl Operand,
    op: impl FnOnce(Expr, Expr) -> Result<Expr>,
) -> ExprResult {
    let rhs = typed_after(rhs.classify(), &lhs);
    match (lhs.0, rhs.0) {
        (Ok(a), Ok(b)) => ExprResult::checked(op(a.into(), b.into())),
        (Err(error), _) | (_, Err(error)) => ExprResult(Err(error)),
    }
}

/// `lhs <op> rhs` for a literal `lhs`, typed after `rhs`.
fn binary_literal(
    lhs: impl Operand,
    rhs: ExprResult,
    op: impl FnOnce(Expr, Expr) -> Result<Expr>,
) -> ExprResult {
    let lhs = typed_after(lhs.classify(), &rhs);
    binary(lhs, rhs, op)
}

mod operators {
    //! The operator traits of [`ExprResult`], and of the expressions and literals that start one.

    use std::ops::{
        Add, AddAssign, BitAnd, BitOr, BitXor, Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
    };

    use super::{binary, binary_literal, ExprResult, Operand};

    macro_rules! binary_operators {
        ($($Trait:ident::$method:ident => $op:ident),+ $(,)?) => {
            $(
                impl<R: Operand> $Trait<R> for ExprResult {
                    type Output = ExprResult;
                    fn $method(self, rhs: R) -> ExprResult {
                        binary(self, rhs, super::$op)
                    }
                }
                binary_operators!(@lhs $Trait::$method => $op; &ExprResult);
                binary_operators!(@literal $Trait::$method => $op; i32, i64, f32, f64);
            )+
        };
        (@lhs $Trait:ident::$method:ident => $op:ident; $($lhs:ty),+) => {
            $(
                impl<R: Operand> $Trait<R> for $lhs {
                    type Output = ExprResult;
                    fn $method(self, rhs: R) -> ExprResult {
                        binary(self.into(), rhs, super::$op)
                    }
                }
            )+
        };
        (@literal $Trait:ident::$method:ident => $op:ident; $($lit:ty),+) => {
            $(
                binary_operators!(@literal_rhs $Trait::$method => $op; $lit; ExprResult, &ExprResult);
            )+
        };
        (@literal_rhs $Trait:ident::$method:ident => $op:ident; $lit:ty; $($rhs:ty),+) => {
            $(
                impl $Trait<$rhs> for $lit {
                    type Output = ExprResult;
                    fn $method(self, rhs: $rhs) -> ExprResult {
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
                impl<R: Operand> $Trait<R> for ExprResult {
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
                    type Output = ExprResult;
                    fn neg(self) -> ExprResult {
                        ExprResult::checked(super::neg(ExprResult::from(self)))
                    }
                }
                /// C++ `!` (`logical_not`).
                impl Not for $ty {
                    type Output = ExprResult;
                    fn not(self) -> ExprResult {
                        ExprResult::checked(super::logical_not(ExprResult::from(self)))
                    }
                }
            )+
        };
    }

    unary_operators!(ExprResult, &ExprResult);
}

// ---------------------------------------------------------------------------------------------
// `PrimOps`: the operations Rust has no operator for

/// [`PrimOps`] methods that are a function of one expression.
macro_rules! unary_methods {
    ($($(#[$doc:meta])* $name:ident;)+) => {
        $(
            $(#[$doc])*
            fn $name(&self) -> ExprResult {
                ExprResult::checked($name(self.to_expr_result()))
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
            fn $name(&self, rhs: impl Operand) -> ExprResult {
                binary(self.to_expr_result(), rhs, $name)
            }
        )+
    };
}

/// The operations Rust has no operator for, as methods of [`ExprResult`] and every expression
/// handle.
///
/// The comparisons are named after the C++ functions (`less`, `equal`), since Rust's `eq` and `lt`
/// return a `bool`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::PrimOps;
/// use tvm_tirx::ir::{PrimExpr, PrimType, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
///
/// let i = Var::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let n = Var::new("n", PrimType::int(32), UNKNOWN_LOC);
/// let in_bounds = PrimExpr::try_from(i.greater_equal(0).logical_and(i.less(&n)))?;
/// assert_eq!(script(&in_bounds)?, "i >= 0 and i < n");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub trait PrimOps {
    /// Return this expression as an [`ExprResult`].
    fn to_expr_result(&self) -> ExprResult;

    /// Build the boolean `self < rhs` ([`lt`]).
    fn less(&self, rhs: impl Operand) -> ExprResult {
        binary(self.to_expr_result(), rhs, lt)
    }
    /// Build the boolean `self <= rhs` ([`le`]).
    fn less_equal(&self, rhs: impl Operand) -> ExprResult {
        binary(self.to_expr_result(), rhs, le)
    }
    /// Build the boolean `self > rhs` ([`gt`]).
    fn greater(&self, rhs: impl Operand) -> ExprResult {
        binary(self.to_expr_result(), rhs, gt)
    }
    /// Build the boolean `self >= rhs` ([`ge`]).
    fn greater_equal(&self, rhs: impl Operand) -> ExprResult {
        binary(self.to_expr_result(), rhs, ge)
    }
    /// Build the boolean `self == rhs` ([`eq`]).
    fn equal(&self, rhs: impl Operand) -> ExprResult {
        binary(self.to_expr_result(), rhs, eq)
    }
    /// Build the boolean `self != rhs` ([`ne`]).
    fn not_equal(&self, rhs: impl Operand) -> ExprResult {
        binary(self.to_expr_result(), rhs, ne)
    }
    /// Build the boolean `self && rhs` ([`logical_and`]).
    fn logical_and(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(logical_and(self.to_expr_result(), rhs))
    }
    /// Build the boolean `self || rhs` ([`logical_or`]).
    fn logical_or(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(logical_or(self.to_expr_result(), rhs))
    }

    binary_methods! {
        /// Build [`div`]`(self, rhs)`.
        div;
        /// Build [`truncdiv`]`(self, rhs)`.
        truncdiv;
        /// Build [`truncmod`]`(self, rhs)`.
        truncmod;
        /// Build [`floordiv`]`(self, rhs)`.
        floordiv;
        /// Build [`floormod`]`(self, rhs)`.
        floormod;
        /// Build [`indexdiv`]`(self, rhs)`.
        indexdiv;
        /// Build [`indexmod`]`(self, rhs)`.
        indexmod;
        /// Build [`ceildiv`]`(self, rhs)`.
        ceildiv;
        /// Build [`min`]`(self, rhs)`.
        min;
        /// Build [`max`]`(self, rhs)`.
        max;
    }

    /// Build [`pow`]`(self, rhs)`.
    fn pow(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(pow(self.to_expr_result(), rhs))
    }
    /// Build [`fmod`]`(self, rhs)`.
    fn fmod(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(fmod(self.to_expr_result(), rhs))
    }
    /// Build [`atan2`]`(self, rhs)`.
    fn atan2(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(atan2(self.to_expr_result(), rhs))
    }
    /// Build [`nextafter`]`(self, rhs)`.
    fn nextafter(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(nextafter(self.to_expr_result(), rhs))
    }
    /// Build [`copysign`]`(self, rhs)`.
    fn copysign(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(copysign(self.to_expr_result(), rhs))
    }
    /// Build [`hypot`]`(self, rhs)`.
    fn hypot(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(hypot(self.to_expr_result(), rhs))
    }
    /// Build [`ldexp`]`(self, rhs)`.
    fn ldexp(&self, rhs: impl IntoExpr) -> ExprResult {
        ExprResult::checked(ldexp(self.to_expr_result(), rhs))
    }

    unary_methods! {
        /// Build [`bitwise_not`]`(self)`.
        bitwise_not;
        /// Build [`floor`]`(self)`.
        floor;
        /// Build [`ceil`]`(self)`.
        ceil;
        /// Build [`round`]`(self)`.
        round;
        /// Build [`nearbyint`]`(self)`.
        nearbyint;
        /// Build [`trunc`]`(self)`.
        trunc;
        /// Build [`abs`]`(self)`.
        abs;
        /// Build [`isnan`]`(self)`.
        isnan;
        /// Build [`isinf`]`(self)`.
        isinf;
        /// Build [`isfinite`]`(self)`.
        isfinite;
        /// Build [`likely`]`(self)`.
        likely;
        /// Build [`assume`]`(self)`.
        assume;
        /// Build [`clz`]`(self)`.
        clz;
        /// Build [`exp`]`(self)`.
        exp;
        /// Build [`exp2`]`(self)`.
        exp2;
        /// Build [`exp10`]`(self)`.
        exp10;
        /// Build [`erf`]`(self)`.
        erf;
        /// Build [`sigmoid`]`(self)`.
        sigmoid;
        /// Build [`sqrt`]`(self)`.
        sqrt;
        /// Build [`rsqrt`]`(self)`.
        rsqrt;
        /// Build [`log`]`(self)`.
        log;
        /// Build [`log2`]`(self)`.
        log2;
        /// Build [`log10`]`(self)`.
        log10;
        /// Build [`log1p`]`(self)`.
        log1p;
        /// Build [`popcount`]`(self)`.
        popcount;
        /// Build [`tanh`]`(self)`.
        tanh;
        /// Build [`tan`]`(self)`.
        tan;
        /// Build [`cos`]`(self)`.
        cos;
        /// Build [`cosh`]`(self)`.
        cosh;
        /// Build [`sin`]`(self)`.
        sin;
        /// Build [`sinh`]`(self)`.
        sinh;
        /// Build [`asin`]`(self)`.
        asin;
        /// Build [`acos`]`(self)`.
        acos;
        /// Build [`atan`]`(self)`.
        atan;
        /// Build [`acosh`]`(self)`.
        acosh;
        /// Build [`asinh`]`(self)`.
        asinh;
        /// Build [`atanh`]`(self)`.
        atanh;
    }

    /// Build [`cast`]`(self, ty)`.
    fn cast(&self, ty: impl Into<PrimType>) -> ExprResult {
        ExprResult::checked(cast(self.to_expr_result(), ty))
    }

    /// Build [`if_then_else`]`(self, t, f)`, with `self` the condition.
    fn if_then_else(&self, t: impl IntoExpr, f: impl IntoExpr) -> ExprResult {
        ExprResult::checked(if_then_else(self.to_expr_result(), t, f))
    }

    /// Build the raw node `Select(self, t, f)` ([`Select::new`]), with `self` the condition; unlike
    /// [`if_then_else`](Self::if_then_else), both values are evaluated.
    fn select(&self, t: impl IntoExpr, f: impl IntoExpr) -> ExprResult {
        ExprResult::checked(Select::new(self.to_expr_result(), t, f, UNKNOWN_LOC).map(Expr::from))
    }
}

impl PrimOps for ExprResult {
    fn to_expr_result(&self) -> ExprResult {
        self.clone()
    }
}

// ---------------------------------------------------------------------------------------------
// Operands

/// An operand of an operator: an expression, an [`ExprResult`] or a literal.
///
/// A literal is typed after the expression it meets, as in C++: an integer takes its type, an `f32`
/// is a float32 constant and an `f64` a float64 one.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::add;
/// use tvm_tirx::ir::{PrimType, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
///
/// let n = Var::new("n", PrimType::int(64), UNKNOWN_LOC);
/// assert_eq!(script(&add(&n, 1)?)?, "n + T.int64(1)");
/// assert_eq!(script(&add(&n, 0.5)?)?, "T.Cast(\"float64\", n) + T.float64(0.5)");
/// assert_eq!(script(&add(1, 2)?)?, "3");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub trait Operand {
    #[doc(hidden)]
    fn classify(self) -> OperandKind;
}

/// What an [`Operand`] is.
#[doc(hidden)]
pub enum OperandKind {
    Expr(ExprResult),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
}

impl<T: Into<ExprResult>> Operand for T {
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
fn typed_after(operand: OperandKind, peer: &ExprResult) -> ExprResult {
    let constant = |value: i64| {
        ExprResult::checked(
            peer.dtype().and_then(|dtype| make_const(PrimType::from_dtype(dtype)?, value)),
        )
    };
    match operand {
        OperandKind::Expr(expr) => expr,
        OperandKind::Int32(value) => constant(i64::from(value)),
        OperandKind::Int64(value) => constant(value),
        OperandKind::Float32(value) => ExprResult(Ok(FloatImm::float32(value).into())),
        OperandKind::Float64(value) => ExprResult(Ok(FloatImm::float64(value).into())),
    }
}

/// An operand on its own: a literal is the constant [`From`] makes of it.
fn on_its_own(operand: OperandKind) -> ExprResult {
    match operand {
        OperandKind::Expr(expr) => expr,
        OperandKind::Int32(value) => ExprResult(Ok(IntImm::int32(value).into())),
        OperandKind::Int64(value) => ExprResult(Ok(IntImm::int64(value).into())),
        OperandKind::Float32(value) => ExprResult(Ok(FloatImm::float32(value).into())),
        OperandKind::Float64(value) => ExprResult(Ok(FloatImm::float64(value).into())),
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
    Ok((a.into_owned_expr()?, b.into_owned_expr()?))
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
///
/// Rust integers, `bool`, [`BigInt`], `f32` and `f64` convert into one, so [`make_const`] takes
/// them directly.
#[derive(Clone, Debug)]
pub enum ConstValue {
    /// An integer, exactly.
    Int(BigInt),
    /// A floating-point value.
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
    /// Create the constant `value` of the type `ty`, broadcast over the lanes of a vector type (C++
    /// `prim::MakeConst(ty, value)`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `value` is out of the range of `ty`, or an `InternalError` if it
    /// is negative for an unsigned `ty` or `ty` is not a type of constants, such as `void`.
    make_const(ty: impl Into<PrimType>, value: impl Into<ConstValue>) => make_const_at;
}

/// Build [`make_const`] at the location `loc`.
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

/// Convert `value` into an expression: a Rust `bool`, integer or float becomes a constant of the
/// default type, and an expression is returned as is (`prim.convert`).
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
    if let Some(imm) = value.downcast_ref::<IntImmObj>() {
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
    } else if let Some(imm) = value.downcast_ref::<FloatImmObj>() {
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
    /// Convert `value` to the type `ty`, folding a constant that keeps its value and broadcasting a
    /// scalar to a vector type (C++ `prim::cast(value, t)`, `prim._cast`).
    ///
    /// [`Cast::new`] builds the node without folding.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `value` has the void type, or an `InternalError` if `value` and
    /// `ty` are vectors of different lanes.
    cast(value: impl IntoExpr, ty: impl Into<PrimType>) => cast_at;
}

/// Build [`cast`] at the location `loc`.
pub fn cast_at(
    value: impl IntoExpr,
    ty: impl Into<PrimType>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (value, ty, loc) = (value.into_owned_expr()?, ty.into(), loc.into());
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
    if let Some(broadcast) = value.downcast_ref::<BroadcastObj>() {
        let value = cast_at(broadcast.value.clone(), element, loc.clone())?;
        return Ok(Broadcast::new(value, broadcast.lanes.clone(), loc)?.into());
    }
    if let Some(ramp) = value.downcast_ref::<RampObj>() {
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

/// Bring `a` and `b` to one type, as every binary operator does first (C++
/// `BinaryOpMatchTypes(lhs, rhs)`).
///
/// A scalar next to a vector is broadcast, and both are cast to the promoted type: the wider float,
/// a float over an integer, the wider integer, unsigned over signed.
///
/// # Errors
///
/// Returns an `InternalError` if the lanes of the two differ or their types cannot be promoted to
/// one.
pub fn match_types(a: impl IntoExpr, b: impl IntoExpr) -> Result<(Expr, Expr)> {
    match_types_at(a, b, UNKNOWN_LOC)
}

/// Run [`match_types`] with the casts at the location `loc`.
pub fn match_types_at(
    a: impl IntoExpr,
    b: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<(Expr, Expr)> {
    let (a, b, loc) = (a.into_owned_expr()?, b.into_owned_expr()?, loc.into());
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

            #[doc = concat!("Build [`", stringify!($name), "`] at the location `loc`.")]
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
    /// Build `a + b` (C++ `add`, `prim._OpAdd`).
    add / add_at: const_fold::add => Add,
    /// Build `a - b` (C++ `sub`, `prim._OpSub`).
    sub / sub_at: const_fold::sub => Sub,
    /// Build `a * b` (C++ `mul`, `prim._OpMul`).
    mul / mul_at: const_fold::mul => Mul,
    /// Build `a / b`; integer division rounds towards zero, as in C (C++ `div`, `prim._OpDiv`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `b` is a constant zero.
    div / div_at: const_fold::div => Div,
    /// Build the remainder of [`truncdiv`], with the sign of `a` (C++ `truncmod`,
    /// `prim._OpTruncMod`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if a folded divisor is zero.
    truncmod / truncmod_at: const_fold::truncmod => Mod,
    /// Build the boolean `a > b` (C++ `greater`, `prim._OpGT`).
    gt / gt_at: fold_gt => GT,
    /// Build the boolean `a >= b` (C++ `greater_equal`, `prim._OpGE`).
    ge / ge_at: fold_ge => GE,
    /// Build the boolean `a < b` (C++ `less`, `prim._OpLT`).
    lt / lt_at: fold_lt => LT,
    /// Build the boolean `a <= b` (C++ `less_equal`, `prim._OpLE`).
    le / le_at: fold_le => LE,
    /// Build the boolean `a != b` (C++ `not_equal`, `prim._OpNE`).
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
    /// Build the boolean `a == b` (C++ `equal`, `prim._OpEQ`).
    eq(a: impl Operand, b: impl Operand) => eq_at;
    /// Build `-a`, folding a constant (C++ `neg`).
    neg(a: impl IntoExpr) => neg_at;
    /// Build `a / b`, rounding towards zero, of two integers (C++ `truncdiv`, `prim._OpTruncDiv`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer or `b` is a constant zero.
    truncdiv(a: impl Operand, b: impl Operand) => truncdiv_at;
    /// Build `a / b`, rounding down, of two integers (C++ `floordiv`, `prim._OpFloorDiv`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer or `b` is a constant zero.
    floordiv(a: impl Operand, b: impl Operand) => floordiv_at;
    /// Build the remainder of [`floordiv`], with the sign of `b` (C++ `floormod`,
    /// `prim._OpFloorMod`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer or a folded divisor is zero.
    floormod(a: impl Operand, b: impl Operand) => floormod_at;
    /// Build `a / b` over non-negative indices, which is [`floordiv`] (C++ `indexdiv`,
    /// `prim._OpIndexDiv`).
    indexdiv(a: impl Operand, b: impl Operand) => indexdiv_at;
    /// Build `a % b` over non-negative indices, which is [`floormod`] (C++ `indexmod`,
    /// `prim._OpIndexMod`).
    indexmod(a: impl Operand, b: impl Operand) => indexmod_at;
    /// Build `a / b`, rounding up, of two integers: `floordiv(a + b - 1, b)` (C++ `ceildiv`,
    /// `prim._OpCeilDiv`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer or `b` is a constant zero.
    ceildiv(a: impl Operand, b: impl Operand) => ceildiv_at;
    /// Build the smaller of `a` and `b`; a symbolic infinity on either side decides it (C++ `min`,
    /// `prim._OpMin`).
    min(a: impl Operand, b: impl Operand) => min_at;
    /// Build the larger of `a` and `b`; a symbolic infinity on either side decides it (C++ `max`,
    /// `prim._OpMax`).
    max(a: impl Operand, b: impl Operand) => max_at;
}

/// Build [`eq`] at the location `loc`.
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

/// Build [`neg`] at the location `loc`.
pub fn neg_at(a: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let (a, loc) = (a.into_owned_expr()?, loc.into());
    let dtype = support::prim_dtype(&a)?;
    if let Some(imm) = a.downcast_ref::<IntImmObj>() {
        let value = -imm.value.clone();
        if const_fold::is_representable(&value, dtype) {
            return Ok(IntImm::from_dtype(dtype, value, loc)?.into());
        }
    }
    if let Some(imm) = a.downcast_ref::<FloatImmObj>() {
        return Ok(FloatImm::from_dtype(dtype, -imm.value, loc)?.into());
    }
    sub(make_const_at(PrimType::from_dtype(dtype)?, 0, loc)?, a)
}

/// Build [`truncdiv`] at the location `loc`.
pub fn truncdiv_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let (a, b) = operands(a, b)?;
    check_integers(&a, &b)?;
    div_at(a, b, loc)
}

/// Build [`floordiv`] at the location `loc`.
pub fn floordiv_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    check_integers(&a, &b)?;
    let (a, b) = match_types_at(a, b, loc.clone())?;
    match const_fold::floordiv(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(FloorDiv::new(a, b, loc)?.into()),
    }
}

/// Build [`floormod`] at the location `loc`.
pub fn floormod_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    let ((a, b), loc) = (operands(a, b)?, loc.into());
    check_integers(&a, &b)?;
    let (a, b) = match_types_at(a, b, loc.clone())?;
    match const_fold::floormod(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(FloorMod::new(a, b, loc)?.into()),
    }
}

/// Build [`indexdiv`] at the location `loc`.
pub fn indexdiv_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    floordiv_at(a, b, loc)
}

/// Build [`indexmod`] at the location `loc`.
pub fn indexmod_at(a: impl Operand, b: impl Operand, loc: impl Into<Location>) -> Result<Expr> {
    floormod_at(a, b, loc)
}

/// Build [`ceildiv`] at the location `loc`.
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

/// Build [`min`] at the location `loc`.
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

/// Build [`max`] at the location `loc`.
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
    /// Build the boolean `a && b`; a constant side folds (C++ `logical_and`, `prim._OpAnd`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not a boolean.
    logical_and(a: impl IntoExpr, b: impl IntoExpr) => logical_and_at;
    /// Build the boolean `a || b`; a constant side folds (C++ `logical_or`, `prim._OpOr`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not a boolean.
    logical_or(a: impl IntoExpr, b: impl IntoExpr) => logical_or_at;
    /// Build the boolean `!a`; a constant folds (C++ `logical_not`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `a` is not a boolean.
    logical_not(a: impl IntoExpr) => logical_not_at;
    /// Build the conjunction of `conds` (Python `tvm_tirx.ir.prim.op.all`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `conds` is empty.
    all(conds: impl IntoIterator<Item = impl IntoExpr>) => all_at;
    /// Build the disjunction of `conds` (Python `tvm_tirx.ir.prim.op.any`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `conds` is empty.
    any(conds: impl IntoIterator<Item = impl IntoExpr>) => any_at;
    /// Build `true_value` where `cond` holds and `false_value` elsewhere, evaluating only the
    /// chosen one (C++ `if_then_else`, `prim._OpIfThenElse`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `cond` is not a boolean or the values cannot be matched.
    if_then_else(cond: impl IntoExpr, true_value: impl IntoExpr, false_value: impl IntoExpr) => if_then_else_at;
}

/// Build [`logical_and`] at the location `loc`.
pub fn logical_and_at(
    a: impl IntoExpr,
    b: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
    check_operands(&a, &b, "&&", "logical AND", "boolean", (&[support::BOOL], BOOL_CODES))?;
    match const_fold::and(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(And::new(a, b, loc)?.into()),
    }
}

/// Build [`logical_or`] at the location `loc`.
pub fn logical_or_at(a: impl IntoExpr, b: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
    check_operands(&a, &b, "||", "logical OR", "boolean", (&[support::BOOL], BOOL_CODES))?;
    match const_fold::or(&a, &b)? {
        Some(folded) => Ok(folded),
        None => Ok(Or::new(a, b, loc)?.into()),
    }
}

/// Build [`logical_not`] at the location `loc`.
pub fn logical_not_at(a: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let a = a.into_owned_expr()?;
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

/// Build [`all`] at the location `loc`.
pub fn all_at(
    conds: impl IntoIterator<Item = impl IntoExpr>,
    loc: impl Into<Location>,
) -> Result<Expr> {
    fold_conditions(conds, loc.into(), "all", logical_and_at)
}

/// Build [`any`] at the location `loc`.
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
    let mut conds = conds.into_iter().map(IntoExpr::into_owned_expr);
    let Some(first) = conds.next() else {
        let message = format!("{name} must take at least 1 argument");
        return Err(Error::new(VALUE_ERROR, &message, ""));
    };
    conds.try_fold(first?, |acc, cond| op(acc, cond?, loc.clone()))
}

/// Build [`if_then_else`] at the location `loc`.
pub fn if_then_else_at(
    cond: impl IntoExpr,
    true_value: impl IntoExpr,
    false_value: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (cond, loc) = (cond.into_owned_expr()?, loc.into());
    if support::prim_dtype(&cond)?.code != support::BOOL {
        let condition = "cond.ty().MatchesCode(DLDataTypeCode::kDLBool)";
        let message = "if_then_else only accept the condition to be boolean type.";
        return Err(support::check_failed(INTERNAL_ERROR, condition, message));
    }
    let (true_value, false_value) = match_types_at(true_value, false_value, loc.clone())?;
    if let Some(imm) = cond.downcast_ref::<IntImmObj>() {
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
    /// Build `a << b` of two integers (C++ `left_shift`, `prim.left_shift`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer, or a constant shift amount is
    /// negative or not less than the width.
    shift_left(a: impl Operand, b: impl Operand) => shift_left_at;
    /// Build `a >> b` of two integers, sign-extending (C++ `right_shift`, `prim.right_shift`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer, or a constant shift amount is
    /// negative or not less than the width.
    shift_right(a: impl Operand, b: impl Operand) => shift_right_at;
    /// Build the node `~a` of an integer or boolean, which neither promotes nor folds (C++
    /// `bitwise_neg`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `a` is not an integer or boolean.
    bitwise_not(a: impl IntoExpr) => bitwise_not_at;
}

/// Build [`shift_left`] at the location `loc`.
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

/// Build [`shift_right`] at the location `loc`.
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

            #[doc = concat!("Build [`", stringify!($name), "`] at the location `loc`.")]
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
    /// Build `a & b` of two integers or booleans (C++ `bitwise_and`, `prim.bitwise_and`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer or boolean.
    bitwise_and / bitwise_and_at: "&", "bitwise AND", & => BitwiseAnd,
    /// Build `a | b` of two integers or booleans (C++ `bitwise_or`, `prim.bitwise_or`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer or boolean.
    bitwise_or / bitwise_or_at: "|", "bitwise OR", | => BitwiseOr,
    /// Build `a ^ b` of two integers or booleans (C++ `bitwise_xor`, `prim.bitwise_xor`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if an operand is not an integer or boolean.
    bitwise_xor / bitwise_xor_at: "^", "bitwise XOR", ^ => BitwiseXor,
);

/// Build [`bitwise_not`] at the location `loc`.
pub fn bitwise_not_at(a: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    let a = a.into_owned_expr()?;
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
    if let Some(imm) = x.downcast_ref::<FloatImmObj>() {
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

            #[doc = concat!("Build [`", stringify!($name), "`] at the location `loc`.")]
            pub fn $at(x: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
                $imp(x.into_owned_expr()?, loc.into())
            }
        )+
    };
}

unary! {
    /// Round `x` up (C++ `ceil`, `prim.ceil`).
    ceil / ceil_at => |x, loc| rounding(x, "prim.ceil", f64::ceil, loc);
    /// Round `x` down (C++ `floor`, `prim.floor`).
    floor / floor_at => |x, loc| rounding(x, "prim.floor", f64::floor, loc);
    /// Round `x` to the nearest integer; a constant folds a tie to even, as `std::nearbyint` does
    /// (C++ `round`, `prim.round`).
    round / round_at => |x, loc| rounding(x, "prim.round", f64::round_ties_even, loc);
    /// Round `x` to the nearest integer under the current rounding mode, a tie to even (C++
    /// `nearbyint`, `prim.nearbyint`).
    nearbyint / nearbyint_at => |x, loc| rounding(x, "prim.nearbyint", f64::round_ties_even, loc);
    /// Round `x` towards zero (C++ `trunc`, `prim.trunc`).
    trunc / trunc_at => |x, loc| rounding(x, "prim.trunc", f64::trunc, loc);
    /// Build the absolute value of `x`; a constant folds (C++ `abs`, `prim.abs`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `x` is not an integer or a float, such as a boolean.
    abs / abs_at => absolute;
    /// Build whether `x` is NaN (C++ `isnan`, `prim.isnan`).
    isnan / isnan_at => |x, loc| call(None, "prim.isnan", vec![x], loc);
    /// Build whether `x` is infinite (C++ `isinf`, `prim.isinf`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `x` is not an integer or a float.
    isinf / isinf_at => infinite;
    /// Build whether `x` is finite (C++ `isfinite`, `prim.isfinite`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `x` is not an integer or a float.
    isfinite / isfinite_at => finite;
    /// Mark `cond` as likely to hold; an integer constant is itself (C++ `likely`, `prim.likely`).
    likely / likely_at => likely_hint;
    /// Build a statement that `cond` holds, for simplifications (C++ `assume`, `prim.assume`).
    assume / assume_at => |cond, loc| {
        let ty: Type = PrimType::bool().into();
        call(Some(&ty), "prim.assume", vec![cond], loc)
    };
    /// Build the number of leading zero bits of `x` (C++ `clz`, `prim.clz`).
    clz / clz_at => |x: Expr, loc| {
        let ty = x.ty.clone();
        call(Some(&ty), "prim.clz", vec![x], loc)
    };
}

fn absolute(x: Expr, loc: Location) -> Result<Expr> {
    let dtype = support::prim_dtype(&x)?;
    match dtype.code {
        support::INT => integer_abs(x, loc),
        support::FLOAT | support::BFLOAT => match x.downcast_ref::<FloatImmObj>() {
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
    if let Some(imm) = x.downcast_ref::<IntImmObj>() {
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
    if cond.downcast_ref::<IntImmObj>().is_some() {
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
    /// Create the largest value of the scalar type `ty` (C++ `max_value`, `prim.max_value`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `ty` is a vector or has no largest value.
    max_value(ty: impl Into<PrimType>) => max_value_at;
    /// Create the smallest value of the scalar type `ty` (C++ `min_value`, `prim.min_value`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `ty` is a vector or has no smallest value.
    min_value(ty: impl Into<PrimType>) => min_value_at;
    /// Create the positive infinity of the scalar `float16`, `float32` or `float64` type `ty` (C++
    /// `infinity`, `prim.infinity`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `ty` is a vector or not one of those types.
    infinity(ty: impl Into<PrimType>) => infinity_at;
    /// Build the `int32` call `vscale()`, the factor of the lanes of a scalable vector
    /// (`prim.vscale`).
    vscale() => vscale_at;
}

/// Build [`max_value`] at the location `loc`.
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

/// Build [`min_value`] at the location `loc`.
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

/// Build [`infinity`] at the location `loc`.
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

/// Build [`vscale`] at the location `loc`.
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
    /// Build `x` to the power `y`, of two matched floats (C++ `pow`, `prim._OpPow`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if the operands are not floats or cannot be matched.
    pow(x: impl IntoExpr, y: impl IntoExpr) => pow_at;
    /// Build the floating-point remainder of `x / y`, of two matched floats (C++ `fmod`).
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if the operands are not floats or cannot be matched.
    fmod(x: impl IntoExpr, y: impl IntoExpr) => fmod_at;
    /// Build `x * y + z` in one rounding (C++ `fma`, `prim.fma`).
    fma(x: impl IntoExpr, y: impl IntoExpr, z: impl IntoExpr) => fma_at;
}

/// Build [`pow`] at the location `loc`.
pub fn pow_at(x: impl IntoExpr, y: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    float_binary(x.into_owned_expr()?, y.into_owned_expr()?, "prim.pow", "power", loc.into())
}

/// Build [`fmod`] at the location `loc`.
pub fn fmod_at(x: impl IntoExpr, y: impl IntoExpr, loc: impl Into<Location>) -> Result<Expr> {
    float_binary(x.into_owned_expr()?, y.into_owned_expr()?, "prim.fmod", "fmod", loc.into())
}

/// Build [`fma`] at the location `loc`.
pub fn fma_at(
    x: impl IntoExpr,
    y: impl IntoExpr,
    z: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let x = x.into_owned_expr()?;
    let ty = x.ty.clone();
    call(Some(&ty), "prim.fma", vec![x, y.into_owned_expr()?, z.into_owned_expr()?], loc.into())
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
    /// Build `e` to the power `x` (C++ `exp`, `prim.exp`).
    exp / exp_at => "prim.exp", false;
    /// Build `2` to the power `x` (C++ `exp2`, `prim.exp2`).
    exp2 / exp2_at => "prim.exp2", false;
    /// Build `10` to the power `x` (C++ `exp10`, `prim.exp10`).
    exp10 / exp10_at => "prim.exp10", false;
    /// Build the error function of `x` (C++ `erf`, `prim.erf`).
    erf / erf_at => "prim.erf", false;
    /// Build the sigmoid of `x` (C++ `sigmoid`, `prim.sigmoid`).
    sigmoid / sigmoid_at => "prim.sigmoid", false;
    /// Build the square root of `x` (C++ `sqrt`, `prim.sqrt`).
    sqrt / sqrt_at => "prim.sqrt", false;
    /// Build the reciprocal square root of `x` (C++ `rsqrt`, `prim.rsqrt`).
    rsqrt / rsqrt_at => "prim.rsqrt", false;
    /// Build the natural logarithm of `x` (C++ `log`, `prim.log`).
    log / log_at => "prim.log", false;
    /// Build the base-2 logarithm of `x` (C++ `log2`, `prim.log2`).
    log2 / log2_at => "prim.log2", false;
    /// Build the base-10 logarithm of `x` (C++ `log10`, `prim.log10`).
    log10 / log10_at => "prim.log10", false;
    /// Build `log(1 + x)` (C++ `log1p`, `prim.log1p`).
    log1p / log1p_at => "prim.log1p", false;
    /// Build the number of set bits of `x` (C++ `popcount`, `prim.popcount`).
    popcount / popcount_at => "prim.popcount", false;
    /// Build the hyperbolic tangent of the float `x` (C++ `tanh`, `prim.tanh`).
    tanh / tanh_at => "prim.tanh", true;
    /// Build the tangent of the float `x` (C++ `tan`, `prim.tan`).
    tan / tan_at => "prim.tan", true;
    /// Build the cosine of the float `x` (C++ `cos`, `prim.cos`).
    cos / cos_at => "prim.cos", true;
    /// Build the hyperbolic cosine of the float `x` (C++ `cosh`, `prim.cosh`).
    cosh / cosh_at => "prim.cosh", true;
    /// Build the sine of the float `x` (C++ `sin`, `prim.sin`).
    sin / sin_at => "prim.sin", true;
    /// Build the hyperbolic sine of the float `x` (C++ `sinh`, `prim.sinh`).
    sinh / sinh_at => "prim.sinh", true;
    /// Build the arc sine of the float `x` (C++ `asin`, `prim.asin`).
    asin / asin_at => "prim.asin", true;
    /// Build the arc cosine of the float `x` (C++ `acos`, `prim.acos`).
    acos / acos_at => "prim.acos", true;
    /// Build the arc tangent of the float `x` (C++ `atan`, `prim.atan`).
    atan / atan_at => "prim.atan", true;
    /// Build the inverse hyperbolic cosine of the float `x` (C++ `acosh`, `prim.acosh`).
    acosh / acosh_at => "prim.acosh", true;
    /// Build the inverse hyperbolic sine of the float `x` (C++ `asinh`, `prim.asinh`).
    asinh / asinh_at => "prim.asinh", true;
    /// Build the inverse hyperbolic tangent of the float `x` (C++ `atanh`, `prim.atanh`).
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

            #[doc = concat!("Build [`", stringify!($name), "`] at the location `loc`.")]
            pub fn $at(
                x: impl IntoExpr,
                y: impl IntoExpr,
                loc: impl Into<Location>,
            ) -> Result<Expr> {
                let x = x.into_owned_expr()?;
                let ty = x.ty.clone();
                call(Some(&ty), $op, vec![x, y.into_owned_expr()?], loc.into())
            }
        )+
    };
}

binary_intrinsics! {
    /// Build the arc tangent of `x / y`, by the signs of both (C++ `atan2`, `prim.atan2`).
    atan2 / atan2_at => "prim.atan2";
    /// Build the next representable value after `x` towards `y` (C++ `nextafter`,
    /// `prim.nextafter`).
    nextafter / nextafter_at => "prim.nextafter";
    /// Build `x` with the sign of `y` (C++ `copysign`, `prim.copysign`).
    copysign / copysign_at => "prim.copysign";
    /// Build `sqrt(x * x + y * y)` (C++ `hypot`, `prim.hypot`).
    hypot / hypot_at => "prim.hypot";
    /// Build `x * 2 ^ y` (C++ `ldexp`, `prim.ldexp`).
    ldexp / ldexp_at => "prim.ldexp";
}

// ---------------------------------------------------------------------------------------------
// Data type queries and operand checks the operators share

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
    let Some(call) = expr.downcast_ref::<CallObj>() else { return Ok(false) };
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
