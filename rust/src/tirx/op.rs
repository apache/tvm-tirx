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

//! Attributes of TIRx operator calls.
//!
//! The functions build TIRx operator calls; a [`CallFFIKernelAttr`] holds the launch parameters of
//! a `call_ffi_kernel`.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{PrimType, PrimVar, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//! use tvm_tirx::tirx::{logaddexp, reinterpret};
//!
//! let x = PrimVar::new("x", PrimType::float(32), UNKNOWN_LOC);
//! let y = PrimVar::new("y", PrimType::float(32), UNKNOWN_LOC);
//! assert_eq!(script(&logaddexp(&x, &y)?)?, "T.log(T.exp(x) + T.exp(y))");
//! assert_eq!(script(&reinterpret(PrimType::int(32), &x)?)?, "T.reinterpret(x, ty=T.int32)");
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::location::Location
// tvm-ffi-stubgen(import-object): crate::ir::r#type::Type
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::Call
// tvm-ffi-stubgen(import-object): crate::ir::expr::FloatImm
// tvm-ffi-stubgen(import-object): crate::ir::expr::StringImmObj
// tvm-ffi-stubgen(import-object): crate::ir::location::UNKNOWN_LOC
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::ExprResult
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::PrimOps
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::add
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::exp
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::log
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::match_types_at
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PtrTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataType
// tvm-ffi-stubgen(import-object): tvm_ffi::DLDataTypeCode
// tvm-ffi-stubgen(import-object): tvm_ffi::INTERNAL_ERROR
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::TYPE_ERROR
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::attrs::AttrsObj;
use crate::ir::expr::Call;
use crate::ir::expr::Expr;
use crate::ir::expr::FloatImm;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::StringImmObj;
use crate::ir::location::Location;
use crate::ir::location::UNKNOWN_LOC;
use crate::ir::prim::op::ExprResult;
use crate::ir::prim::op::PrimOps;
use crate::ir::prim::op::add;
use crate::ir::prim::op::exp;
use crate::ir::prim::op::log;
use crate::ir::prim::op::match_types_at;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::PrimTypeObj;
use crate::ir::r#type::PtrTypeObj;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::DLDataType;
use tvm_ffi::DLDataTypeCode;
use tvm_ffi::INTERNAL_ERROR;
use tvm_ffi::Map;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::TYPE_ERROR;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.CallFFIKernelAttr
/// Complete: reflected fields fill [24, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.CallFFIKernelAttr"]
#[type_final]
pub struct CallFFIKernelAttrObj {
    base: AttrsObj,
    pub launch_params: Array<String>,
    pub launch_fields: Array<String>,
    pub num_kernel_args: i64,
    pub kernel_attrs: Map<String, i64>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallFFIKernelAttrObj>() == 56);
    assert!(::core::mem::align_of::<CallFFIKernelAttrObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CallFFIKernelAttr {
    base: ObjectArc<CallFFIKernelAttrObj>,
}

impl Deref for CallFFIKernelAttr {
    type Target = CallFFIKernelAttrObj;
    fn deref(&self) -> &CallFFIKernelAttrObj {
        &self.base
    }
}

impl Deref for CallFFIKernelAttrObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base
    }
}

impl CallFFIKernelAttrObj {
    pub(crate) fn new(
        launch_params: Array<String>,
        launch_fields: Array<String>,
        num_kernel_args: i64,
        kernel_attrs: Map<String, i64>,
    ) -> Self {
        let base = AttrsObj::new();
        Self { base, launch_params, launch_fields, num_kernel_args, kernel_attrs }
    }
}

impl CallFFIKernelAttr {
    /// Lossless complete-field allocation.
    pub fn new(
        launch_params: impl Into<Array<String>>,
        launch_fields: impl Into<Array<String>>,
        num_kernel_args: i64,
        kernel_attrs: impl Into<Map<String, i64>>,
    ) -> Self {
        let obj = CallFFIKernelAttrObj::new(
            launch_params.into(),
            launch_fields.into(),
            num_kernel_args,
            kernel_attrs.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for CallFFIKernelAttrObj {}

tvm_ffi::unsafe_impl_object_upcast!(CallFFIKernelAttr => Attrs);
// tvm-ffi-stubgen(end)

// The native `tirx.CallFFIKernelAttr` stores its fields as given, as the generated
// `CallFFIKernelAttr::new` does.

/// Build `log(exp(a) + exp(b))` of the floating-point `a` and `b`, matched to one type
/// (C++ `tirx::logaddexp`, `tirx._OpLogAddExp`).
///
/// # Errors
///
/// Returns an error if an operand is not floating-point.
pub fn logaddexp(a: impl IntoExpr<PrimExpr>, b: impl IntoExpr<PrimExpr>) -> Result<PrimExpr> {
    logaddexp_at(a, b, UNKNOWN_LOC)
}

/// Build [`logaddexp`] with the casts at `loc`.
pub fn logaddexp_at(
    a: impl IntoExpr<PrimExpr>,
    b: impl IntoExpr<PrimExpr>,
    loc: impl Into<Location>,
) -> Result<PrimExpr> {
    let (a, b) = (a.into_owned_expr()?, b.into_owned_expr()?);
    for (operand, side) in [(&a, "a"), (&b, "b")] {
        if operand.dtype().code != support::FLOAT {
            let condition = format!("{side}.ty().MatchesCode(DLDataTypeCode::kDLFloat)");
            let message = support::repr(operand);
            return Err(support::check_failed(INTERNAL_ERROR, &condition, &message));
        }
    }
    let (a, b) = match_types_at(a, b, loc)?;
    log(add(exp(a)?, exp(b)?)?)
}

/// Whether `dtype` is a scalar 64-bit integer, which a pointer reinterprets as.
fn is_scalar_int64(dtype: DLDataType) -> bool {
    support::is_scalar(dtype)
        && dtype.bits == 64
        && matches!(dtype.code, support::INT | support::UINT)
}

/// C++ `PrimType::StorageBytes()`.
fn storage_bytes(dtype: DLDataType) -> Result<u64> {
    Ok((u64::from(dtype.bits) * support::lanes(dtype)? as u64).div_ceil(8))
}

/// Reinterpret `value` bit for bit as the primitive or pointer type `ty`
/// (C++ `tirx::reinterpret(target_ty, value, loc)`, `tirx.reinterpret`).
///
/// Besides primitive types of the same size, a string reinterprets as a pointer and a pointer as a
/// scalar `int64`, and back.
///
/// # Errors
///
/// Returns an error for any other pair of types, or two primitive types of different sizes.
pub fn reinterpret(ty: impl Into<Type>, value: impl IntoExpr) -> Result<Expr> {
    reinterpret_at(ty, value, UNKNOWN_LOC)
}

/// Build [`reinterpret`] at `loc`.
pub fn reinterpret_at(
    ty: impl Into<Type>,
    value: impl IntoExpr,
    loc: impl Into<Location>,
) -> Result<Expr> {
    let (ty, value, loc) = (ty.into(), value.into_owned_expr()?, loc.into());
    let target_is_pointer = ty.downcast_ref::<PtrTypeObj>().is_some();
    let source_is_pointer = value.ty.downcast_ref::<PtrTypeObj>().is_some();
    let type_error = |condition: &str, message: std::string::String| {
        Err(support::check_failed(TYPE_ERROR, condition, &message))
    };
    if value.downcast_ref::<StringImmObj>().is_some() {
        if !target_is_pointer {
            let message = format!(
                "String reinterpret requires a pointer target, but got {}",
                support::repr(&ty)
            );
            return type_error("target_ty.as<PtrTypeObj>()", message);
        }
    } else if let Some(target) = ty.downcast_ref::<PrimTypeObj>() {
        let target_dtype = target.dtype;
        if let Some(source) = value.ty.downcast_ref::<PrimTypeObj>() {
            let source_dtype = source.dtype;
            if source_dtype == target_dtype {
                return Ok(value);
            }
            if !support::is_scalable_vector(target_dtype)
                && !support::is_scalable_vector(source_dtype)
            {
                let bits = |dtype: DLDataType| {
                    Ok::<i64, tvm_ffi::Error>(
                        i64::from(dtype.bits) * i64::from(support::lanes(dtype)?),
                    )
                };
                let float4 = DLDataTypeCode::kDLFloat4_e2m1fn as u8;
                let same_size = bits(source_dtype)? == bits(target_dtype)?
                    || ((source_dtype.code == float4 || target_dtype.code == float4)
                        && storage_bytes(source_dtype)? == storage_bytes(target_dtype)?);
                if !same_size {
                    let condition = "value_bits == target_bits || \
                                     ((value_dtype.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) || \
                                     target_prim.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn)) && \
                                     value_dtype.StorageBytes() == target_prim.StorageBytes())";
                    let message = format!(
                        "Reinterpret requires size match {} vs {}",
                        support::repr(&ty),
                        support::repr(&value.ty)
                    );
                    return Err(support::check_failed(INTERNAL_ERROR, condition, &message));
                }
            }
        } else {
            if !source_is_pointer {
                let message = format!(
                    "Reinterpret source must be PrimType or PtrType, but got {}",
                    support::repr(&value.ty)
                );
                return type_error("value->ty.as<PtrTypeObj>()", message);
            }
            if !is_scalar_int64(target_dtype) {
                let condition = "target_dtype.value().IsScalar() && target_dtype.value().bits() == 64 && \
                                 target_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)";
                let message = format!(
                    "Pointer reinterpret requires a scalar 64-bit integer target, but got {}",
                    support::repr(&ty)
                );
                return type_error(condition, message);
            }
        }
    } else {
        if !target_is_pointer {
            let message = format!(
                "Reinterpret target must be PrimType or PtrType, but got {}",
                support::repr(&ty)
            );
            return type_error("target_ty.as<PtrTypeObj>()", message);
        }
        if let Some(source) = value.ty.downcast_ref::<PrimTypeObj>() {
            if !is_scalar_int64(source.dtype) {
                let condition = "source_dtype.value().IsScalar() && source_dtype.value().bits() == 64 && \
                                 source_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)";
                let message = format!(
                    "Pointer reinterpret requires a scalar 64-bit integer source, but got {}",
                    support::repr(&value.ty)
                );
                return type_error(condition, message);
            }
        } else if !source_is_pointer {
            let message = format!(
                "Reinterpret source must be PrimType or PtrType, but got {}",
                support::repr(&value.ty)
            );
            return type_error("value->ty.as<PtrTypeObj>()", message);
        }
    }
    let op = support::cached_op("tirx.reinterpret")?;
    Ok(Call::new(Some(&ty), op, vec![value], None, Vec::<Type>::new(), loc)?.into())
}

/// Build a rational approximation of `erf(arg)` on `[-4, 4]` in `bits`-bit floats, as the Metal and
/// WebGPU backends lower `erf` (C++ `tirx::fast_erf_float_expr(arg, bits)`).
pub fn fast_erf_float_expr(arg: impl IntoExpr<PrimExpr>, bits: u8) -> Result<PrimExpr> {
    let ty = PrimType::float(bits);
    // The C++ coefficients are `float` literals.
    let constant =
        |value: f32| -> ExprResult { FloatImm::new(&ty, f64::from(value), UNKNOWN_LOC).into() };
    // alpha_1, alpha_3, ..., alpha_13: the odd coefficients of the numerator.
    #[allow(clippy::excessive_precision)]
    let alpha: [f32; 7] = [
        -1.60960333262415e-02,
        -2.95459980854025e-03,
        -7.34990630326855e-04,
        -5.69250639462346e-05,
        -2.10102402082508e-06,
        2.77068142495902e-08,
        -2.72614225801306e-10,
    ];
    // beta_0, beta_2, ..., beta_8: the even coefficients of the denominator.
    #[allow(clippy::excessive_precision)]
    let beta: [f32; 5] = [
        -1.42647390514189e-02,
        -7.37332916720468e-03,
        -1.68282697438203e-03,
        -2.13374055278905e-04,
        -1.45660718464996e-05,
    ];
    let x = arg.into_owned_expr()?.min(constant(4.0)).max(constant(-4.0));
    let x2 = &x * &x;
    // Horner's rule from the highest coefficient down, as C++ writes it out.
    let mut p = &x2 * constant(alpha[6]) + constant(alpha[5]);
    for coefficient in alpha[..5].iter().rev() {
        p = &x2 * p + constant(*coefficient);
    }
    p = &x * p;
    let mut q = &x2 * constant(beta[4]) + constant(beta[3]);
    for coefficient in beta[..3].iter().rev() {
        q = &x2 * q + constant(*coefficient);
    }
    p.div(q).try_into()
}

/// Build a return from the current GPU thread, without a value (`tirx.gpu_thread_return`).
pub fn gpu_thread_return() -> Result<PrimExpr> {
    gpu_thread_return_at(UNKNOWN_LOC)
}

/// Build [`gpu_thread_return`] at `loc`.
pub fn gpu_thread_return_at(loc: impl Into<Location>) -> Result<PrimExpr> {
    let loc: Location = loc.into();
    tvm_ffi::cached_global_func!("tirx.gpu_thread_return").call_tuple((&loc,))?.try_into()
}
