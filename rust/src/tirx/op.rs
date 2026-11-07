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

//! Operators of the dialect. The functions build calls of its operators, such as [`logaddexp`]
//! and [`reinterpret`]; a [`CallFFIKernelAttr`] holds the launch metadata of a `call_ffi_kernel`
//! call.

// tvm-ffi-stubgen(missing-docs): allow
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
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PtrTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::support
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
use crate::ir::r#type::PtrTypeObj;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::DLDataTypeCode;
use tvm_ffi::INTERNAL_ERROR;
use tvm_ffi::Map;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::TYPE_ERROR;
// tvm-ffi-stubgen(end)

/// The launch metadata of a `call_ffi_kernel` call, whose arguments are the kernel symbol, the
/// kernel operands and then the launch values.
///
/// `launch_params` tags the launch values in order (a flag-only tag takes no value, and the dynamic
/// shared-memory size, when present, comes last), `launch_fields` names the CUDA launch
/// configuration field each launch value sets, `num_kernel_args` counts the kernel operands, and
/// `kernel_attrs` holds the compile-time requirements of the kernel.
// tvm-ffi-stubgen(begin): object/tirx.CallFFIKernelAttr
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CallFFIKernelAttr {
    data_: ObjectArc<CallFFIKernelAttrObj>,
}

/// The object behind [`CallFFIKernelAttr`].
///
/// Complete: reflected fields fill [24, 56) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.CallFFIKernelAttr"]
#[type_final]
pub struct CallFFIKernelAttrObj {
    base_: AttrsObj,
    pub launch_params: Array<String>,
    pub launch_fields: Array<String>,
    pub num_kernel_args: i64,
    pub kernel_attrs: Map<String, i64>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallFFIKernelAttrObj>() == 56);
    assert!(::core::mem::align_of::<CallFFIKernelAttrObj>() == 8);
};

impl Deref for CallFFIKernelAttrObj {
    type Target = AttrsObj;
    fn deref(&self) -> &AttrsObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl CallFFIKernelAttrObj {
    pub(crate) fn new(
        launch_params: Array<String>,
        launch_fields: Array<String>,
        num_kernel_args: i64,
        kernel_attrs: Map<String, i64>,
    ) -> Self {
        let base_ = AttrsObj::new();
        Self { base_, launch_params, launch_fields, num_kernel_args, kernel_attrs }
    }
}

#[allow(missing_docs)]
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
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for CallFFIKernelAttrObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(CallFFIKernelAttr => Attrs);
// tvm-ffi-stubgen(end)

// The native `tirx.CallFFIKernelAttr` stores its fields as given, as the generated
// `CallFFIKernelAttr::new` does.

/// Build `log(exp(a) + exp(b))` of the floating-point `a` and `b`, matched to one type.
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
        if !operand.prim_ty().matches_code(&[DLDataTypeCode::kDLFloat]) {
            let condition = format!("{side}.ty().MatchesCode(DLDataTypeCode::kDLFloat)");
            let message = operand.to_string();
            return Err(support::check_failed(INTERNAL_ERROR, &condition, &message));
        }
    }
    let (a, b) = match_types_at(a, b, loc)?;
    log(add(exp(a)?, exp(b)?)?)
}

/// Reinterpret `value` bit for bit as the primitive or pointer type `ty` (`tirx.reinterpret`).
///
/// Besides primitive types of the same size, a string reinterprets as a pointer and a pointer as a
/// scalar `int64`, and back.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{PrimType, PrimVar, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::tirx::reinterpret;
///
/// let x = PrimVar::new("x", PrimType::float(32), UNKNOWN_LOC);
/// assert_eq!(script(&reinterpret(PrimType::int(32), &x)?)?, "T.reinterpret(x, ty=T.int32)");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
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
            let message = format!("String reinterpret requires a pointer target, but got {ty}");
            return type_error("target_ty.as<PtrTypeObj>()", message);
        }
    } else if let Some(target) = ty.downcast_ref::<PrimType>() {
        if let Some(source) = value.ty.downcast_ref::<PrimType>() {
            if source.dtype == target.dtype {
                return Ok(value);
            }
            if !target.is_scalable_vector() && !source.is_scalable_vector() {
                let bits =
                    |ty: &PrimType| ty.lanes().map(|lanes| u32::from(ty.bits()) * u32::from(lanes));
                let float4 = [DLDataTypeCode::kDLFloat4_e2m1fn];
                let same_size = bits(source) == bits(target)
                    || ((source.matches_code(&float4) || target.matches_code(&float4))
                        && source.storage_bytes() == target.storage_bytes());
                if !same_size {
                    let condition = "value_bits == target_bits || \
                                     ((value_dtype.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) || \
                                     target_prim.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn)) && \
                                     value_dtype.StorageBytes() == target_prim.StorageBytes())";
                    let message = format!("Reinterpret requires size match {} vs {}", ty, value.ty);
                    return Err(support::check_failed(INTERNAL_ERROR, condition, &message));
                }
            }
        } else {
            if !source_is_pointer {
                let message =
                    format!("Reinterpret source must be PrimType or PtrType, but got {}", value.ty);
                return type_error("value->ty.as<PtrTypeObj>()", message);
            }
            if !is_scalar_int64(target) {
                let condition = "target_dtype.value().IsScalar() && target_dtype.value().bits() == 64 && \
                                 target_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)";
                let message = format!(
                    "Pointer reinterpret requires a scalar 64-bit integer target, but got {ty}"
                );
                return type_error(condition, message);
            }
        }
    } else {
        if !target_is_pointer {
            let message = format!("Reinterpret target must be PrimType or PtrType, but got {ty}");
            return type_error("target_ty.as<PtrTypeObj>()", message);
        }
        if let Some(source) = value.ty.downcast_ref::<PrimType>() {
            if !is_scalar_int64(source) {
                let condition = "source_dtype.value().IsScalar() && source_dtype.value().bits() == 64 && \
                                 source_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)";
                let message = format!(
                    "Pointer reinterpret requires a scalar 64-bit integer source, but got {}",
                    value.ty
                );
                return type_error(condition, message);
            }
        } else if !source_is_pointer {
            let message =
                format!("Reinterpret source must be PrimType or PtrType, but got {}", value.ty);
            return type_error("value->ty.as<PtrTypeObj>()", message);
        }
    }
    let op = support::cached_op("tirx.reinterpret")?;
    Ok(Call::new(Some(&ty), op, vec![value], None, Vec::<Type>::new(), loc)?.into())
}

/// Build a rational approximation of `erf(arg)` on `[-4, 4]` in `bits`-bit floats, as the Metal and
/// WebGPU backends lower `erf`.
pub fn fast_erf_float_expr(arg: impl IntoExpr<PrimExpr>, bits: u8) -> Result<PrimExpr> {
    let ty = PrimType::float(bits);
    // The coefficients are single-precision values.
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
    // Horner's rule from the highest coefficient down.
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

/// Whether `ty` is a scalar 64-bit integer, which a pointer reinterprets as.
fn is_scalar_int64(ty: &PrimType) -> bool {
    ty.is_scalar()
        && ty.bits() == 64
        && ty.matches_code(&[DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt])
}
