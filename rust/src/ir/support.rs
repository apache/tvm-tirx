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

//! Constructor support. The hand-written constructors raise their check errors in the text the
//! native checks use, query a `PrimType` through these helpers, and call the native hooks that
//! the operator and type registries hold.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;
use std::thread::LocalKey;

use tvm_ffi::tvm_ffi_sys::{TVMFFIFieldFlagBitMask, TVMFFIGetTypeInfo};
use tvm_ffi::{
    Any, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, ErrorKind, Result, INTERNAL_ERROR,
};

use tvm_ffi::reflection::NativeFunction;

use super::{Op, OpAttrMap, OpObj, PrimType};

pub(crate) const INT: u8 = DLDataTypeCode::kDLInt as u8;
pub(crate) const UINT: u8 = DLDataTypeCode::kDLUInt as u8;
pub(crate) const FLOAT: u8 = DLDataTypeCode::kDLFloat as u8;
pub(crate) const BFLOAT: u8 = DLDataTypeCode::kDLBfloat as u8;
pub(crate) const BOOL: u8 = DLDataTypeCode::kDLBool as u8;

/// The largest finite float16.
pub(crate) const MAX_FLOAT16: f64 = 65504.0;
/// The largest finite bfloat16, `0x1.fep127`.
pub(crate) const MAX_BFLOAT16: f64 = 3.389_531_389_251_535_5e38;

/// The largest finite value of each float8 type code, and whether the type is unsigned
/// (`support::kMax*` in `src/support/limits.h`).
pub(crate) fn float8_bound(code: u8) -> Option<(f64, bool)> {
    use tvm_ffi::DLDataTypeCode::*;
    let bound = match code {
        c if c == kDLFloat8_e3m4 as u8 => (31.0, false),
        c if c == kDLFloat8_e4m3 as u8 => (448.0, false),
        c if c == kDLFloat8_e4m3b11fnuz as u8 => (30.0, true),
        c if c == kDLFloat8_e4m3fn as u8 => (448.0, false),
        c if c == kDLFloat8_e4m3fnuz as u8 => (448.0, true),
        c if c == kDLFloat8_e5m2 as u8 => (57344.0, false),
        c if c == kDLFloat8_e5m2fnuz as u8 => (57344.0, true),
        // `support::kMaxE8M0FNU`, 2^128.
        c if c == kDLFloat8_e8m0fnu as u8 => (3.402_823_669_209_385e38, true),
        _ => return None,
    };
    Some(bound)
}

/// The largest finite value of each float6 type code.
pub(crate) fn float6_bound(code: u8) -> Option<f64> {
    use tvm_ffi::DLDataTypeCode::*;
    match code {
        c if c == kDLFloat6_e2m3fn as u8 => Some(7.5),
        c if c == kDLFloat6_e3m2fn as u8 => Some(28.0),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Errors

/// The value of a native call that cannot fail: the crate links `libtvm_tirx_compiler`, so the
/// function is registered, and the arguments are ones it accepts without a check. A failure means
/// the linked library does not match the sources the crate was built from, which is an error of
/// the build, so it panics.
pub(crate) fn infallible<T>(result: Result<T>) -> T {
    result.expect("a native call that cannot fail with a matching libtvm_tirx_compiler failed")
}

/// The error of a failed check, in the text the native `TVM_FFI_CHECK(condition, kind) << message`
/// raises; `TVM_FFI_ICHECK` is `INTERNAL_ERROR`.
pub(crate) fn check_failed(kind: ErrorKind, condition: &str, message: &str) -> Error {
    Error::new(kind, &format!("Check failed: ({condition}) is false: {message}"), "")
}

/// The error of a failed comparison, in the text the native `TVM_FFI_CHECK_EQ(lhs, rhs, kind)` and
/// its siblings raise, which prints both sides.
pub(crate) fn compare_failed(
    kind: ErrorKind,
    condition: &str,
    lhs: impl Display,
    rhs: impl Display,
    message: &str,
) -> Error {
    Error::new(kind, &format!("Check failed: {condition} ({lhs} vs. {rhs}) : {message}"), "")
}

/// A dtype as error messages print it, such as `int32`.
pub(crate) fn dtype_name(dtype: DLDataType) -> std::string::String {
    dtype.to_string()
}

/// The default value the reflection of the type of `type_index` registers for its own field
/// `name`, or `None` when it registers none or a factory.
pub(crate) fn field_default(type_index: i32, name: &str) -> Option<Any> {
    use TVMFFIFieldFlagBitMask::*;
    // SAFETY: the type information of a registered type is static, and its `num_fields` fields
    // are initialized.
    let field = unsafe {
        let info = &*TVMFFIGetTypeInfo(type_index);
        let fields = std::slice::from_raw_parts(info.fields, info.num_fields as usize);
        fields.iter().find(|field| field.name.as_str() == name)?
    };
    let has_default = field.flags & kTVMFFIFieldFlagBitMaskHasDefault as i64 != 0;
    let from_factory = field.flags & kTVMFFIFieldFlagBitMaskDefaultFromFactory as i64 != 0;
    if !has_default || from_factory {
        return None;
    }
    // SAFETY: the registry owns the default value; it is copied, never released.
    let value = std::mem::ManuallyDrop::new(unsafe {
        Any::from_raw_ffi_any(field.default_value_or_factory)
    });
    Some((*value).clone())
}

/// A double as C++ `std::ostream` prints it by default: `%g`, six significant digits.
#[derive(Clone, Copy)]
pub(crate) struct CppDouble(pub f64);

impl Display for CppDouble {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.0;
        if value.is_nan() {
            return f.write_str(if value.is_sign_negative() { "-nan" } else { "nan" });
        }
        if value.is_infinite() {
            return f.write_str(if value < 0.0 { "-inf" } else { "inf" });
        }
        if value == 0.0 {
            return f.write_str(if value.is_sign_negative() { "-0" } else { "0" });
        }
        let trim = |text: &str| -> std::string::String {
            match text.contains('.') {
                true => text.trim_end_matches('0').trim_end_matches('.').to_string(),
                false => text.to_string(),
            }
        };
        let scientific = format!("{value:.5e}");
        let (mantissa, exponent) = scientific.split_once('e').expect("an exponent");
        let exponent: i32 = exponent.parse().expect("an integer exponent");
        if !(-4..6).contains(&exponent) {
            let sign = if exponent < 0 { '-' } else { '+' };
            return write!(f, "{}e{sign}{:02}", trim(mantissa), exponent.abs());
        }
        let decimals = (5 - exponent) as usize;
        f.write_str(&trim(&format!("{value:.decimals$}")))
    }
}

// ---------------------------------------------------------------------------------------------
// Shared nodes

/// A node the native library shares, such as the unknown-location singleton, fetched once per
/// thread so that every node Rust builds refers to the same object as natively built nodes. The
/// fetch is a native call that cannot fail (see [`infallible`]).
pub(crate) fn shared<T: Clone + 'static>(
    cell: &'static LocalKey<OnceCell<T>>,
    fetch: impl FnOnce() -> Result<T>,
) -> T {
    if let Some(value) = cell.with(|cell| cell.get().cloned()) {
        return value;
    }
    let value = infallible(fetch());
    cell.with(|cell| cell.get_or_init(|| value.clone()).clone());
    value
}

// ---------------------------------------------------------------------------------------------
// Primitive types

/// The number of lanes of `ty`; a scalable vector has no fixed number and raises an
/// `InternalError`.
pub(crate) fn lanes(ty: &PrimType) -> Result<i32> {
    match ty.lanes() {
        Some(lanes) => Ok(i32::from(lanes)),
        None => {
            let message = "Can't fetch the lanes of a scalable vector at a compile time.";
            Err(Error::new(INTERNAL_ERROR, message, ""))
        }
    }
}

/// The scalable-vector dtype of `factor` times `vscale` lanes, with the code and bits of `dtype`;
/// `factor` must be in `2..32768`.
pub(crate) fn scalable_vector(dtype: DLDataType, factor: i32) -> Result<DLDataType> {
    if factor <= 1 {
        let message = format!("Invalid value for vscale factor {factor}");
        return Err(compare_failed(INTERNAL_ERROR, "lanes > 1", factor, 1, &message));
    }
    if factor >= 32768 {
        return Err(compare_failed(INTERNAL_ERROR, "lanes < 32768", factor, 32768, ""));
    }
    Ok(DLDataType { code: dtype.code, bits: dtype.bits, lanes: (-factor) as u16 })
}

// ---------------------------------------------------------------------------------------------
// Registry hooks

/// The hook `op` registers in the attribute column `column`, if any.
///
/// # Safety
///
/// A native function in the column must take `Args` and return `R`, as its declaration says.
pub(crate) unsafe fn op_hook<Args, R>(
    op: &OpObj,
    column: &'static str,
) -> Result<Option<NativeFunction<Args, R>>> {
    // SAFETY: the caller vouches for the signature.
    op_attr(op, column)?.map(|hook| unsafe { NativeFunction::from_any(hook) }).transpose()
}

thread_local! {
    /// The operator attribute columns, fetched once per thread. A column no operator registered yet
    /// is looked up again.
    static OP_ATTR_COLUMNS: RefCell<HashMap<&'static str, Rc<OpAttrMap<Any>>>> =
        RefCell::new(HashMap::new());
}

/// The value `op` registers in the attribute column `column`, of whatever type, or `None`.
pub(crate) fn op_attr(op: &OpObj, column: &'static str) -> Result<Option<Any>> {
    let cached = OP_ATTR_COLUMNS.with(|columns| columns.borrow().get(column).cloned());
    let map = match cached {
        Some(map) => map,
        None => {
            let Some(map) = Op::attr_column::<Any>(column)? else { return Ok(None) };
            let map = Rc::new(map);
            OP_ATTR_COLUMNS.with(|columns| columns.borrow_mut().insert(column, map.clone()));
            map
        }
    };
    Ok(map.get_any(op))
}

/// The registered operator `name`, fetched once per thread.
pub(crate) fn cached_op(name: &'static str) -> Result<Op> {
    thread_local! {
        static OPS: RefCell<HashMap<&'static str, Op>> = RefCell::new(HashMap::new());
    }
    if let Some(op) = OPS.with(|ops| ops.borrow().get(name).cloned()) {
        return Ok(op);
    }
    let op = Op::get(name)?;
    OPS.with(|ops| ops.borrow_mut().insert(name, op.clone()));
    Ok(op)
}
