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

//! What the constructors written in Rust share: the errors of the C++ checks they reproduce, the
//! queries C++ makes of a `PrimType`, and calls to the hooks the registries hold.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::Display;
use std::rc::Rc;
use std::sync::OnceLock;
use std::thread::LocalKey;

use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIFieldFlagBitMask, TVMFFIGetTypeInfo, TVMFFIObject, TVMFFITypeIndex,
};
use tvm_ffi::{
    Any, AnyView, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, ErrorKind, Function,
    ObjectArc, ObjectRefCore, Result, TypeAttrColumn, INTERNAL_ERROR, TYPE_ERROR,
};

use super::{Expr, Op, OpAttrMap, OpObj, PrimType, PrimTypeObj};

pub(crate) const INT: u8 = DLDataTypeCode::kDLInt as u8;
pub(crate) const UINT: u8 = DLDataTypeCode::kDLUInt as u8;
pub(crate) const FLOAT: u8 = DLDataTypeCode::kDLFloat as u8;
pub(crate) const BFLOAT: u8 = DLDataTypeCode::kDLBfloat as u8;
pub(crate) const BOOL: u8 = DLDataTypeCode::kDLBool as u8;

/// C++ `support::kMaxFloat16`, the largest finite float16.
pub(crate) const MAX_FLOAT16: f64 = 65504.0;
/// C++ `support::kMaxBFloat16` (`0x1.fep127`), the largest finite bfloat16.
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
/// function is registered, and the arguments are ones its C++ counterpart accepts without a
/// check. A failure means the linked library does not match the sources the crate was built
/// from, which is an error of the build, so it panics.
pub(crate) fn infallible<T>(result: Result<T>) -> T {
    result.expect("a native call that cannot fail with a matching libtvm_tirx_compiler failed")
}

/// The error of a failed C++ `TVM_FFI_CHECK(condition, kind) << message`; `TVM_FFI_ICHECK` is
/// `INTERNAL_ERROR`.
pub(crate) fn check_failed(kind: ErrorKind, condition: &str, message: &str) -> Error {
    Error::new(kind, &format!("Check failed: ({condition}) is false: {message}"), "")
}

/// The error of a failed C++ `TVM_FFI_CHECK_EQ(lhs, rhs, kind)` and its siblings, which print both
/// sides.
pub(crate) fn compare_failed(
    kind: ErrorKind,
    condition: &str,
    lhs: impl Display,
    rhs: impl Display,
    message: &str,
) -> Error {
    Error::new(kind, &format!("Check failed: {condition} ({lhs} vs. {rhs}) : {message}"), "")
}

/// A value as C++ `operator<<` prints it into an error message (`ffi.ReprPrint`).
pub(crate) fn repr<T>(value: &T) -> std::string::String
where
    for<'a> AnyView<'a>: From<&'a T>,
{
    tvm_ffi::cached_global_func!("ffi.ReprPrint")
        .call_packed(&[AnyView::from(value)])
        .and_then(tvm_ffi::String::try_from)
        .map(|text| text.as_str().to_string())
        .unwrap_or_default()
}

/// A dtype as C++ prints it, such as `int32`.
pub(crate) fn dtype_name(dtype: DLDataType) -> std::string::String {
    DLDataTypeExt::to_string(&dtype).as_str().to_string()
}

/// The registered key of a type index (C++ `TypeIndexToTypeKey`).
pub(crate) fn type_key(type_index: i32) -> std::string::String {
    // SAFETY: the type index of a live value is registered, and its information is static.
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        (*info).type_key.as_str().to_string()
    }
}

/// The default value the reflection of the type of `type_index` registers for its own field
/// `name` (C++ `refl::DefaultValue`), or `None` when it registers none or a factory.
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

/// A node C++ shares, such as the `UnknownLoc()` singleton, fetched once per thread so that every
/// node Rust builds refers to the same object as the nodes C++ builds. The fetch is a native call
/// that cannot fail (see [`infallible`]).
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

/// The type of a primitive expression. C++ takes such an argument as a `PrimExpr`, whose
/// conversion accepts only an expression of a `PrimType`.
pub(crate) fn prim_type(expr: &(impl AsRef<Expr> + ?Sized)) -> Result<PrimType> {
    let expr = expr.as_ref();
    match expr.ty.downcast_ref::<PrimTypeObj>() {
        Some(_) => Ok(tvm_ffi::ObjectRefCast::try_cast(expr.ty.clone()).expect("a PrimType")),
        None => Err(not_a_prim_expr(expr)),
    }
}

/// The dtype of a primitive expression ([`prim_type`]).
pub(crate) fn prim_dtype(expr: &(impl AsRef<Expr> + ?Sized)) -> Result<DLDataType> {
    let expr = expr.as_ref();
    match expr.ty.downcast_ref::<PrimTypeObj>() {
        Some(ty) => Ok(ty.dtype),
        None => Err(not_a_prim_expr(expr)),
    }
}

fn not_a_prim_expr(expr: &Expr) -> Error {
    let message = format!(
        "Expected `ir.PrimExpr` but got `{}[ty={}]`",
        type_key(AnyView::from(expr).type_index()),
        type_key(AnyView::from(&expr.ty).type_index()),
    );
    Error::new(TYPE_ERROR, &message, "")
}

/// The lanes of a dtype as C++ decodes them: negative for a scalable vector.
pub(crate) fn encoded_lanes(dtype: DLDataType) -> i16 {
    dtype.lanes as i16
}

/// C++ `PrimType::IsScalar()`.
pub(crate) fn is_scalar(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) == 1
}

/// C++ `PrimType::IsScalableVector()`.
pub(crate) fn is_scalable_vector(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) < -1
}

/// C++ `PrimType::IsFixedLengthVector()`.
pub(crate) fn is_fixed_length_vector(dtype: DLDataType) -> bool {
    encoded_lanes(dtype) > 1
}

/// C++ `PrimType::lanes()`, which a scalable vector does not have.
pub(crate) fn lanes(dtype: DLDataType) -> Result<i32> {
    let lanes = encoded_lanes(dtype);
    if lanes < 0 {
        let message = "Can't fetch the lanes of a scalable vector at a compile time.";
        return Err(Error::new(INTERNAL_ERROR, message, ""));
    }
    Ok(lanes as i32)
}

/// C++ `PrimType::ScalableVector(code, bits, factor)`: `factor` times `vscale` lanes.
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
// Sequences

/// The object header followed by `TVMFFISeqCell`, the layout `ffi.List` and `ffi.Array` share
/// (`c_api.h`).
#[repr(C)]
pub(crate) struct SeqCell {
    pub(crate) object: tvm_ffi::Object,
    pub(crate) data: *const TVMFFIAny,
    pub(crate) size: i64,
    pub(crate) capacity: i64,
    pub(crate) data_deleter: Option<unsafe extern "C" fn(*mut c_void)>,
}

/// The elements of an `ffi.Array`, read in place.
pub(crate) fn array_cells<T: tvm_ffi::type_traits::ContainerElement + Clone>(
    array: &Array<T>,
) -> &[TVMFFIAny] {
    // SAFETY: an `ffi.Array` is the object header followed by `TVMFFISeqCell`, and `array` keeps
    // its `size` elements alive.
    unsafe {
        let cell = &*(ObjectArc::as_raw(ObjectRefCore::data(array)) as *const SeqCell);
        if cell.size == 0 {
            return &[];
        }
        std::slice::from_raw_parts(cell.data, cell.size as usize)
    }
}

// ---------------------------------------------------------------------------------------------
// Registry hooks

/// A C++ `NativeFunctionView<R(const Node*)>` hook that a registry holds: the C ABI adapter its
/// opaque pointer names, or a packed function a frontend registered in its place.
pub(crate) enum NativeHook {
    Adapter(extern "C" fn(*const c_void) -> TVMFFIAny),
    Packed(Function),
}

impl NativeHook {
    /// The hook `value` holds.
    pub(crate) fn from_any(value: Any) -> Result<Self> {
        if value.type_index() != TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
            return Ok(Self::Packed(value.try_into()?));
        }
        let address: *mut c_void = value.try_into()?;
        // SAFETY: the opaque pointer of a `NativeFunctionView` is its C ABI adapter.
        let adapter = unsafe {
            std::mem::transmute::<*mut c_void, extern "C" fn(*const c_void) -> TVMFFIAny>(address)
        };
        Ok(Self::Adapter(adapter))
    }

    /// The hook `op` registers in the attribute column `column`, if any.
    pub(crate) fn of_op(op: &OpObj, column: &'static str) -> Result<Option<Self>> {
        op_attr(op, column)?.map(Self::from_any).transpose()
    }

    /// C++ `CallExpected(node)`: the hook's result on `node`, or the error it reports.
    pub(crate) fn call<T: ObjectRefCore>(&self, node: &T) -> Result<Any>
    where
        for<'a> AnyView<'a>: From<&'a T>,
    {
        let result = match self {
            // SAFETY: `node` keeps the object alive for the call, and the adapter returns an
            // owned value.
            Self::Adapter(adapter) => unsafe {
                let node = ObjectArc::as_raw(ObjectRefCore::data(node)) as *const c_void;
                Any::from_raw_ffi_any(adapter(node))
            },
            Self::Packed(function) => function.call_packed(&[AnyView::from(node)])?,
        };
        expected(result)
    }

    /// [`NativeHook::call`] of a hook that returns an `R`; a packed hook's result of another type
    /// is the error C++ raises for it.
    pub(crate) fn call_as<T: ObjectRefCore, R: TryFrom<Any>>(&self, node: &T) -> Result<R>
    where
        for<'a> AnyView<'a>: From<&'a T>,
    {
        R::try_from(self.call(node)?).map_err(|_| hook_result_mismatch())
    }

    /// [`NativeHook::call`] of a hook that returns nothing.
    pub(crate) fn call_void<T: ObjectRefCore>(&self, node: &T) -> Result<()>
    where
        for<'a> AnyView<'a>: From<&'a T>,
    {
        match self.call(node)?.type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
            true => Ok(()),
            false => Err(hook_result_mismatch()),
        }
    }
}

/// C++ `CallExpected(dest, indices, value)` of a hook that takes three object references, such as
/// `NativeFunctionView<void(const Expr&, const Array<PrimExpr>&, const PrimExpr&)>`. The C ABI
/// adapter takes each reference as the address of an object pointer; a packed function takes the
/// values.
pub(crate) fn call_with_refs<A, B, C>(hook: Any, a: &A, b: &B, c: &C) -> Result<Any>
where
    A: ObjectRefCore,
    B: ObjectRefCore,
    C: ObjectRefCore,
    for<'x> AnyView<'x>: From<&'x A> + From<&'x B> + From<&'x C>,
{
    if hook.type_index() != TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
        let function: Function = hook.try_into()?;
        return expected(function.call_packed(&[
            AnyView::from(a),
            AnyView::from(b),
            AnyView::from(c),
        ])?);
    }
    type Adapter = extern "C" fn(
        *const *const TVMFFIObject,
        *const *const TVMFFIObject,
        *const *const TVMFFIObject,
    ) -> TVMFFIAny;
    fn object<T: ObjectRefCore>(value: &T) -> *const TVMFFIObject {
        // SAFETY: the pointer is only read while `value` keeps the object alive.
        unsafe { ObjectArc::as_raw(ObjectRefCore::data(value)) as *const TVMFFIObject }
    }
    let address: *mut c_void = hook.try_into()?;
    let (a, b, c) = (object(a), object(b), object(c));
    // SAFETY: the opaque pointer of a `NativeFunctionView` is its C ABI adapter; a C++ object
    // reference is the address of its single object pointer, which outlives the call; and the
    // adapter returns an owned value.
    let result = unsafe {
        let adapter = std::mem::transmute::<*mut c_void, Adapter>(address);
        Any::from_raw_ffi_any(adapter(&a, &b, &c))
    };
    expected(result)
}

/// The value of an `Expected` result, or its error.
fn expected(result: Any) -> Result<Any> {
    match result.type_index() == TVMFFITypeIndex::kTVMFFIError as i32 {
        true => Err(result.try_into()?),
        false => Ok(result),
    }
}

fn hook_result_mismatch() -> Error {
    Error::new(TYPE_ERROR, "Packed hook result type mismatch", "")
}

thread_local! {
    /// The operator attribute columns, fetched once per thread as C++ keeps them in `static`s.
    /// A column no operator registered yet is looked up again.
    static OP_ATTR_COLUMNS: RefCell<HashMap<&'static str, Rc<OpAttrMap<Any>>>> =
        RefCell::new(HashMap::new());
}

/// The value `op` registers in the attribute column `column` (C++ `Op::GetAttrMap(column)[op]`),
/// of whatever type, or `None`.
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

/// The registered operator `name` (C++ `Op::Get(name)`), fetched once per thread as C++ keeps an
/// operator in a `static`.
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

/// A type attribute column, looked up once it exists, as C++ keeps a `TypeAttrColumn` in a
/// `static`.
pub(crate) struct TypeAttr {
    name: &'static str,
    column: OnceLock<TypeAttrColumn>,
}

impl TypeAttr {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self { name, column: OnceLock::new() }
    }

    /// The value the type of `type_index` registers in this column, if any.
    pub(crate) fn get(&self, type_index: i32) -> Option<Any> {
        let column = match self.column.get() {
            Some(column) => *column,
            None => {
                let column = TypeAttrColumn::new(self.name)?;
                *self.column.get_or_init(|| column)
            }
        };
        column.get(type_index)
    }
}

/// `ffi::StructuralEqual()(lhs, rhs)`, which does not map free variables.
pub(crate) fn structural_equal<L, R>(lhs: &L, rhs: &R) -> Result<bool>
where
    for<'a> AnyView<'a>: From<&'a L> + From<&'a R>,
{
    let (map_free_vars, skip_tensor_content) = (false, false);
    tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&map_free_vars),
            AnyView::from(&skip_tensor_content),
        ])?
        .try_into()
}
