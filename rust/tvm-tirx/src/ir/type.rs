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

//! Types. Every expression carries a [`Type`]; a [`PrimType`] types the primitive scalars and
//! vectors that arithmetic works on, and a dialect extends `Type` with its own kinds, as
//! [`TensorType`](crate::tirx::TensorType) does.

// tvm-ffi-stubgen(file): mod(missing_docs = allow)
// `custom_new` reserves `new` for a hand-written constructor (defaults, checks, the shared
// `MissingType`, `NoneType` and `TensorRegionType` nodes); the generated `from_fields`
// allocates the fields as given.
// tvm-ffi-stubgen(object): ir.MissingType(custom_new)
// tvm-ffi-stubgen(object): ir.NoneType(custom_new)
// tvm-ffi-stubgen(object): ir.TensorRegionType(custom_new)
// tvm-ffi-stubgen(object): ir.PtrType(custom_new)
// tvm-ffi-stubgen(object): ir.TupleType(custom_new)
// tvm-ffi-stubgen(object): ir.FuncType(custom_new)
// tvm-ffi-stubgen(object): ir.PrimType(custom_new)
// tvm-ffi-stubgen(object): ir.OpaqueType(custom_new)
// tvm-ffi-stubgen(object): ir.StringType(custom_new)
// `no_new` generates no allocator for the abstract base: a value is one of its subtypes.
// tvm-ffi-stubgen(object): ir.Type(no_new)
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import): tvm_ffi::Result
// tvm-ffi-stubgen(import): crate::ir::location::UNKNOWN_LOC
// tvm-ffi-stubgen(import): crate::ir::support
// tvm-ffi-stubgen(import): crate::ir::dtype
// tvm-ffi-stubgen(import): tvm_ffi::Error
// tvm-ffi-stubgen(import): tvm_ffi::DLDataTypeCode
// tvm-ffi-stubgen(import): tvm_ffi::VALUE_ERROR
// tvm-ffi-stubgen(import): tvm_ffi::IntoArg

// tvm-ffi-stubgen(begin): import-section
use crate::ir::dtype;
use crate::ir::location::Location;
use crate::ir::location::UNKNOWN_LOC;
use crate::ir::support;
use crate::runtime::MemoryScope;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::DLDataType;
use tvm_ffi::DLDataTypeCode;
use tvm_ffi::Error;
use tvm_ffi::IntoArg;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::VALUE_ERROR;
// tvm-ffi-stubgen(end)

/// The base of every type; `loc` is the source location the type was written at.
// tvm-ffi-stubgen(begin): object/ir.Type
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Type {
    data_: ObjectArc<TypeObj>,
}

/// The object behind [`Type`].
// Complete: reflected fields fill [24, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.Type", immutable, rs_parent)]
pub struct TypeObj {
    base_: Object,
    pub loc: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<TypeObj>() == 32);
    assert!(::core::mem::align_of::<TypeObj>() == 8);
};

#[allow(missing_docs)]
impl TypeObj {
    pub fn new(loc: Location) -> Self {
        let base_ = Object::new();
        Self { base_, loc }
    }
}
// tvm-ffi-stubgen(end)

/// The type of a node whose type has not been given or inferred yet.
///
/// It is a placeholder, not a type of any value: unlike [`AnyType`], it must be resolved before
/// the IR is fully typed.
// tvm-ffi-stubgen(begin): object/ir.MissingType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct MissingType {
    data_: ObjectArc<MissingTypeObj>,
}

/// The object behind [`MissingType`].
// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.MissingType", final, immutable)]
pub struct MissingTypeObj {
    base_: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<MissingTypeObj>() == 32);
    assert!(::core::mem::align_of::<MissingTypeObj>() == 8);
};

impl Deref for MissingTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl MissingTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl MissingType {
    /// Create a `MissingType` from its field values.
    pub fn from_fields(loc: impl IntoArg<Location>) -> Self {
        let obj = MissingTypeObj::new(loc.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(MissingType => Type);
// tvm-ffi-stubgen(end)

impl MissingType {
    /// Return the shared type of a node whose type is not known yet, fetched once per thread
    /// (`ir.MissingType`).
    // Not `Default`: `new` fetches a shared FFI singleton rather than building a value.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        thread_local! {
            static MISSING_TYPE: std::cell::OnceCell<MissingType> = const { std::cell::OnceCell::new() };
        }
        support::shared(&MISSING_TYPE, || {
            tvm_ffi::cached_global_func!("ir.MissingType").call_tuple(())?.try_into()
        })
    }
}

/// The type whose only value is `None`.
// tvm-ffi-stubgen(begin): object/ir.NoneType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct NoneType {
    data_: ObjectArc<NoneTypeObj>,
}

/// The object behind [`NoneType`].
// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.NoneType", final, immutable)]
pub struct NoneTypeObj {
    base_: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<NoneTypeObj>() == 32);
    assert!(::core::mem::align_of::<NoneTypeObj>() == 8);
};

impl Deref for NoneTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl NoneTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl NoneType {
    /// Create a `NoneType` from its field values.
    pub fn from_fields(loc: impl IntoArg<Location>) -> Self {
        let obj = NoneTypeObj::new(loc.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(NoneType => Type);
// tvm-ffi-stubgen(end)

impl NoneType {
    /// Return the shared type of the `None` value, fetched once per thread (`ir.NoneType`).
    // Not `Default`: `new` fetches a shared FFI singleton rather than building a value.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        thread_local! {
            static NONE_TYPE: std::cell::OnceCell<NoneType> = const { std::cell::OnceCell::new() };
        }
        support::shared(&NONE_TYPE, || {
            tvm_ffi::cached_global_func!("ir.NoneType").call_tuple(())?.try_into()
        })
    }
}

/// The top type, which admits any value: the type of a value with no more specific type, such
/// as a data type constant ([`DataTypeImm`](crate::ir::DataTypeImm)).
// tvm-ffi-stubgen(begin): object/ir.AnyType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AnyType {
    data_: ObjectArc<AnyTypeObj>,
}

/// The object behind [`AnyType`].
// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.AnyType", final, immutable)]
pub struct AnyTypeObj {
    base_: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<AnyTypeObj>() == 32);
    assert!(::core::mem::align_of::<AnyTypeObj>() == 8);
};

impl Deref for AnyTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl AnyTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl AnyType {
    /// Create a `AnyType` from its field values.
    pub fn new(loc: impl IntoArg<Location>) -> Self {
        let obj = AnyTypeObj::new(loc.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(AnyType => Type);
// tvm-ffi-stubgen(end)

// `ir.AnyType` stores its location as given: the generated `AnyType::new` is the same.

/// The type of a [`TensorRegion`](crate::ir::TensorRegion), a multi-dimensional region of a
/// tensor.
// tvm-ffi-stubgen(begin): object/ir.TensorRegionType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorRegionType {
    data_: ObjectArc<TensorRegionTypeObj>,
}

/// The object behind [`TensorRegionType`].
// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.TensorRegionType", final, immutable)]
pub struct TensorRegionTypeObj {
    base_: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorRegionTypeObj>() == 32);
    assert!(::core::mem::align_of::<TensorRegionTypeObj>() == 8);
};

impl Deref for TensorRegionTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl TensorRegionTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl TensorRegionType {
    /// Create a `TensorRegionType` from its field values.
    pub fn from_fields(loc: impl IntoArg<Location>) -> Self {
        let obj = TensorRegionTypeObj::new(loc.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(TensorRegionType => Type);
// tvm-ffi-stubgen(end)

impl TensorRegionType {
    /// Return the shared type of a tensor region, fetched once per thread (`ir.TensorRegionType`).
    // Not `Default`: `new` fetches a shared FFI singleton rather than building a value.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        thread_local! {
            static TENSOR_REGION_TYPE: std::cell::OnceCell<TensorRegionType> =
                const { std::cell::OnceCell::new() };
        }
        support::shared(&TENSOR_REGION_TYPE, || {
            tvm_ffi::cached_global_func!("ir.TensorRegionType").call_tuple(())?.try_into()
        })
    }
}

/// The type of opaque construction-time values, such as an
/// [`OpaqueExpr`](crate::ir::OpaqueExpr), which must be removed before the IR is finished.
// tvm-ffi-stubgen(begin): object/ir.OpaqueType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct OpaqueType {
    data_: ObjectArc<OpaqueTypeObj>,
}

/// The object behind [`OpaqueType`].
// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.OpaqueType", final, immutable)]
pub struct OpaqueTypeObj {
    base_: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<OpaqueTypeObj>() == 32);
    assert!(::core::mem::align_of::<OpaqueTypeObj>() == 8);
};

impl Deref for OpaqueTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl OpaqueTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl OpaqueType {
    /// Create a `OpaqueType` from its field values.
    pub fn from_fields(loc: impl IntoArg<Location>) -> Self {
        let obj = OpaqueTypeObj::new(loc.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(OpaqueType => Type);
// tvm-ffi-stubgen(end)

impl OpaqueType {
    /// Create an opaque type at the unknown location.
    // Not `Default`, as no IR node is.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::from_fields(UNKNOWN_LOC)
    }
}

/// A primitive type: a scalar or a vector of a DLPack data type `dtype`, or `void`.
///
/// It types the operands of arithmetic. [`PrimType::int`], [`PrimType::float`] and their
/// siblings give the common scalars, [`with_lanes`](PrimType::with_lanes) and
/// [`with_scalable_lanes`](PrimType::with_scalable_lanes) the vectors.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::PrimType;
/// use tvm_tirx::script::script;
///
/// let half4 = PrimType::float(16).with_lanes(4)?;
/// assert_eq!(half4.lanes(), Some(4));
/// assert_eq!(script(&half4)?, "T.float16x4");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.PrimType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PrimType {
    data_: ObjectArc<PrimTypeObj>,
}

/// The object behind [`PrimType`].
// Complete: reflected fields fill [32, 40) exactly (alignment padding [36, 40)).
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.PrimType", final, immutable)]
pub struct PrimTypeObj {
    base_: TypeObj,
    pub dtype: DLDataType,
}

const _: () = {
    assert!(::core::mem::size_of::<PrimTypeObj>() == 40);
    assert!(::core::mem::align_of::<PrimTypeObj>() == 8);
};

impl Deref for PrimTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl PrimTypeObj {
    pub(crate) fn new(loc: Location, dtype: DLDataType) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_, dtype }
    }
}

#[allow(missing_docs)]
impl PrimType {
    /// Create a `PrimType` from its field values.
    pub fn from_fields(loc: impl IntoArg<Location>, dtype: impl Into<DLDataType>) -> Self {
        let obj = PrimTypeObj::new(loc.into_owned(), dtype.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(PrimType => Type);
// tvm-ffi-stubgen(end)

/// The primitive type of a dtype, checked to be a valid primitive type (`ir.PrimType`).
///
/// # Errors
///
/// Returns a `TypeError` if the dtype is an opaque handle other than `void`: a pointer is a
/// [`PtrType`].
impl TryFrom<DLDataType> for PrimType {
    type Error = Error;

    fn try_from(dtype: DLDataType) -> Result<Self> {
        thread_local! {
            static PRIM_TYPES: std::cell::RefCell<std::collections::HashMap<(u8, u8, u16), PrimType>> =
                std::cell::RefCell::new(std::collections::HashMap::new());
        }
        let key = (dtype.code, dtype.bits, dtype.lanes);
        if let Some(ty) = PRIM_TYPES.with(|types| types.borrow().get(&key).cloned()) {
            return Ok(ty);
        }
        let ty: Self =
            tvm_ffi::cached_global_func!("ir.PrimType").call_tuple((dtype,))?.try_into()?;
        PRIM_TYPES.with(|types| types.borrow_mut().insert(key, ty.clone()));
        Ok(ty)
    }
}

impl PrimType {
    /// The primitive type of a dtype that is always valid, which fails only when the compiler
    /// library is not loaded.
    fn builtin(dtype: DLDataType) -> Self {
        support::infallible(Self::try_from(dtype))
    }

    /// Return the scalar signed integer type of `bits` bits.
    pub fn int(bits: u8) -> Self {
        Self::builtin(dtype::int(bits))
    }

    /// Return the scalar unsigned integer type of `bits` bits.
    pub fn uint(bits: u8) -> Self {
        Self::builtin(dtype::uint(bits))
    }

    /// Return the scalar IEEE floating-point type of `bits` bits.
    pub fn float(bits: u8) -> Self {
        Self::builtin(dtype::float(bits))
    }

    /// Return the scalar brain floating-point type of `bits` bits.
    pub fn bfloat(bits: u8) -> Self {
        Self::builtin(dtype::bfloat(bits))
    }

    /// Return the scalar boolean type.
    pub fn bool() -> Self {
        Self::builtin(dtype::bool())
    }

    /// Return the void type.
    pub fn void() -> Self {
        Self::builtin(dtype::void())
    }

    /// Return the vector of `lanes` elements of this scalar type.
    ///
    /// [`with_scalable_lanes`](Self::with_scalable_lanes) builds the scalable vectors.
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `lanes` is 32768 or more, or a `TypeError` if the vector is of an
    /// opaque handle, as that of `void` is.
    pub fn with_lanes(&self, lanes: u16) -> Result<Self> {
        if i16::try_from(lanes).is_err() {
            let message = format!("a fixed-length vector has fewer than 32768 lanes, not {lanes}");
            return Err(Error::new(VALUE_ERROR, &message));
        }
        Self::try_from(dtype::with_lanes(self.dtype, lanes))
    }

    /// Return the scalable vector of `factor` times `vscale` elements of this scalar type.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `factor` is 1 or less, or 32768 or more, or a `TypeError` if
    /// the vector is of an opaque handle.
    pub fn with_scalable_lanes(&self, factor: u16) -> Result<Self> {
        Self::try_from(support::scalable_vector(self.dtype, i32::from(factor))?)
    }

    /// Return this type with the type code `code`, keeping the bits and the lanes.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the result is an opaque handle other than `void`.
    pub fn with_code(&self, code: DLDataTypeCode) -> Result<Self> {
        Self::try_from(DLDataType { code: code as u8, ..self.dtype })
    }

    /// Return this type with elements of `bits` bits, keeping the type code and the lanes.
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the result is an opaque handle other than `void`.
    pub fn with_bits(&self, bits: u8) -> Result<Self> {
        Self::try_from(DLDataType { bits, ..self.dtype })
    }

    /// Return whether the type code is any of `codes`; the bits and the lanes are ignored.
    #[inline(always)]
    pub fn matches_code(&self, codes: &[DLDataTypeCode]) -> bool {
        codes.iter().any(|&code| self.dtype.code == code as u8)
    }

    /// The bit width of one element.
    #[inline(always)]
    pub fn bits(&self) -> u8 {
        self.dtype.bits
    }

    /// Return whether this type is a scalar: neither a fixed-length nor a scalable vector, nor
    /// `void`.
    #[inline(always)]
    pub fn is_scalar(&self) -> bool {
        self.encoded_lanes() == 1
    }

    /// Return whether this type is `void`, the opaque handle of no bits and no lanes.
    #[inline(always)]
    pub fn is_void(&self) -> bool {
        self.dtype == dtype::void()
    }

    /// Return whether this type is a scalable vector, of a multiple of `vscale` lanes.
    ///
    /// See also [`with_scalable_lanes`](Self::with_scalable_lanes).
    #[inline(always)]
    pub fn is_scalable_vector(&self) -> bool {
        self.encoded_lanes() < -1
    }

    /// Return whether this type is a vector of a fixed number of lanes.
    ///
    /// See also [`with_lanes`](Self::with_lanes).
    #[inline(always)]
    pub fn is_fixed_length_vector(&self) -> bool {
        self.encoded_lanes() > 1
    }

    /// Return the number of lanes, 1 for a scalar, or `None` for a scalable vector, whose lanes
    /// are known only at run time.
    #[inline(always)]
    pub fn lanes(&self) -> Option<u16> {
        u16::try_from(self.encoded_lanes()).ok()
    }

    /// Return the factor of `vscale` in the lanes of a scalable vector, or `None` for any other
    /// type.
    #[inline(always)]
    pub fn vscale_factor(&self) -> Option<u16> {
        match self.is_scalable_vector() {
            true => Some(self.encoded_lanes().unsigned_abs()),
            false => None,
        }
    }

    /// Return the number of bytes that store a value of this type, its sub-byte elements packed,
    /// or `None` for a scalable vector.
    #[inline(always)]
    pub fn storage_bytes(&self) -> Option<usize> {
        let lanes = usize::from(self.lanes()?);
        Some((usize::from(self.bits()) * lanes).div_ceil(8))
    }

    /// The lanes field read as signed: negative for a scalable vector.
    #[inline(always)]
    fn encoded_lanes(&self) -> i16 {
        self.dtype.lanes as i16
    }
}

/// The type of a string value, such as a [`StringImm`](crate::ir::StringImm); how a string is
/// stored depends on where it is used.
// tvm-ffi-stubgen(begin): object/ir.StringType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StringType {
    data_: ObjectArc<StringTypeObj>,
}

/// The object behind [`StringType`].
// Complete: reflected fields fill [32, 32) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.StringType", final, immutable)]
pub struct StringTypeObj {
    base_: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<StringTypeObj>() == 32);
    assert!(::core::mem::align_of::<StringTypeObj>() == 8);
};

impl Deref for StringTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl StringTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl StringType {
    /// Create a `StringType` from its field values.
    pub fn from_fields(loc: impl IntoArg<Location>) -> Self {
        let obj = StringTypeObj::new(loc.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(StringType => Type);
// tvm-ffi-stubgen(end)

impl StringType {
    /// Create the string type at the unknown location.
    // Not `Default`, as no IR node is.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::from_fields(UNKNOWN_LOC)
    }
}

/// The type of a pointer to values of `element_type` in the memory `scope`, a low-level type
/// for code generation.
// tvm-ffi-stubgen(begin): object/ir.PtrType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PtrType {
    data_: ObjectArc<PtrTypeObj>,
}

/// The object behind [`PtrType`].
// Complete: reflected fields fill [32, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.PtrType", final, immutable)]
pub struct PtrTypeObj {
    base_: TypeObj,
    pub element_type: Type,
    pub scope: MemoryScope,
}

const _: () = {
    assert!(::core::mem::size_of::<PtrTypeObj>() == 48);
    assert!(::core::mem::align_of::<PtrTypeObj>() == 8);
};

impl Deref for PtrTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl PtrTypeObj {
    pub(crate) fn new(loc: Location, element_type: Type, scope: MemoryScope) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_, element_type, scope }
    }
}

#[allow(missing_docs)]
impl PtrType {
    /// Create a `PtrType` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        element_type: impl IntoArg<Type>,
        scope: impl IntoArg<MemoryScope>,
    ) -> Self {
        let obj = PtrTypeObj::new(loc.into_owned(), element_type.into_owned(), scope.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(PtrType => Type);
// tvm-ffi-stubgen(end)

impl PtrType {
    /// Create the type of a pointer to values of `element_type` in the memory `scope`, at the
    /// unknown location.
    ///
    /// # Errors
    ///
    /// Returns an `InternalError` if `element_type` is a [`MissingType`].
    pub fn new(element_type: impl IntoArg<Type>, scope: impl IntoArg<MemoryScope>) -> Result<Self> {
        let element_type = element_type.into_owned();
        if element_type.downcast_ref::<MissingTypeObj>().is_some() {
            let condition = "!element_type.as<MissingType>().has_value()";
            let message = "PtrType element_type cannot be Type::Missing()";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, message));
        }
        Ok(Self::from_fields(UNKNOWN_LOC, element_type, scope.into_owned()))
    }
}

/// The type of a tuple: the types of its `fields`, in order.
// tvm-ffi-stubgen(begin): object/ir.TupleType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TupleType {
    data_: ObjectArc<TupleTypeObj>,
}

/// The object behind [`TupleType`].
// Complete: reflected fields fill [32, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.TupleType", final, immutable)]
pub struct TupleTypeObj {
    base_: TypeObj,
    pub fields: Array<Type>,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleTypeObj>() == 40);
    assert!(::core::mem::align_of::<TupleTypeObj>() == 8);
};

impl Deref for TupleTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl TupleTypeObj {
    pub(crate) fn new(loc: Location, fields: Array<Type>) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_, fields }
    }
}

#[allow(missing_docs)]
impl TupleType {
    /// Create a `TupleType` from its field values.
    pub fn from_fields(loc: impl IntoArg<Location>, fields: impl Into<Array<Type>>) -> Self {
        let obj = TupleTypeObj::new(loc.into_owned(), fields.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(TupleType => Type);
// tvm-ffi-stubgen(end)

impl TupleType {
    /// Create the type of a tuple whose fields have the types `fields`, at `loc`.
    pub fn new(
        fields: impl IntoIterator<Item = impl IntoArg<Type>>,
        loc: impl IntoArg<Location>,
    ) -> Self {
        let fields = fields.into_iter().map(IntoArg::into_owned).collect::<Array<Type>>();
        Self::from_fields(loc.into_owned(), fields)
    }
}

/// The type of a function from arguments of `arg_types` to a result of `ret_type`.
// tvm-ffi-stubgen(begin): object/ir.FuncType
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FuncType {
    data_: ObjectArc<FuncTypeObj>,
}

/// The object behind [`FuncType`].
// Complete: reflected fields fill [32, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ir.FuncType", final, immutable)]
pub struct FuncTypeObj {
    base_: TypeObj,
    pub arg_types: Array<Type>,
    pub ret_type: Type,
}

const _: () = {
    assert!(::core::mem::size_of::<FuncTypeObj>() == 48);
    assert!(::core::mem::align_of::<FuncTypeObj>() == 8);
};

impl Deref for FuncTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl FuncTypeObj {
    pub(crate) fn new(loc: Location, arg_types: Array<Type>, ret_type: Type) -> Self {
        let base_ = TypeObj::new(loc);
        Self { base_, arg_types, ret_type }
    }
}

#[allow(missing_docs)]
impl FuncType {
    /// Create a `FuncType` from its field values.
    pub fn from_fields(
        loc: impl IntoArg<Location>,
        arg_types: impl Into<Array<Type>>,
        ret_type: impl IntoArg<Type>,
    ) -> Self {
        let obj = FuncTypeObj::new(loc.into_owned(), arg_types.into(), ret_type.into_owned());
        Self { data_: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_ref_upcast!(FuncType => Type);
// tvm-ffi-stubgen(end)

impl FuncType {
    /// Create the type of a function from arguments of `arg_types` to a result of `ret_type`, at
    /// the unknown location.
    pub fn new(
        arg_types: impl IntoIterator<Item = impl IntoArg<Type>>,
        ret_type: impl IntoArg<Type>,
    ) -> Self {
        let arg_types = arg_types.into_iter().map(IntoArg::into_owned).collect::<Array<Type>>();
        Self::from_fields(UNKNOWN_LOC, arg_types, ret_type.into_owned())
    }
}
