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

//! Types: primitive, pointer, tuple, function and the special types.
//!
//! The objects `python/tvm_tirx/ir/type.py` defines. [`Type`] is the base of every type; a dialect
//! extends it, as [`TensorType`](crate::tirx::TensorType) does. The constructors are `new`, written
//! in Rust after the C++ ones; the shared types ([`MissingType::new`], [`NoneType::new`],
//! [`TensorRegionType::new`] and [`PrimType::from_dtype`]) are the nodes C++ shares, fetched once
//! per thread. The common
//! primitive types are [`PrimType::int`], [`PrimType::float`], [`PrimType::bool`] and their
//! siblings, as C++ writes `PrimType::Int(32)`.

// `custom-new` reserves `new` for a constructor with the C++ one's semantics (defaults, the shared
// `MissingType`, `NoneType` and `TensorRegionType` nodes); the generated `from_fields`
// allocates the fields as given.
// tvm-ffi-stubgen(custom-new): ir.MissingType
// tvm-ffi-stubgen(custom-new): ir.NoneType
// tvm-ffi-stubgen(custom-new): ir.TensorRegionType
// tvm-ffi-stubgen(custom-new): ir.PtrType
// tvm-ffi-stubgen(custom-new): ir.TupleType
// tvm-ffi-stubgen(custom-new): ir.FuncType
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::location::UnknownLoc
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::dtype
// tvm-ffi-stubgen(import-object): tvm_ffi::Error
// tvm-ffi-stubgen(import-object): tvm_ffi::VALUE_ERROR

// tvm-ffi-stubgen(begin): import-section
use crate::ir::dtype;
use crate::ir::location::Location;
use crate::ir::location::UnknownLoc;
use crate::ir::support;
use crate::runtime::MemoryScope;
use std::ops::Deref;
use tvm_ffi::Array;
use tvm_ffi::DLDataType;
use tvm_ffi::Error;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::ShallowCopy;
use tvm_ffi::VALUE_ERROR;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Type
/// Complete: reflected fields fill [24, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Type"]
pub struct TypeObj {
    base: Object,
    pub loc: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<TypeObj>() == 32);
    assert!(::core::mem::align_of::<TypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Type {
    base: ObjectArc<TypeObj>,
}

impl Deref for Type {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl TypeObj {
    pub fn new(loc: Location) -> Self {
        let base = Object::new();
        Self { base, loc }
    }
}

impl Type {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = TypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for TypeObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base), loc: Clone::clone(&self.loc) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.MissingType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.MissingType"]
#[type_final]
pub struct MissingTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<MissingTypeObj>() == 32);
    assert!(::core::mem::align_of::<MissingTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct MissingType {
    base: ObjectArc<MissingTypeObj>,
}

impl Deref for MissingType {
    type Target = MissingTypeObj;
    fn deref(&self) -> &MissingTypeObj {
        &self.base
    }
}

impl Deref for MissingTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl MissingTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = TypeObj::new(loc);
        Self { base }
    }
}

impl MissingType {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>) -> Self {
        let obj = MissingTypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(MissingType => Type);
// tvm-ffi-stubgen(end)

impl MissingType {
    /// The shared type of a node whose type is not known yet (`ir.MissingType`), fetched once per
    /// thread.
    pub fn new() -> Self {
        thread_local! {
            static MISSING_TYPE: std::cell::OnceCell<MissingType> = const { std::cell::OnceCell::new() };
        }
        support::shared(&MISSING_TYPE, || {
            tvm_ffi::cached_global_func!("ir.MissingType").call_tuple(())?.try_into()
        })
    }
}

// tvm-ffi-stubgen(begin): object/ir.NoneType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.NoneType"]
#[type_final]
pub struct NoneTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<NoneTypeObj>() == 32);
    assert!(::core::mem::align_of::<NoneTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct NoneType {
    base: ObjectArc<NoneTypeObj>,
}

impl Deref for NoneType {
    type Target = NoneTypeObj;
    fn deref(&self) -> &NoneTypeObj {
        &self.base
    }
}

impl Deref for NoneTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl NoneTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = TypeObj::new(loc);
        Self { base }
    }
}

impl NoneType {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>) -> Self {
        let obj = NoneTypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(NoneType => Type);
// tvm-ffi-stubgen(end)

impl NoneType {
    /// The shared type of the `None` value (`ir.NoneType`).
    pub fn new() -> Self {
        thread_local! {
            static NONE_TYPE: std::cell::OnceCell<NoneType> = const { std::cell::OnceCell::new() };
        }
        support::shared(&NONE_TYPE, || {
            tvm_ffi::cached_global_func!("ir.NoneType").call_tuple(())?.try_into()
        })
    }
}

// tvm-ffi-stubgen(begin): object/ir.AnyType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.AnyType"]
#[type_final]
pub struct AnyTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<AnyTypeObj>() == 32);
    assert!(::core::mem::align_of::<AnyTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct AnyType {
    base: ObjectArc<AnyTypeObj>,
}

impl Deref for AnyType {
    type Target = AnyTypeObj;
    fn deref(&self) -> &AnyTypeObj {
        &self.base
    }
}

impl Deref for AnyTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl AnyTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = TypeObj::new(loc);
        Self { base }
    }
}

impl AnyType {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = AnyTypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(AnyType => Type);
// tvm-ffi-stubgen(end)

// `ir.AnyType` stores its location as given: the generated `AnyType::new` is the same.

// tvm-ffi-stubgen(begin): object/ir.TensorRegionType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorRegionType"]
#[type_final]
pub struct TensorRegionTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorRegionTypeObj>() == 32);
    assert!(::core::mem::align_of::<TensorRegionTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorRegionType {
    base: ObjectArc<TensorRegionTypeObj>,
}

impl Deref for TensorRegionType {
    type Target = TensorRegionTypeObj;
    fn deref(&self) -> &TensorRegionTypeObj {
        &self.base
    }
}

impl Deref for TensorRegionTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl TensorRegionTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = TypeObj::new(loc);
        Self { base }
    }
}

impl TensorRegionType {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>) -> Self {
        let obj = TensorRegionTypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorRegionType => Type);
// tvm-ffi-stubgen(end)

impl TensorRegionType {
    /// The shared type of a tensor region (`ir.TensorRegionType`).
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

// tvm-ffi-stubgen(begin): object/ir.OpaqueType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.OpaqueType"]
#[type_final]
pub struct OpaqueTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<OpaqueTypeObj>() == 32);
    assert!(::core::mem::align_of::<OpaqueTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct OpaqueType {
    base: ObjectArc<OpaqueTypeObj>,
}

impl Deref for OpaqueType {
    type Target = OpaqueTypeObj;
    fn deref(&self) -> &OpaqueTypeObj {
        &self.base
    }
}

impl Deref for OpaqueTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl OpaqueTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = TypeObj::new(loc);
        Self { base }
    }
}

impl OpaqueType {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = OpaqueTypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(OpaqueType => Type);
// tvm-ffi-stubgen(end)

// `ir.OpaqueType` allocates a fresh type at the unknown location: the generated
// `OpaqueType::new(Location::unknown())` is the same.

// tvm-ffi-stubgen(begin): object/ir.PrimType
/// Complete: reflected fields fill [32, 40) exactly (alignment padding [36, 40)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.PrimType"]
#[type_final]
pub struct PrimTypeObj {
    base: TypeObj,
    pub dtype: DLDataType,
}

const _: () = {
    assert!(::core::mem::size_of::<PrimTypeObj>() == 40);
    assert!(::core::mem::align_of::<PrimTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PrimType {
    base: ObjectArc<PrimTypeObj>,
}

impl Deref for PrimType {
    type Target = PrimTypeObj;
    fn deref(&self) -> &PrimTypeObj {
        &self.base
    }
}

impl Deref for PrimTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl PrimTypeObj {
    pub(crate) fn new(loc: Location, dtype: DLDataType) -> Self {
        let base = TypeObj::new(loc);
        Self { base, dtype }
    }
}

impl PrimType {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, dtype: impl Into<DLDataType>) -> Self {
        let obj = PrimTypeObj::new(loc.into(), dtype.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(PrimType => Type);
// tvm-ffi-stubgen(end)

impl PrimType {
    /// The primitive type of `dtype`, checked to be a valid primitive type (`ir.PrimType`). C++
    /// keeps one node per dtype and thread, so the node of each dtype is fetched once per thread.
    pub fn from_dtype(dtype: DLDataType) -> Result<Self> {
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

    /// [`PrimType::from_dtype`] of a dtype C++ always accepts, which fails only when the compiler
    /// library is not loaded.
    fn builtin(dtype: DLDataType) -> Self {
        support::infallible(Self::from_dtype(dtype))
    }

    /// The signed integer type of `bits` bits (C++ `PrimType::Int(bits)`).
    pub fn int(bits: u8) -> Self {
        Self::builtin(dtype::int(bits))
    }

    /// The unsigned integer type of `bits` bits (C++ `PrimType::UInt(bits)`).
    pub fn uint(bits: u8) -> Self {
        Self::builtin(dtype::uint(bits))
    }

    /// The IEEE floating-point type of `bits` bits (C++ `PrimType::Float(bits)`).
    pub fn float(bits: u8) -> Self {
        Self::builtin(dtype::float(bits))
    }

    /// The brain floating-point type of `bits` bits (C++ `PrimType::BFloat(bits)`).
    pub fn bfloat(bits: u8) -> Self {
        Self::builtin(dtype::bfloat(bits))
    }

    /// The boolean type (C++ `PrimType::Bool()`).
    pub fn bool() -> Self {
        Self::builtin(dtype::bool())
    }

    /// The void type (C++ `PrimType::Void()`).
    pub fn void() -> Self {
        Self::builtin(dtype::void())
    }

    /// The vector of `lanes` elements of this scalar type (C++ `PrimType::Int(bits, lanes)` and
    /// `WithLanes(lanes)`), with the C++ checks of the result: `PrimType::float(16).with_lanes(4)`
    /// is `float16x4`, and `PrimType::void().with_lanes(4)` is an error. `lanes` is a fixed
    /// count, below 32768; [`with_scalable_lanes`](Self::with_scalable_lanes) builds the scalable
    /// vectors.
    pub fn with_lanes(&self, lanes: u16) -> Result<Self> {
        if i16::try_from(lanes).is_err() {
            let message = format!("a fixed-length vector has fewer than 32768 lanes, not {lanes}");
            return Err(Error::new(VALUE_ERROR, &message, ""));
        }
        Self::from_dtype(dtype::with_lanes(self.dtype, lanes))
    }

    /// The scalable vector of `factor` times `vscale` elements of this scalar type, with the C++
    /// checks of the factor (C++ `PrimType::ScalableVector(code, bits, factor)`).
    pub fn with_scalable_lanes(&self, factor: u16) -> Result<Self> {
        Self::from_dtype(support::scalable_vector(self.dtype, i32::from(factor))?)
    }
}

// tvm-ffi-stubgen(begin): object/ir.StringType
/// Complete: reflected fields fill [32, 32) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.StringType"]
#[type_final]
pub struct StringTypeObj {
    base: TypeObj,
}

const _: () = {
    assert!(::core::mem::size_of::<StringTypeObj>() == 32);
    assert!(::core::mem::align_of::<StringTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StringType {
    base: ObjectArc<StringTypeObj>,
}

impl Deref for StringType {
    type Target = StringTypeObj;
    fn deref(&self) -> &StringTypeObj {
        &self.base
    }
}

impl Deref for StringTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl StringTypeObj {
    pub(crate) fn new(loc: Location) -> Self {
        let base = TypeObj::new(loc);
        Self { base }
    }
}

impl StringType {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>) -> Self {
        let obj = StringTypeObj::new(loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(StringType => Type);
// tvm-ffi-stubgen(end)

// `ir.StringType` allocates a fresh type at the unknown location: the generated
// `StringType::new(Location::unknown())` is the same.

// tvm-ffi-stubgen(begin): object/ir.PtrType
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.PtrType"]
#[type_final]
pub struct PtrTypeObj {
    base: TypeObj,
    pub element_type: Type,
    pub scope: MemoryScope,
}

const _: () = {
    assert!(::core::mem::size_of::<PtrTypeObj>() == 48);
    assert!(::core::mem::align_of::<PtrTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PtrType {
    base: ObjectArc<PtrTypeObj>,
}

impl Deref for PtrType {
    type Target = PtrTypeObj;
    fn deref(&self) -> &PtrTypeObj {
        &self.base
    }
}

impl Deref for PtrTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl PtrTypeObj {
    pub(crate) fn new(loc: Location, element_type: Type, scope: MemoryScope) -> Self {
        let base = TypeObj::new(loc);
        Self { base, element_type, scope }
    }
}

impl PtrType {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        element_type: impl Into<Type>,
        scope: impl Into<MemoryScope>,
    ) -> Self {
        let obj = PtrTypeObj::new(loc.into(), element_type.into(), scope.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(PtrType => Type);
// tvm-ffi-stubgen(end)

impl PtrType {
    /// A pointer to `element_type` in the memory `scope`; the element type must not be missing
    /// (C++ `PtrType(element_type, scope)`).
    pub fn new(element_type: impl Into<Type>, scope: &MemoryScope) -> Result<Self> {
        let element_type = element_type.into();
        if element_type.as_node::<MissingTypeObj>().is_some() {
            let condition = "!element_type.as<MissingType>().has_value()";
            let message = "PtrType element_type cannot be Type::Missing()";
            return Err(support::check_failed(tvm_ffi::INTERNAL_ERROR, condition, message));
        }
        Ok(Self::from_fields(UnknownLoc::new(), element_type, scope.clone()))
    }
}

// tvm-ffi-stubgen(begin): object/ir.TupleType
/// Complete: reflected fields fill [32, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TupleType"]
#[type_final]
pub struct TupleTypeObj {
    base: TypeObj,
    pub fields: Array<Type>,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleTypeObj>() == 40);
    assert!(::core::mem::align_of::<TupleTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TupleType {
    base: ObjectArc<TupleTypeObj>,
}

impl Deref for TupleType {
    type Target = TupleTypeObj;
    fn deref(&self) -> &TupleTypeObj {
        &self.base
    }
}

impl Deref for TupleTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl TupleTypeObj {
    pub(crate) fn new(loc: Location, fields: Array<Type>) -> Self {
        let base = TypeObj::new(loc);
        Self { base, fields }
    }
}

impl TupleType {
    /// Lossless complete-field allocation.
    pub fn from_fields(loc: impl Into<Location>, fields: impl Into<Array<Type>>) -> Self {
        let obj = TupleTypeObj::new(loc.into(), fields.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TupleType => Type);
// tvm-ffi-stubgen(end)

impl TupleType {
    /// The type of a tuple of `fields` (C++ `TupleType(fields, loc)`).
    pub fn new(
        fields: impl IntoIterator<Item = impl Into<Type>>,
        loc: impl Into<Location>,
    ) -> Self {
        let fields = fields.into_iter().map(Into::into).collect::<Array<Type>>();
        Self::from_fields(loc.into(), fields)
    }
}

// tvm-ffi-stubgen(begin): object/ir.FuncType
/// Complete: reflected fields fill [32, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.FuncType"]
#[type_final]
pub struct FuncTypeObj {
    base: TypeObj,
    pub arg_types: Array<Type>,
    pub ret_type: Type,
}

const _: () = {
    assert!(::core::mem::size_of::<FuncTypeObj>() == 48);
    assert!(::core::mem::align_of::<FuncTypeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FuncType {
    base: ObjectArc<FuncTypeObj>,
}

impl Deref for FuncType {
    type Target = FuncTypeObj;
    fn deref(&self) -> &FuncTypeObj {
        &self.base
    }
}

impl Deref for FuncTypeObj {
    type Target = TypeObj;
    fn deref(&self) -> &TypeObj {
        &self.base
    }
}

impl FuncTypeObj {
    pub(crate) fn new(loc: Location, arg_types: Array<Type>, ret_type: Type) -> Self {
        let base = TypeObj::new(loc);
        Self { base, arg_types, ret_type }
    }
}

impl FuncType {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        arg_types: impl Into<Array<Type>>,
        ret_type: impl Into<Type>,
    ) -> Self {
        let obj = FuncTypeObj::new(loc.into(), arg_types.into(), ret_type.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FuncType => Type);
// tvm-ffi-stubgen(end)

impl FuncType {
    /// The type of a function from `arg_types` to `ret_type` (C++ `FuncType(arg_types, ret_type)`).
    pub fn new(
        arg_types: impl IntoIterator<Item = impl Into<Type>>,
        ret_type: impl Into<Type>,
    ) -> Self {
        let arg_types = arg_types.into_iter().map(Into::into).collect::<Array<Type>>();
        Self::from_fields(UnknownLoc::new(), arg_types, ret_type.into())
    }
}
