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

//! Expressions: variables, constants, calls, tuples and tensor accesses.
//!
//! The objects `python/tvm_tirx/ir/expr.py` defines. [`Expr`] is the base of every expression and
//! carries its type and location. The primitive arithmetic and vector expressions are in
//! [`prim`](super::prim). The constructors are `new`, or `from_dtype` and the like where `new`
//! allocates the fields as given; they are written in Rust after the C++ constructors, with their
//! checks, defaults and errors. The constants of the common types are [`IntImm::int32`],
//! [`IntImm::int64`], [`IntImm::bool`], [`FloatImm::float32`] and [`FloatImm::float64`], and an
//! `i32`, `i64`, `bool`, `f32` or `f64` converts into an [`Expr`] as C++ converts it implicitly.

// `custom-new` reserves `new` for a constructor with the C++ one's semantics (defaults, implicit
// conversions, a derived type); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): ir.GlobalVar
// tvm-ffi-stubgen(custom-new): ir.Tuple
// tvm-ffi-stubgen(custom-new): ir.TensorLoad
// tvm-ffi-stubgen(custom-new): ir.StringImm
// tvm-ffi-stubgen(custom-new): ir.Call
// tvm-ffi-stubgen(custom-new): ir.Var
// tvm-ffi-stubgen(custom-new): ir.LambdaExpr
// tvm-ffi-stubgen(custom-new): ir.Range
// Constants and loads are typed by a `PrimType`: their allocators take one, and they convert into
// a `PrimExpr` without a check.
// tvm-ffi-stubgen(field): ir.IntImm.ty -> PrimType
// tvm-ffi-stubgen(field): ir.FloatImm.ty -> PrimType
// tvm-ffi-stubgen(field): ir.TensorLoad.ty -> PrimType
// tvm-ffi-stubgen(field): ir.TensorLoad.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(upcast): ir.IntImm -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.FloatImm -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.TensorLoad -> PrimExpr
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::prim::op::ExprResult
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimTypeObj
// tvm-ffi-stubgen(import-object): tvm_ffi::Error
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCast
// tvm-ffi-stubgen(import-object): tvm_ffi::object::is_instance_of
// tvm-ffi-stubgen(import-object): tvm_ffi::tvm_ffi_sys::TVMFFIAny
// tvm-ffi-stubgen(import-object): tvm_ffi::type_traits::AnyCompatible
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex
// tvm-ffi-stubgen(import-object): crate::ir::location::UnknownLoc
// tvm-ffi-stubgen(import-object): crate::ir::op::OpObj
// tvm-ffi-stubgen(import-object): crate::ir::prim::AddObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::AnyType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::FuncType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::FuncTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::MissingType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::MissingTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::r#type::PrimType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::StringType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::TupleType
// tvm-ffi-stubgen(import-object): crate::ir::r#type::TupleTypeObj
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::dtype

// tvm-ffi-stubgen(begin): import-section
use crate::ir::attrs::Attrs;
use crate::ir::dtype;
use crate::ir::location::Location;
use crate::ir::location::UnknownLoc;
use crate::ir::op::OpObj;
use crate::ir::prim::AddObj;
use crate::ir::prim::op::ExprResult;
use crate::ir::r#type::AnyType;
use crate::ir::r#type::FuncType;
use crate::ir::r#type::FuncTypeObj;
use crate::ir::r#type::MissingType;
use crate::ir::r#type::MissingTypeObj;
use crate::ir::r#type::PrimType;
use crate::ir::r#type::PrimTypeObj;
use crate::ir::r#type::StringType;
use crate::ir::r#type::TupleType;
use crate::ir::r#type::TupleTypeObj;
use crate::ir::r#type::Type;
use crate::ir::support;
use std::ops::Deref;
use std::ptr::addr_of_mut;
use tvm_ffi::Any;
use tvm_ffi::AnyView;
use tvm_ffi::Array;
use tvm_ffi::BigInt;
use tvm_ffi::DLDataType;
use tvm_ffi::Error;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectRefCast;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::ShallowCopy;
use tvm_ffi::String;
use tvm_ffi::UnsafeInit;
use tvm_ffi::object::is_instance_of;
use tvm_ffi::reflection::drop_if_defined;
use tvm_ffi::tvm_ffi_sys::TVMFFIAny;
use tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex;
use tvm_ffi::type_traits::AnyCompatible;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.Expr
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: Object,
    pub loc: Location,
    pub ty: Type,
}

const _: () = {
    assert!(::core::mem::size_of::<ExprObj>() == 40);
    assert!(::core::mem::align_of::<ExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Expr {
    base: ObjectArc<ExprObj>,
}

impl Deref for Expr {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ExprObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = Object::new();
        Self { base, loc, ty }
    }
}

impl Expr {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = ExprObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for ExprObj {
    fn shallow_copy(&self) -> Self {
        Self {
            base: ShallowCopy::shallow_copy(&self.base),
            loc: Clone::clone(&self.loc),
            ty: Clone::clone(&self.ty),
        }
    }
}

unsafe impl UnsafeInit for ExprObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base));
            drop_if_defined(addr_of_mut!((*this).loc));
            drop_if_defined(addr_of_mut!((*this).ty));
        }
    }
}
// tvm-ffi-stubgen(end)

/// One index of [`Expr::subscript`]: an expression, or a `start:stop:step` slice whose missing
/// bounds are `None`.
#[derive(Clone)]
pub enum Subscript {
    Index(Expr),
    Slice(Option<Expr>, Option<Expr>, Option<Expr>),
}

impl From<&Subscript> for Any {
    fn from(subscript: &Subscript) -> Self {
        match subscript {
            Subscript::Index(index) => Any::from(index.clone()),
            Subscript::Slice(start, stop, step) => Any::from(
                [start, stop, step]
                    .into_iter()
                    .map(|bound| Any::from(bound.clone()))
                    .collect::<Array<Any>>(),
            ),
        }
    }
}

impl Expr {
    /// `self[indices]`, as the subscript rule of its type realizes it (`ir.SubscriptExprRealize`).
    pub fn subscript(
        &self,
        indices: impl IntoIterator<Item = Subscript>,
        loc: impl Into<Location>,
    ) -> Result<Any> {
        let indices = indices.into_iter().map(|index| Any::from(&index)).collect::<Array<Any>>();
        tvm_ffi::cached_global_func!("ir.SubscriptExprRealize").call_tuple((
            self,
            &indices,
            &loc.into(),
        ))
    }
}

// tvm-ffi-stubgen(begin): object/ir.StagingExpr
/// Complete: reflected fields fill [40, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.StagingExpr"]
pub struct StagingExprObj {
    base: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<StagingExprObj>() == 40);
    assert!(::core::mem::align_of::<StagingExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StagingExpr {
    base: ObjectArc<StagingExprObj>,
}

impl Deref for StagingExpr {
    type Target = StagingExprObj;
    fn deref(&self) -> &StagingExprObj {
        &self.base
    }
}

impl Deref for StagingExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl StagingExprObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl StagingExpr {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = StagingExprObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for StagingExprObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}

unsafe impl UnsafeInit for StagingExprObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base));
        }
    }
}

tvm_ffi::impl_object_upcast!(StagingExpr => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.OpaqueExpr
/// Complete: reflected fields fill [40, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.OpaqueExpr"]
pub struct OpaqueExprObj {
    base: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<OpaqueExprObj>() == 40);
    assert!(::core::mem::align_of::<OpaqueExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct OpaqueExpr {
    base: ObjectArc<OpaqueExprObj>,
}

impl Deref for OpaqueExpr {
    type Target = OpaqueExprObj;
    fn deref(&self) -> &OpaqueExprObj {
        &self.base
    }
}

impl Deref for OpaqueExprObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl OpaqueExprObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl OpaqueExpr {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = OpaqueExprObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for OpaqueExprObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}

unsafe impl UnsafeInit for OpaqueExprObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base));
        }
    }
}

tvm_ffi::impl_object_upcast!(OpaqueExpr => Expr);
// tvm-ffi-stubgen(end)

// ---------------------------------------------------------------------------------------------
// Typed views: `PrimExpr` and `IntExpr` (C++ `TypedExpr`)

/// Define a view: a reference to an `ExprObj` whose type `$accepts` accepts, named `$type_str`
/// like the C++ type, which an `Any` holding no expression converts into by `$fallback`.
macro_rules! typed_expr_view {
    (
        $(#[$meta:meta])*
        $name:ident, $type_str:literal, accepts = $accepts:path, fallback = $fallback:path
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name {
            base: ObjectArc<ExprObj>,
        }

        // SAFETY: `base` is the reference; `from_data` requires its caller to have checked it.
        unsafe impl ObjectRefCore for $name {
            type ContainerType = ExprObj;

            #[inline]
            fn data(this: &Self) -> &ObjectArc<ExprObj> {
                &this.base
            }

            #[inline]
            fn into_data(this: Self) -> ObjectArc<ExprObj> {
                this.base
            }

            /// The view of `data`.
            ///
            /// # Safety
            ///
            /// The type of `data` is one the view accepts.
            #[inline]
            unsafe fn from_data(data: ObjectArc<ExprObj>) -> Self {
                Self { base: data }
            }
        }

        // SAFETY: a view is an `Expr` whose `check_any_strict` accepts fewer values; the other
        // conversions are the `Expr` ones.
        unsafe impl AnyCompatible for $name {
            #[inline]
            unsafe fn is_undefined(field: *const Self) -> bool {
                unsafe { ObjectArc::is_null(::core::ptr::addr_of!((*field).base)) }
            }

            fn type_str() -> std::string::String {
                $type_str.into()
            }

            fn get_mismatch_type_info(data: &TVMFFIAny) -> std::string::String {
                typed_expr_mismatch(data)
            }

            #[inline(always)]
            unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
                unsafe { <Expr as AnyCompatible>::copy_to_any_view(src.as_expr(), data) }
            }

            #[inline(always)]
            unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
                unsafe { <Expr as AnyCompatible>::move_to_any(src.into(), data) }
            }

            #[inline(always)]
            unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
                unsafe { prim_dtype(data) }.is_some_and($accepts)
            }

            unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
                let expr = unsafe { <Expr as AnyCompatible>::copy_from_any_view_after_check(data) };
                Self { base: ObjectRefCore::into_data(expr) }
            }

            unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
                let expr = unsafe { <Expr as AnyCompatible>::move_from_any_after_check(data) };
                Self { base: ObjectRefCore::into_data(expr) }
            }

            unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
                unsafe {
                    if Self::check_any_strict(data) {
                        Ok(Self::copy_from_any_view_after_check(data))
                    } else {
                        $fallback(data).ok_or(())
                    }
                }
            }
        }

        tvm_ffi::impl_try_from_any!($name);
        tvm_ffi::impl_arg_into_ref!($name);
        tvm_ffi::impl_into_arg_holder_default!($name);
        crate::def_expr_op_over_loads!($name);

        impl From<&$name> for $name {
            fn from(value: &$name) -> Self {
                value.clone()
            }
        }

        /// The expression object, to read, as [`Expr`] dereferences to it.
        impl Deref for $name {
            type Target = ExprObj;

            fn deref(&self) -> &ExprObj {
                &self.base
            }
        }

        impl $name {
            /// This expression as an [`Expr`], borrowed.
            #[inline]
            pub fn as_expr(&self) -> &Expr {
                // SAFETY: `Expr` is a `repr(C)` struct of the one `ObjectArc<ExprObj>` the view
                // holds, and every value of the view is an `Expr`.
                unsafe { &*(self as *const Self as *const Expr) }
            }

            /// The type of this expression (C++ `PrimExpr::ty()`), which the view guarantees.
            pub fn ty(&self) -> PrimType {
                self.base.ty.clone().try_cast().expect("a view is typed by a `PrimType`")
            }

            /// The dtype of this expression's type.
            pub fn dtype(&self) -> DLDataType {
                let ty = self.base.ty.downcast_ref::<PrimTypeObj>();
                ty.expect("a view is typed by a `PrimType`").dtype
            }
        }

        /// The view of an expression whose type it accepts; a TypeError names any other.
        impl TryFrom<Expr> for $name {
            type Error = Error;

            fn try_from(value: Expr) -> Result<Self> {
                value.try_cast()
            }
        }

        impl TryFrom<&Expr> for $name {
            type Error = Error;

            fn try_from(value: &Expr) -> Result<Self> {
                value.clone().try_cast()
            }
        }
    };
}

typed_expr_view! {
    /// An expression typed by a `PrimType` (C++ `PrimExpr`): a constant, a variable, an operator
    /// on scalars or vectors, or a call that returns one.
    ///
    /// `PrimExpr` and [`IntExpr`] are typed views (C++ `TypedExpr`). A view holds an `ir.Expr` as
    /// [`Expr`] does, and adds what the object class does not say: the type of the expression. An
    /// `ir.Call` is a `PrimExpr` when its `ty` is a `PrimType`, and an `IntExpr` when that type is
    /// a scalar integer. The views are checked when they are made, so a function that takes one
    /// needs no check of its own:
    ///
    /// - A view converts into the views and the [`Expr`] it refines for free (`From`): an
    ///   `IntExpr` into a `PrimExpr` and both into an `Expr`. Borrowed, an `IntExpr` is a
    ///   `PrimExpr` through `AsRef`, and both views read their object through `Deref`, as an
    ///   `Expr` does.
    /// - An `Expr` converts into a view only through a check (`TryFrom`,
    ///   `ObjectRefCast::try_cast`), as an `Any` does. A failed check names the expression and its
    ///   type, `ir.StringImm[ty=ir.StringType]`.
    /// - [`IntImm`], [`FloatImm`] and [`TensorLoad`] are typed by a `PrimType` when they are made,
    ///   so they convert into a `PrimExpr` for free. Rust literals convert as they convert into an
    ///   `Expr`: an `i32` and an `f32` as C++ `PrimExpr(int32_t)` and `PrimExpr(float)` convert
    ///   them.
    /// - The operators build a `PrimExpr`: each step of an [`ExprResult`] is checked, so
    ///   `PrimExpr::try_from(a + b)?` reports only the errors of the operators.
    /// - The view holds its object privately and gives it out only to read, so its type cannot be
    ///   changed; the one way past the check is the `unsafe` [`ObjectRefCore::from_data`].
    ///
    /// ```compile_fail
    /// # use tvm_tirx::ir::{Expr, PrimExpr};
    /// fn widen(expr: Expr) -> PrimExpr {
    ///     expr // An `Expr` is not a `PrimExpr` until it is checked.
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// # use tvm_tirx::ir::{IntExpr, PrimExpr};
    /// fn narrow(expr: PrimExpr) -> IntExpr {
    ///     expr.into() // Nor is a `PrimExpr` an `IntExpr`.
    /// }
    /// ```
    PrimExpr, "ir.PrimExpr", accepts = any_prim, fallback = prim_literal
}

typed_expr_view! {
    /// A [`PrimExpr`] of a scalar signed or unsigned integer type of any width (C++ `IntExpr`),
    /// such as an index or an extent.
    IntExpr, "ir.IntExpr", accepts = scalar_int, fallback = int_literal
}

tvm_ffi::impl_object_upcast!(PrimExpr => Expr, IntExpr => Expr, IntExpr => PrimExpr);

/// An `IntExpr` borrowed as the `PrimExpr` it is, as C++ `IntExpr` derives from `PrimExpr`.
impl AsRef<PrimExpr> for IntExpr {
    fn as_ref(&self) -> &PrimExpr {
        // SAFETY: both views are `repr(transparent)` over an `ObjectArc<ExprObj>`, and every
        // scalar integer type is a `PrimType`.
        unsafe { &*(self as *const IntExpr as *const PrimExpr) }
    }
}

impl TryFrom<PrimExpr> for IntExpr {
    type Error = Error;

    fn try_from(value: PrimExpr) -> Result<Self> {
        value.try_cast()
    }
}

impl TryFrom<&PrimExpr> for IntExpr {
    type Error = Error;

    fn try_from(value: &PrimExpr) -> Result<Self> {
        value.clone().try_cast()
    }
}

/// The built expression, checked to be an integer, or the first error raised while building it.
impl TryFrom<ExprResult> for IntExpr {
    type Error = Error;

    fn try_from(value: ExprResult) -> Result<Self> {
        PrimExpr::try_from(value)?.try_cast()
    }
}

// Rust literals, as they convert into an `Expr`. C++ has the constructors `PrimExpr(int32_t)` and
// `PrimExpr(float)`; the other literals are constants of their own type.

/// An `i32` is an `int32` constant (C++ `PrimExpr(int32_t)`).
impl From<i32> for PrimExpr {
    fn from(value: i32) -> Self {
        IntImm::int32(value).into()
    }
}

/// An `i64` is an `int64` constant. (C++ converts an `int64_t` held by an `Any` to `int32` when
/// the value fits.)
impl From<i64> for PrimExpr {
    fn from(value: i64) -> Self {
        IntImm::int64(value).into()
    }
}

/// A `bool` is a boolean constant, as C++ converts a `bool` held by an `Any`. (C++ promotes a
/// `bool` argument to `PrimExpr(int32_t)`, an `int32` constant.)
impl From<bool> for PrimExpr {
    fn from(value: bool) -> Self {
        IntImm::bool(value).into()
    }
}

/// An `f32` is a `float32` constant (C++ `PrimExpr(float)`).
impl From<f32> for PrimExpr {
    fn from(value: f32) -> Self {
        FloatImm::float32(value).into()
    }
}

/// An `f64` is a `float64` constant, as C++ types a `double` next to an expression. (C++ converts
/// a `double` held by an `Any` to `float32`.)
impl From<f64> for PrimExpr {
    fn from(value: f64) -> Self {
        FloatImm::float64(value).into()
    }
}

/// An `i32` is an `int32` constant. (C++ `IntExpr` converts only an int held by an `Any`, as
/// `PrimExpr` does.)
impl From<i32> for IntExpr {
    fn from(value: i32) -> Self {
        int_constant(IntImm::int32(value))
    }
}

/// An `i64` is an `int64` constant.
impl From<i64> for IntExpr {
    fn from(value: i64) -> Self {
        int_constant(IntImm::int64(value))
    }
}

/// The `IntExpr` of an integer constant, which `int32` and `int64` constants are.
fn int_constant(constant: IntImm) -> IntExpr {
    debug_assert!(scalar_int(PrimExpr::from(&constant).dtype()));
    // SAFETY: the constant is typed by a scalar integer type.
    unsafe { IntExpr::from_data(ObjectRefCore::into_data(Expr::from(constant))) }
}

// The checks and conversions of the views.

/// Every `PrimType`.
fn any_prim(_: DLDataType) -> bool {
    true
}

/// A scalar signed or unsigned integer type.
fn scalar_int(dtype: DLDataType) -> bool {
    dtype.lanes == 1 && matches!(dtype.code, support::INT | support::UINT)
}

/// The expression `data` holds, if it holds one.
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn expr_of(data: &TVMFFIAny) -> Option<&ExprObj> {
    let is_object = data.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32;
    // SAFETY: an object of an `ir.Expr` type is an `ExprObj`.
    (is_object && is_instance_of::<ExprObj>(data.type_index))
        .then(|| unsafe { &*(data.data_union.v_obj as *const ExprObj) })
}

/// The dtype of the expression `data` holds, if it holds one typed by a `PrimType`.
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn prim_dtype(data: &TVMFFIAny) -> Option<DLDataType> {
    Some(unsafe { expr_of(data) }?.ty.downcast_ref::<PrimTypeObj>()?.dtype)
}

/// What a view names a value it rejects (C++ `TypeTraits<TypedExpr<T>>::GetMismatchTypeInfo`):
/// an expression with its type, `ir.StringImm[ty=ir.StringType]`, or the type key of another value.
fn typed_expr_mismatch(data: &TVMFFIAny) -> std::string::String {
    // SAFETY: a mismatch is reported for a live value.
    let Some(expr) = (unsafe { expr_of(data) }) else { return support::type_key(data.type_index) };
    let ty = match expr.ty.downcast_ref::<PrimTypeObj>() {
        Some(prim) => support::dtype_name(prim.dtype),
        None => support::type_key(AnyView::from(&expr.ty).type_index()),
    };
    format!("{}[ty={ty}]", support::type_key(data.type_index))
}

/// The constant of the int, bool or float `data` holds, as C++ converts an `Any` into a
/// `PrimExpr`: a bool is a boolean constant, an int an `int32` constant when it fits and an
/// `int64` one otherwise, and a float a `float32` constant.
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn prim_literal(data: &TVMFFIAny) -> Option<PrimExpr> {
    let index = data.type_index;
    // SAFETY: the type index says which member of the union holds the value.
    unsafe {
        if index == TVMFFITypeIndex::kTVMFFIBool as i32 {
            Some(IntImm::bool(data.data_union.v_int64 != 0).into())
        } else if index == TVMFFITypeIndex::kTVMFFIInt as i32 {
            Some(int_imm(data.data_union.v_int64).into())
        } else if index == TVMFFITypeIndex::kTVMFFIFloat as i32 {
            let value = data.data_union.v_float64;
            FloatImm::from_dtype(dtype::float(32), value, UnknownLoc::new()).ok().map(Into::into)
        } else {
            None
        }
    }
}

/// The constant of the int `data` holds, as [`prim_literal`] makes it.
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn int_literal(data: &TVMFFIAny) -> Option<IntExpr> {
    // SAFETY: an int is held in `v_int64`.
    (data.type_index == TVMFFITypeIndex::kTVMFFIInt as i32)
        .then(|| int_constant(int_imm(unsafe { data.data_union.v_int64 })))
}

/// An `int32` constant when `value` fits, and an `int64` one otherwise (C++
/// `TypeTraits<IntImm>::ConvertFallbackValue`).
fn int_imm(value: i64) -> IntImm {
    match i32::try_from(value) {
        Ok(value) => IntImm::int32(value),
        Err(_) => IntImm::int64(value),
    }
}

// tvm-ffi-stubgen(begin): object/ir.GlobalVar
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GlobalVar"]
#[type_final]
pub struct GlobalVarObj {
    base: ExprObj,
    pub name_hint: String,
}

const _: () = {
    assert!(::core::mem::size_of::<GlobalVarObj>() == 56);
    assert!(::core::mem::align_of::<GlobalVarObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GlobalVar {
    base: ObjectArc<GlobalVarObj>,
}

impl Deref for GlobalVar {
    type Target = GlobalVarObj;
    fn deref(&self) -> &GlobalVarObj {
        &self.base
    }
}

impl Deref for GlobalVarObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl GlobalVarObj {
    pub(crate) fn new(loc: Location, ty: Type, name_hint: String) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, name_hint }
    }
}

impl GlobalVar {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        name_hint: impl Into<String>,
    ) -> Self {
        let obj = GlobalVarObj::new(loc.into(), ty.into(), name_hint.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GlobalVar => Expr);
// tvm-ffi-stubgen(end)

impl GlobalVar {
    /// The global variable `name_hint`, of a missing type (C++ `GlobalVar(name_hint, loc)`).
    pub fn new(name_hint: &str, loc: impl Into<Location>) -> Self {
        let ty = MissingType::new();
        Self::from_fields(loc.into(), ty, String::from(name_hint))
    }
}

// tvm-ffi-stubgen(begin): object/ir.Tuple
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
    pub fields: Array<Expr>,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleObj>() == 48);
    assert!(::core::mem::align_of::<TupleObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Tuple {
    base: ObjectArc<TupleObj>,
}

impl Deref for Tuple {
    type Target = TupleObj;
    fn deref(&self) -> &TupleObj {
        &self.base
    }
}

impl Deref for TupleObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TupleObj {
    pub(crate) fn new(loc: Location, ty: Type, fields: Array<Expr>) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, fields }
    }
}

impl Tuple {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        fields: impl Into<Array<Expr>>,
    ) -> Self {
        let obj = TupleObj::new(loc.into(), ty.into(), fields.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Tuple => Expr);
// tvm-ffi-stubgen(end)

impl Tuple {
    /// The tuple of `fields`, typed by the types of its fields when they are all known
    /// (C++ `Tuple(fields, loc)`).
    pub fn new(
        fields: impl IntoIterator<Item = impl Into<Expr>>,
        loc: impl Into<Location>,
    ) -> Self {
        let fields = fields.into_iter().map(Into::into).collect::<Array<Expr>>();
        let ty: Type =
            match fields.iter().any(|field| field.ty.downcast_ref::<MissingTypeObj>().is_some()) {
                true => MissingType::new().into(),
                false => {
                    let field_types = fields.iter().map(|field| field.ty.clone());
                    TupleType::new(field_types, UnknownLoc::new()).into()
                }
            };
        Self::from_fields(loc.into(), ty, fields)
    }
}

// tvm-ffi-stubgen(begin): object/ir.TupleGetItem
/// Complete: reflected fields fill [40, 56) exactly (alignment padding [52, 56)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TupleGetItem"]
#[type_final]
pub struct TupleGetItemObj {
    base: ExprObj,
    pub tuple_value: Expr,
    pub index: i32,
}

const _: () = {
    assert!(::core::mem::size_of::<TupleGetItemObj>() == 56);
    assert!(::core::mem::align_of::<TupleGetItemObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TupleGetItem {
    base: ObjectArc<TupleGetItemObj>,
}

impl Deref for TupleGetItem {
    type Target = TupleGetItemObj;
    fn deref(&self) -> &TupleGetItemObj {
        &self.base
    }
}

impl Deref for TupleGetItemObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TupleGetItemObj {
    pub(crate) fn new(loc: Location, ty: Type, tuple_value: Expr, index: i32) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, tuple_value, index }
    }
}

impl TupleGetItem {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        tuple_value: impl Into<Expr>,
        index: i32,
    ) -> Self {
        let obj = TupleGetItemObj::new(loc.into(), ty.into(), tuple_value.into(), index);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TupleGetItem => Expr);
// tvm-ffi-stubgen(end)

impl TupleGetItem {
    /// Field `index` of `tuple_value`, typed by the field type and checked against the tuple type
    /// when it is known (C++ `TupleGetItem(tuple, index, loc)`).
    pub fn from_tuple(
        tuple_value: impl IntoExpr,
        index: i32,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let tuple_value = tuple_value.into_expr()?;
        if index < 0 {
            let message = format!(
                "Index out of bounds: Tuple {} cannot be accessed with negative index {index}",
                support::repr(&tuple_value)
            );
            return Err(support::compare_failed(
                tvm_ffi::INDEX_ERROR,
                "index >= 0",
                index,
                0,
                &message,
            ));
        }
        let ty: Type = match tuple_value.ty.downcast_ref::<TupleTypeObj>() {
            Some(tuple_type) => {
                let size = tuple_type.fields.len();
                if index as usize >= size {
                    let message = format!(
                        "Index out of bounds: Tuple {} is of size {size}, and cannot be accessed \
                         with index {index}",
                        support::repr(&tuple_value)
                    );
                    let condition = "index < tuple_type->fields.size()";
                    let error = support::compare_failed(
                        tvm_ffi::INDEX_ERROR,
                        condition,
                        index,
                        size,
                        &message,
                    );
                    return Err(error);
                }
                tuple_type.fields.get(index as usize)?
            }
            None => MissingType::new().into(),
        };
        Ok(Self::new(loc.into(), ty, tuple_value, index))
    }
}

// tvm-ffi-stubgen(begin): object/ir.TensorLoad
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorLoad"]
#[type_final]
pub struct TensorLoadObj {
    base: ExprObj,
    pub source: Expr,
    pub indices: Array<PrimExpr>,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorLoadObj>() == 56);
    assert!(::core::mem::align_of::<TensorLoadObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorLoad {
    base: ObjectArc<TensorLoadObj>,
}

impl Deref for TensorLoad {
    type Target = TensorLoadObj;
    fn deref(&self) -> &TensorLoadObj {
        &self.base
    }
}

impl Deref for TensorLoadObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TensorLoadObj {
    pub(crate) fn new(loc: Location, ty: PrimType, source: Expr, indices: Array<PrimExpr>) -> Self {
        let base = ExprObj::new(loc, ty.into());
        Self { base, source, indices }
    }
}

impl TensorLoad {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        source: impl Into<Expr>,
        indices: impl Into<Array<PrimExpr>>,
    ) -> Self {
        let obj = TensorLoadObj::new(loc.into(), ty.into(), source.into(), indices.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorLoad => Expr, TensorLoad => PrimExpr);
// tvm-ffi-stubgen(end)

impl TensorLoad {
    /// The element of `source` at `indices`, typed by the element type of `source`: the type the
    /// type of `source` infers (C++ `TensorLoad(source, indices, loc)`).
    pub fn new(
        source: impl IntoExpr,
        indices: impl IntoIterator<Item = impl IntoExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        static INFER_TYPE: support::TypeAttr =
            support::TypeAttr::new("__tvm_tensor_load_infer_type__");
        let (source, loc) = (source.into_expr()?, loc.into());
        let indices = indices
            .into_iter()
            .map(|index| PrimExpr::try_from(index.into_expr()?))
            .collect::<Result<Array<PrimExpr>>>()?;
        let ty_index = AnyView::from(&source.ty).type_index();
        let Some(infer) = INFER_TYPE.get(ty_index) else {
            let message =
                format!("Type {} does not support TensorLoad", support::type_key(ty_index));
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, "infer.has_value()", &message));
        };
        // The hook types a load that has no type yet. Such a load is no `TensorLoad`, which is
        // typed by a `PrimType`, so it is made as an `Expr`.
        let base = ExprObj::new(loc.clone(), MissingType::new().into());
        let untyped = TensorLoadObj { base, source: source.clone(), indices: indices.clone() };
        let untyped = Expr::from(Self { base: ObjectArc::new(untyped) });
        let ty: PrimType = support::NativeHook::from_any(infer)?.call_as(&untyped)?;
        Ok(Self::from_fields(loc, ty, source, indices))
    }
}

// tvm-ffi-stubgen(begin): object/ir.Constant
/// Complete: reflected fields fill [40, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Constant"]
pub struct ConstantObj {
    base: ExprObj,
}

const _: () = {
    assert!(::core::mem::size_of::<ConstantObj>() == 40);
    assert!(::core::mem::align_of::<ConstantObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Constant {
    base: ObjectArc<ConstantObj>,
}

impl Deref for Constant {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl Deref for ConstantObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl ConstantObj {
    pub fn new(loc: Location, ty: Type) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base }
    }
}

impl Constant {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>) -> Self {
        let obj = ConstantObj::new(loc.into(), ty.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for ConstantObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}

unsafe impl UnsafeInit for ConstantObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base));
        }
    }
}

tvm_ffi::impl_object_upcast!(Constant => Expr);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.FloatImm
/// Complete: reflected fields fill [40, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.FloatImm"]
#[type_final]
pub struct FloatImmObj {
    base: ConstantObj,
    pub value: f64,
}

const _: () = {
    assert!(::core::mem::size_of::<FloatImmObj>() == 48);
    assert!(::core::mem::align_of::<FloatImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct FloatImm {
    base: ObjectArc<FloatImmObj>,
}

impl Deref for FloatImm {
    type Target = FloatImmObj;
    fn deref(&self) -> &FloatImmObj {
        &self.base
    }
}

impl Deref for FloatImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl FloatImmObj {
    pub(crate) fn new(loc: Location, ty: PrimType, value: f64) -> Self {
        let base = ConstantObj::new(loc, ty.into());
        Self { base, value }
    }
}

impl FloatImm {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<PrimType>, value: f64) -> Self {
        let obj = FloatImmObj::new(loc.into(), ty.into(), value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloatImm => Expr, FloatImm => Constant, FloatImm => PrimExpr);
// tvm-ffi-stubgen(end)

impl FloatImm {
    /// The floating-point constant `value` of type `dtype`, checked to fit it
    /// (C++ `FloatImm(PrimType(dtype), value, loc)`).
    pub fn from_dtype(dtype: DLDataType, value: f64, loc: impl Into<Location>) -> Result<Self> {
        let ty = PrimType::from_dtype(dtype)?;
        check_float_imm(dtype, value)?;
        Ok(Self::new(loc.into(), ty, value))
    }

    /// The `float32` constant `value` at the unknown location, which every `f32` is
    /// (C++ `FloatImm(PrimType::Float(32), value)`).
    pub fn float32(value: f32) -> Self {
        Self::builtin(dtype::float(32), value as f64)
    }

    /// The `float64` constant `value` at the unknown location, which every `f64` is
    /// (C++ `FloatImm(PrimType::Float(64), value)`).
    pub fn float64(value: f64) -> Self {
        Self::builtin(dtype::float(64), value)
    }

    /// [`FloatImm::from_dtype`] of a value its type holds, which fails only when the compiler
    /// library is not loaded.
    fn builtin(dtype: DLDataType, value: f64) -> Self {
        support::infallible(Self::from_dtype(dtype, value, UnknownLoc::new()))
    }
}

/// An `f32` is a `float32` constant, as C++ converts a `float` to a `PrimExpr`.
impl From<f32> for FloatImm {
    fn from(value: f32) -> Self {
        Self::float32(value)
    }
}

/// An `f64` is a `float64` constant. (C++ converts a `double` held by an `Any` to `float32`, as
/// [`prim::convert`](crate::ir::prim::convert) does.)
impl From<f64> for FloatImm {
    fn from(value: f64) -> Self {
        Self::float64(value)
    }
}

impl From<f32> for Expr {
    fn from(value: f32) -> Self {
        FloatImm::float32(value).into()
    }
}

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        FloatImm::float64(value).into()
    }
}

/// The checks of C++ `FloatImm(ty, value)`: a scalar floating-point type whose range holds a
/// finite `value`.
fn check_float_imm(dtype: DLDataType, value: f64) -> Result<()> {
    use support::{check_failed, compare_failed, CppDouble};
    use tvm_ffi::VALUE_ERROR;
    const FLOAT4_E2M1FN: u8 = tvm_ffi::DLDataTypeCode::kDLFloat4_e2m1fn as u8;
    /// `ffi::DLExtDataTypeCode::kDLExtCustomBegin`.
    const CUSTOM_BEGIN: u8 = 129;
    if support::is_scalable_vector(dtype) || support::is_fixed_length_vector(dtype) {
        let condition = "!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector()";
        return Err(check_failed(VALUE_ERROR, condition, "FloatImm can only take scalar."));
    }
    let (code, bits) = (dtype.code, dtype.bits);
    let float8 = support::float8_bound(code);
    let float6 = support::float6_bound(code);
    let supported = code == support::FLOAT
        || float8.is_some()
        || float6.is_some()
        || (code == support::BFLOAT && bits == 16)
        || (code == FLOAT4_E2M1FN && bits == 4)
        || code >= CUSTOM_BEGIN;
    let name = || support::dtype_name(dtype);
    if !supported {
        let condition = "value_ty.MatchesCode(DLDataTypeCode::kDLFloat, \
                         DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3, \
                         DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn, \
                         DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2, \
                         DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu, \
                         DLDataTypeCode::kDLFloat6_e2m3fn, DLDataTypeCode::kDLFloat6_e3m2fn) || \
                         value_ty.MatchesElementType(DLDataTypeCode::kDLBfloat, 16) || \
                         value_ty.MatchesElementType(DLDataTypeCode::kDLFloat4_e2m1fn, 4) || \
                         static_cast<int>(code) >= \
                         static_cast<int>(ffi::DLExtDataTypeCode::kDLExtCustomBegin)";
        let message = format!("FloatImm supports only float, but {} was supplied.", name());
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    if value.is_infinite() || value.is_nan() {
        return Ok(());
    }
    // The bound checked from below and how its message says so, and the bound from above.
    let (low, high): ((&str, f64, &str), (&str, f64)) = if bits == 32 {
        (
            (
                "value >= std::numeric_limits<float>::lowest()",
                f32::MIN as f64,
                "exceeds minimum of",
            ),
            ("value <= std::numeric_limits<float>::max()", f32::MAX as f64),
        )
    } else if code == support::FLOAT && bits == 16 {
        (
            ("value >= -support::kMaxFloat16", -support::MAX_FLOAT16, "exceeds minimum of"),
            ("value <= support::kMaxFloat16", support::MAX_FLOAT16),
        )
    } else if code == support::BFLOAT && bits == 16 {
        (
            ("value >= -support::kMaxBFloat16", -support::MAX_BFLOAT16, "exceeds minimum of"),
            ("value <= support::kMaxBFloat16", support::MAX_BFLOAT16),
        )
    } else if let Some((bound, nonneg)) = float8 {
        let low = match nonneg {
            true => ("value >= 0", 0.0, "below zero for unsigned"),
            false => ("value >= -bound", -bound, "below minimum of"),
        };
        (low, ("value <= bound", bound))
    } else if let Some(bound) = float6 {
        (("value >= -bound", -bound, "below minimum of"), ("value <= bound", bound))
    } else if code == FLOAT4_E2M1FN {
        (("value >= -bound", -6.0, "below minimum of"), ("value <= bound", 6.0))
    } else {
        return Ok(());
    };
    let failed = |condition: &str, bound: f64, what: &str| {
        let message = format!("Literal value {} {what} {}", CppDouble(value), name());
        compare_failed(VALUE_ERROR, condition, CppDouble(value), CppDouble(bound), &message)
    };
    if value < low.1 {
        return Err(failed(low.0, low.1, low.2));
    }
    if value > high.1 {
        return Err(failed(high.0, high.1, "exceeds maximum of"));
    }
    Ok(())
}

// tvm-ffi-stubgen(begin): object/ir.IntImm
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.IntImm"]
#[type_final]
pub struct IntImmObj {
    base: ConstantObj,
    pub value: BigInt,
}

const _: () = {
    assert!(::core::mem::size_of::<IntImmObj>() == 56);
    assert!(::core::mem::align_of::<IntImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct IntImm {
    base: ObjectArc<IntImmObj>,
}

impl Deref for IntImm {
    type Target = IntImmObj;
    fn deref(&self) -> &IntImmObj {
        &self.base
    }
}

impl Deref for IntImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl IntImmObj {
    pub(crate) fn new(loc: Location, ty: PrimType, value: BigInt) -> Self {
        let base = ConstantObj::new(loc, ty.into());
        Self { base, value }
    }
}

impl IntImm {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl Into<Location>,
        ty: impl Into<PrimType>,
        value: impl Into<BigInt>,
    ) -> Self {
        let obj = IntImmObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(IntImm => Expr, IntImm => Constant, IntImm => PrimExpr);
// tvm-ffi-stubgen(end)

impl IntImm {
    /// The integer constant `value` of the scalar integer or boolean type `dtype`, checked to fit
    /// it (C++ `IntImm(PrimType(dtype), value, loc)`).
    pub fn from_dtype(
        dtype: DLDataType,
        value: impl Into<BigInt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (value, ty) = (value.into(), PrimType::from_dtype(dtype)?);
        check_int_imm(dtype, &value)?;
        Ok(Self::new(loc.into(), ty, value))
    }

    /// The `int32` constant `value` at the unknown location, which every `i32` is
    /// (C++ `IntImm::Int32(value)`).
    pub fn int32(value: i32) -> Self {
        Self::builtin(dtype::int(32), value)
    }

    /// The `int64` constant `value` at the unknown location, which every `i64` is
    /// (C++ `IntImm::Int64(value)`).
    pub fn int64(value: i64) -> Self {
        Self::builtin(dtype::int(64), value)
    }

    /// The boolean constant `value` at the unknown location (C++ `IntImm::Bool(value)`).
    pub fn bool(value: bool) -> Self {
        Self::builtin(dtype::bool(), value)
    }

    /// [`IntImm::from_dtype`] of a value its type holds, which fails only when the compiler
    /// library is not loaded.
    fn builtin(dtype: DLDataType, value: impl Into<BigInt>) -> Self {
        support::infallible(Self::from_dtype(dtype, value, UnknownLoc::new()))
    }
}

/// An `i32` is an `int32` constant, as C++ converts an `int` to a `PrimExpr`.
impl From<i32> for IntImm {
    fn from(value: i32) -> Self {
        Self::int32(value)
    }
}

/// An `i64` is an `int64` constant. (C++ converts an `int64_t` held by an `Any` to `int32` when
/// the value fits, as [`prim::convert`](crate::ir::prim::convert) does.)
impl From<i64> for IntImm {
    fn from(value: i64) -> Self {
        Self::int64(value)
    }
}

/// A `bool` is a boolean constant, as C++ converts a `bool` to a `PrimExpr`.
impl From<bool> for IntImm {
    fn from(value: bool) -> Self {
        Self::bool(value)
    }
}

/// What converts to an expression, possibly failing: anything that converts into an [`Expr`] (a
/// node, a `Var`, a Rust `i32`, `f32` and the like), and an [`ExprResult`], an
/// expression under construction, which gives the first error any of its steps raised. The
/// constructors and functions that can fail take their expression operands as `impl IntoExpr`, so
/// an `ExprResult` passes straight in and `?` on their result covers it.
pub trait IntoExpr {
    fn into_expr(self) -> Result<Expr>;
}

impl<T: Into<Expr>> IntoExpr for T {
    fn into_expr(self) -> Result<Expr> {
        Ok(self.into())
    }
}

impl From<i32> for Expr {
    fn from(value: i32) -> Self {
        IntImm::int32(value).into()
    }
}

impl From<i64> for Expr {
    fn from(value: i64) -> Self {
        IntImm::int64(value).into()
    }
}

impl From<bool> for Expr {
    fn from(value: bool) -> Self {
        IntImm::bool(value).into()
    }
}

/// The checks of C++ `IntImm(ty, value)`: a scalar integer or boolean type whose range holds
/// `value`. The messages are built only when a check fails.
fn check_int_imm(dtype: DLDataType, value: &BigInt) -> Result<()> {
    use support::{check_failed, compare_failed};
    use tvm_ffi::VALUE_ERROR;
    let name = || support::dtype_name(dtype);
    if support::is_scalable_vector(dtype) || support::is_fixed_length_vector(dtype) {
        let condition = "!value_ty.IsScalableVector() && !value_ty.IsFixedLengthVector()";
        let message = format!("IntImm can only take scalar, but {} was supplied.", name());
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    if !matches!(dtype.code, support::INT | support::UINT | support::BOOL) {
        let condition = "value_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt, \
                         DLDataTypeCode::kDLBool)";
        let message =
            format!("IntImm supports only int or uint or bool type, but {} was supplied.", name());
        return Err(check_failed(VALUE_ERROR, condition, &message));
    }
    let bits = dtype.bits as u32;
    if bits == 0 {
        let message = "IntImm requires a positive integer width";
        return Err(compare_failed(VALUE_ERROR, "bits > 0", bits, 0, message));
    }
    let zero = BigInt::from(0i64);
    if dtype.code == support::UINT {
        if *value < zero {
            let message =
                format!("Literal value {value} is negative for unsigned integer type {}", name());
            return Err(compare_failed(VALUE_ERROR, "value >= 0", value, 0, &message));
        }
        let message = || format!("Literal value {value} exceeds maximum of {}", name());
        if bits <= 64 {
            if !value.to_u64().is_some_and(|small| bits == 64 || small < 1u64 << bits) {
                let condition =
                    "small.has_value() && (bits == 64 || *small < (uint64_t{1} << bits))";
                return Err(check_failed(VALUE_ERROR, condition, &message()));
            }
        } else {
            let limit = BigInt::from(1i64).try_shl(bits as i64)?;
            if *value >= limit {
                let condition = "value < ffi::BigInt(1) << bits";
                return Err(compare_failed(VALUE_ERROR, condition, value, &limit, &message()));
            }
        }
    } else if bits == 1 || dtype.code == support::BOOL {
        if *value != zero && *value != BigInt::from(1i64) {
            let message = format!("{value} exceeds range of {}", name());
            return Err(check_failed(VALUE_ERROR, "value == 0 || value == 1", &message));
        }
    } else {
        let message = || format!("Literal value {value} exceeds range of {}", name());
        if bits <= 64 {
            let half = 1i128 << (bits - 1);
            let fits = value.to_i64().is_some_and(|small| {
                bits == 64 || ((small as i128) >= -half && (small as i128) < half)
            });
            if !fits {
                let condition = "small.has_value() && (bits == 64 || (*small >= -(int64_t{1} << \
                                 (bits - 1)) && *small < (int64_t{1} << (bits - 1))))";
                return Err(check_failed(VALUE_ERROR, condition, &message()));
            }
        } else {
            let limit = BigInt::from(1i64).try_shl(bits as i64 - 1)?;
            if *value < -limit.clone() || *value >= limit {
                let condition = "value >= -limit && value < limit";
                return Err(check_failed(VALUE_ERROR, condition, &message()));
            }
        }
    }
    Ok(())
}

// tvm-ffi-stubgen(begin): object/ir.GenericConst
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GenericConst"]
#[type_final]
pub struct GenericConstObj {
    base: ConstantObj,
    pub value: Any,
}

const _: () = {
    assert!(::core::mem::size_of::<GenericConstObj>() == 56);
    assert!(::core::mem::align_of::<GenericConstObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GenericConst {
    base: ObjectArc<GenericConstObj>,
}

impl Deref for GenericConst {
    type Target = GenericConstObj;
    fn deref(&self) -> &GenericConstObj {
        &self.base
    }
}

impl Deref for GenericConstObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl GenericConstObj {
    pub(crate) fn new(loc: Location, ty: Type, value: Any) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl GenericConst {
    /// Lossless complete-field allocation.
    pub fn new(loc: impl Into<Location>, ty: impl Into<Type>, value: impl Into<Any>) -> Self {
        let obj = GenericConstObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GenericConst => Expr, GenericConst => Constant);
// tvm-ffi-stubgen(end)

impl GenericConst {
    /// The constant `value` of type `ty`, checked to have a type and not to be a primitive literal
    /// (C++ `GenericConst(ty, value, loc)`).
    pub fn checked(
        ty: impl Into<Type>,
        value: impl Into<Any>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        use TVMFFITypeIndex::*;
        let (ty, value) = (ty.into(), value.into());
        if ty.downcast_ref::<MissingTypeObj>().is_some() {
            let condition = "!ty.as<MissingType>().has_value()";
            let message = "GenericConst requires an expression type";
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, message));
        }
        let primitive =
            [kTVMFFIInt, kTVMFFIBigInt, kTVMFFIBool, kTVMFFIFloat, kTVMFFISmallStr, kTVMFFIStr]
                .into_iter()
                .any(|index| value.type_index() == index as i32);
        if primitive {
            let condition =
                "!value.as<ffi::BigInt>() && !value.as<bool>() && !value.as<double>() && \
                             !value.as<ffi::String>()";
            let message = "Primitive literals use IntImm, FloatImm, or StringImm";
            return Err(support::check_failed(tvm_ffi::TYPE_ERROR, condition, message));
        }
        Ok(Self::new(loc.into(), ty, value))
    }
}

// tvm-ffi-stubgen(begin): object/ir.DataTypeImm
/// Complete: reflected fields fill [40, 48) exactly (alignment padding [44, 48)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.DataTypeImm"]
#[type_final]
pub struct DataTypeImmObj {
    base: ConstantObj,
    pub value: DLDataType,
}

const _: () = {
    assert!(::core::mem::size_of::<DataTypeImmObj>() == 48);
    assert!(::core::mem::align_of::<DataTypeImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct DataTypeImm {
    base: ObjectArc<DataTypeImmObj>,
}

impl Deref for DataTypeImm {
    type Target = DataTypeImmObj;
    fn deref(&self) -> &DataTypeImmObj {
        &self.base
    }
}

impl Deref for DataTypeImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl DataTypeImmObj {
    pub(crate) fn new(loc: Location, ty: Type, value: DLDataType) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl DataTypeImm {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        value: impl Into<DLDataType>,
    ) -> Self {
        let obj = DataTypeImmObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(DataTypeImm => Expr, DataTypeImm => Constant);
// tvm-ffi-stubgen(end)

impl DataTypeImm {
    /// The data type `value` as a constant of type [`AnyType`]
    /// (C++ `DataTypeImm(value, loc)`).
    pub fn from_dtype(value: DLDataType, loc: impl Into<Location>) -> Self {
        let ty = AnyType::new(UnknownLoc::new());
        Self::new(loc.into(), ty, value)
    }
}

// tvm-ffi-stubgen(begin): object/ir.StringImm
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ConstantObj,
    pub value: String,
}

const _: () = {
    assert!(::core::mem::size_of::<StringImmObj>() == 56);
    assert!(::core::mem::align_of::<StringImmObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StringImm {
    base: ObjectArc<StringImmObj>,
}

impl Deref for StringImm {
    type Target = StringImmObj;
    fn deref(&self) -> &StringImmObj {
        &self.base
    }
}

impl Deref for StringImmObj {
    type Target = ConstantObj;
    fn deref(&self) -> &ConstantObj {
        &self.base
    }
}

impl StringImmObj {
    pub(crate) fn new(loc: Location, ty: Type, value: String) -> Self {
        let base = ConstantObj::new(loc, ty);
        Self { base, value }
    }
}

impl StringImm {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        value: impl Into<String>,
    ) -> Self {
        let obj = StringImmObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(StringImm => Expr, StringImm => Constant);
// tvm-ffi-stubgen(end)

impl StringImm {
    /// The string constant `value`, of type [`StringType`]
    /// (C++ `StringImm(value, loc)`).
    pub fn new(value: &str, loc: impl Into<Location>) -> Self {
        let ty = StringType::new(UnknownLoc::new());
        Self::from_fields(loc.into(), ty, String::from(value))
    }
}

// tvm-ffi-stubgen(begin): object/ir.Call
/// Complete: reflected fields fill [40, 72) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Call"]
#[type_final]
pub struct CallObj {
    base: ExprObj,
    pub op: Expr,
    pub args: Array<Expr>,
    pub attrs: Option<Attrs>,
    pub ty_args: Array<Type>,
}

const _: () = {
    assert!(::core::mem::size_of::<CallObj>() == 72);
    assert!(::core::mem::align_of::<CallObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Call {
    base: ObjectArc<CallObj>,
}

impl Deref for Call {
    type Target = CallObj;
    fn deref(&self) -> &CallObj {
        &self.base
    }
}

impl Deref for CallObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl CallObj {
    pub(crate) fn new(
        loc: Location,
        ty: Type,
        op: Expr,
        args: Array<Expr>,
        attrs: Option<Attrs>,
        ty_args: Array<Type>,
    ) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, op, args, attrs, ty_args }
    }
}

impl Call {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        op: impl Into<Expr>,
        args: impl Into<Array<Expr>>,
        attrs: impl Into<Option<Attrs>>,
        ty_args: impl Into<Array<Type>>,
    ) -> Self {
        let obj = CallObj::new(
            loc.into(),
            ty.into(),
            op.into(),
            args.into(),
            attrs.into(),
            ty_args.into(),
        );
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(Call => Expr);
// tvm-ffi-stubgen(end)

impl Call {
    /// The call of `op` on `args`, of type `ty` or the type [`Call::reinfer_type`] gives it when
    /// `None` (C++ `Call(ty, op, args, attrs, ty_args, loc)`). The call is not validated.
    pub fn new(
        ty: Option<&Type>,
        op: impl IntoExpr,
        args: impl IntoIterator<Item = impl IntoExpr>,
        attrs: Option<&Attrs>,
        ty_args: impl IntoIterator<Item = impl Into<Type>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (op, attrs, loc) = (op.into_expr()?, attrs.cloned(), loc.into());
        let args = args.into_iter().map(IntoExpr::into_expr).collect::<Result<Array<Expr>>>()?;
        let ty_args = ty_args.into_iter().map(Into::into).collect::<Array<Type>>();
        if let Some(ty) = ty {
            return Ok(Self::from_fields(loc, ty.clone(), op, args, attrs, ty_args));
        }
        let call = Self::from_fields(loc, MissingType::new(), op, args, attrs, ty_args);
        let ty = call.reinfer_type()?;
        Ok(Self::from_fields(
            call.loc.clone(),
            ty,
            call.op.clone(),
            call.args.clone(),
            call.attrs.clone(),
            call.ty_args.clone(),
        ))
    }

    /// Check the call against the contract its operator registered: the operator's validator, if
    /// it has one (C++ `Call::Validate`).
    pub fn validate(&self) -> Result<()> {
        let Some(op) = self.op.downcast_ref::<OpObj>() else { return Ok(()) };
        if op.validator().type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(());
        }
        support::NativeHook::from_any(op.validator().clone())?.call_void(self)
    }

    /// The result type the operator infers from the current inputs, without changing the call: its
    /// `TFixedReturnType`, else what its `FInferType` infers; the return type of a callee of a
    /// function type; otherwise a missing type (C++ `Call::ReinferType`).
    pub fn reinfer_type(&self) -> Result<Type> {
        let Some(op) = self.op.downcast_ref::<OpObj>() else {
            return match self.op.ty.downcast_ref::<FuncTypeObj>() {
                Some(func_type) => Ok(func_type.ret_type.clone()),
                None => Ok(MissingType::new().into()),
            };
        };
        if let Some(fixed) = support::op_attr(op, "TFixedReturnType")? {
            return fixed.try_into();
        }
        match support::NativeHook::of_op(op, "FInferType")? {
            Some(infer) => infer.call_as(self),
            None => Ok(MissingType::new().into()),
        }
    }
}

// tvm-ffi-stubgen(begin): object/ir.TensorRegion
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.TensorRegion"]
#[type_final]
pub struct TensorRegionObj {
    base: ExprObj,
    pub source: Expr,
    pub region: Array<Range>,
}

const _: () = {
    assert!(::core::mem::size_of::<TensorRegionObj>() == 56);
    assert!(::core::mem::align_of::<TensorRegionObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct TensorRegion {
    base: ObjectArc<TensorRegionObj>,
}

impl Deref for TensorRegion {
    type Target = TensorRegionObj;
    fn deref(&self) -> &TensorRegionObj {
        &self.base
    }
}

impl Deref for TensorRegionObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl TensorRegionObj {
    pub(crate) fn new(loc: Location, ty: Type, source: Expr, region: Array<Range>) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, source, region }
    }
}

impl TensorRegion {
    /// Lossless complete-field allocation.
    pub fn new(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        source: impl Into<Expr>,
        region: impl Into<Array<Range>>,
    ) -> Self {
        let obj = TensorRegionObj::new(loc.into(), ty.into(), source.into(), region.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(TensorRegion => Expr);
// tvm-ffi-stubgen(end)

// `ir.TensorRegion` stores its arguments as given: the generated `TensorRegion::new` is the same.

// tvm-ffi-stubgen(begin): object/ir.Var
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Var"]
pub struct VarObj {
    base: ExprObj,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<VarObj>() == 56);
    assert!(::core::mem::align_of::<VarObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Var {
    base: ObjectArc<VarObj>,
}

impl Deref for Var {
    type Target = VarObj;
    fn deref(&self) -> &VarObj {
        &self.base
    }
}

impl Deref for VarObj {
    type Target = ExprObj;
    fn deref(&self) -> &ExprObj {
        &self.base
    }
}

impl VarObj {
    pub fn new(loc: Location, ty: Type, name: String) -> Self {
        let base = ExprObj::new(loc, ty);
        Self { base, name }
    }
}

impl Var {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        name: impl Into<String>,
    ) -> Self {
        let obj = VarObj::new(loc.into(), ty.into(), name.into());
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for VarObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base), name: Clone::clone(&self.name) }
    }
}

unsafe impl UnsafeInit for VarObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base));
            drop_if_defined(addr_of_mut!((*this).name));
        }
    }
}

tvm_ffi::impl_object_upcast!(Var => Expr);
// tvm-ffi-stubgen(end)

impl Var {
    /// The variable `name` of type `ty` (C++ `Var(name, ty, loc)`); a variable whose type is not
    /// known yet takes [`MissingType::new`].
    pub fn new(name: &str, ty: impl Into<Type>, loc: impl Into<Location>) -> Self {
        Self::from_fields(loc.into(), ty.into(), String::from(name))
    }
}

// tvm-ffi-stubgen(begin): object/ir.LambdaExpr
/// Complete: reflected fields fill [40, 56) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.LambdaExpr"]
#[type_final]
pub struct LambdaExprObj {
    base: StagingExprObj,
    pub vars: Array<Var>,
    pub body: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<LambdaExprObj>() == 56);
    assert!(::core::mem::align_of::<LambdaExprObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct LambdaExpr {
    base: ObjectArc<LambdaExprObj>,
}

impl Deref for LambdaExpr {
    type Target = LambdaExprObj;
    fn deref(&self) -> &LambdaExprObj {
        &self.base
    }
}

impl Deref for LambdaExprObj {
    type Target = StagingExprObj;
    fn deref(&self) -> &StagingExprObj {
        &self.base
    }
}

impl LambdaExprObj {
    pub(crate) fn new(loc: Location, ty: Type, vars: Array<Var>, body: Expr) -> Self {
        let base = StagingExprObj::new(loc, ty);
        Self { base, vars, body }
    }
}

impl LambdaExpr {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        vars: impl Into<Array<Var>>,
        body: impl Into<Expr>,
    ) -> Self {
        let obj = LambdaExprObj::new(loc.into(), ty.into(), vars.into(), body.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(LambdaExpr => Expr, LambdaExpr => StagingExpr);
// tvm-ffi-stubgen(end)

impl LambdaExpr {
    /// The lambda of `vars` that evaluates to `body`, of the function type from the types of
    /// `vars` to the type of `body` (C++ `LambdaExpr(vars, body, loc)`).
    pub fn new(
        vars: impl IntoIterator<Item = impl Into<Var>>,
        body: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Self {
        let (vars, body) = (vars.into_iter().map(Into::into).collect::<Array<Var>>(), body.into());
        let arg_types = vars.iter().map(|var| var.ty.clone()).collect::<Vec<Type>>();
        let ty = FuncType::new(arg_types, body.ty.clone());
        Self::from_fields(loc.into(), ty, vars, body)
    }

    /// The body with `arguments` substituted for the variables (`ir.LambdaExprApply`).
    pub fn apply(&self, arguments: impl IntoIterator<Item = impl IntoExpr>) -> Result<Expr> {
        let arguments =
            arguments.into_iter().map(IntoExpr::into_expr).collect::<Result<Array<Expr>>>()?;
        tvm_ffi::cached_global_func!("ir.LambdaExprApply")
            .call_tuple((self, &arguments))?
            .try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.Range
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Range"]
#[type_final]
pub struct RangeObj {
    base: Object,
    pub begin: Expr,
    pub extent: Expr,
    pub loc: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<RangeObj>() == 48);
    assert!(::core::mem::align_of::<RangeObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Range {
    base: ObjectArc<RangeObj>,
}

impl Deref for Range {
    type Target = RangeObj;
    fn deref(&self) -> &RangeObj {
        &self.base
    }
}

impl RangeObj {
    pub(crate) fn new(begin: Expr, extent: Expr, loc: Location) -> Self {
        let base = Object::new();
        Self { base, begin, extent, loc }
    }
}

impl Range {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        begin: impl Into<Expr>,
        extent: impl Into<Expr>,
        loc: impl Into<Location>,
    ) -> Self {
        let obj = RangeObj::new(begin.into(), extent.into(), loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl Range {
    /// The range `[begin, end)` (C++ `Range(begin, end, loc)`);
    /// [`from_min_extent`](Self::from_min_extent) takes the extent instead of the end.
    pub fn new(begin: impl IntoExpr, end: impl IntoExpr, loc: impl Into<Location>) -> Result<Self> {
        let (begin, end) = (begin.into_expr()?, end.into_expr()?);
        let extent = Self::get_extent_from_begin_end(begin.clone(), end)?;
        Ok(Self::from_fields(begin, extent, loc.into()))
    }

    /// The range of `extent` values from `min_value` (C++ `Range::FromMinExtent`).
    pub fn from_min_extent(
        min_value: impl IntoExpr,
        extent: impl IntoExpr,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (min_value, extent) = (min_value.into_expr()?, extent.into_expr()?);
        support::prim_dtype(&min_value)?;
        support::prim_dtype(&extent)?;
        Ok(Self::from_fields(min_value, extent, loc.into()))
    }

    /// The extent `end - begin` of the range `[begin, end)`: the other operand when `end` adds
    /// `begin` to it, `end` when `begin` is zero, else `end - begin` folded
    /// (C++ `Range::GetExtentFromBeginEnd`).
    pub fn get_extent_from_begin_end(begin: impl IntoExpr, end: impl IntoExpr) -> Result<Expr> {
        let (begin, end) = (begin.into_expr()?, end.into_expr()?);
        support::prim_dtype(&begin)?;
        support::prim_dtype(&end)?;
        if let Some(add) = end.downcast_ref::<AddObj>() {
            if crate::ir::prim::expr_deep_equal(add.a.clone(), begin.clone())? {
                return Ok(add.b.clone());
            }
            if crate::ir::prim::expr_deep_equal(add.b.clone(), begin.clone())? {
                return Ok(add.a.clone());
            }
        }
        let is_zero =
            begin.downcast_ref::<IntImmObj>().is_some_and(|imm| imm.value == BigInt::from(0i64));
        match is_zero {
            true => Ok(end),
            false => crate::ir::prim::op::sub(end, begin),
        }
    }
}

// Operator attachment is separate from generated layouts and upcasts. Keep the native
// primitive-expression counterparts here, plus Expr, Var and Call for general handles whose
// primitive type is checked when an operator is applied.
crate::def_expr_op_over_loads!(Expr);
crate::def_expr_op_over_loads!(Var);
crate::def_expr_op_over_loads!(Call);
crate::def_expr_op_over_loads!(IntImm);
crate::def_expr_op_over_loads!(FloatImm);
crate::def_expr_op_over_loads!(TensorLoad);

// Primitive operator and vector reference classes inherit C++ PrimExpr.
crate::def_expr_op_over_loads!(crate::ir::prim::Cast);
crate::def_expr_op_over_loads!(crate::ir::prim::LShift);
crate::def_expr_op_over_loads!(crate::ir::prim::RShift);
crate::def_expr_op_over_loads!(crate::ir::prim::BitwiseAnd);
crate::def_expr_op_over_loads!(crate::ir::prim::BitwiseOr);
crate::def_expr_op_over_loads!(crate::ir::prim::BitwiseXor);
crate::def_expr_op_over_loads!(crate::ir::prim::BitwiseNot);
crate::def_expr_op_over_loads!(crate::ir::prim::Add);
crate::def_expr_op_over_loads!(crate::ir::prim::Sub);
crate::def_expr_op_over_loads!(crate::ir::prim::Mul);
crate::def_expr_op_over_loads!(crate::ir::prim::Div);
crate::def_expr_op_over_loads!(crate::ir::prim::Mod);
crate::def_expr_op_over_loads!(crate::ir::prim::FloorDiv);
crate::def_expr_op_over_loads!(crate::ir::prim::FloorMod);
crate::def_expr_op_over_loads!(crate::ir::prim::Min);
crate::def_expr_op_over_loads!(crate::ir::prim::Max);
crate::def_expr_op_over_loads!(crate::ir::prim::EQ);
crate::def_expr_op_over_loads!(crate::ir::prim::NE);
crate::def_expr_op_over_loads!(crate::ir::prim::LT);
crate::def_expr_op_over_loads!(crate::ir::prim::LE);
crate::def_expr_op_over_loads!(crate::ir::prim::GT);
crate::def_expr_op_over_loads!(crate::ir::prim::GE);
crate::def_expr_op_over_loads!(crate::ir::prim::And);
crate::def_expr_op_over_loads!(crate::ir::prim::Or);
crate::def_expr_op_over_loads!(crate::ir::prim::Not);
crate::def_expr_op_over_loads!(crate::ir::prim::Select);
crate::def_expr_op_over_loads!(crate::ir::prim::Let);
crate::def_expr_op_over_loads!(crate::ir::prim::Ramp);
crate::def_expr_op_over_loads!(crate::ir::prim::Broadcast);
crate::def_expr_op_over_loads!(crate::ir::prim::Shuffle);

// These symbolic reference classes also inherit C++ PrimExpr. The opaque Rust mirrors retain
// their existing allocation restrictions; attaching operators only forwards their handles.
crate::def_expr_op_over_loads!(crate::sym::iter_affine_map::IterMapExpr);
crate::def_expr_op_over_loads!(crate::sym::iter_affine_map::IterSplitExpr);
crate::def_expr_op_over_loads!(crate::sym::iter_affine_map::IterSumExpr);
crate::def_expr_op_over_loads!(crate::sym::analyzer::SplitExpr);
crate::def_expr_op_over_loads!(crate::sym::analyzer::SumExpr);
