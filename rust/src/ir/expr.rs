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
//! The objects `python/tvm_tirx/ir/expr.py` defines. [`Expr`] is the base of every expression;
//! [`PrimExpr`] and [`IntExpr`] are views of one checked to have a primitive or integer type. The
//! primitive arithmetic and vector expressions are in [`prim`](super::prim).
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{Call, Op, PrimExpr, PrimType, PrimVar, Type, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//!
//! let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
//! let f = PrimVar::new("f", PrimType::float(32), UNKNOWN_LOC);
//! let index = PrimExpr::try_from(&i * 4 + 1)?;
//! assert_eq!(script(&index)?, "i * 4 + 1");
//!
//! let exp = Call::new(None, Op::get("prim.exp")?, [&f], None, Vec::<Type>::new(), UNKNOWN_LOC)?;
//! assert_eq!(script(&exp)?, "T.exp(f)");
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// `custom-new` reserves `new` for a constructor with the C++ one's arguments and semantics
// (argument order, checks, defaults, implicit conversions, a derived type); the generated
// `from_fields` allocates the fields as given, inherited fields first.
// tvm-ffi-stubgen(custom-new): ir.GlobalVar
// tvm-ffi-stubgen(custom-new): ir.Tuple
// tvm-ffi-stubgen(custom-new): ir.TensorLoad
// tvm-ffi-stubgen(custom-new): ir.StringImm
// tvm-ffi-stubgen(custom-new): ir.Call
// tvm-ffi-stubgen(custom-new): ir.Var
// tvm-ffi-stubgen(custom-new): ir.LambdaExpr
// tvm-ffi-stubgen(custom-new): ir.Range
// tvm-ffi-stubgen(custom-new): ir.TupleGetItem
// tvm-ffi-stubgen(custom-new): ir.FloatImm
// tvm-ffi-stubgen(custom-new): ir.IntImm
// tvm-ffi-stubgen(custom-new): ir.GenericConst
// tvm-ffi-stubgen(custom-new): ir.DataTypeImm
// tvm-ffi-stubgen(custom-new): ir.TensorRegion
// Constants and loads are typed by a `PrimType`: their allocators take one, and they convert into
// a `PrimExpr` without a check.
// tvm-ffi-stubgen(field): ir.IntImm.ty -> PrimType
// tvm-ffi-stubgen(field): ir.FloatImm.ty -> PrimType
// tvm-ffi-stubgen(field): ir.TensorLoad.ty -> PrimType
// tvm-ffi-stubgen(field): ir.TensorLoad.indices -> Array<PrimExpr>
// tvm-ffi-stubgen(upcast): ir.IntImm -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.FloatImm -> PrimExpr
// tvm-ffi-stubgen(upcast): ir.TensorLoad -> PrimExpr
// A range is bounded by `PrimExpr`s, as C++ `RangeObj` holds `PrimExpr begin, extent`.
// tvm-ffi-stubgen(field): ir.Range.begin -> PrimExpr
// tvm-ffi-stubgen(field): ir.Range.extent -> PrimExpr
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
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// tvm-ffi-stubgen(import-object): tvm_ffi::object::IntoObjectRef

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
use std::borrow::Borrow;
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
use tvm_ffi::object::IntoObjectRef;
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

/// An index of [`Expr::subscript`]: an expression, or a `start:stop:step` slice.
#[derive(Clone)]
pub enum Subscript {
    /// The point index of an expression.
    Index(Expr),
    /// The slice `start:stop:step`, whose missing bounds are `None`.
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
    /// Return `self[indices]`, as the subscript rule of its type realizes it: a [`TupleGetItem`] of
    /// a tuple, a [`TensorLoad`] or [`TensorRegion`] of a tensor (`ir.SubscriptExprRealize`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the type of the expression has no subscript rule, or the error the
    /// rule raises.
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

// ---------------------------------------------------------------------------------------------
// Conversions into an expression

/// A value that converts into the expression type `T`, possibly failing: a `T` or a subtype of it,
/// owned or borrowed, a Rust literal, or an [`ExprResult`] under construction.
///
/// `T` is [`Expr`] by default, and [`PrimExpr`] where C++ takes a `PrimExpr`. The subtypes of `T`
/// are the types with an upcast to it, for which tvm-ffi generates an [`IntoObjectRef<T>`]; every
/// `IntoObjectRef<T>` is an `IntoExpr<T>`, so a new expression type converts into each type it
/// upcasts to without a list to maintain. The conversion never checks a type: an [`Expr`] or a
/// [`Var`] is not a `PrimExpr` until it is checked, as in C++, by `downcast_ref`, which borrows it
/// in place (C++ `as`), or `try_cast`, which takes it (C++ `as_or_throw`). Its one error is the
/// first error an [`ExprResult`] operand raised.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::Add;
/// use tvm_tirx::ir::{PrimExpr, PrimType, PrimVar, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::tvm_ffi::ObjectRefCore;
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let sum = Add::new(&x + 1, 2, UNKNOWN_LOC)?;
/// assert_eq!(script(&sum)?, "x + 1 + 2");
///
/// let f = PrimVar::new("f", PrimType::float(32), UNKNOWN_LOC);
/// assert!(Add::new(&f << 1, 2, UNKNOWN_LOC).is_err());
///
/// // A variable of any type is checked first, here borrowed in place.
/// let var = Var::new("y", PrimType::int(32), UNKNOWN_LOC);
/// let y = var.downcast_ref::<PrimVar>().expect("an int32 variable");
/// assert_eq!(script(&Add::new(y, &x, UNKNOWN_LOC)?)?, "y + x");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
///
/// An [`Expr`] is not an `IntoExpr<PrimExpr>`:
///
/// ```compile_fail
/// # use tvm_tirx::ir::prim::Add;
/// # use tvm_tirx::ir::{Expr, UNKNOWN_LOC};
/// fn add(a: &Expr, b: &Expr) -> tvm_tirx::tvm_ffi::Result<Add> {
///     Add::new(a, b, UNKNOWN_LOC)
/// }
/// ```
pub trait IntoExpr<T = Expr> {
    /// The expression: an owned `T` for an owned value or a literal, or a borrowed `&T` for a
    /// reference, which converts without a reference-count change.
    ///
    /// A function that only reads the expression borrows it with `.borrow()`; one that keeps
    /// it takes ownership with [`into_owned_expr`](Self::into_owned_expr).
    type Output: Borrow<T> + Into<T>;

    /// Convert into an expression, or return the first error an [`ExprResult`] raised.
    fn into_expr(self) -> Result<Self::Output>;

    /// Convert into an owned expression, as [`Cow::into_owned`](std::borrow::Cow::into_owned)
    /// does: an owned value moves, and only a borrowed expression is cloned.
    fn into_owned_expr(self) -> Result<T>
    where
        Self: Sized,
    {
        self.into_expr().map(Into::into)
    }
}

// A `T` or a subtype of it, owned or borrowed, through the `IntoObjectRef<T>` its upcast to `T`
// generates. Rust literals join through their own `IntoObjectRef` below, and `ExprResult`
// implements `IntoExpr` for each target it builds.
impl<T, S: IntoObjectRef<T>> IntoExpr<T> for S {
    type Output = S::Output;

    fn into_expr(self) -> Result<S::Output> {
        Ok(self.into_object_ref())
    }
}

// Rust literals are expressions too, owned and built on conversion, as `From` builds them.
macro_rules! impl_literal_into_expr {
    ($target:ty: $($t:ty),*) => {$(
        impl IntoObjectRef<$target> for $t {
            type Output = $target;

            fn into_object_ref(self) -> $target {
                self.into()
            }
        }
    )*};
}
impl_literal_into_expr!(Expr: i32, i64, bool, f32, f64);
impl_literal_into_expr!(PrimExpr: i32, i64, bool, f32, f64);

// Rust literals, as C++ converts them into an expression: an `i32` and an `f32` through the
// constructors `PrimExpr(int32_t)` and `PrimExpr(float)`, the others as constants of their type.

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

// ---------------------------------------------------------------------------------------------
// Typed views: `PrimExpr` and `IntExpr` (C++ `TypedExpr`)

crate::r#macro::typed_expr_view! {
    /// An expression typed by a `PrimType` (C++ `PrimExpr`): a constant, a variable, an operator
    /// on scalars or vectors, or a call that returns one.
    ///
    /// `PrimExpr` and [`IntExpr`] are typed views (C++ `TypedExpr`): an [`Expr`] checked when the
    /// view is made, so a function that takes one needs no check of its own. A view converts into
    /// an `Expr` for free; an `Expr` converts into a view only through `TryFrom`, whose error names
    /// the expression and its type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tvm_tirx::ir::{dtype, Expr, PrimExpr, PrimType, PrimVar, StringImm, Var, UNKNOWN_LOC};
    ///
    /// fn is_scalar(expr: &PrimExpr) -> bool {
    ///     expr.dtype().lanes == 1
    /// }
    ///
    /// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
    /// assert!(is_scalar(&PrimExpr::try_from(&x + 1)?));
    ///
    /// let text = Expr::from(StringImm::new("hi", UNKNOWN_LOC));
    /// let error = PrimExpr::try_from(text).err().unwrap();
    /// assert!(error.message().contains("ir.StringImm[ty=ir.StringType]"));
    /// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
    /// ```
    ///
    /// An `Expr` is not a `PrimExpr` until it is checked, nor a `PrimExpr` an `IntExpr`:
    ///
    /// ```compile_fail
    /// # use tvm_tirx::ir::{Expr, PrimExpr};
    /// fn widen(expr: Expr) -> PrimExpr {
    ///     expr
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// # use tvm_tirx::ir::{IntExpr, PrimExpr};
    /// fn narrow(expr: PrimExpr) -> IntExpr {
    ///     expr.into()
    /// }
    /// ```
    PrimExpr, "ir.PrimExpr", base = Expr(ExprObj), check = is_prim_expr,
    mismatch = typed_expr_mismatch, fallback = prim_literal
}

crate::r#macro::typed_expr_view! {
    /// A [`PrimExpr`] of a scalar signed or unsigned integer type of any width (C++ `IntExpr`),
    /// such as an index or an extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tvm_tirx::ir::{IntExpr, PrimType, PrimVar, Var, UNKNOWN_LOC};
    /// use tvm_tirx::script::script;
    ///
    /// let i = PrimVar::new("i", PrimType::int(64), UNKNOWN_LOC);
    /// let offset = IntExpr::try_from(&i * 4)?;
    /// assert_eq!(script(&offset)?, "i * T.int64(4)");
    ///
    /// let f = PrimVar::new("f", PrimType::float(32), UNKNOWN_LOC);
    /// assert!(IntExpr::try_from(&f * 4).is_err());
    /// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
    /// ```
    IntExpr, "ir.IntExpr", base = Expr(ExprObj), check = is_int_expr,
    mismatch = typed_expr_mismatch, fallback = int_literal
}

tvm_ffi::impl_object_upcast!(PrimExpr => Expr, IntExpr => Expr, IntExpr => PrimExpr);

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

/// An `i64` is an `int64` constant.
impl From<i64> for PrimExpr {
    fn from(value: i64) -> Self {
        IntImm::int64(value).into()
    }
}

/// A `bool` is a boolean constant.
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

/// An `f64` is a `float64` constant.
impl From<f64> for PrimExpr {
    fn from(value: f64) -> Self {
        FloatImm::float64(value).into()
    }
}

/// An `i32` is an `int32` constant.
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

/// Whether `data` holds an expression typed by a `PrimType`.
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn is_prim_expr(data: &TVMFFIAny) -> bool {
    unsafe { prim_dtype(data) }.is_some()
}

/// Whether `data` holds an expression typed by a scalar integer type.
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn is_int_expr(data: &TVMFFIAny) -> bool {
    unsafe { prim_dtype(data) }.is_some_and(scalar_int)
}

/// Whether `data` holds a `Var`, not a subclass, typed by a `PrimType` (C++
/// `TypeTraits<PrimVar>::CheckAnyStrict`).
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn is_prim_var(data: &TVMFFIAny) -> bool {
    crate::ir::op_def::sig::var_of_type::<PrimTypeObj>(data)
}

/// The type key of a value a [`PrimVar`] rejects (C++ `TypeTraits<PrimVar>::GetMismatchTypeInfo`).
fn type_key_mismatch(data: &TVMFFIAny) -> std::string::String {
    support::type_key(data.type_index)
}

/// No value that is not a variable converts into a [`PrimVar`].
///
/// # Safety
///
/// `data` is a live FFI value.
unsafe fn no_literal<T>(_: &TVMFFIAny) -> Option<T> {
    None
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

// ---------------------------------------------------------------------------------------------
// Expression nodes

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
    /// Create the global variable `name_hint`, of a missing type (C++ `GlobalVar(name_hint, loc)`).
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
    /// Create the tuple of `fields`, typed by the types of its fields when they are all known
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
    pub fn from_fields(
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
    /// Create the access to field `index` of `tuple_value`, typed by the field type and checked
    /// against the tuple type when it is known (C++ `TupleGetItem(tuple, index, loc)`).
    ///
    /// # Errors
    ///
    /// Returns an `IndexError` if `index` is negative, or not less than the number of fields of a
    /// tuple of known type.
    pub fn new(tuple_value: impl IntoExpr, index: i32, loc: impl Into<Location>) -> Result<Self> {
        let tuple_value = tuple_value.into_owned_expr()?;
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
        Ok(Self::from_fields(loc, ty, tuple_value, index))
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
    /// Create the load of the element of `source` at `indices`, typed by the element type that the
    /// type of `source` infers (C++ `TensorLoad(source, indices, loc)`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if the type of `source` does not support `TensorLoad`, or the error
    /// its type inference raises.
    pub fn new(
        source: impl IntoExpr,
        indices: impl IntoIterator<Item = impl IntoExpr<PrimExpr>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        static INFER_TYPE: support::TypeAttr =
            support::TypeAttr::new("__tvm_tensor_load_infer_type__");
        let (source, loc) = (source.into_owned_expr()?, loc.into());
        let indices = indices
            .into_iter()
            .map(IntoExpr::<PrimExpr>::into_owned_expr)
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
    pub fn from_fields(loc: impl Into<Location>, ty: impl Into<PrimType>, value: f64) -> Self {
        let obj = FloatImmObj::new(loc.into(), ty.into(), value);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(FloatImm => Expr, FloatImm => Constant, FloatImm => PrimExpr);
// tvm-ffi-stubgen(end)

impl FloatImm {
    /// Create the floating-point constant `value` of type `ty`, checked to fit it
    /// (C++ `FloatImm(ty, value, loc)`).
    ///
    /// ```
    /// use tvm_tirx::ir::expr::FloatImm;
    /// use tvm_tirx::ir::location::UnknownLoc;
    /// use tvm_tirx::ir::r#type::PrimType;
    /// let half = FloatImm::new(PrimType::float(16), 0.5, UnknownLoc::new())?;
    /// assert_eq!(half.value, 0.5);
    /// assert!(FloatImm::new(PrimType::int(32), 0.5, UnknownLoc::new()).is_err());
    /// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `ty` is not a scalar floating-point type, or a finite `value` is
    /// out of its range.
    pub fn new(ty: impl Into<PrimType>, value: f64, loc: impl Into<Location>) -> Result<Self> {
        let ty = ty.into();
        check_float_imm(ty.dtype, value)?;
        Ok(Self::from_fields(loc, ty, value))
    }

    /// [`FloatImm::new`] of the primitive type of `dtype`.
    pub(crate) fn from_dtype(
        dtype: DLDataType,
        value: f64,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        Self::new(PrimType::from_dtype(dtype)?, value, loc)
    }

    /// Create the `float32` constant `value` at the unknown location
    /// (C++ `FloatImm(PrimType::Float(32), value)`).
    pub fn float32(value: f32) -> Self {
        Self::builtin(dtype::float(32), value as f64)
    }

    /// Create the `float64` constant `value` at the unknown location
    /// (C++ `FloatImm(PrimType::Float(64), value)`).
    pub fn float64(value: f64) -> Self {
        Self::builtin(dtype::float(64), value)
    }

    /// [`FloatImm::new`] of a value its type holds, which fails only when the compiler
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

/// An `f64` is a `float64` constant.
impl From<f64> for FloatImm {
    fn from(value: f64) -> Self {
        Self::float64(value)
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
    pub fn from_fields(
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
    /// Create the integer constant `value` of the scalar integer or boolean type `ty`, checked to
    /// fit it (C++ `IntImm(ty, value, loc)`).
    ///
    /// ```
    /// use tvm_tirx::ir::expr::IntImm;
    /// use tvm_tirx::ir::location::UnknownLoc;
    /// use tvm_tirx::ir::r#type::PrimType;
    /// let byte = IntImm::new(PrimType::uint(8), 255, UnknownLoc::new())?;
    /// assert_eq!(byte.value.to_string(), "255");
    /// assert!(IntImm::new(PrimType::uint(8), 256, UnknownLoc::new()).is_err());
    /// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if `ty` is not a scalar integer or boolean type, or `value` is out of
    /// its range.
    pub fn new(
        ty: impl Into<PrimType>,
        value: impl Into<BigInt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (ty, value) = (ty.into(), value.into());
        check_int_imm(ty.dtype, &value)?;
        Ok(Self::from_fields(loc, ty, value))
    }

    /// [`IntImm::new`] of the primitive type of `dtype`.
    pub(crate) fn from_dtype(
        dtype: DLDataType,
        value: impl Into<BigInt>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        Self::new(PrimType::from_dtype(dtype)?, value, loc)
    }

    /// Create the `int32` constant `value` at the unknown location (C++ `IntImm::Int32(value)`).
    pub fn int32(value: i32) -> Self {
        Self::builtin(dtype::int(32), value)
    }

    /// Create the `int64` constant `value` at the unknown location (C++ `IntImm::Int64(value)`).
    pub fn int64(value: i64) -> Self {
        Self::builtin(dtype::int(64), value)
    }

    /// Create the boolean constant `value` at the unknown location (C++ `IntImm::Bool(value)`).
    pub fn bool(value: bool) -> Self {
        Self::builtin(dtype::bool(), value)
    }

    /// [`IntImm::new`] of a value its type holds, which fails only when the compiler
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

/// An `i64` is an `int64` constant.
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
    pub fn from_fields(
        loc: impl Into<Location>,
        ty: impl Into<Type>,
        value: impl Into<Any>,
    ) -> Self {
        let obj = GenericConstObj::new(loc.into(), ty.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(GenericConst => Expr, GenericConst => Constant);
// tvm-ffi-stubgen(end)

impl GenericConst {
    /// Create the constant `value` of type `ty`, checked to have a type and not to be a primitive
    /// literal (C++ `GenericConst(ty, value, loc)`).
    ///
    /// # Errors
    ///
    /// Returns a `TypeError` if `ty` is a [`MissingType`], or `value` is an integer, a boolean, a
    /// float or a string.
    pub fn new(
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
        Ok(Self::from_fields(loc, ty, value))
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
    pub fn from_fields(
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
    /// Create the data type `value` as a constant of type [`AnyType`] (C++ `DataTypeImm(value,
    /// loc)`).
    pub fn new(value: DLDataType, loc: impl Into<Location>) -> Self {
        let ty = AnyType::new(UnknownLoc::new());
        Self::from_fields(loc, ty, value)
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
    /// Create the string constant `value`, of type [`StringType`] (C++ `StringImm(value, loc)`).
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
    /// Create the call of `op` on `args`, of type `ty` or the type
    /// [`reinfer_type`](Self::reinfer_type) gives it when `None` (C++ `Call(ty, op, args, attrs,
    /// ty_args, loc)`).
    ///
    /// The call is not validated; [`validate`](Self::validate) checks it.
    ///
    /// # Errors
    ///
    /// Returns the error of the type inference when `ty` is `None`.
    pub fn new(
        ty: Option<&Type>,
        op: impl IntoExpr,
        args: impl IntoIterator<Item = impl IntoExpr>,
        attrs: Option<&Attrs>,
        ty_args: impl IntoIterator<Item = impl Into<Type>>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (op, attrs, loc) = (op.into_owned_expr()?, attrs.cloned(), loc.into());
        let args =
            args.into_iter().map(IntoExpr::into_owned_expr).collect::<Result<Array<Expr>>>()?;
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

    /// Check the call against the contract its operator registered: the operator's validator, if it
    /// has one (C++ `Call::Validate`).
    ///
    /// # Errors
    ///
    /// Returns the error the validator raises.
    pub fn validate(&self) -> Result<()> {
        let Some(op) = self.op.downcast_ref::<OpObj>() else { return Ok(()) };
        if op.validator().type_index() == TVMFFITypeIndex::kTVMFFINone as i32 {
            return Ok(());
        }
        support::NativeHook::from_any(op.validator().clone())?.call_void(self)
    }

    /// Return the result type the operator infers from the current inputs, without changing the
    /// call (C++ `Call::ReinferType`).
    ///
    /// # Errors
    ///
    /// Returns the error of the operator's `FInferType`, or a `TypeError` if its `TFixedReturnType`
    /// is not a type.
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
    pub fn from_fields(
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

impl TensorRegion {
    /// Create the region `region` of `source`, of type `ty` (C++ `TensorRegion(source, region, ty,
    /// loc)`).
    pub fn new(
        source: impl Into<Expr>,
        region: impl Into<Array<Range>>,
        ty: impl Into<Type>,
        loc: impl Into<Location>,
    ) -> Self {
        Self::from_fields(loc, ty, source, region)
    }
}

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
    /// Create the variable `name` of type `ty` (C++ `Var(name, ty, loc)`); a variable whose type is
    /// not known yet takes [`MissingType::new`].
    pub fn new(name: &str, ty: impl Into<Type>, loc: impl Into<Location>) -> Self {
        Self::from_fields(loc.into(), ty.into(), String::from(name))
    }
}

crate::r#macro::typed_expr_view! {
    /// A [`Var`] typed by a `PrimType` (C++ `PrimVar`): a scalar or vector variable, which takes
    /// part in arithmetic as a [`PrimExpr`].
    ///
    /// A `PrimVar` is a view of a `Var`, as [`PrimExpr`] is of an `Expr`: [`PrimVar::new`] types it
    /// by a `PrimType`, and a `Var` becomes one only through the check of C++ `PrimVar`, by
    /// `downcast_ref`, which borrows it in place, or `try_cast`, which takes it. It converts into a
    /// `Var`, a `PrimExpr` and an `Expr` for free, and reads the variable through `Deref`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tvm_tirx::ir::{MissingType, PrimExpr, PrimType, PrimVar, Var, UNKNOWN_LOC};
    /// use tvm_tirx::script::script;
    /// use tvm_tirx::tvm_ffi::{ObjectRefCast, ObjectRefCore};
    ///
    /// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
    /// assert_eq!(i.name.as_str(), "i");
    /// assert_eq!(script(&PrimExpr::try_from(&i * 4 + 1)?)?, "i * 4 + 1");
    ///
    /// let var: Var = i.into();
    /// assert!(var.downcast_ref::<PrimVar>().is_some());
    /// let untyped = Var::new("u", MissingType::new(), UNKNOWN_LOC);
    /// assert!(untyped.try_cast::<PrimVar>().is_err());
    /// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
    /// ```
    ///
    /// A `Var` takes no part in arithmetic until it is checked, as in C++:
    ///
    /// ```compile_fail
    /// # use tvm_tirx::ir::{PrimType, Var, UNKNOWN_LOC};
    /// let x = Var::new("x", PrimType::int(32), UNKNOWN_LOC);
    /// let _ = &x + 1;
    /// ```
    PrimVar, "ir.Var", base = Var(VarObj), check = is_prim_var, mismatch = type_key_mismatch,
    fallback = no_literal
}

tvm_ffi::impl_object_upcast!(PrimVar => Expr, PrimVar => Var, PrimVar => PrimExpr);

impl PrimVar {
    /// Create the variable `name` of the primitive type `ty` (C++ `PrimVar(name, ty, loc)`).
    pub fn new(name: &str, ty: impl Into<PrimType>, loc: impl Into<Location>) -> Self {
        let var = Var::new(name, ty.into(), loc);
        // SAFETY: the variable is typed by a `PrimType`.
        unsafe { Self::from_data(ObjectRefCore::into_data(var)) }
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
    /// Create the lambda of `vars` that evaluates to `body`, of the function type from the types of
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

    /// Return the body with `arguments` substituted for the variables (`ir.LambdaExprApply`).
    ///
    /// # Errors
    ///
    /// Returns a `ValueError` if the number of arguments is not the number of variables.
    pub fn apply(&self, arguments: impl IntoIterator<Item = impl IntoExpr>) -> Result<Expr> {
        let arguments = arguments
            .into_iter()
            .map(IntoExpr::into_owned_expr)
            .collect::<Result<Array<Expr>>>()?;
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
    pub begin: PrimExpr,
    pub extent: PrimExpr,
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
    pub(crate) fn new(begin: PrimExpr, extent: PrimExpr, loc: Location) -> Self {
        let base = Object::new();
        Self { base, begin, extent, loc }
    }
}

impl Range {
    /// Lossless complete-field allocation.
    pub fn from_fields(
        begin: impl Into<PrimExpr>,
        extent: impl Into<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Self {
        let obj = RangeObj::new(begin.into(), extent.into(), loc.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl Range {
    /// Create the range `[begin, end)` (C++ `Range(begin, end, loc)`);
    /// [`from_min_extent`](Self::from_min_extent) takes the extent instead of the end.
    pub fn new(
        begin: impl IntoExpr<PrimExpr>,
        end: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (begin, end) = (begin.into_owned_expr()?, end.into_owned_expr()?);
        let extent = Self::get_extent_from_begin_end(&begin, end)?;
        Ok(Self::from_fields(begin, extent, loc.into()))
    }

    /// Create the range of `extent` values from `min_value` (C++ `Range::FromMinExtent`).
    pub fn from_min_extent(
        min_value: impl IntoExpr<PrimExpr>,
        extent: impl IntoExpr<PrimExpr>,
        loc: impl Into<Location>,
    ) -> Result<Self> {
        let (min_value, extent) = (min_value.into_owned_expr()?, extent.into_owned_expr()?);
        Ok(Self::from_fields(min_value, extent, loc.into()))
    }

    /// Return the extent `end - begin` of the range `[begin, end)`, folded
    /// (C++ `Range::GetExtentFromBeginEnd`).
    pub fn get_extent_from_begin_end(
        begin: impl IntoExpr<PrimExpr>,
        end: impl IntoExpr<PrimExpr>,
    ) -> Result<PrimExpr> {
        let (begin, end) = (begin.into_owned_expr()?, end.into_owned_expr()?);
        if let Some(add) = end.downcast_ref::<AddObj>() {
            if crate::ir::prim::expr_deep_equal(&add.a, &begin)? {
                return Ok(add.b.clone());
            }
            if crate::ir::prim::expr_deep_equal(&add.b, &begin)? {
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

// The operators apply to the expressions typed by a `PrimType`, the subtypes of `PrimExpr`, as the
// C++ operators take `PrimExpr`; the typed views get theirs from `typed_expr_view!`. An `Expr`, a
// `Var` or a `Call` is checked first, with `downcast_ref` or `try_cast`.
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
