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
//! The typed expression views, [`PrimExpr`](crate::ir::PrimExpr),
//! [`IntExpr`](crate::ir::IntExpr) and [`PrimVar`](crate::ir::PrimVar) (C++ `TypedExpr`).

/// Define a view: a reference to a `$obj`, the object of the reference `$base`, that `$check`
/// accepts, named `$type_str` and describing a rejected value by `$mismatch` like the C++ type
/// traits, which an `Any` holding no such value converts into by `$fallback`.
macro_rules! typed_expr_view {
    (
        $(#[$meta:meta])*
        $name:ident, $type_str:literal, base = $base:ident($obj:ident), check = $check:path,
        mismatch = $mismatch:path, fallback = $fallback:path
    ) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name {
            base: ObjectArc<$obj>,
        }

        // SAFETY: `base` is the reference; `from_data` requires its caller to have checked it.
        unsafe impl ObjectRefCore for $name {
            type ContainerType = $obj;

            #[inline]
            fn data(this: &Self) -> &ObjectArc<$obj> {
                &this.base
            }

            #[inline]
            fn into_data(this: Self) -> ObjectArc<$obj> {
                this.base
            }

            /// The view of `data`.
            ///
            /// # Safety
            ///
            /// The type of `data` is one the view accepts.
            #[inline]
            unsafe fn from_data(data: ObjectArc<$obj>) -> Self {
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
                $mismatch(data)
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
                unsafe { $check(data) }
            }

            unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
                let expr =
                    unsafe { <$base as AnyCompatible>::copy_from_any_view_after_check(data) };
                Self { base: ObjectRefCore::into_data(expr) }
            }

            unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
                let expr = unsafe { <$base as AnyCompatible>::move_from_any_after_check(data) };
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

        // SAFETY: the view is `repr(transparent)` over one object pointer, and
        // `check_any_strict` is its type check.
        tvm_ffi::unsafe_impl_object_ref_downcast_ref!($name);

        impl AsRef<$name> for $name {
            #[inline(always)]
            fn as_ref(&self) -> &Self {
                self
            }
        }

        // As `#[derive(ObjectRef)]` generates: the view converts into itself, owned or borrowed,
        // so `$name` and `&$name` are an `IntoExpr<$name>`.
        impl tvm_ffi::object::IntoObjectRef<$name> for $name {
            type Output = $name;

            #[inline(always)]
            fn into_object_ref(self) -> $name {
                self
            }
        }

        impl<'a> tvm_ffi::object::IntoObjectRef<$name> for &'a $name {
            type Output = &'a $name;

            #[inline(always)]
            fn into_object_ref(self) -> &'a $name {
                self
            }
        }

        tvm_ffi::impl_any_compatible_conversions!($name);

        // The type key and address, as `#[derive(ObjectRef)]` prints a reference, so a
        // `downcast` result can be unwrapped.
        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                tvm_ffi::object::fmt_object_ref(self, f)
            }
        }
        crate::def_expr_op_over_loads!($name);

        impl From<&$name> for $name {
            fn from(value: &$name) -> Self {
                value.clone()
            }
        }

        /// Dereference to the object, to read, as the reference it views does.
        impl Deref for $name {
            type Target = $obj;

            fn deref(&self) -> &$obj {
                &self.base
            }
        }

        impl $name {
            /// Borrow this expression as an [`Expr`], without a reference-count increment.
            #[inline]
            pub fn as_expr(&self) -> &Expr {
                // SAFETY: `Expr` is a `repr(C)` struct of one object pointer, as the view is, and
                // every value of the view is an `Expr`.
                unsafe { &*(self as *const Self as *const Expr) }
            }

            /// Return the type of this expression, the [`PrimType`] the view guarantees (C++
            /// `PrimExpr::ty()`).
            pub fn ty(&self) -> PrimType {
                self.base.ty.downcast_ref::<PrimType>().cloned().expect("a view is typed by a `PrimType`")
            }

            /// Return the dtype of this expression's type.
            pub fn dtype(&self) -> DLDataType {
                let ty = self.base.ty.downcast_ref::<PrimTypeObj>();
                ty.expect("a view is typed by a `PrimType`").dtype
            }
        }
    };
}

pub(crate) use typed_expr_view;
