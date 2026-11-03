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
//! The typed expression views, [`PrimExpr`](crate::ir::PrimExpr) and
//! [`IntExpr`](crate::ir::IntExpr) (C++ `TypedExpr`).

/// Define a view: a reference to an `ExprObj` whose type `$accepts` accepts, named `$type_str`
/// like the C++ type, which an `Any` holding no expression converts into by `$fallback`.
///
/// `ir/expr.rs` defines `PrimExpr` and `IntExpr` with it; the expansion names the types and
/// traits that file imports.
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

pub(crate) use typed_expr_view;
