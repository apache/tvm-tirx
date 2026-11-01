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
//! Expression operators for existing reference wrappers.

/// Add primitive-expression operators to an existing reference wrapper.
///
/// Pass a module-scope path to a type implementing `Clone` and `Into<Expr>`. Define its object
/// layout, reference wrapper and ancestor upcasts separately, using `tvm_ffi::def_object_ref!` or
/// generated bindings. For a deeper parent, also provide the upcast to `Expr`.
///
/// Owned operands move their handle; borrowed operands clone the handle. Operators return
/// [`ExprResult`](crate::ir::prim::ExprResult), preserving contextual literal typing and the
/// first construction error. Import [`PrimOps`](crate::ir::prim::PrimOps) for named operations
/// such as `floordiv` and `less`. Assignment operators apply only to `ExprResult`.
///
/// ```no_run
/// # fn main() {}
/// use tvm_tirx::{def_ex_op_overloads, ir::{Expr, ExprObj}, tvm_ffi};
///
/// #[repr(C)]
/// #[derive(tvm_ffi::derive::Object)]
/// #[type_key = "example.MyExpr"]
/// struct MyExprObj {
///     base: ExprObj,
/// }
/// tvm_ffi::def_object_ref!(MyExpr, Expr, MyExprObj);
/// def_ex_op_overloads!(MyExpr);
///
/// fn twice(value: &MyExpr) -> tvm_ffi::Result<Expr> {
///     (value * 2).try_into()
/// }
/// ```
#[macro_export]
macro_rules! def_ex_op_overloads {
    (crate::$($ty:ident)::+ $(,)?) => {
        $crate::def_ex_op_overloads!(@impl crate::$($ty)::+);
    };
    (self::$($ty:ident)::+ $(,)?) => {
        $crate::def_ex_op_overloads!(@impl self::$($ty)::+);
    };
    (super::$($ty:ident)::+ $(,)?) => {
        $crate::def_ex_op_overloads!(@impl super::$($ty)::+);
    };
    (::$($ty:ident)::+ $(,)?) => {
        $crate::def_ex_op_overloads!(@impl ::$($ty)::+);
    };
    ($($ty:ident)::+ $(,)?) => {
        // Anchor caller names before introducing generic operator parameters.
        $crate::def_ex_op_overloads!(@impl self::$($ty)::+);
    };
    (@impl $ty:ty) => {
        impl ::core::convert::From<$ty> for $crate::ir::prim::ExprResult {
            fn from(value: $ty) -> Self {
                Self::from(::core::result::Result::<$crate::ir::Expr, $crate::tvm_ffi::Error>::Ok(
                    ::core::convert::Into::<$crate::ir::Expr>::into(value),
                ))
            }
        }

        impl ::core::convert::From<&$ty> for $crate::ir::prim::ExprResult {
            fn from(value: &$ty) -> Self {
                Self::from(<$ty as ::core::clone::Clone>::clone(value))
            }
        }

        impl $crate::ir::prim::PrimOps for $ty {
            fn to_expr_result(&self) -> $crate::ir::prim::ExprResult {
                ::core::convert::From::from(self)
            }
        }

        $crate::def_ex_op_overloads!(@binary $ty;
            Add::add, Sub::sub, Mul::mul, BitAnd::bitand, BitOr::bitor,
            BitXor::bitxor, Shl::shl, Shr::shr
        );
        $crate::def_ex_op_overloads!(@unary $ty; Neg::neg, Not::not);
    };
    (@binary $ty:ty; $($trait:ident::$method:ident),+ $(,)?) => {
        $(
            impl<R: $crate::ir::prim::Operand> ::core::ops::$trait<R> for $ty {
                type Output = $crate::ir::prim::ExprResult;
                fn $method(self, rhs: R) -> Self::Output {
                    <Self::Output as ::core::ops::$trait<R>>::$method(
                        ::core::convert::From::from(self), rhs,
                    )
                }
            }

            impl<R: $crate::ir::prim::Operand> ::core::ops::$trait<R> for &$ty {
                type Output = $crate::ir::prim::ExprResult;
                fn $method(self, rhs: R) -> Self::Output {
                    <Self::Output as ::core::ops::$trait<R>>::$method(
                        ::core::convert::From::from(self), rhs,
                    )
                }
            }

            $crate::def_ex_op_overloads!(@literal $ty, $trait::$method;
                ::core::primitive::i32, ::core::primitive::i64,
                ::core::primitive::f32, ::core::primitive::f64);
        )+
    };
    (@literal $ty:ty, $trait:ident::$method:ident; $($literal:ty),+ $(,)?) => {
        $(
            impl ::core::ops::$trait<$ty> for $literal {
                type Output = $crate::ir::prim::ExprResult;
                fn $method(self, rhs: $ty) -> Self::Output {
                    <Self as ::core::ops::$trait<Self::Output>>::$method(
                        self, ::core::convert::From::from(rhs),
                    )
                }
            }

            impl ::core::ops::$trait<&$ty> for $literal {
                type Output = $crate::ir::prim::ExprResult;
                fn $method(self, rhs: &$ty) -> Self::Output {
                    <Self as ::core::ops::$trait<Self::Output>>::$method(
                        self, ::core::convert::From::from(rhs),
                    )
                }
            }
        )+
    };
    (@unary $ty:ty; $($trait:ident::$method:ident),+ $(,)?) => {
        $(
            impl ::core::ops::$trait for $ty {
                type Output = $crate::ir::prim::ExprResult;
                fn $method(self) -> Self::Output {
                    <Self::Output as ::core::ops::$trait>::$method(
                        ::core::convert::From::from(self),
                    )
                }
            }

            impl ::core::ops::$trait for &$ty {
                type Output = $crate::ir::prim::ExprResult;
                fn $method(self) -> Self::Output {
                    <Self::Output as ::core::ops::$trait>::$method(
                        ::core::convert::From::from(self),
                    )
                }
            }
        )+
    };
}
