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

/// Add the primitive-expression operators to an existing reference wrapper.
///
/// The type, a module-scope path to a reference wrapper implementing `Clone` and
/// `Into<PrimExpr>`, gets the operators `+`, `-`, `*`, `&`, `|`, `^`, `<<`, `>>`, unary `-` and `!`
/// and [`PrimOps`](crate::ir::prim::PrimOps), all returning an
/// [`ExprResult`](crate::ir::prim::ExprResult). The operators take
/// [`PrimExpr`](crate::ir::PrimExpr) operands, so the type must convert into one: its upcast to
/// `PrimExpr` declares that every value of it is typed by a `PrimType`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{Expr, ExprObj, PrimExpr, PrimType, UNKNOWN_LOC};
/// use tvm_tirx::sym::Analyzer;
/// use tvm_tirx::tvm_ffi::{self, rs_object, ObjectRefCore};
///
/// #[rs_object("my_ext.ThreadIndex", final, immutable)]
/// pub struct ThreadIndexObj {
///     base_: ExprObj,
///     pub axis: i64,
/// }
/// #[derive(Clone, tvm_ffi::derive::ObjectRef)]
/// #[object_ref(parent = Expr)]
/// pub struct ThreadIndex {
///     data_: tvm_ffi::ObjectArc<ThreadIndexObj>,
/// }
///
/// impl ThreadIndex {
///     /// The `int32` index of the thread along `axis`.
///     pub fn new(axis: i64) -> Self {
///         let base = ExprObj::new(UNKNOWN_LOC.into(), PrimType::int(32).into());
///         Self::from_object(ThreadIndexObj::new(base, axis))
///     }
/// }
///
/// // Every thread index is an `int32`, as `new` makes it.
/// tvm_ffi::impl_object_ref_upcast!(unsafe ThreadIndex => PrimExpr);
/// tvm_tirx::def_expr_op_over_loads!(ThreadIndex);
///
/// # fn main() -> tvm_tirx::tvm_ffi::Result<()> {
/// let tx = ThreadIndex::new(0);
/// let analyzer = Analyzer::new();
/// assert!(analyzer.can_prove_equal(&tx * 4 + &tx, &tx * 5)?);
/// # Ok(())
/// # }
/// ```
// `crate::` matches the caller's path and re-emits it, so it names the caller's crate.
#[allow(clippy::crate_in_macro_def)]
#[macro_export]
macro_rules! def_expr_op_over_loads {
    (crate::$($ty:ident)::+ $(,)?) => {
        $crate::def_expr_op_over_loads!(@impl crate::$($ty)::+);
    };
    (self::$($ty:ident)::+ $(,)?) => {
        $crate::def_expr_op_over_loads!(@impl self::$($ty)::+);
    };
    (super::$($ty:ident)::+ $(,)?) => {
        $crate::def_expr_op_over_loads!(@impl super::$($ty)::+);
    };
    (::$($ty:ident)::+ $(,)?) => {
        $crate::def_expr_op_over_loads!(@impl ::$($ty)::+);
    };
    ($($ty:ident)::+ $(,)?) => {
        // Anchor caller names before introducing generic operator parameters.
        $crate::def_expr_op_over_loads!(@impl self::$($ty)::+);
    };
    (@impl $ty:ty) => {
        impl ::core::convert::From<$ty> for $crate::ir::prim::ExprResult {
            fn from(value: $ty) -> Self {
                let value = ::core::convert::Into::<$crate::ir::PrimExpr>::into(value);
                Self::from(::core::result::Result::<_, $crate::tvm_ffi::Error>::Ok(value))
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

        $crate::def_expr_op_over_loads!(@binary $ty;
            Add::add, Sub::sub, Mul::mul, BitAnd::bitand, BitOr::bitor,
            BitXor::bitxor, Shl::shl, Shr::shr
        );
        $crate::def_expr_op_over_loads!(@unary $ty; Neg::neg, Not::not);
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

            $crate::def_expr_op_over_loads!(@literal $ty, $trait::$method;
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
