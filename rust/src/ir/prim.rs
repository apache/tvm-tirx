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

//! Primitive expressions: scalar arithmetic, comparisons and vector lanes.
//!
//! [`expr`] holds the scalar expressions, [`vector_expr`] the vector ones and [`op`] the
//! type-promoting operators.

// tvm-ffi-stubgen(missing-docs): allow
// An object registered under `prim` that no file has a block for yet is generated here, to be moved
// to the file of the Python module that defines it.
// tvm-ffi-stubgen(prefix): prim
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr

mod const_fold;
pub mod expr;
pub mod op;
pub mod vector_expr;

pub use expr::*;
pub use op::*;
pub use vector_expr::*;

// tvm-ffi-stubgen(import-object): std::borrow::Borrow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use std::borrow::Borrow;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

/// Return whether `lhs` and `rhs` are the same expression tree, without remapping variables: use
/// structural equality for that (`prim.expr_deep_equal`).
pub fn expr_deep_equal(lhs: impl IntoExpr<PrimExpr>, rhs: impl IntoExpr<PrimExpr>) -> Result<bool> {
    tvm_ffi::cached_global_func!("prim.expr_deep_equal")
        .call_tuple((lhs.into_expr()?.borrow(), rhs.into_expr()?.borrow()))?
        .try_into()
}
