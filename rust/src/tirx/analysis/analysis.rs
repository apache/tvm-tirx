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

//! The analyses of tensor-level IR (`tirx.analysis.*`).
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{Expr, PrimType, PrimVar, Var, UNKNOWN_LOC};
//! use tvm_tirx::tirx::analysis;
//! use tvm_tirx::tvm_ffi::ObjectRefCore;
//!
//! let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
//! let y = PrimVar::new("y", PrimType::int(32), UNKNOWN_LOC);
//! let sum: Expr = (&x + &y).try_into()?;
//! let undefined = analysis::undefined_vars(sum, [&x])?;
//! assert!(undefined.len() == 1 && undefined.get(0)?.same_as(&y));
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): crate::ir::expr::Expr
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule
// tvm-ffi-stubgen(import-object): crate::tirx::function::Function
// tvm-ffi-stubgen(import-object): tvm_ffi::Any
// tvm-ffi-stubgen(import-object): tvm_ffi::Array
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): std::borrow::Borrow

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::Var;
use crate::ir::module::IRModule;
use crate::tirx::function::Function;
use std::borrow::Borrow;
use tvm_ffi::Any;
use tvm_ffi::Array;
use tvm_ffi::IntoObjectRef;
use tvm_ffi::Result;
// tvm-ffi-stubgen(end)

/// Return whether `func` has no illegal direct memory access from the host
/// (`tirx.analysis.VerifyMemory`).
///
/// A function without a target attribute, or not of the default calling convention, is not checked
/// and passes.
pub fn verify_memory(func: impl IntoObjectRef<Function>) -> Result<bool> {
    tvm_ffi::cached_global_func!("tirx.analysis.VerifyMemory")
        .call_tuple((func.into_object_ref().borrow(),))?
        .try_into()
}

/// Return the variables `node`, a statement or an expression, uses but neither defines nor finds in
/// `defs` (`tirx.analysis.UndefinedVars`).
pub fn undefined_vars(
    node: impl Into<Any>,
    defs: impl IntoIterator<Item = impl Into<Var>>,
) -> Result<Array<Var>> {
    let defs = defs.into_iter().map(Into::into).collect::<Array<Var>>();
    tvm_ffi::cached_global_func!("tirx.analysis.UndefinedVars")
        .call_tuple((&node.into(), &defs))?
        .try_into()
}

/// Return whether `func` is well formed; invalid IR raises an error instead if `assert_mode`
/// (`tirx.analysis.CheckWellFormed`).
pub fn check_well_formed(func: impl IntoObjectRef<Function>, assert_mode: bool) -> Result<bool> {
    tvm_ffi::cached_global_func!("tirx.analysis.CheckWellFormed")
        .call_tuple((func.into_object_ref().borrow(), assert_mode))?
        .try_into()
}

/// Return whether the TIRx functions of `module` are well formed; invalid IR raises an error
/// instead if `assert_mode` (`tirx.analysis.CheckWellFormed`).
pub fn check_well_formed_module(
    module: impl IntoObjectRef<IRModule>,
    assert_mode: bool,
) -> Result<bool> {
    tvm_ffi::cached_global_func!("tirx.analysis.CheckWellFormed")
        .call_tuple((module.into_object_ref().borrow(), assert_mode))?
        .try_into()
}

/// Return the coefficients `c` of `expr = sum(vars[i] * c[i]) + c[n]`, each invariant of `vars`, or
/// an empty array if `expr` has no such form (`tirx.analysis.DetectLinearEquation`).
pub fn detect_linear_equation(
    expr: impl IntoExpr,
    vars: impl IntoIterator<Item = impl Into<Var>>,
) -> Result<Array<Expr>> {
    let vars = vars.into_iter().map(Into::into).collect::<Array<Var>>();
    tvm_ffi::cached_global_func!("tirx.analysis.DetectLinearEquation")
        .call_tuple((expr.into_expr()?.borrow(), &vars))?
        .try_into()
}

/// Return the minimum and maximum of each of `vars` that the clip bound `expr` sets, concatenated
/// with `None` for no bound, or an empty array if `expr` is not a clip bound
/// (`tirx.analysis.DetectClipBound`).
pub fn detect_clip_bound(
    expr: impl IntoExpr,
    vars: impl IntoIterator<Item = impl Into<Var>>,
) -> Result<Array<Option<Expr>>> {
    let vars = vars.into_iter().map(Into::into).collect::<Array<Var>>();
    tvm_ffi::cached_global_func!("tirx.analysis.DetectClipBound")
        .call_tuple((expr.into_expr()?.borrow(), &vars))?
        .try_into()
}
