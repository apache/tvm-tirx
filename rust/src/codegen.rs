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

//! Code generation. [`build_module`] compiles an [`IRModule`] into a runtime [`Module`] with the
//! code generator of the target's kind, and [`llvm`] queries the LLVM backend.

// tvm-ffi-stubgen(missing-docs): allow
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Module
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): crate::ir::module::IRModule
// tvm-ffi-stubgen(import-object): crate::target::Target

pub mod llvm;

// tvm-ffi-stubgen(begin): import-section
use crate::ir::module::IRModule;
use crate::target::Target;
use tvm_ffi::Module;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

/// Compile `module` into a runtime module with the code generator of the kind of `target`
/// (`codegen.Build`).
///
/// # Errors
///
/// Returns an error if the code generator of the kind is not built in or loaded, or the error it
/// raises.
///
/// # Examples
///
/// ```
/// use tvm_tirx::codegen;
/// use tvm_tirx::ir::{DictAttrs, GlobalVar, IRModule, PrimVar, UNKNOWN_LOC};
/// use tvm_tirx::target::Target;
/// use tvm_tirx::tvm_ffi::Map;
/// # use tvm_tirx::ir::{Expr, PrimType, Return, SeqStmt, Var};
/// # use tvm_tirx::tirx::Function;
/// # use tvm_tirx::tvm_ffi::{Any, String};
/// # let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// # let body = SeqStmt::new([Return::new(Expr::try_from(&x + 1)?, UNKNOWN_LOC)], UNKNOWN_LOC);
/// # let name = Any::from(String::from("add_one"));
/// # let attrs = DictAttrs::new(Map::from_iter([(String::from("global_symbol"), name)]));
/// # let add_one = Function::new([&x], &body, PrimType::int(32), attrs, UNKNOWN_LOC);
///
/// let functions = Map::from_iter([(GlobalVar::new("add_one", UNKNOWN_LOC), add_one.into())]);
/// let module = IRModule::new(&functions, DictAttrs::new(Map::new()), &Map::new())?;
///
/// let compiled = codegen::build_module(&module, Target::new("c")?)?;
/// assert!(compiled.implements_function_with_imports("add_one")?);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn build_module(module: impl Into<IRModule>, target: impl Into<Target>) -> Result<Module> {
    tvm_ffi::cached_global_func!("codegen.Build")
        .call_tuple((&module.into(), &target.into()))?
        .try_into()
}

/// Return whether the LLVM target `target`, or the current target, has the CPU feature `feature`,
/// or `false` for a target of another kind (`codegen.target_has_feature`).
///
/// # Errors
///
/// Returns an error if the LLVM backend, which registers the query, is not loaded, or if `target`
/// is `None` and there is no current target.
pub fn target_has_feature(feature: &str, target: Option<&Target>) -> Result<bool> {
    tvm_ffi::Function::get_global("codegen.target_has_feature")?
        .call_tuple((String::from(feature), target.cloned()))?
        .try_into()
}
