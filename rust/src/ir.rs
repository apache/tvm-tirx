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

//! The shared IR: expressions, statements, types, functions, modules and operators.
//!
//! [`Expr`], [`Stmt`] and [`Type`] are the bases of the three node hierarchies, and [`IRModule`]
//! holds the functions of a program.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::{
//!     BaseFunc, DictAttrs, GlobalVar, IRModule, PrimType, PrimVar, Return, SeqStmt,
//! };
//! use tvm_tirx::ir::{PrimExpr, Var, UNKNOWN_LOC};
//! use tvm_tirx::script::script;
//! use tvm_tirx::tirx::Function;
//! use tvm_tirx::tvm_ffi::Map;
//!
//! let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
//! let y = PrimExpr::try_from(&x * 2 + 1)?;
//!
//! let body = SeqStmt::new([Return::new(&y, UNKNOWN_LOC)], UNKNOWN_LOC);
//! let attrs = DictAttrs::new(Map::new());
//! let func = Function::new([x], &body, PrimType::int(32), attrs, UNKNOWN_LOC);
//!
//! let functions = Map::from_iter([(GlobalVar::new("main", UNKNOWN_LOC), BaseFunc::from(func))]);
//! let module = IRModule::new(&functions, DictAttrs::new(Map::new()), &Map::new())?;
//! assert!(script(&module)?.contains("def main(x: T.int32) -> T.int32:"));
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// An object registered under `ir` that no file has a block for yet is generated here, to be moved
// to the file of the Python module that defines it.
// tvm-ffi-stubgen(prefix): ir
// `ir.Op` is hand-written in `op.rs`.
// tvm-ffi-stubgen(skip): ir.Op

pub mod attrs;
pub mod base;
pub mod dtype;
pub mod expr;
pub mod function;
pub mod global_info;
pub mod location;
pub mod module;
pub mod op;
pub mod op_def;
pub mod prim;
pub mod stmt;
pub(crate) mod support;
pub mod transform;
pub mod r#type;
pub mod type_def;
pub mod unique_name_supply;

pub use attrs::*;
pub use base::*;
pub use expr::*;
pub use function::*;
pub use global_info::*;
pub use location::*;
pub use module::*;
pub use op::*;
pub use op_def::*;
pub use stmt::*;
pub use r#type::*;
pub use type_def::*;
pub use unique_name_supply::*;

// tvm-ffi-stubgen(begin): import-section
// tvm-ffi-stubgen(end)
