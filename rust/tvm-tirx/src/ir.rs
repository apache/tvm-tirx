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

//! The IR shared by every dialect. Expressions ([`Expr`]), statements ([`Stmt`]) and types
//! ([`Type`]) are the three node hierarchies; functions are expressions, and an [`IRModule`] binds
//! them to global names, the unit of compilation that the passes of [`transform`] rewrite.

// tvm-ffi-stubgen(missing-docs): allow
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
