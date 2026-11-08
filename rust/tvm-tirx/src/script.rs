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

//! TVMScript printing. [`script`] prints an IR object as TVMScript, the Python-syntax text form
//! of the IR.

// tvm-ffi-stubgen(missing-docs): allow
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyCompatible
// tvm-ffi-stubgen(import-object): tvm_ffi::AnyView
// tvm-ffi-stubgen(import-object): tvm_ffi::ObjectRefCore
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String

// tvm-ffi-stubgen(begin): import-section
use tvm_ffi::AnyCompatible;
use tvm_ffi::AnyView;
use tvm_ffi::ObjectRefCore;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

/// Print the IR object `obj` as TVMScript, with the default printer config
/// (`script.printer.Script`).
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{
///     DictAttrs, Expr, PrimExpr, PrimType, PrimVar, Return, SeqStmt, Var, UNKNOWN_LOC,
/// };
/// use tvm_tirx::script::script;
/// use tvm_tirx::tirx::Function;
/// use tvm_tirx::tvm_ffi::Map;
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// assert_eq!(script(&PrimExpr::try_from(&x * 2 + 1)?)?, "x * 2 + 1");
///
/// let body = SeqStmt::new([Return::new(Expr::try_from(&x + 1)?, UNKNOWN_LOC)], UNKNOWN_LOC);
/// let attrs = DictAttrs::new(Map::new());
/// let func = Function::new([&x], &body, PrimType::int(32), attrs, UNKNOWN_LOC);
/// let text = script(&func)?;
/// assert!(text.contains("def main(x: T.int32) -> T.int32:\n    return x + 1"));
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn script<T: ObjectRefCore + AnyCompatible>(obj: &T) -> Result<String> {
    tvm_ffi::cached_global_func!("script.printer.Script")
        .call_packed(&[AnyView::from(obj), AnyView::new()])?
        .try_into()
}
