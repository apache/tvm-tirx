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

//! Tests of `tvm_tirx::ir`, laid out like `src/ir/`: the tests of `src/ir/<path>.rs` are in
//! `tests/ir/<path>.rs`.

#[path = "../common/mod.rs"]
mod common;

mod expr;
mod function;
mod location;
mod module;
mod op;
mod op_def;
mod prim;
mod stmt;
mod transform;
mod r#type;
mod unique_name_supply;

use std::sync::OnceLock;

use tvm_tirx::ir::*;
use tvm_tirx::runtime::MemoryScope;
use tvm_tirx::tvm_ffi::{
    AnyView, Array, DLDataType, DLDataTypeExt, IntoObjectRef, Map, String as FfiString,
};

use common::global;

/// The dtype a test names, for the cases that range over dtypes.
fn parse_dtype(name: &str) -> DLDataType {
    DLDataType::try_from_str(name).unwrap()
}

fn int32(value: i64) -> PrimExpr {
    IntImm::new(PrimType::int(32), value, UNKNOWN_LOC).unwrap().into()
}

fn int32_var(name: &str) -> PrimVar {
    PrimVar::new(name, PrimType::int(32), UNKNOWN_LOC)
}

fn op(name: &str, a: impl Into<Expr>, b: impl Into<Expr>) -> PrimExpr {
    global(name).call_tuple((a.into(), b.into(), UnknownLoc::new())).unwrap().try_into().unwrap()
}

/// A memory scope registered once for all the tests: Rust has no lookup of the builtin ones.
fn scope() -> MemoryScope {
    struct Shared(MemoryScope);
    // SAFETY: the scope is an immutable native object with an atomic reference count.
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}
    static SCOPE: OnceLock<Shared> = OnceLock::new();
    let shared = SCOPE.get_or_init(|| {
        let name = FfiString::from("rust.tests.ir.scope");
        Shared(
            global("runtime.MemoryScopeRegister").call_tuple((name,)).unwrap().try_into().unwrap(),
        )
    });
    shared.0.clone()
}

/// A variable of type `float32[16]`.
fn tensor_var(name: &str) -> Var {
    let shape: Array<PrimExpr> = [int32(16)].into_iter().collect();
    let f32_ty = PrimType::float(32);
    let (scope, alignment, layout) = (scope(), 0i32, Option::<Expr>::None);
    let ty: Type = global("tirx.TensorType")
        .call_packed(&[
            AnyView::from(&shape),
            AnyView::from(&f32_ty),
            AnyView::from(&scope),
            AnyView::from(&alignment),
            AnyView::from(&layout),
            AnyView::from(&UnknownLoc::new()),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    Var::new(name, ty, UNKNOWN_LOC)
}

fn seq(stmts: impl IntoIterator<Item = impl IntoObjectRef<Stmt>>) -> SeqStmt {
    SeqStmt::new(stmts, UNKNOWN_LOC)
}

/// A function without parameters that returns `value`.
fn function(value: i64) -> BaseFunc {
    let params = Array::<Var>::from_vec(vec![]);
    let body = Some(seq([Return::new(int32(value), UNKNOWN_LOC)]));
    let ret_type: Type = PrimType::int(32).into();
    let attrs = DictAttrs::new(Map::new());
    global("tirx.Function")
        .call_tuple((params, body, ret_type, attrs, UnknownLoc::new()))
        .unwrap()
        .try_into()
        .unwrap()
}
