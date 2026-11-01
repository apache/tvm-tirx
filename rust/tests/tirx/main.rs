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

//! Tests of `tvm_tirx::tirx`, laid out like `src/tirx/`: the tests of `src/tirx/<path>.rs` are in
//! `tests/tirx/<path>.rs`.

#[path = "../common/mod.rs"]
mod common;

mod analysis;
mod function;
mod index_map;
mod layout;
mod op;
mod tensor;
mod thread_scope;
mod tile;
mod transform;

use std::sync::atomic::{AtomicUsize, Ordering};

use tvm_tirx::ir::{
    Location, UNKNOWN_LOC, dtype, BaseFunc, DictAttrs, Expr, GlobalInfo, GlobalVar, IRModule,
    IntImm, Range, SeqStmt, Stmt, Type, Var,
};
use tvm_tirx::runtime::MemoryScope;
use tvm_tirx::tirx::{Axis, Function, Iter, TileLayout};
use tvm_tirx::tvm_ffi::{Array, Map, ObjectRefCast, String as FfiString};

use common::global;

fn int32(value: i64) -> Expr {
    IntImm::from_dtype(dtype::int(32), value, UNKNOWN_LOC).unwrap().into()
}

fn var(name: &str, ty: impl Into<Type>) -> Var {
    global("ir.Var")
        .call_tuple((FfiString::from(name), Some(ty.into()), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap()
}

fn op(name: &str, a: impl Into<Expr>, b: impl Into<Expr>) -> Expr {
    global(name).call_tuple((a.into(), b.into(), Location::unknown())).unwrap().try_into().unwrap()
}

fn range(extent: i64) -> Range {
    global("ir.Range")
        .call_tuple((int32(0), Some(int32(extent)), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap()
}

fn as_i64(expr: &Expr) -> i64 {
    expr.clone().try_cast::<IntImm>().unwrap().value.to_i64().unwrap()
}

/// A new scope of the test's own, since the builtin ones cannot be registered again.
fn memory_scope() -> MemoryScope {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let index = COUNT.fetch_add(1, Ordering::Relaxed);
    MemoryScope::register(&format!("rust.tests.tirx.{index}")).unwrap()
}

/// The row-major layout of an 8 x 8 tile.
fn tile_8x8() -> TileLayout {
    let m = Axis::get("m").unwrap();
    TileLayout::new(
        [Iter::new(int32(8), int32(8), m.clone()), Iter::new(int32(8), int32(1), m)],
        Vec::<Iter>::new(),
        &Map::new(),
    )
}

fn void_function(params: Vec<Var>) -> Function {
    let zero = global("ir.Evaluate").call_tuple((int32(0), Location::unknown())).unwrap();
    let body: SeqStmt = global("ir.SeqStmt")
        .call_tuple((Array::<Stmt>::new(vec![zero.try_into().unwrap()]), Some(Location::unknown())))
        .unwrap()
        .try_into()
        .unwrap();
    let missing: Type = global("ir.MissingType").call_tuple(()).unwrap().try_into().unwrap();
    Function::new(params, &body, missing, DictAttrs::new(Map::new()), UNKNOWN_LOC)
}

fn module_of(name: &str, func: &Function) -> IRModule {
    let gv: GlobalVar = global("ir.GlobalVar")
        .call_tuple((FfiString::from(name), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap();
    let functions: Map<GlobalVar, BaseFunc> = [(gv, BaseFunc::from(func))].into_iter().collect();
    IRModule::from_functions(&functions, None, &Map::<FfiString, Array<GlobalInfo>>::new()).unwrap()
}
