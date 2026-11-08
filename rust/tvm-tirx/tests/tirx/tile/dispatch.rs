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

//! Tests of `tirx::tile::dispatch`: the context tile instructions are dispatched in.

use tvm_tirx::ir::{UnknownLoc, UNKNOWN_LOC, PrimExpr, PrimType, Stmt, Var};
use tvm_tirx::target::Target;
use tvm_tirx::tirx::tile::DispatchContext;
use tvm_tirx::tirx::{TensorType, ThreadScope};
use tvm_tirx::tvm_ffi::{Any, EnumRef, Map, ObjectRefCore, String as FfiString};

use crate::common::{global, serial, structural_equal};
use crate::{int32, memory_scope, prim_var, var};

#[test]
fn dispatch_contexts() {
    let _serial = serial();
    let target = Target::new("cuda").unwrap();
    let warp = ThreadScope::get("cuda.warp").unwrap();
    let tx_var = prim_var("tx", PrimType::int(32));
    let ctx = DispatchContext::new(
        &target,
        &warp,
        [("threadIdx.x", tx_var.clone(), int32(32))],
        Map::new(),
        true,
        Map::new(),
        Map::new(),
        Map::new(),
        Map::new(),
        "warp",
    )
    .unwrap();
    assert_eq!(ctx.scope_kind().unwrap().as_str(), "warp");
    assert!(ctx.alloc_only().unwrap());

    assert!(ctx.cache_get("key").unwrap().is_none());
    ctx.cache_set("key", &tx_var).unwrap();
    let cached = ctx.cache_get("key").unwrap().unwrap();
    assert!(structural_equal(&cached.downcast::<Var>().unwrap(), &tx_var));

    let stmt: Stmt = global("ir.Evaluate")
        .call_tuple((int32(0), UnknownLoc::new()))
        .unwrap()
        .try_into()
        .unwrap();
    ctx.add_init_stmt(&stmt, false).unwrap();
    ctx.add_init_stmt(&stmt, true).unwrap();
    let scope = memory_scope();
    let ty =
        TensorType::new([int32(4)], PrimType::float(32), &scope, 0, None, UNKNOWN_LOC).unwrap();
    let tensor = var("T", &ty);
    ctx.add_alloc_tensor(&tensor, Vec::<PrimExpr>::new(), Map::<FfiString, Any>::new()).unwrap();
    ctx.add_post_tensor_def_stmt(&tensor, &stmt).unwrap();
    assert_eq!(ctx.callbacks().unwrap().len(), 4);
}
