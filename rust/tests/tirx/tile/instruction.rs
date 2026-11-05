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

//! Tests of `tirx::tile::instruction`: the tensor regions and validators of tile instructions.

use tvm_tirx::ir::{Location, UNKNOWN_LOC, Expr, Op, PrimType};
use tvm_tirx::tirx::tile::{as_tensor_region, configure_tensor_instruction};
use tvm_tirx::tirx::TensorType;
use tvm_tirx::tvm_ffi::Array;

use crate::common::{global, serial, structural_equal};
use crate::{int32, memory_scope, var};

#[test]
fn tensor_regions() {
    let _serial = serial();
    let scope = memory_scope();
    let ty =
        TensorType::new([int32(4)], PrimType::float(32), &scope, 0, None, UNKNOWN_LOC).unwrap();
    let tensor = var("T", &ty);
    let load: Expr = global("ir.TensorLoad")
        .call_tuple((Expr::from(&tensor), Array::from_vec(vec![int32(1)]), Location::unknown()))
        .unwrap()
        .try_into()
        .unwrap();
    let region = as_tensor_region(&load).unwrap();
    assert!(structural_equal(&as_tensor_region(&region).unwrap(), &region));
}

#[test]
fn tensor_instructions() {
    let _serial = serial();
    Op::register("rust.tests.tirx.instruction", "").unwrap();
    let instruction = Op::get("rust.tests.tirx.instruction").unwrap();
    configure_tensor_instruction(&instruction, |_call| Ok(())).unwrap();
    assert!(configure_tensor_instruction(&instruction, |_call| Ok(())).is_err());
}
