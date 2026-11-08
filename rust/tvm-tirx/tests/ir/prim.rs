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

//! Tests of `ir::prim`, laid out like `src/ir/prim/`: the node constructors and the
//! type-promoting operators.

mod expr;
mod op;
mod vector_expr;

use tvm_tirx::ir::prim;
use tvm_tirx::ir::{UNKNOWN_LOC, Expr, IntImm, PrimType, PrimVar};
use tvm_tirx::tvm_ffi::{DLDataTypeExt, ObjectRefCore};

use crate::common::serial;

/// A variable of the primitive type `ty`.
fn var(name: &str, ty: PrimType) -> PrimVar {
    PrimVar::new(name, ty, UNKNOWN_LOC)
}

/// The dtype of a primitive expression, by name.
fn dtype(expr: &impl AsRef<Expr>) -> String {
    let ty: PrimType = expr.as_ref().ty.clone().downcast().unwrap();
    ty.dtype.to_string().as_str().to_owned()
}

fn int_value(expr: impl Into<Expr>) -> i64 {
    let imm: IntImm = expr.into().downcast().unwrap();
    imm.value.to_i64().unwrap()
}

#[test]
fn expr_deep_equal() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let a = prim::op::add(&x, 1).unwrap();
    let b = prim::op::add(&x, 1).unwrap();
    assert!(prim::expr_deep_equal(&a, &b).unwrap());
    assert!(!prim::expr_deep_equal(&a, var("x", PrimType::int(32))).unwrap());
}
