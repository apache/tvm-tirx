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

//! Tests of `tirx::analysis::analysis`: memory accesses, undefined variables, linear patterns and
//! well-formedness.

use tvm_tirx::ir::PrimType;
use tvm_tirx::tirx::analysis;

use crate::common::{serial, structural_equal};
use crate::{as_i64, int32, module_of, op, prim_var, var, void_function};

#[test]
fn well_formedness() {
    let _serial = serial();
    let func = void_function(vec![var("x", PrimType::int(32))]);
    assert!(analysis::verify_memory(&func).unwrap());
    assert!(analysis::check_well_formed(&func, false).unwrap());
    assert!(analysis::check_well_formed_module(module_of("main", &func), false).unwrap());
}

#[test]
fn expression_analyses() {
    let _serial = serial();
    let (x, y) = (prim_var("x", PrimType::int(32)), prim_var("y", PrimType::int(32)));
    let undefined = analysis::undefined_vars(op("prim._OpAdd", &x, &y), [x.clone()]).unwrap();
    assert_eq!(undefined.len(), 1);
    assert!(structural_equal(&undefined.get(0).unwrap(), &y));

    let linear = op("prim._OpAdd", op("prim._OpMul", &x, int32(4)), int32(2));
    let coeffs = analysis::detect_linear_equation(linear, [x.clone()]).unwrap();
    assert_eq!((as_i64(&coeffs.get(0).unwrap()), as_i64(&coeffs.get(1).unwrap())), (4, 2));
    let clip = op("prim._OpAnd", op("prim._OpLE", int32(1), &x), op("prim._OpLT", &x, int32(8)));
    let bounds = analysis::detect_clip_bound(clip, [x.clone()]).unwrap();
    assert_eq!(bounds.len(), 2);
    assert!(bounds.get(0).unwrap().is_some());
}
