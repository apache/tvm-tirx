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

//! Tests of `tirx::function`.

use tvm_tirx::ir::{
    UNKNOWN_LOC, Location, DictAttrs, MissingType, PrimType, Return, SeqStmt, Type, Var,
};
use tvm_tirx::tirx::{renew_def, Function};
use tvm_tirx::tvm_ffi::{Any, Array, Map, String as FfiString};

use super::{var, void_function};
use crate::common::{agree, native, serial, structural_equal};

#[test]
fn functions() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let func = void_function(vec![x.clone()]);
    assert_eq!(func.params.len(), 1);
    let renewed = renew_def(&func).unwrap();
    // The same function, alpha-renamed: a fresh parameter in place of `x`.
    assert!(structural_equal(&renewed, &func));
    assert!(!structural_equal(&renewed.params.get(0).unwrap(), &x));
}

#[test]
fn functions_agree_with_cpp() {
    let _serial = serial();
    let x = var("x", PrimType::int(32));
    let params: Array<Var> = [x.clone()].into_iter().collect();
    let body = SeqStmt::new([Return::new(&x, UNKNOWN_LOC)], UNKNOWN_LOC);
    let int32: Type = PrimType::int(32).into();
    let missing: Type = MissingType::new().into();
    let note: Map<FfiString, Any> =
        [(FfiString::from("note"), Any::from(1i64))].into_iter().collect();
    let attrs = DictAttrs::new(note);
    for (what, body, ret_type) in [
        ("Function", body.clone(), &int32),
        ("Function(void)", body.clone(), &missing),
        (
            "Function(empty body)",
            SeqStmt::new(Vec::<tvm_tirx::ir::Stmt>::new(), UNKNOWN_LOC),
            &int32,
        ),
    ] {
        let native_function = native(
            "tirx.Function",
            &[
                (&params).into(),
                (&body).into(),
                ret_type.into(),
                (&attrs).into(),
                (&Location::unknown()).into(),
            ],
        );
        agree(
            what,
            Ok(Function::new(params.iter(), &body, ret_type, &attrs, UNKNOWN_LOC)),
            native_function,
        );
    }
}
