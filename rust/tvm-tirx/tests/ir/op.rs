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

//! Tests of `ir::op`.

use tvm_tirx::ir::Op;

use crate::common::serial;

#[test]
fn ops() {
    let _serial = serial();
    let op_names = Op::list_op_names().unwrap();
    assert!(op_names.iter().any(|name| name.as_str() == "prim.exp"));
    assert!(!Op::list_attr_names().unwrap().is_empty());
    let effect_kinds = Op::attr_map::<i64>("TCallEffectKind").unwrap().unwrap();
    assert!(effect_kinds.get(&Op::get("prim.exp").unwrap()).unwrap().is_some());
    assert!(Op::attr_map::<i64>("no_such_attr").unwrap().is_none());
    Op::register("rust.tests.ir.op", "An operator the tests register.").unwrap();
    assert_eq!(Op::get("rust.tests.ir.op").unwrap().name.as_str(), "rust.tests.ir.op");
    assert!(Op::register("rust.tests.ir.op", "Twice.").is_err());
    // The map is live: it sees the operator registered after it was read, which has no value.
    assert!(effect_kinds.get(&Op::get("rust.tests.ir.op").unwrap()).unwrap().is_none());
}
