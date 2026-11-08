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

//! Tests of `ir::function`.

use tvm_tirx::tvm_ffi::{Any, Map, String as FfiString};

use super::{function, int32};
use crate::common::serial;

#[test]
fn function_attrs() {
    let _serial = serial();
    let func = function(0);
    let copy = func.copy().unwrap();
    let with = copy.with_attr("note", 7i64).unwrap();
    assert!(with.attrs.dict.get(&FfiString::from("note")).is_ok());
    assert!(func.attrs.dict.is_empty());
    let attrs: Map<FfiString, Any> =
        [(FfiString::from("other"), Any::from(int32(1)))].into_iter().collect();
    let with = with.with_attrs(&attrs).unwrap();
    assert_eq!(with.attrs.dict.len(), 2);
    let without = with.without_attr("note").unwrap();
    assert_eq!(without.attrs.dict.len(), 1);
}
