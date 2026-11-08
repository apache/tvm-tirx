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

//! Tests of `ir::unique_name_supply`.

use tvm_tirx::ir::UniqueNameSupply;

use crate::common::serial;

#[test]
fn unique_names() {
    let _serial = serial();
    let names = UniqueNameSupply::new("p");
    let fresh = names.fresh_name("x", true, true).unwrap();
    assert!(names.contains_name(fresh.as_str(), false).unwrap());
    let again = names.fresh_name("x", true, true).unwrap();
    assert_ne!(fresh.as_str(), again.as_str());
    assert_eq!(names.reserve_name("y", false).unwrap().as_str(), "y");
    assert!(names.contains_name("y", false).unwrap());
}
