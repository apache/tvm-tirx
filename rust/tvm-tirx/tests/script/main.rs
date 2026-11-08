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

//! Tests of `tvm_tirx::script`, which is the single file `src/script.rs`: printing IR objects as
//! TVMScript.

#[path = "../common/mod.rs"]
mod common;

use tvm_tirx::script;

use common::empty_module;

#[test]
fn print_module() {
    let text = script::script(&empty_module()).unwrap();
    assert!(text.as_str().contains("class Module"), "{}", text.as_str());
}
