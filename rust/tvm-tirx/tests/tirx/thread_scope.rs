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

//! Tests of `tirx::thread_scope`.

use tvm_tirx::tirx::ThreadScope;
use tvm_tirx::tvm_ffi::EnumRef;

use crate::common::serial;

#[test]
fn thread_scopes() {
    let _serial = serial();
    let warp = ThreadScope::get("cuda.warp").unwrap();
    let cta = ThreadScope::get("cuda.cta").unwrap();
    assert_eq!(warp.backend().unwrap().as_str(), "cuda");
    assert_eq!(warp.local_name().unwrap().as_str(), "warp");
    assert!(cta.rank().unwrap() < warp.rank().unwrap());
    assert!(cta.is_higher_than(&warp).unwrap());
}
