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

//! Tests of `tirx::transform::function_pass`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tvm_tirx::ir::transform::{Pass, PassInfo};
use tvm_tirx::tirx::transform::FunctionPass;

use crate::common::serial;
use crate::{module_of, void_function};

#[test]
fn function_passes() {
    let _serial = serial();
    let func = void_function(vec![]);
    let runs = Arc::new(AtomicUsize::new(0));
    let counter = runs.clone();
    let counting = FunctionPass::new(
        move |func, _module, _ctx| {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(Some(func))
        },
        PassInfo::new(0, "rust.Count"),
    );
    let out = Pass::from(&counting).run(module_of("main", &func)).unwrap();
    assert_eq!(runs.load(Ordering::Relaxed), 1);
    assert!(out.contains("main").unwrap());
    let dropping =
        FunctionPass::new(|_func, _module, _ctx| Ok(None), PassInfo::new(0, "rust.Drop"));
    assert!(!Pass::from(&dropping)
        .run(module_of("main", &func))
        .unwrap()
        .contains("main")
        .unwrap());
}
