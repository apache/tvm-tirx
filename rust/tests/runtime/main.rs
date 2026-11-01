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

//! Tests of `tvm_tirx::runtime`, which is the single file `src/runtime.rs`: tensors, memory
//! scopes and runtime queries.

#[path = "../common/mod.rs"]
mod common;

use tvm_tirx::runtime::{self, MemoryScope};
use tvm_tirx::ir::dtype;
use tvm_tirx::tvm_ffi::{DLDevice, DLDeviceType, TypeIndex};

#[test]
fn tensors() {
    let cpu = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let float32 = dtype::float(32);
    let tensor = runtime::empty(&[2, 8], float32, cpu, None).unwrap();
    assert_eq!(tensor.shape(), &[2, 8]);
    let int8 = dtype::int(8);
    let view = runtime::create_view(&tensor, &[4, 4], int8, 4 * 4).unwrap();
    assert_eq!(view.shape(), &[4, 4]);
    assert_eq!(view.dtype(), int8);
    assert!(runtime::create_view(&tensor, &[1024], float32, 0).is_err());
}

#[test]
fn memory_scopes() {
    let scope = MemoryScope::register("rust_test.scratch").unwrap();
    let again = MemoryScope::register("rust_test.scratch");
    assert!(again.is_err());
    // The CPU allocator supports the global scope only.
    let cpu = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let float32 = dtype::float(32);
    assert!(runtime::empty(&[4], float32, cpu, Some(&scope)).is_err());
}

#[test]
fn queries() {
    assert!(runtime::enabled("cpu").unwrap());
    // The thread pool is thread-local. On Windows it joins its workers when the test's thread
    // exits, under the loader lock that the workers need to exit, so the test would never end.
    if cfg!(not(windows)) {
        assert!(runtime::num_threads().unwrap() >= 1);
    }
}

#[test]
fn device_attrs() {
    let cpu = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let exist = runtime::get_device_attr(cpu, "exist").unwrap();
    assert!(bool::try_from(exist).unwrap());
    let missing = runtime::get_device_attr(cpu, "no_such_attr").unwrap();
    assert_eq!(missing.type_index(), TypeIndex::kTVMFFINone as i32);
}

#[test]
fn cuda_queries() {
    // CPU workers do not load a CUDA runtime; skip its device queries there.
    if !runtime::enabled("cuda").unwrap() {
        return;
    }
    if let Ok(count) = runtime::get_cuda_device_count() {
        if count > 0 {
            assert!(runtime::get_cuda_free_memory().unwrap().as_str().contains("bytes free"));
        }
    }
}
