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
#![allow(dead_code, unused_imports)]

//! Module-level information that is not a function.
//!
//! The objects `python/tvm_tirx/ir/global_info.py` defines. [`GlobalInfo`] is the base of the
//! entries an [`IRModule`](super::IRModule) keeps by name in `global_infos`.

// tvm-ffi-stubgen(begin): import-section
use std::ops::Deref;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ShallowCopy;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.GlobalInfo
/// Complete: reflected fields fill [24, 24) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GlobalInfo"]
pub struct GlobalInfoObj {
    base: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<GlobalInfoObj>() == 24);
    assert!(::core::mem::align_of::<GlobalInfoObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GlobalInfo {
    base: ObjectArc<GlobalInfoObj>,
}

impl Deref for GlobalInfo {
    type Target = GlobalInfoObj;
    fn deref(&self) -> &GlobalInfoObj {
        &self.base
    }
}

impl GlobalInfoObj {
    pub fn new() -> Self {
        let base = Object::new();
        Self { base }
    }
}

impl GlobalInfo {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = GlobalInfoObj::new();
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for GlobalInfoObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}
// tvm-ffi-stubgen(end)
