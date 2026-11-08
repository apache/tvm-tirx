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
// The generated `new` of a field-less base needs no `Default`.
#![allow(clippy::new_without_default)]

//! Module-level information. A [`GlobalInfo`] is information an [`IRModule`](super::IRModule)
//! holds besides its functions, kept by name in `global_infos`.

// tvm-ffi-stubgen(missing-docs): allow

// tvm-ffi-stubgen(begin): import-section
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
// tvm-ffi-stubgen(end)

/// The base of the module-level information an [`IRModule`](super::IRModule) keeps by name in
/// `global_infos`.
// tvm-ffi-stubgen(begin): object/ir.GlobalInfo
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct GlobalInfo {
    data_: ObjectArc<GlobalInfoObj>,
}

/// The object behind [`GlobalInfo`].
///
/// Complete: reflected fields fill [24, 24) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.GlobalInfo"]
#[type_immutable]
#[type_rs_parent]
pub struct GlobalInfoObj {
    base_: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<GlobalInfoObj>() == 24);
    assert!(::core::mem::align_of::<GlobalInfoObj>() == 8);
};

#[allow(missing_docs)]
impl GlobalInfoObj {
    pub fn new() -> Self {
        let base_ = Object::new();
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl GlobalInfo {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = GlobalInfoObj::new();
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)
