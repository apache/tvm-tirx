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

//! Unique names. A [`UniqueNameSupply`] hands out names that do not collide with the ones it has
//! given out or reserved, for building and rewriting IR.

// tvm-ffi-stubgen(missing-docs): allow
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::support;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

/// A set of names in use, with a prefix, from which [`fresh_name`](Self::fresh_name) derives
/// names that do not collide.
///
/// A supply is mutable: every handle sees the names it has given out or reserved.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::UniqueNameSupply;
///
/// let names = UniqueNameSupply::new("fused");
/// names.reserve_name("x", false)?;
///
/// assert_eq!(names.fresh_name("x", false, true)?, "x_1");
///
/// assert_eq!(names.fresh_name("y", true, true)?, "fused_y");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.UniqueNameSupply
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct UniqueNameSupply {
    data_: ObjectArc<UniqueNameSupplyObj>,
}

/// The object behind [`UniqueNameSupply`].
///
/// Opaque: bytes [24, 64) of [24, 64) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.UniqueNameSupply"]
#[type_final]
pub struct UniqueNameSupplyObj {
    base_: Object,
}
// tvm-ffi-stubgen(end)

impl UniqueNameSupply {
    /// Create an empty supply with the prefix `prefix`, which `add_prefix` prepends to a name as
    /// `<prefix>_<name>` (`ir.UniqueNameSupply`).
    pub fn new(prefix: &str) -> Self {
        support::infallible(
            tvm_ffi::cached_global_func!("ir.UniqueNameSupply")
                .call_tuple((String::from(prefix),))
                .and_then(|any| any.try_into()),
        )
    }

    /// Return a name derived from `name` that the supply has not given out, and reserve it
    /// (`ir.UniqueNameSupplyFreshName`).
    ///
    /// The name is `name` with `.` replaced by `_`, and a numeric suffix when the supply already
    /// has it; `add_prefix` prepends the prefix and `add_underscore` puts `_` before the suffix.
    pub fn fresh_name(&self, name: &str, add_prefix: bool, add_underscore: bool) -> Result<String> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupplyFreshName")
            .call_tuple((self, String::from(name), add_prefix, add_underscore))?
            .try_into()
    }

    /// Reserve `name`, with the prefix if `add_prefix`, so that [`fresh_name`](Self::fresh_name)
    /// does not give it out, and return the reserved name (`ir.UniqueNameSupplyReserveName`).
    pub fn reserve_name(&self, name: &str, add_prefix: bool) -> Result<String> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupplyReserveName")
            .call_tuple((self, String::from(name), add_prefix))?
            .try_into()
    }

    /// Return whether the supply has given out or reserved `name`, with the prefix if `add_prefix`
    /// (`ir.UniqueNameSupplyContainsName`).
    pub fn contains_name(&self, name: &str, add_prefix: bool) -> Result<bool> {
        tvm_ffi::cached_global_func!("ir.UniqueNameSupplyContainsName")
            .call_tuple((self, String::from(name), add_prefix))?
            .try_into()
    }
}
