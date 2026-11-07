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

//! Thread scopes. A [`ThreadScope`] is one level of a backend's thread hierarchy, such as
//! `cuda.warp`; each backend registers its levels, ranked from the widest down.

// tvm-ffi-stubgen(missing-docs): allow
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// tvm-ffi-stubgen(import-object): tvm_ffi::IntoObjectRef

// tvm-ffi-stubgen(begin): import-section
use std::borrow::Borrow;
use tvm_ffi::IntoObjectRef;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::enum_ref::EnumObj;
// tvm-ffi-stubgen(end)

/// A level of a backend's thread hierarchy, named `backend.level`, such as `cuda.warp`.
///
/// Backends register their scopes; [`EnumRef::get`] only looks one up. Within a backend,
/// [`rank`](ThreadScope::rank) orders the levels, a wider scope ranking lower.
///
/// # Examples
///
/// ```
/// use tvm_tirx::tirx::ThreadScope;
/// use tvm_tirx::tvm_ffi::EnumRef;
///
/// let warp = ThreadScope::get("cuda.warp")?;
/// assert_eq!((warp.backend()?, warp.local_name()?), ("cuda".into(), "warp".into()));
///
/// let cta = ThreadScope::get("cuda.cta")?;
/// assert!(cta.is_higher_than(&warp)?);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
///
/// [`EnumRef::get`]: tvm_ffi::EnumRef::get
// tvm-ffi-stubgen(begin): object/tirx.ThreadScope
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ThreadScope {
    data_: ObjectArc<ThreadScopeObj>,
}

/// The object behind [`ThreadScope`].
///
/// Complete: reflected fields fill [48, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.ThreadScope"]
#[type_final]
#[type_immutable]
pub struct ThreadScopeObj {
    base_: EnumObj,
}

const _: () = {
    assert!(::core::mem::size_of::<ThreadScopeObj>() == 48);
    assert!(::core::mem::align_of::<ThreadScopeObj>() == 8);
};

impl tvm_ffi::EnumRef for ThreadScope {}
// tvm-ffi-stubgen(end)

impl ThreadScope {
    /// Return the backend that registered the scope, such as `cuda` (`tirx.ThreadScopeGetBackend`).
    pub fn backend(&self) -> Result<String> {
        tvm_ffi::cached_global_func!("tirx.ThreadScopeGetBackend").call_tuple((self,))?.try_into()
    }

    /// Return the name of the scope within its backend, such as `warp`
    /// (`tirx.ThreadScopeGetLocalName`).
    pub fn local_name(&self) -> Result<String> {
        tvm_ffi::cached_global_func!("tirx.ThreadScopeGetLocalName").call_tuple((self,))?.try_into()
    }

    /// Return the level of the scope in its backend's hierarchy; outer scopes rank lower
    /// (`tirx.ThreadScopeGetRank`).
    pub fn rank(&self) -> Result<i64> {
        tvm_ffi::cached_global_func!("tirx.ThreadScopeGetRank").call_tuple((self,))?.try_into()
    }

    /// Return whether this scope is above `other` in the hierarchy of their backend
    /// (`tirx.ScopeHigher`).
    ///
    /// # Errors
    ///
    /// Returns an error if the two scopes are of different backends.
    pub fn is_higher_than(&self, other: impl IntoObjectRef<ThreadScope>) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.ScopeHigher")
            .call_tuple((self, other.into_object_ref().borrow()))?
            .try_into()
    }
}
