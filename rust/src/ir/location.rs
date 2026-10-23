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

//! Source locations of IR nodes, for diagnostics.
//!
//! The objects `python/tvm_tirx/ir/location.py` defines. A [`Location`] is an [`UnknownLoc`], a
//! [`SourceLoc`] (a span in a [`SourceName`]), or a [`CallSiteLoc`] (a callee location seen from a
//! caller). [`Location::unknown`] and [`SourceName::get`] wrap the native constructors.

// `custom-new` reserves `new` for a constructor with the native one's semantics (the shared
// `UnknownLoc` instance); the generated `from_fields` allocates the fields as given.
// tvm-ffi-stubgen(custom-new): ir.UnknownLoc
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::support;
use std::ops::Deref;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::ShallowCopy;
use tvm_ffi::String;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.SourceName
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SourceName"]
#[type_final]
pub struct SourceNameObj {
    base: Object,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<SourceNameObj>() == 40);
    assert!(::core::mem::align_of::<SourceNameObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SourceName {
    base: ObjectArc<SourceNameObj>,
}

impl Deref for SourceName {
    type Target = SourceNameObj;
    fn deref(&self) -> &SourceNameObj {
        &self.base
    }
}

impl SourceNameObj {
    pub(crate) fn new(name: String) -> Self {
        let base = Object::new();
        Self { base, name }
    }
}

impl SourceName {
    /// Lossless complete-field allocation.
    pub fn new(name: impl Into<String>) -> Self {
        let obj = SourceNameObj::new(name.into());
        Self { base: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

impl SourceName {
    /// The source name `name`, interned so that equal names share one object (`ir.SourceName`).
    pub fn get(name: &str) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.SourceName").call_tuple((String::from(name),))?.try_into()
    }
}

// tvm-ffi-stubgen(begin): object/ir.Location
/// Complete: reflected fields fill [24, 24) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Location"]
pub struct LocationObj {
    base: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<LocationObj>() == 24);
    assert!(::core::mem::align_of::<LocationObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Location {
    base: ObjectArc<LocationObj>,
}

impl Deref for Location {
    type Target = LocationObj;
    fn deref(&self) -> &LocationObj {
        &self.base
    }
}

impl LocationObj {
    pub fn new() -> Self {
        let base = Object::new();
        Self { base }
    }
}

impl Location {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = LocationObj::new();
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for LocationObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/ir.UnknownLoc
/// Complete: reflected fields fill [24, 24) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.UnknownLoc"]
#[type_final]
pub struct UnknownLocObj {
    base: LocationObj,
}

const _: () = {
    assert!(::core::mem::size_of::<UnknownLocObj>() == 24);
    assert!(::core::mem::align_of::<UnknownLocObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct UnknownLoc {
    base: ObjectArc<UnknownLocObj>,
}

impl Deref for UnknownLoc {
    type Target = UnknownLocObj;
    fn deref(&self) -> &UnknownLocObj {
        &self.base
    }
}

impl Deref for UnknownLocObj {
    type Target = LocationObj;
    fn deref(&self) -> &LocationObj {
        &self.base
    }
}

impl UnknownLocObj {
    pub(crate) fn new() -> Self {
        let base = LocationObj::new();
        Self { base }
    }
}

impl UnknownLoc {
    /// Lossless complete-field allocation.
    pub fn from_fields() -> Self {
        let obj = UnknownLocObj::new();
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(UnknownLoc => Location);
// tvm-ffi-stubgen(end)

impl UnknownLoc {
    /// The shared unknown location (`ir.UnknownLoc`), which IR constructors default to, fetched
    /// once per thread.
    pub fn new() -> Self {
        thread_local! {
            static UNKNOWN_LOC: std::cell::OnceCell<UnknownLoc> = const { std::cell::OnceCell::new() };
        }
        support::shared(&UNKNOWN_LOC, || {
            tvm_ffi::cached_global_func!("ir.UnknownLoc").call_tuple(())?.try_into()
        })
    }
}

impl Location {
    /// The shared [`UnknownLoc`] as a [`Location`] value; [`UNKNOWN_LOC`] is the same location as
    /// a constructor argument.
    pub fn unknown() -> Self {
        UnknownLoc::new().into()
    }
}

/// The unknown location as the `loc` argument of a constructor, which takes any
/// `impl Into<Location>`: `Var::new("x", ty, UNKNOWN_LOC)`. [`Location::unknown`] is the same
/// location as a value.
#[derive(Clone, Copy, Debug)]
pub struct Unknown;

/// The unknown location, as Python's `UNKNOWN_LOC`.
pub const UNKNOWN_LOC: Unknown = Unknown;

impl From<Unknown> for Location {
    fn from(_: Unknown) -> Self {
        Location::unknown()
    }
}

// tvm-ffi-stubgen(begin): object/ir.SourceLoc
/// Complete: reflected fields fill [24, 48) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SourceLoc"]
#[type_final]
pub struct SourceLocObj {
    base: LocationObj,
    pub source_name: SourceName,
    pub start_line: i32,
    pub start_column: i32,
    pub end_line: i32,
    pub end_column: i32,
}

const _: () = {
    assert!(::core::mem::size_of::<SourceLocObj>() == 48);
    assert!(::core::mem::align_of::<SourceLocObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SourceLoc {
    base: ObjectArc<SourceLocObj>,
}

impl Deref for SourceLoc {
    type Target = SourceLocObj;
    fn deref(&self) -> &SourceLocObj {
        &self.base
    }
}

impl Deref for SourceLocObj {
    type Target = LocationObj;
    fn deref(&self) -> &LocationObj {
        &self.base
    }
}

impl SourceLocObj {
    pub(crate) fn new(
        source_name: SourceName,
        start_line: i32,
        start_column: i32,
        end_line: i32,
        end_column: i32,
    ) -> Self {
        let base = LocationObj::new();
        Self { base, source_name, start_line, start_column, end_line, end_column }
    }
}

impl SourceLoc {
    /// Lossless complete-field allocation.
    pub fn new(
        source_name: impl Into<SourceName>,
        start_line: i32,
        start_column: i32,
        end_line: i32,
        end_column: i32,
    ) -> Self {
        let obj =
            SourceLocObj::new(source_name.into(), start_line, start_column, end_line, end_column);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(SourceLoc => Location);
// tvm-ffi-stubgen(end)

// `ir.SourceLoc` stores its arguments as given: the generated `SourceLoc::new` is the same.

// tvm-ffi-stubgen(begin): object/ir.CallSiteLoc
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.CallSiteLoc"]
#[type_final]
pub struct CallSiteLocObj {
    base: LocationObj,
    pub callee: Location,
    pub caller: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<CallSiteLocObj>() == 40);
    assert!(::core::mem::align_of::<CallSiteLocObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CallSiteLoc {
    base: ObjectArc<CallSiteLocObj>,
}

impl Deref for CallSiteLoc {
    type Target = CallSiteLocObj;
    fn deref(&self) -> &CallSiteLocObj {
        &self.base
    }
}

impl Deref for CallSiteLocObj {
    type Target = LocationObj;
    fn deref(&self) -> &LocationObj {
        &self.base
    }
}

impl CallSiteLocObj {
    pub(crate) fn new(callee: Location, caller: Location) -> Self {
        let base = LocationObj::new();
        Self { base, callee, caller }
    }
}

impl CallSiteLoc {
    /// Lossless complete-field allocation.
    pub fn new(callee: impl Into<Location>, caller: impl Into<Location>) -> Self {
        let obj = CallSiteLocObj::new(callee.into(), caller.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(CallSiteLoc => Location);
// tvm-ffi-stubgen(end)

// `ir.CallSiteLoc` stores its arguments as given: the generated `CallSiteLoc::new` is the same.
