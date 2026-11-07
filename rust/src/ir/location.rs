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
// The generated `new` of a field-less base needs no `Default`, and `UnknownLoc::new` fetches a
// shared FFI singleton.
#![allow(clippy::new_without_default)]

//! Source locations. Every IR node carries a [`Location`] for diagnostics: an [`UnknownLoc`], a
//! [`SourceLoc`] span in a source, or a [`CallSiteLoc`] that pairs a location with its caller's.

// tvm-ffi-stubgen(missing-docs): allow
// C++ has one `UnknownLoc`, which `UnknownLoc::new` returns: `opaque` generates no allocator for
// another one. It has no fields, so it loses no field access.
// tvm-ffi-stubgen(opaque): ir.UnknownLoc
// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): crate::ir::support

// tvm-ffi-stubgen(begin): import-section
use crate::ir::support;
use std::ops::Deref;
use std::ptr::addr_of_mut;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::ShallowCopy;
use tvm_ffi::String;
use tvm_ffi::UnsafeInit;
// tvm-ffi-stubgen(end)

/// Where an IR node comes from, for diagnostics: an [`UnknownLoc`], a [`SourceLoc`] or a
/// [`CallSiteLoc`].
///
/// A constructor takes its `loc` as any `impl Into<Location>`, such as a `SourceLoc` or
/// [`UNKNOWN_LOC`].
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{PrimType, PrimVar, SourceLoc, SourceName, Var};
/// use tvm_tirx::tvm_ffi::ObjectRefCore;
///
/// let span = SourceLoc::new(SourceName::get("kernel.py"), 3, 1, 3, 10);
///
/// let x = PrimVar::new("x", PrimType::int(32), &span);
/// assert!(x.loc.same_as(&span));
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/ir.Location
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Location {
    data_: ObjectArc<LocationObj>,
}

/// The object behind [`Location`].
///
/// Complete: reflected fields fill [24, 24) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.Location"]
pub struct LocationObj {
    base_: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<LocationObj>() == 24);
    assert!(::core::mem::align_of::<LocationObj>() == 8);
};

#[allow(missing_docs)]
impl LocationObj {
    pub fn new() -> Self {
        let base_ = Object::new();
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl Location {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = LocationObj::new();
        Self { data_: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for LocationObj {
    fn shallow_copy(&self) -> Self {
        Self { base_: ShallowCopy::shallow_copy(&self.base_) }
    }
}

unsafe impl UnsafeInit for LocationObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base_));
        }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for LocationObj {}
// tvm-ffi-stubgen(end)

/// The location of a node with no source information, a single shared object.
// tvm-ffi-stubgen(begin): object/ir.UnknownLoc
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct UnknownLoc {
    data_: ObjectArc<UnknownLocObj>,
}

/// The object behind [`UnknownLoc`].
///
/// Opaque: vetoed by directive although the layout is reproducible. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.UnknownLoc"]
#[type_final]
pub struct UnknownLocObj {
    base_: LocationObj,
}

impl Deref for UnknownLocObj {
    type Target = LocationObj;
    fn deref(&self) -> &LocationObj {
        &self.base_
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for UnknownLocObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(UnknownLoc => Location);
// tvm-ffi-stubgen(end)

impl UnknownLoc {
    /// Return the shared unknown location, which IR constructors default to, fetched once per
    /// thread (`ir.UnknownLoc`).
    pub fn new() -> Self {
        thread_local! {
            static UNKNOWN_LOC: std::cell::OnceCell<UnknownLoc> = const { std::cell::OnceCell::new() };
        }
        support::shared(&UNKNOWN_LOC, || {
            tvm_ffi::cached_global_func!("ir.UnknownLoc").call_tuple(())?.try_into()
        })
    }
}

/// A marker for the unknown location as the `loc` argument of a constructor, which takes any
/// `impl Into<Location>`: `Var::new("x", ty, UNKNOWN_LOC)`.
///
/// [`UnknownLoc::new`] is the same location as a value.
#[derive(Clone, Copy, Debug)]
pub struct Unknown;

/// The unknown location, as the `loc` argument of a constructor.
pub const UNKNOWN_LOC: Unknown = Unknown;

impl From<Unknown> for Location {
    fn from(_: Unknown) -> Self {
        UnknownLoc::new().into()
    }
}

/// The span from `start_line`, `start_column` to `end_line`, `end_column` in the source
/// `source_name`, in the line and column numbering of the frontend that produced it.
// tvm-ffi-stubgen(begin): object/ir.SourceLoc
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SourceLoc {
    data_: ObjectArc<SourceLocObj>,
}

/// The object behind [`SourceLoc`].
///
/// Complete: reflected fields fill [24, 48) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SourceLoc"]
#[type_final]
pub struct SourceLocObj {
    base_: LocationObj,
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

impl Deref for SourceLocObj {
    type Target = LocationObj;
    fn deref(&self) -> &LocationObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl SourceLocObj {
    pub(crate) fn new(
        source_name: SourceName,
        start_line: i32,
        start_column: i32,
        end_line: i32,
        end_column: i32,
    ) -> Self {
        let base_ = LocationObj::new();
        Self { base_, source_name, start_line, start_column, end_line, end_column }
    }
}

#[allow(missing_docs)]
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
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for SourceLocObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(SourceLoc => Location);
// tvm-ffi-stubgen(end)

// `ir.SourceLoc` stores its arguments as given: the generated `SourceLoc::new` is the same.

/// The location `callee` of a node, together with the location `caller` of the call site it
/// comes from.
// tvm-ffi-stubgen(begin): object/ir.CallSiteLoc
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct CallSiteLoc {
    data_: ObjectArc<CallSiteLocObj>,
}

/// The object behind [`CallSiteLoc`].
///
/// Complete: reflected fields fill [24, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.CallSiteLoc"]
#[type_final]
pub struct CallSiteLocObj {
    base_: LocationObj,
    pub callee: Location,
    pub caller: Location,
}

const _: () = {
    assert!(::core::mem::size_of::<CallSiteLocObj>() == 40);
    assert!(::core::mem::align_of::<CallSiteLocObj>() == 8);
};

impl Deref for CallSiteLocObj {
    type Target = LocationObj;
    fn deref(&self) -> &LocationObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl CallSiteLocObj {
    pub(crate) fn new(callee: Location, caller: Location) -> Self {
        let base_ = LocationObj::new();
        Self { base_, callee, caller }
    }
}

#[allow(missing_docs)]
impl CallSiteLoc {
    /// Lossless complete-field allocation.
    pub fn new(callee: impl Into<Location>, caller: impl Into<Location>) -> Self {
        let obj = CallSiteLocObj::new(callee.into(), caller.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for CallSiteLocObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(CallSiteLoc => Location);
// tvm-ffi-stubgen(end)

// `ir.CallSiteLoc` stores its arguments as given: the generated `CallSiteLoc::new` is the same.

/// The name of a source, such as a file name; [`SourceName::get`] returns the shared object for a
/// name.
// tvm-ffi-stubgen(begin): object/ir.SourceName
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct SourceName {
    data_: ObjectArc<SourceNameObj>,
}

/// The object behind [`SourceName`].
///
/// Complete: reflected fields fill [24, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ir.SourceName"]
#[type_final]
pub struct SourceNameObj {
    base_: Object,
    pub name: String,
}

const _: () = {
    assert!(::core::mem::size_of::<SourceNameObj>() == 40);
    assert!(::core::mem::align_of::<SourceNameObj>() == 8);
};

#[allow(missing_docs)]
impl SourceNameObj {
    pub(crate) fn new(name: String) -> Self {
        let base_ = Object::new();
        Self { base_, name }
    }
}

#[allow(missing_docs)]
impl SourceName {
    /// Lossless complete-field allocation.
    pub fn new(name: impl Into<String>) -> Self {
        let obj = SourceNameObj::new(name.into());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for SourceNameObj {}
// tvm-ffi-stubgen(end)

impl SourceName {
    /// Return the source name `name`, interned so that equal names share one object
    /// (`ir.SourceName`).
    pub fn get(name: &str) -> Self {
        support::infallible(
            tvm_ffi::cached_global_func!("ir.SourceName")
                .call_tuple((String::from(name),))
                .and_then(|any| any.try_into()),
        )
    }
}
