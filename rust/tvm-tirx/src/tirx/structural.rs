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

//! Constraint-aware traversal. A [`StructuralVisitorWithConstraintContext`] visits values and a
//! [`StructuralMutatorWithConstraintContext`] rewrites them, each keeping the [`ConstraintContext`]
//! of the bindings, ranges and guards in scope at the current value.

// tvm-ffi-stubgen(file): mod(missing_docs = allow)

// tvm-ffi-stubgen(begin): import-section
use crate::sym::constraint::ConstraintContext;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::cached_field_getter;

/// Header-only stand-in for the builtin `ffi.StructuralMutator`; it only carries the ancestor depth.
#[doc(hidden)]
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ffi.StructuralMutator")]
pub struct FfiStructuralMutatorObj {
    base_: Object,
}

/// Header-only stand-in for the builtin `ffi.StructuralVisitor`; it only carries the ancestor depth.
#[doc(hidden)]
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "ffi.StructuralVisitor")]
pub struct FfiStructuralVisitorObj {
    base_: Object,
}
// tvm-ffi-stubgen(end)

/// A structural mutator that keeps the [`ConstraintContext`] in scope at the current value, read
/// with `constraints`; Rust code runs one with
/// [`structural_mutate_with_constraint_context`](mod@super::structural_mutate_with_constraint_context).
// tvm-ffi-stubgen(begin): object/tirx.StructuralMutatorWithConstraintContext
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StructuralMutatorWithConstraintContext {
    data_: ObjectArc<StructuralMutatorWithConstraintContextObj>,
}

/// The object behind [`StructuralMutatorWithConstraintContext`].
// Opaque: parent 'ffi.StructuralMutator' is opaque (no-mirror). Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.StructuralMutatorWithConstraintContext")]
pub struct StructuralMutatorWithConstraintContextObj {
    base_: FfiStructuralMutatorObj,
}

#[allow(missing_docs)]
impl StructuralMutatorWithConstraintContextObj {
    pub fn constraints(&self) -> Result<ConstraintContext> {
        cached_field_getter!(StructuralMutatorWithConstraintContextObj, "constraints").get(self)
    }
}
// tvm-ffi-stubgen(end)

/// A structural visitor that keeps the [`ConstraintContext`] in scope at the current value, read
/// with `constraints`; Rust code runs one with
/// [`structural_visit_with_constraint_context`](mod@super::structural_visit_with_constraint_context).
// tvm-ffi-stubgen(begin): object/tirx.StructuralVisitorWithConstraintContext
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StructuralVisitorWithConstraintContext {
    data_: ObjectArc<StructuralVisitorWithConstraintContextObj>,
}

/// The object behind [`StructuralVisitorWithConstraintContext`].
// Opaque: parent 'ffi.StructuralVisitor' is opaque (no-mirror). Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "tirx.StructuralVisitorWithConstraintContext")]
pub struct StructuralVisitorWithConstraintContextObj {
    base_: FfiStructuralVisitorObj,
}

#[allow(missing_docs)]
impl StructuralVisitorWithConstraintContextObj {
    pub fn constraints(&self) -> Result<ConstraintContext> {
        cached_field_getter!(StructuralVisitorWithConstraintContextObj, "constraints").get(self)
    }
}
// tvm-ffi-stubgen(end)
