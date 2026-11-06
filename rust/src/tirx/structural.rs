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

//! Structural traversal with the facts in scope.
//!
//! A [`StructuralVisitorWithConstraintContext`] visits values and a
//! [`StructuralMutatorWithConstraintContext`] transforms them, keeping the [`ConstraintContext`] of
//! the bindings, ranges and guards in effect at the current value. From Rust, run them with the
//! `structural_*_with_constraint_context` entry points.

// tvm-ffi-stubgen(begin): import-section
use crate::sym::constraint::ConstraintContext;
use tvm_ffi::FieldGetter;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ObjectCore;
use tvm_ffi::Result;

/// Header-only stand-in for the builtin `ffi.StructuralMutator`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.StructuralMutator"]
struct FfiStructuralMutatorObj {
    base: Object,
}

/// Header-only stand-in for the builtin `ffi.StructuralVisitor`; it only carries the ancestor depth.
#[allow(dead_code)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "ffi.StructuralVisitor"]
struct FfiStructuralVisitorObj {
    base: Object,
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.StructuralMutatorWithConstraintContext
/// Opaque: parent 'ffi.StructuralMutator' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.StructuralMutatorWithConstraintContext"]
pub struct StructuralMutatorWithConstraintContextObj {
    base: FfiStructuralMutatorObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StructuralMutatorWithConstraintContext {
    base: ObjectArc<StructuralMutatorWithConstraintContextObj>,
}

impl StructuralMutatorWithConstraintContextObj {
    pub fn constraints(&self) -> Result<ConstraintContext> {
        FieldGetter::new(Self::type_index(), "constraints")?.get(self)
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/tirx.StructuralVisitorWithConstraintContext
/// Opaque: parent 'ffi.StructuralVisitor' is opaque (no-mirror). Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "tirx.StructuralVisitorWithConstraintContext"]
pub struct StructuralVisitorWithConstraintContextObj {
    base: FfiStructuralVisitorObj,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct StructuralVisitorWithConstraintContext {
    base: ObjectArc<StructuralVisitorWithConstraintContextObj>,
}

impl StructuralVisitorWithConstraintContextObj {
    pub fn constraints(&self) -> Result<ConstraintContext> {
        FieldGetter::new(Self::type_index(), "constraints")?.get(self)
    }
}
// tvm-ffi-stubgen(end)
