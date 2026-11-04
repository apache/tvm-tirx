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

//! Constraints: the facts a traversal collects about the variables in scope.
//!
//! The objects `python/tvm_tirx/sym/constraint.py` defines. A [`Constraint`] is a premise, with a
//! lifetime the collector defines: a [`BindingConstraint`] records the value a `var` is bound to,
//! a [`RangeConstraint`] the half-open `range` a `var` lies in, and a [`PredicateConstraint`] a
//! boolean `condition` its collector vouches for. A [`ConstraintContext`] holds the constraints in
//! effect at a point of a traversal, ancestors' first; the structural visitor and mutator of
//! [`tirx::structural`](crate::tirx::structural) keep one for each value they reach.
//!
//! The context representation mirrors `include/tvm/sym/constraint.h`, including
//! its private parent, captured prefix and append-only array. It requires matching
//! native headers/builds: stub generation cannot certify these private fields.
//! Context handles use ordinary FFI ownership. Their native-mutated array makes
//! typed handles neither Send nor Sync. Erasing them into ObjectRef or a packed Function
//! does not enforce that confinement; callers retain the ordinary FFI obligation
//! to prevent concurrent context access. Constraint and IR nodes remain shared.

// tvm-ffi-stubgen(ty-map): sym.ConstraintContext -> crate::sym::constraint::ConstraintContext

use std::cell::UnsafeCell;
use std::mem::{align_of, offset_of, size_of};
use tvm_ffi::{AnyView, Array, Error, ObjectCore, Result, VALUE_ERROR};

// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::Expr;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use std::ops::Deref;
use std::ptr::addr_of_mut;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::ShallowCopy;
use tvm_ffi::UnsafeInit;
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.Constraint
/// Complete: reflected fields fill [24, 24) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.Constraint"]
pub struct ConstraintObj {
    base: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<ConstraintObj>() == 24);
    assert!(::core::mem::align_of::<ConstraintObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Constraint {
    base: ObjectArc<ConstraintObj>,
}

impl Deref for Constraint {
    type Target = ConstraintObj;
    fn deref(&self) -> &ConstraintObj {
        &self.base
    }
}

impl ConstraintObj {
    pub fn new() -> Self {
        let base = Object::new();
        Self { base }
    }
}

impl Constraint {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = ConstraintObj::new();
        Self { base: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for ConstraintObj {
    fn shallow_copy(&self) -> Self {
        Self { base: ShallowCopy::shallow_copy(&self.base) }
    }
}

unsafe impl UnsafeInit for ConstraintObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base));
        }
    }
}
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.BindingConstraint
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.BindingConstraint"]
#[type_final]
pub struct BindingConstraintObj {
    base: ConstraintObj,
    pub var: Var,
    pub value: Expr,
}

const _: () = {
    assert!(::core::mem::size_of::<BindingConstraintObj>() == 40);
    assert!(::core::mem::align_of::<BindingConstraintObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BindingConstraint {
    base: ObjectArc<BindingConstraintObj>,
}

impl Deref for BindingConstraint {
    type Target = BindingConstraintObj;
    fn deref(&self) -> &BindingConstraintObj {
        &self.base
    }
}

impl Deref for BindingConstraintObj {
    type Target = ConstraintObj;
    fn deref(&self) -> &ConstraintObj {
        &self.base
    }
}

impl BindingConstraintObj {
    pub(crate) fn new(var: Var, value: Expr) -> Self {
        let base = ConstraintObj::new();
        Self { base, var, value }
    }
}

impl BindingConstraint {
    /// Lossless complete-field allocation.
    pub fn new(var: impl Into<Var>, value: impl Into<Expr>) -> Self {
        let obj = BindingConstraintObj::new(var.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(BindingConstraint => Constraint);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.PredicateConstraint
/// Complete: reflected fields fill [24, 40) exactly (alignment padding [33, 40)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.PredicateConstraint"]
#[type_final]
pub struct PredicateConstraintObj {
    base: ConstraintObj,
    pub condition: Expr,
    pub assume_always: bool,
}

const _: () = {
    assert!(::core::mem::size_of::<PredicateConstraintObj>() == 40);
    assert!(::core::mem::align_of::<PredicateConstraintObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PredicateConstraint {
    base: ObjectArc<PredicateConstraintObj>,
}

impl Deref for PredicateConstraint {
    type Target = PredicateConstraintObj;
    fn deref(&self) -> &PredicateConstraintObj {
        &self.base
    }
}

impl Deref for PredicateConstraintObj {
    type Target = ConstraintObj;
    fn deref(&self) -> &ConstraintObj {
        &self.base
    }
}

impl PredicateConstraintObj {
    pub(crate) fn new(condition: Expr, assume_always: bool) -> Self {
        let base = ConstraintObj::new();
        Self { base, condition, assume_always }
    }
}

impl PredicateConstraint {
    /// Lossless complete-field allocation.
    pub fn new(condition: impl Into<Expr>, assume_always: bool) -> Self {
        let obj = PredicateConstraintObj::new(condition.into(), assume_always);
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(PredicateConstraint => Constraint);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.RangeConstraint
/// Complete: reflected fields fill [24, 40) exactly.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.RangeConstraint"]
#[type_final]
pub struct RangeConstraintObj {
    base: ConstraintObj,
    pub var: Var,
    pub range: Range,
}

const _: () = {
    assert!(::core::mem::size_of::<RangeConstraintObj>() == 40);
    assert!(::core::mem::align_of::<RangeConstraintObj>() == 8);
};

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RangeConstraint {
    base: ObjectArc<RangeConstraintObj>,
}

impl Deref for RangeConstraint {
    type Target = RangeConstraintObj;
    fn deref(&self) -> &RangeConstraintObj {
        &self.base
    }
}

impl Deref for RangeConstraintObj {
    type Target = ConstraintObj;
    fn deref(&self) -> &ConstraintObj {
        &self.base
    }
}

impl RangeConstraintObj {
    pub(crate) fn new(var: Var, range: Range) -> Self {
        let base = ConstraintObj::new();
        Self { base, var, range }
    }
}

impl RangeConstraint {
    /// Lossless complete-field allocation.
    pub fn new(var: impl Into<Var>, range: impl Into<Range>) -> Self {
        let obj = RangeConstraintObj::new(var.into(), range.into());
        Self { base: ObjectArc::new(obj) }
    }
}

tvm_ffi::impl_object_upcast!(RangeConstraint => Constraint);
// tvm-ffi-stubgen(end)

/// Native-compatible stable-identity constraint context.
///
/// Only the local array changes after construction. Its handle is replaced by
/// native append, so shared context inspection must use interior mutability.
/// Parent links and their captured local prefix are immutable and acyclic.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.ConstraintContext"]
#[type_final]
pub struct ConstraintContextObj {
    base: Object,
    parent: Option<ObjectArc<ConstraintContextObj>>,
    parent_prefix_size: i64,
    constraints: UnsafeCell<Array<Constraint>>,
}

/// An owning reference to a live context. Clones share identity and later appends.
/// Use [`ConstraintContextObj::snapshot`] to capture the current prefix.
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ConstraintContext {
    base: ObjectArc<ConstraintContextObj>,
}

impl Deref for ConstraintContext {
    type Target = ConstraintContextObj;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl ConstraintContext {
    /// Construct an empty root using the complete native-compatible layout.
    pub fn new() -> Result<Self> {
        Self::allocate(None, 0)
    }

    /// Capture a prefix of the parent's local entries and start an empty child.
    /// Earlier ancestor prefixes are inherited unchanged; later appends are excluded.
    pub fn with_parent_prefix_size(parent: Self, parent_prefix_size: i64) -> Result<Self> {
        if parent_prefix_size < 0 || parent_prefix_size > parent.local_size() {
            return Err(Error::new(
                VALUE_ERROR,
                "Constraint context parent local prefix is out of bounds",
                "",
            ));
        }
        Self::allocate(Some(parent.base), parent_prefix_size)
    }

    fn allocate(parent: Option<ObjectArc<ConstraintContextObj>>, prefix: i64) -> Result<Self> {
        Ok(Self {
            base: ObjectArc::new(ConstraintContextObj {
                base: Object::new(),
                parent,
                parent_prefix_size: prefix,
                // Start with an empty Rust array. Native Append owns all growth;
                // never install a nonempty Rust allocation in this mutable field.
                constraints: UnsafeCell::new(Array::default()),
            }),
        })
    }
}

impl ConstraintContextObj {
    /// Append through the existing native operation, retaining its array COW behavior.
    /// A running native visitor reserves its active context for semantic hooks;
    /// that visitor contract forbids observer appends during traversal. Standalone
    /// contexts and [`Self::for_each`] permit append.
    /// Premise nodes and referenced expressions must not be mutated through aliases.
    pub fn append(&self, constraint: &Constraint) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.ConstraintContextAppend")
            .call_packed(&[AnyView::from(self), AnyView::from(constraint)])?;
        Ok(())
    }

    /// Number of local facts, excluding captured ancestors.
    pub fn local_size(&self) -> i64 {
        // No callback or mutation occurs during this short borrow.
        unsafe { (&*self.constraints.get()).len() as i64 }
    }

    /// Capture the visible prefix in an independently extensible child.
    pub fn snapshot(&self) -> Result<ConstraintContext> {
        let parent = AnyView::from(self)
            .try_as::<ConstraintContext>()
            .expect("context has its registered native type");
        ConstraintContext::with_parent_prefix_size(parent, self.local_size())
    }

    fn local_array(&self) -> Array<Constraint> {
        // Hold an owning clone before any callback; native append then uses COW.
        unsafe { (&*self.constraints.get()).clone() }
    }

    /// Visit a fixed inherited/local prefix. A callback may append without entering
    /// this iteration; each borrowed fact is valid only during its callback.
    pub fn for_each(&self, mut callback: impl FnMut(&Constraint)) {
        let retained = AnyView::from(self)
            .try_as::<ConstraintContext>()
            .expect("context has its registered native type");
        retained.for_each_prefix(retained.local_size(), &mut callback);
    }

    fn for_each_prefix(&self, prefix: i64, callback: &mut impl FnMut(&Constraint)) {
        let local = self.local_array();
        if let Some(parent) = &self.parent {
            parent.for_each_prefix(self.parent_prefix_size, callback);
        }
        for constraint in local.iter().take(prefix as usize) {
            callback(&constraint);
        }
    }

    /// Return an independent sequence in inherited/local order. Later context
    /// appends do not change it; fact nodes and expressions remain shared.
    pub fn to_array(&self) -> Array<Constraint> {
        if self.parent.is_none() {
            return self.local_array();
        }
        let mut facts = Vec::new();
        self.for_each(|fact| facts.push(fact.clone()));
        Array::new(facts)
    }
}

// These verify the Rust mirror against the matching x86_64 header layout, not
// an arbitrary loaded binary's hidden fields. Stub generation cannot certify them.
const _: () = {
    assert!(size_of::<ConstraintContextObj>() == 48);
    assert!(align_of::<ConstraintContextObj>() == 8);
    assert!(offset_of!(ConstraintContextObj, parent) == 24);
    assert!(offset_of!(ConstraintContextObj, parent_prefix_size) == 32);
    assert!(offset_of!(ConstraintContextObj, constraints) == 40);
};
