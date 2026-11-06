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

//! Constraints: the facts in scope at a point of a traversal.
//!
//! A [`ConstraintContext`] holds [`BindingConstraint`]s, [`RangeConstraint`]s and
//! [`PredicateConstraint`]s, ancestors' first. Its handles are neither `Send` nor `Sync`; do not
//! share a context across threads, also not through an erased `ObjectRef`.
//!
//! # Examples
//!
//! ```
//! use tvm_tirx::ir::prim::op::{ge, lt};
//! use tvm_tirx::ir::{PrimType, PrimVar, Var, UNKNOWN_LOC};
//! use tvm_tirx::sym::{Analyzer, ConstraintContext, PredicateConstraint, ProofStrength};
//!
//! let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
//! let outer = ConstraintContext::new()?;
//! outer.append(&PredicateConstraint::new(ge(&x, 0)?, false).into())?;
//! let inner = outer.snapshot()?;
//! inner.append(&PredicateConstraint::new(lt(&x, 16)?, false).into())?;
//!
//! let analyzer = Analyzer::new();
//! analyzer.sync_constraint_context(&inner)?;
//! assert!(analyzer.can_prove(lt(&x, 16)?, ProofStrength::Default)?);
//! analyzer.sync_constraint_context(&outer)?;
//! assert!(!analyzer.can_prove(lt(&x, 16)?, ProofStrength::Default)?);
//! # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
//! ```

// tvm-ffi-stubgen(ty-map): sym.ConstraintContext -> crate::sym::constraint::ConstraintContext

use std::cell::UnsafeCell;
use std::mem::{align_of, offset_of, size_of};
use tvm_ffi::{AnyView, Array, Error, ObjectCore, Result, VALUE_ERROR};
// The base `Constraint` has no C++ constructor, so no `new`: `custom-new` leaves only the generated
// `from_fields`.
// tvm-ffi-stubgen(custom-new): sym.Constraint
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// A binding's value and a predicate are `PrimExpr`s, as in C++.
// tvm-ffi-stubgen(field): sym.BindingConstraint.value -> PrimExpr
// tvm-ffi-stubgen(field): sym.PredicateConstraint.condition -> PrimExpr
// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::PrimExpr;
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
    pub fn from_fields() -> Self {
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

unsafe impl tvm_ffi::object::ImmutableObject for ConstraintObj {}
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
    pub value: PrimExpr,
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
    pub(crate) fn new(var: Var, value: PrimExpr) -> Self {
        let base = ConstraintObj::new();
        Self { base, var, value }
    }
}

impl BindingConstraint {
    /// Lossless complete-field allocation.
    pub fn new(var: impl Into<Var>, value: impl Into<PrimExpr>) -> Self {
        let obj = BindingConstraintObj::new(var.into(), value.into());
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BindingConstraintObj {}

tvm_ffi::unsafe_impl_object_upcast!(BindingConstraint => Constraint);
// tvm-ffi-stubgen(end)

// tvm-ffi-stubgen(begin): object/sym.PredicateConstraint
/// Complete: reflected fields fill [24, 40) exactly (alignment padding [33, 40)).
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.PredicateConstraint"]
#[type_final]
pub struct PredicateConstraintObj {
    base: ConstraintObj,
    pub condition: PrimExpr,
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
    pub(crate) fn new(condition: PrimExpr, assume_always: bool) -> Self {
        let base = ConstraintObj::new();
        Self { base, condition, assume_always }
    }
}

impl PredicateConstraint {
    /// Lossless complete-field allocation.
    pub fn new(condition: impl Into<PrimExpr>, assume_always: bool) -> Self {
        let obj = PredicateConstraintObj::new(condition.into(), assume_always);
        Self { base: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for PredicateConstraintObj {}

tvm_ffi::unsafe_impl_object_upcast!(PredicateConstraint => Constraint);
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

unsafe impl tvm_ffi::object::ImmutableObject for RangeConstraintObj {}

tvm_ffi::unsafe_impl_object_upcast!(RangeConstraint => Constraint);
// tvm-ffi-stubgen(end)

/// The object of a [`ConstraintContext`], with the layout of the native one and a stable identity.
///
/// Only the local array changes after construction. Its handle is replaced by native append, so
/// shared context inspection must use interior mutability. Parent links and their captured local
/// prefix are immutable and acyclic.
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

/// A context of constraints: the premises in effect at a point of a traversal, ancestors' first.
///
/// Clones share the context's identity and later appends; use
/// [`snapshot`](ConstraintContextObj::snapshot) to capture the current prefix.
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
    /// Create an empty root context, with the complete native-compatible layout.
    pub fn new() -> Result<Self> {
        Self::allocate(None, 0)
    }

    /// Create an empty child that captures the first `parent_prefix_size` local entries of
    /// `parent`.
    ///
    /// Earlier ancestor prefixes are inherited unchanged; later appends to the parent are excluded.
    ///
    /// # Errors
    ///
    /// Returns an error if `parent_prefix_size` is negative or larger than the parent's local size.
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
    /// Append `constraint` to the local entries (`sym.ConstraintContextAppend`).
    ///
    /// Do not append to the active context of a running native visitor.
    pub fn append(&self, constraint: &Constraint) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.ConstraintContextAppend")
            .call_packed(&[AnyView::from(self), AnyView::from(constraint)])?;
        Ok(())
    }

    /// Return the number of local entries, excluding captured ancestors.
    pub fn local_size(&self) -> i64 {
        // No callback or mutation occurs during this short borrow.
        unsafe { (&*self.constraints.get()).len() as i64 }
    }

    /// Capture the visible prefix in an independently extensible child.
    pub fn snapshot(&self) -> Result<ConstraintContext> {
        let parent = ConstraintContext::try_from(AnyView::from(self))
            .expect("context has its registered native type");
        ConstraintContext::with_parent_prefix_size(parent, self.local_size())
    }

    fn local_array(&self) -> Array<Constraint> {
        // Hold an owning clone before any callback; native append then uses COW.
        unsafe { (&*self.constraints.get()).clone() }
    }

    /// Call `callback` on each visible constraint, in inherited and then local order.
    ///
    /// The callback may append without entering this iteration; each borrowed constraint is valid
    /// only during its callback.
    pub fn for_each(&self, mut callback: impl FnMut(&Constraint)) {
        let retained = ConstraintContext::try_from(AnyView::from(self))
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

    /// Return the visible constraints in inherited and then local order, as an independent array.
    ///
    /// Later appends do not change it; constraint nodes and expressions remain shared.
    pub fn to_array(&self) -> Array<Constraint> {
        if self.parent.is_none() {
            return self.local_array();
        }
        let mut facts = Vec::new();
        self.for_each(|fact| facts.push(fact.clone()));
        Array::from_vec(facts)
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
