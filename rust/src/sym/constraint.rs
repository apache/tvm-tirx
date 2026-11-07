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

//! Constraints. A [`Constraint`] is one fact in scope, a [`BindingConstraint`],
//! [`RangeConstraint`] or [`PredicateConstraint`]; a [`ConstraintContext`] holds the facts in
//! scope at a point of a traversal, which an [`Analyzer`](super::Analyzer) proves under.

// tvm-ffi-stubgen(missing-docs): allow
// tvm-ffi-stubgen(ty-map): sym.ConstraintContext -> crate::sym::constraint::ConstraintContext

use std::cell::UnsafeCell;
use std::mem::{align_of, offset_of, size_of};
use tvm_ffi::{AnyView, Array, Error, Result, VALUE_ERROR};
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): tvm_ffi::IntoObjectRef
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// A binding's value and a predicate are `PrimExpr`s, as in C++.
// tvm-ffi-stubgen(field): sym.BindingConstraint.value -> PrimExpr
// tvm-ffi-stubgen(field): sym.PredicateConstraint.condition -> PrimExpr
// tvm-ffi-stubgen(begin): import-section
use crate::ir::expr::PrimExpr;
use crate::ir::expr::Range;
use crate::ir::expr::Var;
use std::borrow::Borrow;
use std::ops::Deref;
use std::ptr::addr_of_mut;
use tvm_ffi::IntoObjectRef;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::reflection::ShallowCopy;
use tvm_ffi::reflection::UnsafeInit;
// tvm-ffi-stubgen(end)

/// The base of every fact a [`ConstraintContext`] holds; it owns no analyzer state.
// tvm-ffi-stubgen(begin): object/sym.Constraint
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Constraint {
    data_: ObjectArc<ConstraintObj>,
}

/// The object behind [`Constraint`].
///
/// Complete: reflected fields fill [24, 24) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.Constraint"]
pub struct ConstraintObj {
    base_: Object,
}

const _: () = {
    assert!(::core::mem::size_of::<ConstraintObj>() == 24);
    assert!(::core::mem::align_of::<ConstraintObj>() == 8);
};

#[allow(missing_docs)]
impl ConstraintObj {
    pub fn new() -> Self {
        let base_ = Object::new();
        Self { base_ }
    }
}

#[allow(missing_docs)]
impl Constraint {
    /// Lossless complete-field allocation.
    pub fn new() -> Self {
        let obj = ConstraintObj::new();
        Self { data_: ObjectArc::new(obj) }
    }
}

impl ShallowCopy for ConstraintObj {
    fn shallow_copy(&self) -> Self {
        Self { base_: ShallowCopy::shallow_copy(&self.base_) }
    }
}

unsafe impl UnsafeInit for ConstraintObj {
    unsafe fn drop_defined_fields(this: *mut Self) {
        unsafe {
            UnsafeInit::drop_defined_fields(addr_of_mut!((*this).base_));
        }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for ConstraintObj {}
// tvm-ffi-stubgen(end)

/// The binding of `var` to `value`. The value is the bound expression as written, memory reads
/// included: it records what `var` was bound to, not an equality that holds afterwards.
// tvm-ffi-stubgen(begin): object/sym.BindingConstraint
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct BindingConstraint {
    data_: ObjectArc<BindingConstraintObj>,
}

/// The object behind [`BindingConstraint`].
///
/// Complete: reflected fields fill [24, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.BindingConstraint"]
#[type_final]
pub struct BindingConstraintObj {
    base_: ConstraintObj,
    pub var: Var,
    pub value: PrimExpr,
}

const _: () = {
    assert!(::core::mem::size_of::<BindingConstraintObj>() == 40);
    assert!(::core::mem::align_of::<BindingConstraintObj>() == 8);
};

impl Deref for BindingConstraintObj {
    type Target = ConstraintObj;
    fn deref(&self) -> &ConstraintObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl BindingConstraintObj {
    pub(crate) fn new(var: Var, value: PrimExpr) -> Self {
        let base_ = ConstraintObj::new();
        Self { base_, var, value }
    }
}

#[allow(missing_docs)]
impl BindingConstraint {
    /// Lossless complete-field allocation.
    pub fn new(var: impl IntoObjectRef<Var>, value: impl IntoObjectRef<PrimExpr>) -> Self {
        let obj =
            BindingConstraintObj::new(var.into_owned_object_ref(), value.into_owned_object_ref());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for BindingConstraintObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(BindingConstraint => Constraint);
// tvm-ffi-stubgen(end)

/// The fact that the boolean `condition` holds. With `assume_always`, the caller guarantees it
/// holds throughout the scope, even after the values it reads change.
// tvm-ffi-stubgen(begin): object/sym.PredicateConstraint
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct PredicateConstraint {
    data_: ObjectArc<PredicateConstraintObj>,
}

/// The object behind [`PredicateConstraint`].
///
/// Complete: reflected fields fill [24, 40) exactly (alignment padding [33, 40)).
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.PredicateConstraint"]
#[type_final]
pub struct PredicateConstraintObj {
    base_: ConstraintObj,
    pub condition: PrimExpr,
    pub assume_always: bool,
}

const _: () = {
    assert!(::core::mem::size_of::<PredicateConstraintObj>() == 40);
    assert!(::core::mem::align_of::<PredicateConstraintObj>() == 8);
};

impl Deref for PredicateConstraintObj {
    type Target = ConstraintObj;
    fn deref(&self) -> &ConstraintObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl PredicateConstraintObj {
    pub(crate) fn new(condition: PrimExpr, assume_always: bool) -> Self {
        let base_ = ConstraintObj::new();
        Self { base_, condition, assume_always }
    }
}

#[allow(missing_docs)]
impl PredicateConstraint {
    /// Lossless complete-field allocation.
    pub fn new(condition: impl IntoObjectRef<PrimExpr>, assume_always: bool) -> Self {
        let obj = PredicateConstraintObj::new(condition.into_owned_object_ref(), assume_always);
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for PredicateConstraintObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(PredicateConstraint => Constraint);
// tvm-ffi-stubgen(end)

/// The fact that `var` lies in the half-open `range`, which may be wider than the values `var`
/// takes.
// tvm-ffi-stubgen(begin): object/sym.RangeConstraint
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RangeConstraint {
    data_: ObjectArc<RangeConstraintObj>,
}

/// The object behind [`RangeConstraint`].
///
/// Complete: reflected fields fill [24, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.RangeConstraint"]
#[type_final]
pub struct RangeConstraintObj {
    base_: ConstraintObj,
    pub var: Var,
    pub range: Range,
}

const _: () = {
    assert!(::core::mem::size_of::<RangeConstraintObj>() == 40);
    assert!(::core::mem::align_of::<RangeConstraintObj>() == 8);
};

impl Deref for RangeConstraintObj {
    type Target = ConstraintObj;
    fn deref(&self) -> &ConstraintObj {
        &self.base_
    }
}

#[allow(missing_docs)]
impl RangeConstraintObj {
    pub(crate) fn new(var: Var, range: Range) -> Self {
        let base_ = ConstraintObj::new();
        Self { base_, var, range }
    }
}

#[allow(missing_docs)]
impl RangeConstraint {
    /// Lossless complete-field allocation.
    pub fn new(var: impl IntoObjectRef<Var>, range: impl IntoObjectRef<Range>) -> Self {
        let obj =
            RangeConstraintObj::new(var.into_owned_object_ref(), range.into_owned_object_ref());
        Self { data_: ObjectArc::new(obj) }
    }
}

unsafe impl tvm_ffi::object::ImmutableObject for RangeConstraintObj {}

tvm_ffi::unsafe_impl_object_ref_upcast!(RangeConstraint => Constraint);
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
    base_: Object,
    parent: Option<ObjectArc<ConstraintContextObj>>,
    parent_prefix_size: i64,
    constraints: UnsafeCell<Array<Constraint>>,
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

/// A context of constraints: the premises in effect at a point of a traversal, ancestors' first.
///
/// Clones share the context's identity and later appends; use
/// [`snapshot`](ConstraintContextObj::snapshot) to capture the current prefix. A context is neither
/// `Send` nor `Sync`: do not share one across threads, also not through an erased `ObjectRef`.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::{ge, lt};
/// use tvm_tirx::ir::{PrimType, PrimVar, Var, UNKNOWN_LOC};
/// use tvm_tirx::sym::{Analyzer, ConstraintContext, PredicateConstraint, ProofStrength};
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let outer = ConstraintContext::new();
/// outer.append(PredicateConstraint::new(ge(&x, 0)?, false))?;
/// let inner = outer.snapshot()?;
/// inner.append(PredicateConstraint::new(lt(&x, 16)?, false))?;
///
/// let analyzer = Analyzer::new();
/// analyzer.sync_constraint_context(&inner)?;
/// assert!(analyzer.can_prove(lt(&x, 16)?, ProofStrength::Default)?);
/// analyzer.sync_constraint_context(&outer)?;
/// assert!(!analyzer.can_prove(lt(&x, 16)?, ProofStrength::Default)?);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ConstraintContext {
    data_: ObjectArc<ConstraintContextObj>,
}

impl ConstraintContextObj {
    /// Append `constraint` to the local entries (`sym.ConstraintContextAppend`).
    ///
    /// Do not append to the active context of a running native visitor.
    pub fn append(&self, constraint: impl IntoObjectRef<Constraint>) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.ConstraintContextAppend").call_packed(&[
            AnyView::from(self),
            AnyView::from(constraint.into_object_ref().borrow()),
        ])?;
        Ok(())
    }

    /// Return the number of local entries, excluding captured ancestors.
    pub fn local_size(&self) -> i64 {
        // SAFETY: nothing can append during this short borrow.
        unsafe { (&*self.constraints.get()).len() as i64 }
    }

    /// Capture the visible prefix in an independently extensible child.
    pub fn snapshot(&self) -> Result<ConstraintContext> {
        let parent = ConstraintContext::try_from(AnyView::from(self))
            .expect("context has its registered native type");
        ConstraintContext::with_parent_prefix_size(parent, self.local_size())
    }

    fn local_array(&self) -> Array<Constraint> {
        // SAFETY: the borrow ends with this clone, before any callback can append; an append then
        // copies the array on write.
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
            callback(constraint);
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

impl ConstraintContext {
    /// Create an empty root context, with the complete native-compatible layout.
    pub fn new() -> Self {
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
    pub fn with_parent_prefix_size(
        parent: impl IntoObjectRef<ConstraintContext>,
        parent_prefix_size: i64,
    ) -> Result<Self> {
        let parent = parent.into_owned_object_ref();
        if parent_prefix_size < 0 || parent_prefix_size > parent.local_size() {
            return Err(Error::new(
                VALUE_ERROR,
                "Constraint context parent local prefix is out of bounds",
                "",
            ));
        }
        Ok(Self::allocate(Some(parent.data_), parent_prefix_size))
    }

    fn allocate(parent: Option<ObjectArc<ConstraintContextObj>>, prefix: i64) -> Self {
        Self {
            data_: ObjectArc::new(ConstraintContextObj {
                base_: Object::new(),
                parent,
                parent_prefix_size: prefix,
                // Start with an empty Rust array. Native Append owns all growth;
                // never install a nonempty Rust allocation in this mutable field.
                constraints: UnsafeCell::new(Array::default()),
            }),
        }
    }
}
