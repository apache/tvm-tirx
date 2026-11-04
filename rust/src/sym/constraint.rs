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

// tvm-ffi-stubgen(begin): object/sym.ConstraintContext
/// Opaque: bytes [24, 48) of [24, 48) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[type_key = "sym.ConstraintContext"]
#[type_final]
pub struct ConstraintContextObj {
    base: Object,
}

#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ConstraintContext {
    base: ObjectArc<ConstraintContextObj>,
}

impl Deref for ConstraintContext {
    type Target = ConstraintContextObj;
    fn deref(&self) -> &ConstraintContextObj {
        &self.base
    }
}
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
