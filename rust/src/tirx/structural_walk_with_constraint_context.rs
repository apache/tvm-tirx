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

//! Constraint-aware observation through the shared Rust visitor engine.
//!
//! This adapter chooses pre/post callback order and translates walk control;
//! the visit engine and native constraint hooks own descent, scopes and reborrows.

use std::borrow::BorrowMut;
use tvm_ffi::{
    Any, AnyView, DefRegionKind, Dispatcher, IntoDispatcher, Result, VisitInterrupt, WalkOrder,
    WalkResult,
};

use crate::sym::ConstraintContext;
use super::structural_visit_with_constraint_context::{
    structural_visit_with_constraint_context, StructuralVisitorWithConstraintContextView,
};

struct WalkVisitor<'a, D, const PRE: bool> {
    observer: &'a mut D,
}

impl<'v, 'c, 'r, 'a, D, const PRE: bool>
    Dispatcher<
        (
            AnyView<'v>,
            &'c mut StructuralVisitorWithConstraintContextView<'r, WalkVisitor<'a, D, PRE>>,
        ),
        Option<Result<Option<VisitInterrupt>>>,
    > for &mut WalkVisitor<'a, D, PRE>
where
    for<'s, 'b, 'ctx> &'s mut D: Dispatcher<
        (AnyView<'b>, &'ctx ConstraintContext, DefRegionKind),
        Option<Result<WalkResult>>,
    >,
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, visitor): (
            AnyView<'v>,
            &'c mut StructuralVisitorWithConstraintContextView<'r, WalkVisitor<'a, D, PRE>>,
        ),
    ) -> Option<Result<Option<VisitInterrupt>>> {
        Some(self.visit(value, visitor))
    }
}

impl<D, const PRE: bool> WalkVisitor<'_, D, PRE>
where
    for<'s, 'v, 'c> &'s mut D:
        Dispatcher<(AnyView<'v>, &'c ConstraintContext, DefRegionKind), Option<Result<WalkResult>>>,
{
    #[inline(always)]
    fn visit(
        &mut self,
        value: AnyView<'_>,
        visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>,
    ) -> Result<Option<VisitInterrupt>> {
        let kind = visitor.def_region_kind();
        if !PRE {
            if let Some(interrupt) = visitor.default_visit(self, value)? {
                core::hint::cold_path();
                return Ok(Some(interrupt));
            }
        }
        // Post callbacks observe the restored context. The observer cannot access
        // this runtime's visitor; a nested walk creates its own runtime. Appending
        // mutates context contents, not this handle. End the field borrow before
        // any default descent can replace it; observers may clone it to retain it.
        let outcome = unsafe {
            (&mut *self.observer).dispatch_tuple((
                value,
                visitor.base().constraints_for_observer(),
                kind,
            ))
        };
        let outcome = match outcome {
            Some(result) => result?,
            None => WalkResult::Advance,
        };
        match outcome {
            WalkResult::Advance if PRE => visitor.default_visit(self, value),
            WalkResult::Advance => Ok(None),
            WalkResult::Skip => {
                core::hint::cold_path();
                Ok(None)
            }
            WalkResult::Interrupt => {
                core::hint::cold_path();
                Ok(Some(VisitInterrupt::with(Any::new())))
            }
            WalkResult::InterruptWith(payload) => {
                core::hint::cold_path();
                Ok(Some(VisitInterrupt::with(payload)))
            }
        }
    }
}

/// Walk every occurrence with its active constraint context and optional definition
/// region. Pre-order Skip prunes descent; post-order Skip completes the callback.
/// Interrupt and errors stop traversal. The native hooks own fact timing and scope
/// restoration; sequence/lazy-call hooks omit intermediate Array callbacks, as in C++.
///
/// As specified by the native visitor, observers must not append to its active
/// context during traversal. Standalone contexts and `for_each` allow append.
/// Observers must not panic across native callbacks.
/// Cloned contexts remain live; capture a snapshot or array for a fixed prefix.
/// Accepts the shared [`IntoDispatcher`] forms: owned or borrowed dispatchers,
/// typed value or node callbacks, and nested callback tuples. Callbacks receive
/// a leading prefix of `(value, &ConstraintContext, DefRegionKind)` and return an
/// `IntoWalkResult`. Named dispatchers use that full argument tuple and return
/// `Option<Result<WalkResult>>`; unmatched values advance normally.
/// The local adapter adds context observation to the shared visitor engine.
pub fn structural_walk_with_constraint_context<'input, R, H, M, D, O>(
    root: R,
    walker: H,
    order: WalkOrder,
) -> Result<Option<VisitInterrupt>>
where
    R: Into<AnyView<'input>>,
    O: BorrowMut<D>,
    for<'v, 'c> H: IntoDispatcher<
        (AnyView<'v>, &'c ConstraintContext, DefRegionKind),
        Option<Result<WalkResult>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c> &'s mut D:
        Dispatcher<(AnyView<'v>, &'c ConstraintContext, DefRegionKind), Option<Result<WalkResult>>>,
{
    let mut storage = walker.into_dispatcher();
    let observer = storage.borrow_mut();
    let root = root.into();
    match order {
        WalkOrder::PreOrder => {
            structural_visit_with_constraint_context(root, WalkVisitor::<D, true> { observer })
        }
        WalkOrder::PostOrder => {
            structural_visit_with_constraint_context(root, WalkVisitor::<D, false> { observer })
        }
    }
}
