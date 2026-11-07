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

//! Constraint-aware structural walking. [`structural_walk_with_constraint_context`] calls a closure
//! on every value with the facts in scope there, to observe the value, not to rewrite it.

use std::borrow::BorrowMut;
use tvm_ffi::extra::structural_visit::DefRegionKind;
use tvm_ffi::{Any, AnyView, Dispatcher, IntoDispatcher, Result, VisitInterrupt, WalkOrder, WalkResult};

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
        // Post callbacks observe the restored context.
        // SAFETY: the observer cannot reach this visitor (a nested walk has its own runtime), and
        // the borrow ends with the callback, before default descent can replace the field.
        let constraints = unsafe { visitor.base().constraints_for_observer() };
        let outcome = (&mut *self.observer).dispatch_tuple((value, constraints, kind));
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
                Ok(Some(VisitInterrupt::new(Any::new())))
            }
            WalkResult::InterruptWith(payload) => {
                core::hint::cold_path();
                Ok(Some(VisitInterrupt::new(payload)))
            }
        }
    }
}

/// Walk every value of `root` with `walker`, in `order`, with the facts in scope at each value.
///
/// A callback takes a leading prefix of `(value, &ConstraintContext, DefRegionKind)` and returns a
/// [`WalkResult`]: in pre-order `Skip` prunes the children, and `Interrupt` or an error stops the
/// walk. Callbacks must not append to the context or panic.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{Evaluate, EvaluateObj, For, ForKind, PrimType, PrimVar, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::sym::{ConstraintContext, RangeConstraintObj};
/// use tvm_tirx::tirx::structural_walk_with_constraint_context as walk;
/// use tvm_tirx::tvm_ffi::{ObjectRefCore, WalkOrder, WalkResult};
///
/// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let body = Evaluate::new(&i, UNKNOWN_LOC);
/// let root = For::new(&i, 0, 16, 1, ForKind::Default, body, None, UNKNOWN_LOC)?;
///
/// let mut ranges = Vec::new();
/// let record = |_: &EvaluateObj, context: &ConstraintContext| {
///     context.for_each(|fact| {
///         if let Some(range) = fact.downcast_ref::<RangeConstraintObj>() {
///             ranges.push(script(&range.range).unwrap());
///         }
///     });
///     WalkResult::Advance
/// };
/// walk::structural_walk_with_constraint_context(&root, record, WalkOrder::PreOrder)?;
/// assert_eq!(ranges, ["T.Range(0, 16)"]);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
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
