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

//! Constraint-aware structural mapping. [`structural_map_with_constraint_context`] calls a closure
//! on every value with the facts in scope there and replaces the value with what it returns,
//! unless that is `Unchanged`.

use std::borrow::BorrowMut;

use tvm_ffi::{
    Any, AnyView, DefRegionKind, Dispatcher, IntoDispatcher, MutationView, Object, Result,
    UnchangedOr, WalkOrder,
};

use super::structural_mutate_with_constraint_context::{
    structural_mutate_with_constraint_context, StructuralMutatorWithConstraintContextView,
};
use crate::sym::ConstraintContext;

struct MapMutator<'a, D, const PRE: bool> {
    mapper: &'a mut D,
}

impl<'v, 'c, 'r, 'a, D, const PRE: bool>
    Dispatcher<
        (
            MutationView<'v>,
            &'c mut StructuralMutatorWithConstraintContextView<'r, MapMutator<'a, D, PRE>>,
        ),
        Option<Result<UnchangedOr<Any>>>,
    > for &mut MapMutator<'a, D, PRE>
where
    for<'s, 'b, 'ctx> &'s mut D: Dispatcher<
        (AnyView<'b>, &'ctx ConstraintContext, DefRegionKind),
        Option<Result<UnchangedOr<Any>>>,
    >,
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, mutator): (
            MutationView<'v>,
            &'c mut StructuralMutatorWithConstraintContextView<'r, MapMutator<'a, D, PRE>>,
        ),
    ) -> Option<Result<UnchangedOr<Any>>> {
        Some(self.map(value, mutator))
    }
}

impl<D, const PRE: bool> MapMutator<'_, D, PRE>
where
    for<'s, 'v, 'c> &'s mut D: Dispatcher<
        (AnyView<'v>, &'c ConstraintContext, DefRegionKind),
        Option<Result<UnchangedOr<Any>>>,
    >,
{
    #[inline(always)]
    fn map(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorWithConstraintContextView<'_, Self>,
    ) -> Result<UnchangedOr<Any>> {
        let kind = mutator.def_region_kind();
        if PRE {
            let mode = value.inplace_mode();
            // SAFETY: the mapper cannot reach this mutator, and the borrow ends with the
            // callback, before descent can replace the field.
            let constraints = unsafe { mutator.base().constraints_for_observer() };
            let result = self.mapper.dispatch_tuple((value.as_any_view(), constraints, kind));
            let Some(result) = result else {
                return mutator.default_mutate(self, value);
            };
            let mapped = result?;
            if mapped.is_unchanged() {
                return mutator.default_mutate(self, value);
            }
            let mut mapped: Any = mapped.into();
            let same_object = value
                .downcast_ref::<Object>()
                .zip(mapped.downcast_ref::<Object>())
                .is_some_and(|(original, replacement)| std::ptr::eq(original, replacement));
            if same_object {
                // An identity callback must not make a unique root appear shared.
                drop(mapped);
                return mutator.default_mutate(self, value);
            }
            // An owned replacement can lend a capability, restricted by the
            // original path's permission. The shared engine checks live aliases.
            let replacement = MutationView::from(&mut mapped).narrow_inplace_mode(mode);
            let result = mutator.default_mutate(self, replacement)?;
            Ok(UnchangedOr::changed(result.value_or_unchanged(mapped)))
        } else {
            // SAFETY: this private adapter runs only inside the shared engine
            // callback. The engine owns the root; recursive callers keep each
            // input field owned until this synchronous callback returns. Default
            // descent can mutate fields but cannot destroy or retype that input.
            // This snapshot holds no typed node reference across descent. Only
            // reborrow it afterward when Unchanged preserves the input, and never
            // let the reborrow escape this callback.
            let original: AnyView<'_> = unsafe { std::mem::transmute(value.as_any_view()) };
            let mapped = mutator.default_mutate(self, value)?;
            let current = if mapped.is_unchanged() { original } else { AnyView::from(&mapped) };
            // Scoped native hooks have restored the incoming context here.
            // SAFETY: as above; descent has ended.
            let constraints = unsafe { mutator.base().constraints_for_observer() };
            let result = self.mapper.dispatch_tuple((current, constraints, kind));
            match result {
                Some(result) => {
                    let result = result?;
                    Ok(if result.is_unchanged() { mapped } else { result })
                }
                None => Ok(mapped),
            }
        }
    }
}

/// Map every value of `root` with `mapper`, in `order`, with the facts in scope at each value.
///
/// A callback takes a leading prefix of `(value, &ConstraintContext, DefRegionKind)` and returns a
/// replacement, `Unchanged` or a `Result` of one; `Unchanged` still descends in pre-order.
/// Callbacks must not append to the context or panic, and an error stops the traversal without
/// undoing in-place changes.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{Bind, Evaluate, PrimType, PrimVar, SeqStmt, Stmt, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::sym::{BindingConstraintObj, ConstraintContext};
/// use tvm_tirx::tirx::structural_map_with_constraint_context as map;
/// use tvm_tirx::tvm_ffi::{Any, ObjectRefCore, Unchanged, WalkOrder};
///
/// let substitute = |var: Var, context: &ConstraintContext| -> Any {
///     let mut value: Any = Unchanged.into();
///     context.for_each(|fact| match fact.downcast_ref::<BindingConstraintObj>() {
///         Some(binding) if binding.var.same_as(&var) => value = binding.value.clone().into(),
///         _ => {}
///     });
///     value
/// };
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let bind = Stmt::from(Bind::new(&x, 4, UNKNOWN_LOC)?);
/// let root = SeqStmt::new([bind, Evaluate::new(&x, UNKNOWN_LOC).into()], UNKNOWN_LOC);
/// let result =
///     map::structural_map_with_constraint_context(root, substitute, WalkOrder::PreOrder)?;
/// assert_eq!(script(&SeqStmt::try_from(result)?)?, "x: T.let[T.int32] = 4\nT.evaluate(4)");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn structural_map_with_constraint_context<R, H, M, D, O>(
    root: R,
    mapper: H,
    order: WalkOrder,
) -> Result<Any>
where
    R: Into<Any>,
    O: BorrowMut<D>,
    for<'v, 'c> H: IntoDispatcher<
        (AnyView<'v>, &'c ConstraintContext, DefRegionKind),
        Option<Result<UnchangedOr<Any>>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c> &'s mut D: Dispatcher<
        (AnyView<'v>, &'c ConstraintContext, DefRegionKind),
        Option<Result<UnchangedOr<Any>>>,
    >,
{
    let mut storage = mapper.into_dispatcher();
    let mapper = storage.borrow_mut();
    match order {
        WalkOrder::PreOrder => {
            structural_mutate_with_constraint_context(root, MapMutator::<D, true> { mapper })
        }
        WalkOrder::PostOrder => {
            structural_mutate_with_constraint_context(root, MapMutator::<D, false> { mapper })
        }
    }
}
