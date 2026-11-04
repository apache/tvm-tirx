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

//! Ordered constraint-aware mapping through the shared mutation engine.
//!
//! This adapter owns callback order only. The mutation engine owns dispatch,
//! remaps and copy-on-write; native hooks own lexical facts and scope restoration.

use std::borrow::BorrowMut;

use tvm_ffi::{
    Any, AnyView, DefRegionKind, Dispatcher, IntoDispatcher, MutationView, Object, Result,
    WalkOrder,
};

use tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex;

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
        Option<Result<Any>>,
    > for &mut MapMutator<'a, D, PRE>
where
    for<'s, 'b, 'ctx> &'s mut D:
        Dispatcher<(AnyView<'b>, &'ctx ConstraintContext, DefRegionKind), Option<Result<Any>>>,
{
    #[inline(always)]
    fn dispatch_tuple(
        self,
        (value, mutator): (
            MutationView<'v>,
            &'c mut StructuralMutatorWithConstraintContextView<'r, MapMutator<'a, D, PRE>>,
        ),
    ) -> Option<Result<Any>> {
        Some(self.map(value, mutator))
    }
}

impl<D, const PRE: bool> MapMutator<'_, D, PRE>
where
    for<'s, 'v, 'c> &'s mut D:
        Dispatcher<(AnyView<'v>, &'c ConstraintContext, DefRegionKind), Option<Result<Any>>>,
{
    #[inline(always)]
    fn map(
        &mut self,
        value: MutationView<'_>,
        mutator: &mut StructuralMutatorWithConstraintContextView<'_, Self>,
    ) -> Result<Any> {
        let kind = mutator.def_region_kind();
        if PRE {
            let mode = value.inplace_mode();
            // The mapper cannot access this runtime. Its context borrow ends
            // before descent, which may replace the interior-mutable field.
            let result = unsafe {
                self.mapper.dispatch_tuple((
                    value.as_any_view(),
                    mutator.base().constraints_for_observer(),
                    kind,
                ))
            };
            let Some(result) = result else {
                return mutator.default_mutate(self, value).map(Into::into);
            };
            let mut mapped = result?;
            if mapped.type_index() == TVMFFITypeIndex::kTVMFFIUnchanged as i32 {
                return mutator.default_mutate(self, value).map(Into::into);
            }
            let same_object = value
                .downcast_ref::<Object>()
                .zip(mapped.downcast_ref::<Object>())
                .is_some_and(|(original, replacement)| std::ptr::eq(original, replacement));
            if same_object {
                // An identity callback must not make a unique root appear shared.
                drop(mapped);
                return mutator.default_mutate(self, value).map(Into::into);
            }
            // An owned replacement can lend a capability, restricted by the
            // original path's permission. The shared engine checks live aliases.
            let replacement = MutationView::from(&mut mapped).with_inplace_mode(mode);
            let result = mutator.default_mutate(self, replacement)?;
            Ok(result.value_or(mapped))
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
            let unchanged = mapped.is_unchanged();
            let mapped: Any = mapped.into();
            let current = if unchanged { original } else { AnyView::from(&mapped) };
            // Scoped native hooks have restored the incoming context here.
            let result = unsafe {
                self.mapper.dispatch_tuple((
                    current,
                    mutator.base().constraints_for_observer(),
                    kind,
                ))
            };
            match result {
                Some(result) => {
                    let result = result?;
                    Ok(if result.type_index() == TVMFFITypeIndex::kTVMFFIUnchanged as i32 {
                        mapped
                    } else {
                        result
                    })
                }
                None => Ok(mapped),
            }
        }
    }
}

/// Map every occurrence with the constraint facts active at that position.
///
/// Pre-order callbacks run before descending into the replacement's children;
/// post-order callbacks see the rewritten value after its scope is restored.
/// Facts for later siblings use the resulting bindings and expressions.
/// `Unchanged` keeps the current value and still descends in pre-order.
/// The root is consumed: moved unique roots may reuse storage, retained roots copy.
/// Errors stop traversal without rolling back completed in-place changes.
///
/// Accepts the shared [`IntoDispatcher`] forms: owned or borrowed dispatchers,
/// typed callbacks, and nested tuples. Callbacks take a leading prefix of
/// `(value, &ConstraintContext, DefRegionKind)` and return a replacement,
/// `Unchanged`, or a `Result` thereof. Observers must not append to the active
/// context or panic across native callbacks. Clone a context to retain its live
/// handle, or use `snapshot()` / `to_array()` for a fixed prefix.
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
        Option<Result<Any>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c> &'s mut D:
        Dispatcher<(AnyView<'v>, &'c ConstraintContext, DefRegionKind), Option<Result<Any>>>,
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
