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

//! Constraint-aware structural mutation through the shared Rust engine.
//!
//! The native semantic base carries only lexical context state. Rust owns its
//! dispatcher and remaps in `StructuralMutateEngine<Base>`; native constraint
//! hooks recurse through that engine's vtable and collect facts from results.

use std::borrow::BorrowMut;
use std::cell::UnsafeCell;

use tvm_ffi::extra::cold_init_lazy_lock::ColdInitLazyLock;
use tvm_ffi::extra::structural_mutate::{StructuralMutateEngine, StructuralMutatorBase};
use tvm_ffi::reflection::TypeAttrColumn;
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::{
    Any, Dispatcher, InplaceMode, IntoDispatcher, MutationView, ObjectArc, ObjectCore,
    ObjectRefCore, Result, StructuralMutatorObj, StructuralMutatorView,
};

use super::StructuralMutatorWithConstraintContextObj;
use crate::sym::ConstraintContext;

/// Complete native-compatible semantic base, without an owning object conversion.
///
/// Native hooks may replace the current context during descent through a shared
/// view. All native-mutated state is interior mutable. This representation
/// requires the matching native header/build, including the private scope flag.
#[repr(C)]
pub struct StructuralMutatorWithConstraintContextBase {
    base: StructuralMutatorObj,
    constraints: UnsafeCell<ConstraintContext>,
    needs_child_context: UnsafeCell<bool>,
}

impl StructuralMutatorWithConstraintContextBase {
    /// Clone the current live context without retaining a borrow of its field.
    /// Use `snapshot()` on the returned handle to freeze its current prefix.
    #[inline]
    pub fn constraints(&self) -> ConstraintContext {
        unsafe { (&*self.constraints.get()).clone() }
    }

    /// Borrow only while a mapper cannot recurse through this mutator.
    /// End this borrow before default descent can replace the context field.
    #[inline]
    pub(super) unsafe fn constraints_for_observer(&self) -> &ConstraintContext {
        &*self.constraints.get()
    }
}

// SAFETY: repr(C) matches the complete native state-only semantic base. The
// shared engine installs its vtable, owns remaps and publishes the dispatcher's
// actual recursive reborrow. Native hooks borrow this runtime synchronously and
// cannot retain it. No shared reference freezes the replaceable context/flag.
// Each default call transfers exactly one owning replacement/Unchanged/Error.
unsafe impl StructuralMutatorBase for StructuralMutatorWithConstraintContextBase {
    fn new() -> Result<Self> {
        Ok(Self {
            base: StructuralMutatorObj::default(),
            constraints: UnsafeCell::new(ConstraintContext::new()?),
            needs_child_context: UnsafeCell::new(false),
        })
    }

    fn type_index() -> i32 {
        StructuralMutatorWithConstraintContextObj::type_index()
    }

    #[inline(always)]
    unsafe fn default_mutate_raw(runtime: *mut Self, value: MutationView<'_>) -> TVMFFIAny {
        static COPYING: ColdInitLazyLock<Option<TypeAttrColumn>> =
            ColdInitLazyLock::new(|| TypeAttrColumn::new("__s_mutate_with_constraint_context__"));
        static INPLACE: ColdInitLazyLock<Option<TypeAttrColumn>> = ColdInitLazyLock::new(|| {
            TypeAttrColumn::new("__s_maybe_inplace_mutate_with_constraint_context__")
        });
        let borrowed = value.as_any_view();
        let lookup = |column: Option<TypeAttrColumn>| {
            column
                .and_then(|column| column.get_raw_ffi_any(borrowed.type_index()))
                .filter(|hook| hook.type_index != TVMFFITypeIndex::kTVMFFINone as i32)
        };
        let hook = if value.inplace_mode() == InplaceMode::Allow {
            lookup(*INPLACE).filter(|_| {
                // Permission alone is insufficient after a callback retains an
                // alias. The supported owning count includes this temporary;
                // release it before invoking a hook or entering native descent.
                borrowed.try_as::<ObjectRef>().is_some_and(|owner| {
                    let unique = ObjectArc::strong_count(ObjectRef::data(&owner)) == 2;
                    drop(owner);
                    unique
                })
            })
        } else {
            None
        }
        .or_else(|| lookup(*COPYING));
        if let Some(hook) = hook {
            // Copy the borrowed registry token; the shared helper acquires
            // Function ownership before packed reentry and handles error context.
            StructuralMutatorObj::call_structural_mutate_hook_raw(
                runtime.cast(),
                *borrowed.as_raw_ffi_any(),
                hook,
            )
        } else {
            // Prefix-only fallback uses installed remap/callback thunks, never
            // an ordinary Rust engine's dispatcher offset.
            StructuralMutatorObj::default_mutate_raw(runtime.cast(), value)
        }
    }
}

/// Scoped constraint mutator view. Pass the handler's actual `self` to recursive
/// operations and use `view.base().constraints()` to inspect resulting premises.
/// Callbacks follow the shared engine's non-escaping, non-unwinding contract.
pub type StructuralMutatorWithConstraintContextView<'a, D> =
    StructuralMutatorView<'a, D, StructuralMutatorWithConstraintContextBase>;

/// Mutate using native lexical binding, predicate and range semantics.
///
/// The first matching handler owns descent. Use `view.default_mutate(self, value)`
/// for constraint-aware default descent and `view.mutate(self, child)` to dispatch
/// recursively. A moved unique root may reuse storage; retained roots copy.
/// Specialized hooks obey the ordinary synchronous mutation ABI and registration
/// trust contract. They must not retain the runtime or append through observer
/// context handles during descent. Errors terminate the traversal.
pub fn structural_mutate_with_constraint_context<R, H, M, D, O>(
    root: R,
    dispatcher: H,
) -> Result<Any>
where
    R: Into<Any>,
    O: BorrowMut<D>,
    for<'v, 'c, 'r> H: IntoDispatcher<
        (MutationView<'v>, &'c mut StructuralMutatorWithConstraintContextView<'r, D>),
        Option<Result<Any>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
        (MutationView<'v>, &'c mut StructuralMutatorWithConstraintContextView<'r, D>),
        Option<Result<Any>>,
    >,
{
    StructuralMutateEngine::<StructuralMutatorWithConstraintContextBase>::mutate_root(
        root, dispatcher,
    )
}
