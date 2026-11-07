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

//! Constraint-aware structural mutation: rewrite values with the facts in scope.

use std::borrow::BorrowMut;
use std::cell::UnsafeCell;

use tvm_ffi::extra::cold_init_lazy_lock::ColdInitLazyLock;
use tvm_ffi::extra::structural_mutate::{StructuralMutateEngine, StructuralMutatorBase};
use tvm_ffi::reflection::TypeAttrColumn;
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::{
    Any, Dispatcher, InplaceMode, IntoDispatcher, MutationView, ObjectArc, ObjectCore,
    ObjectRefCore, Result, StructuralMutatorObj, StructuralMutatorView, UnchangedOr,
};

use super::StructuralMutatorWithConstraintContextObj;
use crate::sym::ConstraintContext;

/// The base of a constraint-aware mutator, holding the constraint context of the current value.
///
/// Its layout mirrors the native one, so it requires the matching native build.
#[repr(C)]
pub struct StructuralMutatorWithConstraintContextBase {
    base_: StructuralMutatorObj,
    constraints: UnsafeCell<ConstraintContext>,
    needs_child_context: UnsafeCell<bool>,
}

impl StructuralMutatorWithConstraintContextBase {
    /// Return the constraint context in effect at the current value; `snapshot()` it to keep its
    /// current prefix.
    #[inline]
    pub fn constraints(&self) -> ConstraintContext {
        // SAFETY: the borrow ends with this clone, before anything can replace the field.
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
            base_: StructuralMutatorObj::default(),
            constraints: UnsafeCell::new(ConstraintContext::new()),
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
                // alias. Borrow in place so the count is the slot's own reference.
                borrowed
                    .downcast_ref::<ObjectRef>()
                    .is_some_and(|owner| ObjectArc::strong_count(ObjectRef::data(owner)) == 1)
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

/// The running constraint-aware mutation, passed to each handler; `view.base().constraints()` gives
/// the facts in scope.
pub type StructuralMutatorWithConstraintContextView<'a, D> =
    StructuralMutatorView<'a, D, StructuralMutatorWithConstraintContextBase>;

/// Mutate `root` with `dispatcher`, which sees the facts of the bindings, guards and ranges in
/// scope.
///
/// The first matching handler owns descent: `view.default_mutate(self, value)` rewrites the
/// children, `view.mutate(self, child)` one child. Handlers must not append to the context or
/// panic; an error stops the traversal.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::{
///     Bind, Evaluate, Expr, PrimExpr, PrimType, PrimVar, SeqStmt, Stmt, Var, UNKNOWN_LOC,
/// };
/// use tvm_tirx::script::script;
/// use tvm_tirx::sym::Analyzer;
/// use tvm_tirx::tirx::structural_mutate_with_constraint_context::{
///     structural_mutate_with_constraint_context as mutate,
///     StructuralMutatorWithConstraintContextView as View,
/// };
/// use tvm_tirx::tvm_ffi::{dispatch, Any, MutationView, Result, UnchangedOr};
///
/// struct Simplify;
///
/// #[dispatch(mutate, (MutationView<'_>, &mut View<'_, Self>), Result<UnchangedOr<Any>>)]
/// impl Simplify {
///     fn mutate_evaluate(
///         &mut self,
///         node: MutationView<'_, Evaluate>,
///         mutator: &mut View<'_, Self>,
///     ) -> Result<UnchangedOr<Any>> {
///         let analyzer = Analyzer::new();
///         analyzer.sync_constraint_context(&mutator.base().constraints())?;
///         let value = PrimExpr::try_from(&node.value)?;
///         Ok(Evaluate::new(analyzer.simplify(value, 2)?, UNKNOWN_LOC).into())
///     }
/// }
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let bind = Stmt::from(Bind::new(&x, 4, UNKNOWN_LOC)?);
/// let use_x = Evaluate::new(Expr::try_from(&x * 2)?, UNKNOWN_LOC);
/// let root = SeqStmt::new([bind, use_x.into()], UNKNOWN_LOC);
/// let result = SeqStmt::try_from(mutate(root, Simplify)?)?;
/// assert_eq!(script(&result)?, "x: T.let[T.int32] = 4\nT.evaluate(8)");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn structural_mutate_with_constraint_context<R, H, M, D, O>(
    root: R,
    dispatcher: H,
) -> Result<Any>
where
    R: Into<Any>,
    O: BorrowMut<D>,
    for<'v, 'c, 'r> H: IntoDispatcher<
        (MutationView<'v>, &'c mut StructuralMutatorWithConstraintContextView<'r, D>),
        Option<Result<UnchangedOr<Any>>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
        (MutationView<'v>, &'c mut StructuralMutatorWithConstraintContextView<'r, D>),
        Option<Result<UnchangedOr<Any>>>,
    >,
{
    StructuralMutateEngine::<StructuralMutatorWithConstraintContextBase>::mutate_root(
        root, dispatcher,
    )
}
