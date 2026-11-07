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

//! Constraint-aware structural visiting. [`structural_visit_with_constraint_context`] visits a
//! value with a dispatcher whose handlers read the facts in scope from the
//! [`StructuralVisitorWithConstraintContextView`] they are given.

use std::borrow::BorrowMut;
use std::cell::UnsafeCell;

use tvm_ffi::extra::cold_init_lazy_lock::ColdInitLazyLock;
use tvm_ffi::extra::structural_visit::{
    StructuralVisitEngine, StructuralVisitorBase, StructuralVisitorObj, StructuralVisitorView,
};
use tvm_ffi::reflection::TypeAttrColumn;
use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::{AnyView, Dispatcher, IntoDispatcher, ObjectCore, Result, VisitInterrupt};

use super::StructuralVisitorWithConstraintContextObj;
use crate::sym::ConstraintContext;

/// The base of a constraint-aware visitor, holding the constraint context of the current value.
///
/// Unlike the opaque [`StructuralVisitorWithConstraintContextObj`], its layout mirrors the native
/// one, so it requires the matching native build.
#[repr(C)]
pub struct StructuralVisitorWithConstraintContextBase {
    base_: StructuralVisitorObj,
    constraints: UnsafeCell<ConstraintContext>,
    needs_child_context: UnsafeCell<bool>,
}

impl StructuralVisitorWithConstraintContextBase {
    /// Return the constraint context in effect at the current value; use
    /// [`ConstraintContextObj::snapshot`](crate::sym::ConstraintContextObj::snapshot) to keep its
    /// current prefix.
    pub fn constraints(&self) -> ConstraintContext {
        // SAFETY: the borrow ends with this clone, before anything can replace the field.
        unsafe { (&*self.constraints.get()).clone() }
    }

    /// Borrow only while an observer has no access to this visitor's descent.
    /// The caller must end the borrow before any operation can replace the context.
    #[inline]
    pub(super) unsafe fn constraints_for_observer(&self) -> &ConstraintContext {
        &*self.constraints.get()
    }
}

// SAFETY: repr(C) mirrors the complete base in the matching native header.
// Stub generation cannot certify its private tail; the definition requires the
// matching native build. All fields begin with the same state as the native
// constructor. Native-mutated fields use UnsafeCell. The type identity is the
// registered specialized type, while the shared engine installs its vtable and
// owns the dispatcher. Default hooks receive that live runtime synchronously and
// return one owning None/interrupt/error token under the ordinary hook contract.
unsafe impl StructuralVisitorBase for StructuralVisitorWithConstraintContextBase {
    fn new() -> Result<Self> {
        Ok(Self {
            base_: StructuralVisitorObj::default(),
            constraints: UnsafeCell::new(ConstraintContext::new()),
            needs_child_context: UnsafeCell::new(false),
        })
    }

    fn type_index() -> i32 {
        StructuralVisitorWithConstraintContextObj::type_index()
    }

    #[inline(always)]
    unsafe fn default_visit_raw(runtime: *mut Self, value: AnyView<'_>) -> TVMFFIAny {
        static HOOKS: ColdInitLazyLock<Option<TypeAttrColumn>> =
            ColdInitLazyLock::new(|| TypeAttrColumn::lookup("__s_visit_with_constraint_context__"));
        // Copy the borrowed token before reentry. Packed dispatch acquires an
        // owning Function before invoking a callback that could register hooks.
        let hook = HOOKS
            .and_then(|column| column.get_raw_ffi_any(value.type_index()))
            .filter(|hook| hook.type_index != TVMFFITypeIndex::kTVMFFINone as i32);
        if let Some(hook) = hook {
            StructuralVisitorObj::call_structural_visit_hook_raw(
                runtime.cast(),
                *value.as_raw_ffi_any(),
                hook,
            )
        } else {
            // Prefix-only default descent calls the installed vtable for each
            // child; it must not project an ordinary engine's dispatcher offset.
            StructuralVisitorObj::default_visit_raw(runtime.cast(), value)
        }
    }
}

/// The running constraint-aware visit, passed to each handler; `view.base().constraints()` gives
/// the facts in scope.
///
/// ```compile_fail
/// use tvm_tirx::tirx::structural_visit_with_constraint_context::StructuralVisitorWithConstraintContextView;
/// fn escape<'a, D>(v: StructuralVisitorWithConstraintContextView<'a, D>)
///     -> StructuralVisitorWithConstraintContextView<'static, D> { v }
/// ```
///
/// ```compile_fail
/// use tvm_tirx::tirx::structural_visit_with_constraint_context::StructuralVisitorWithConstraintContextView;
/// use tvm_tirx::tvm_ffi::Any;
/// fn retain<D>(v: &StructuralVisitorWithConstraintContextView<'_, D>) -> Any {
///     v.base().into()
/// }
/// ```
pub type StructuralVisitorWithConstraintContextView<'a, D> =
    StructuralVisitorView<'a, D, StructuralVisitorWithConstraintContextBase>;

/// Visit `root` with `visitor`, which sees the facts of the bindings, ranges and guards in scope.
///
/// A matched handler owns descent; other values are visited by default. Handlers return failures
/// through `Result` and must not panic: a panic at the native callback boundary aborts the process.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::PrimOps;
/// use tvm_tirx::ir::{
///     Evaluate, EvaluateObj, For, ForKind, PrimExpr, PrimType, PrimVar, UNKNOWN_LOC,
/// };
/// use tvm_tirx::sym::{Analyzer, ProofStrength};
/// use tvm_tirx::tirx::structural_visit_with_constraint_context::{
///     structural_visit_with_constraint_context as visit,
///     StructuralVisitorWithConstraintContextView as View,
/// };
/// use tvm_tirx::tvm_ffi::{dispatch, AnyView, Result, VisitInterrupt};
///
/// #[derive(Default)]
/// struct BelowSixteen(Vec<bool>);
///
/// #[dispatch(visit, (AnyView<'_>, &mut View<'_, Self>), Result<Option<VisitInterrupt>>)]
/// impl BelowSixteen {
///     fn visit_evaluate(
///         &mut self,
///         node: &EvaluateObj,
///         visitor: &mut View<'_, Self>,
///     ) -> Result<Option<VisitInterrupt>> {
///         let analyzer = Analyzer::new();
///         analyzer.sync_constraint_context(&visitor.base().constraints())?;
///         let value = PrimExpr::try_from(&node.value)?;
///         self.0.push(analyzer.can_prove(value.less(16), ProofStrength::Default)?);
///         visitor.default_visit(self, node)
///     }
/// }
///
/// let i = PrimVar::new("i", PrimType::int(32), UNKNOWN_LOC);
/// let body = Evaluate::new(&i, UNKNOWN_LOC);
/// let root = For::new(&i, 0, 16, 1, ForKind::Default, body, None, UNKNOWN_LOC)?;
/// let mut proofs = BelowSixteen::default();
/// visit(&root, &mut proofs)?;
/// assert_eq!(proofs.0, [true]);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
pub fn structural_visit_with_constraint_context<'input, R, H, M, D, O>(
    root: R,
    visitor: H,
) -> Result<Option<VisitInterrupt>>
where
    R: Into<AnyView<'input>>,
    O: BorrowMut<D>,
    for<'v, 'c, 'r> H: IntoDispatcher<
        (AnyView<'v>, &'c mut StructuralVisitorWithConstraintContextView<'r, D>),
        Option<Result<Option<VisitInterrupt>>>,
        M,
        Target = D,
        Output = O,
    >,
    for<'s, 'v, 'c, 'r> &'s mut D: Dispatcher<
        (AnyView<'v>, &'c mut StructuralVisitorWithConstraintContextView<'r, D>),
        Option<Result<Option<VisitInterrupt>>>,
    >,
{
    StructuralVisitEngine::<StructuralVisitorWithConstraintContextBase>::visit_root(root, visitor)
}
