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

//! Structural visiting with the lexical constraint context.
//!
//! The shared Rust structural engine owns dispatch and recursion. This module
//! supplies its complete native-compatible base and sparse constraint-aware
//! default hooks; ordinary default traversal is reused when a hook is absent.
//! Native semantic hooks create no analyzer and do not snapshot mutable IR reads.
//!
//! Hook and global-function registration follows the ordinary visitor/binding
//! trust contract: registered hooks must honor the synchronous ABI, may not retain
//! or concurrently use the visitor, and registration must not race with traversal.

use std::borrow::BorrowMut;
use std::cell::UnsafeCell;

use tvm_ffi::extra::cold_init_lazy_lock::ColdInitLazyLock;
use tvm_ffi::extra::structural_visit::StructuralVisitorBase;
use tvm_ffi::reflection::TypeAttrColumn;
use tvm_ffi::tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};
use tvm_ffi::{
    AnyView, Dispatcher, IntoDispatcher, ObjectCore, Result, StructuralVisitEngine,
    StructuralVisitorObj, StructuralVisitorView, VisitInterrupt,
};

use super::StructuralVisitorWithConstraintContextObj;
use crate::sym::{ConstraintContext, ConstraintContextObj};

/// Complete Rust representation of the native constraint-aware visitor base.
///
/// This is distinct from the generated opaque
/// [`StructuralVisitorWithConstraintContextObj`]. The engine allocates this
/// complete representation and installs its own callback; no generated opaque
/// object is allocated. The base has no owning object conversion.
///
/// Native semantic hooks can replace the active context and update the private
/// scope flag through a shared [`StructuralVisitorView::base`] reference. Those
/// fields use interior mutability; context access clones the current handle.
#[repr(C)]
pub struct StructuralVisitorWithConstraintContextBase {
    base: StructuralVisitorObj,
    constraints: UnsafeCell<ConstraintContext>,
    needs_child_context: UnsafeCell<bool>,
}

impl StructuralVisitorWithConstraintContextBase {
    /// Read the current live context. This ordinary owning handle may
    /// outlive the visitor. Use [`ConstraintContextObj::snapshot`] to retain the
    /// current prefix or [`ConstraintContextObj::to_array`] for a detached sequence.
    pub fn constraints(&self) -> ConstraintContext {
        // No reference to the replaceable field survives this short clone.
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
// Stub generation cannot certify its private tail; the binding requires the
// matching native build. All fields begin with the same state as the native
// constructor. Native-mutated fields use UnsafeCell. The type identity is the
// registered specialized type, while the shared engine installs its vtable and
// owns the dispatcher. Default hooks receive that live runtime synchronously and
// return one owning None/interrupt/error token under the ordinary hook contract.
unsafe impl StructuralVisitorBase for StructuralVisitorWithConstraintContextBase {
    fn new() -> Result<Self> {
        Ok(Self {
            base: StructuralVisitorObj::default(),
            constraints: UnsafeCell::new(ConstraintContext::new()?),
            needs_child_context: UnsafeCell::new(false),
        })
    }

    fn type_index() -> i32 {
        StructuralVisitorWithConstraintContextObj::type_index()
    }

    #[inline(always)]
    unsafe fn default_visit_raw(runtime: *mut Self, value: AnyView<'_>) -> TVMFFIAny {
        static HOOKS: ColdInitLazyLock<Option<TypeAttrColumn>> =
            ColdInitLazyLock::new(|| TypeAttrColumn::new("__s_visit_with_constraint_context__"));
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

/// Scoped access to the shared Rust engine with a constraint-aware base.
///
/// Pass the handler's `self` to recursive operations so the engine publishes the
/// actual exclusive dispatcher reborrow. Inspect premises via `view.base()`.
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

/// Visit with native binding, range and guard semantics through the shared Rust
/// engine. A matched dispatcher handler owns descent. Unmatched values use the
/// base's specialized hook or ordinary structural fallback.
///
/// ```
/// use tvm_tirx::tirx::structural_visit_with_constraint_context::{
///     structural_visit_with_constraint_context as visit,
///     StructuralVisitorWithConstraintContextView,
/// };
/// use tvm_tirx::tvm_ffi::{dispatch, AnyView, Result, VisitInterrupt};
///
/// #[derive(Default)]
/// struct Count(usize);
/// #[dispatch(visit, (AnyView<'_>, &mut StructuralVisitorWithConstraintContextView<'_, Self>), Result<Option<VisitInterrupt>>)]
/// impl Count {
///     fn visit_number(&mut self, _value: &i64, visitor: &mut StructuralVisitorWithConstraintContextView<'_, Self>)
///         -> Result<Option<VisitInterrupt>> {
///         assert_eq!(visitor.base().constraints().local_size(), 0);
///         self.0 += 1;
///         Ok(None)
///     }
/// }
/// let mut count = Count::default();
/// visit(&7_i64, &mut count)?;
/// assert_eq!(count.0, 1);
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
///
/// Callbacks must return failures through `Result` and must not panic: a panic
/// reaching the non-unwinding native callback boundary aborts the process.
/// Hooks and reflected metadata retain the ordinary visitor's trust obligations.
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
