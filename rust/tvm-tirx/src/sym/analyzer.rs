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

//! The analyzer. An [`Analyzer`] proves, bounds and simplifies integer expressions under the facts
//! of a [`ConstraintContext`]; its bounds are [`ConstIntBound`]s and [`ModularSet`]s.

// tvm-ffi-stubgen(missing-docs): allow
// Python has no class for the `RewriteSimplifierStats` that `Analyzer.rewrite_simplify_stats()`
// returns; it is here, with the analyzer it belongs to.

// Names the hand-written wrappers use: the import section keeps them even when no generated block
// does.
// tvm-ffi-stubgen(import-object): tvm_ffi::Error
// tvm-ffi-stubgen(import-object): tvm_ffi::Map
// tvm-ffi-stubgen(import-object): tvm_ffi::Result
// tvm-ffi-stubgen(import-object): tvm_ffi::String
// tvm-ffi-stubgen(import-object): tvm_ffi::VALUE_ERROR
// tvm-ffi-stubgen(import-object): crate::ExitOnDrop
// tvm-ffi-stubgen(import-object): crate::ir::expr::Var
// tvm-ffi-stubgen(import-object): crate::sym::int_set::IntSet
// tvm-ffi-stubgen(import-object): crate::sym::constraint::ConstraintContext
// tvm-ffi-stubgen(import-object): crate::ir::support
// tvm-ffi-stubgen(import-object): crate::ir::expr::IntoExpr
// tvm-ffi-stubgen(import-object): crate::ir::expr::PrimExpr
// tvm-ffi-stubgen(import-object): std::borrow::Borrow
// tvm-ffi-stubgen(import-object): tvm_ffi::IntoArg

// tvm-ffi-stubgen(begin): import-section
use crate::ExitOnDrop;
use crate::ir::expr::IntoExpr;
use crate::ir::expr::PrimExpr;
use crate::ir::expr::Var;
use crate::ir::support;
use crate::sym::constraint::ConstraintContext;
use crate::sym::int_set::IntSet;
use std::borrow::Borrow;
use tvm_ffi::Error;
use tvm_ffi::IntoArg;
use tvm_ffi::Map;
use tvm_ffi::Object;
use tvm_ffi::ObjectArc;
use tvm_ffi::Result;
use tvm_ffi::String;
use tvm_ffi::VALUE_ERROR;
// tvm-ffi-stubgen(end)

// ---------------------------------------------------------------------------------------------
// The analyzer

/// An analyzer of integer expressions: it proves, bounds and simplifies them under the facts it
/// knows, which [`sync_constraint_context`](Analyzer::sync_constraint_context) sets from a
/// [`ConstraintContext`].
///
/// A clone shares the analyzer and its facts; [`deep_clone`](Analyzer::deep_clone) copies them.
/// When the build includes the Z3 prover, default-strength proofs fall back to it unless
/// [`set_z3_default_fallback`](Analyzer::set_z3_default_fallback) turns that off.
///
/// # Examples
///
/// ```
/// use tvm_tirx::ir::prim::op::floordiv;
/// use tvm_tirx::ir::{PrimType, PrimVar, Range, Var, UNKNOWN_LOC};
/// use tvm_tirx::script::script;
/// use tvm_tirx::sym::{Analyzer, ConstraintContext, RangeConstraint};
///
/// let x = PrimVar::new("x", PrimType::int(32), UNKNOWN_LOC);
/// let context = ConstraintContext::new();
/// let domain = Range::from_min_extent(0, 16, UNKNOWN_LOC)?;
/// context.append(RangeConstraint::new(x.clone(), domain))?;
/// let analyzer = Analyzer::new();
/// analyzer.sync_constraint_context(&context)?;
///
/// assert_eq!(script(&analyzer.simplify(floordiv(&x, 16)?, 2)?)?, "0");
///
/// assert_eq!(script(&Analyzer::new().simplify(floordiv(&x, 16)?, 2)?)?, "x // 16");
/// # Ok::<(), tvm_tirx::tvm_ffi::Error>(())
/// ```
// tvm-ffi-stubgen(begin): object/sym.Analyzer
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct Analyzer {
    data_: ObjectArc<AnalyzerObj>,
}

/// The object behind [`Analyzer`].
///
/// Opaque: bytes [24, 120) of [24, 120) are not accounted for by reflected fields. Fields are read through the C ABI getters.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.Analyzer", final)]
pub struct AnalyzerObj {
    base_: Object,
}
// tvm-ffi-stubgen(end)

impl Analyzer {
    /// Create an analyzer that knows no facts (`sym.Analyzer`).
    // Not `Default`: creating an analyzer allocates a stateful FFI object.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        support::infallible(
            tvm_ffi::cached_global_func!("sym.Analyzer")
                .call_tuple(())
                .and_then(|any| any.try_into()),
        )
    }

    /// Set whether default-strength queries try Z3 after native analysis fails
    /// (`sym.AnalyzerSetZ3DefaultFallback`).
    ///
    /// Enabled by default and preserved by [`deep_clone`](Self::deep_clone). Stronger queries and
    /// direct Z3 APIs are unaffected. This does not clear accepted premises or solver state.
    pub fn set_z3_default_fallback(&self, enabled: bool) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.AnalyzerSetZ3DefaultFallback")
            .call_tuple((self, enabled))?;
        Ok(())
    }

    /// Return a copy of the analyzer and the facts it knows, which learns facts independently
    /// (`sym.AnalyzerClone`).
    pub fn deep_clone(&self) -> Result<Self> {
        tvm_ffi::cached_global_func!("sym.AnalyzerClone").call_tuple((self,))?.try_into()
    }

    /// Synchronize the analyzer's facts with `context` at its current position
    /// (`sym.AnalyzerSyncConstraintContext`).
    ///
    /// Later appends to the context take effect only at the next synchronization.
    ///
    /// # Errors
    ///
    /// Returns an error for a reentrant synchronization, or if `context` defines a variable twice.
    pub fn sync_constraint_context(&self, context: impl IntoArg<ConstraintContext>) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.AnalyzerSyncConstraintContext").call_packed(&[
            tvm_ffi::AnyView::from(self),
            tvm_ffi::AnyView::from(context.into_arg().borrow()),
        ])?;
        Ok(())
    }

    /// Snapshot the accepted premises in an independently extensible context
    /// (`sym.AnalyzerGetConstraintContext`).
    ///
    /// Fact and expression nodes remain shared. Appending to this context changes the analyzer only
    /// after [`sync_constraint_context`](Self::sync_constraint_context) is called again.
    pub fn get_constraint_context(&self) -> Result<ConstraintContext> {
        tvm_ffi::cached_global_func!("sym.AnalyzerGetConstraintContext")
            .call_tuple((self,))?
            .try_into()
    }

    /// Return the constant bounds of `expr` under the known facts (`sym.AnalyzerConstIntBound`).
    pub fn const_int_bound(&self, expr: impl IntoExpr<PrimExpr>) -> Result<ConstIntBound> {
        tvm_ffi::cached_global_func!("sym.AnalyzerConstIntBound")
            .call_tuple((self, expr.into_expr()?.borrow()))?
            .try_into()
    }

    /// Return whether `var` has constant bounds recorded (`sym.AnalyzerConstIntBoundIsBound`).
    pub fn const_int_bound_is_bound(&self, var: impl IntoArg<Var>) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.AnalyzerConstIntBoundIsBound")
            .call_tuple((self, var.into_arg().borrow()))?
            .try_into()
    }

    /// Return the modular set `coeff * x + base` of `expr` (`sym.AnalyzerModularSet`).
    pub fn modular_set(&self, expr: impl IntoExpr<PrimExpr>) -> Result<ModularSet> {
        tvm_ffi::cached_global_func!("sym.AnalyzerModularSet")
            .call_tuple((self, expr.into_expr()?.borrow()))?
            .try_into()
    }

    /// Return the integer set `expr` covers, with the variables of `dom_map` over their sets
    /// (`sym.AnalyzerIntSet`).
    pub fn int_set(
        &self,
        expr: impl IntoExpr<PrimExpr>,
        dom_map: impl Into<Option<Map<Var, IntSet>>>,
    ) -> Result<IntSet> {
        tvm_ffi::cached_global_func!("sym.AnalyzerIntSet")
            .call_tuple((self, expr.into_expr()?.borrow(), dom_map.into()))?
            .try_into()
    }

    /// Simplify `expr`, alternating the rewrite and canonical simplifiers `steps` times
    /// (`sym.AnalyzerSimplify`).
    pub fn simplify(&self, expr: impl IntoExpr<PrimExpr>, steps: i32) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("sym.AnalyzerSimplify")
            .call_tuple((self, expr.into_expr()?.borrow(), steps))?
            .try_into()
    }

    /// Simplify `expr` by rewrite rules (`sym.AnalyzerRewriteSimplify`).
    pub fn rewrite_simplify(&self, expr: impl IntoExpr<PrimExpr>) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("sym.AnalyzerRewriteSimplify")
            .call_tuple((self, expr.into_expr()?.borrow()))?
            .try_into()
    }

    /// Simplify `expr` into canonical form (`sym.AnalyzerCanonicalSimplify`).
    pub fn canonical_simplify(&self, expr: impl IntoExpr<PrimExpr>) -> Result<PrimExpr> {
        tvm_ffi::cached_global_func!("sym.AnalyzerCanonicalSimplify")
            .call_tuple((self, expr.into_expr()?.borrow()))?
            .try_into()
    }

    /// Return the counters of the rewrite simplifier (`sym.AnalyzerGetRewriteSimplifyStats`).
    pub fn rewrite_simplify_stats(&self) -> Result<RewriteSimplifierStats> {
        tvm_ffi::cached_global_func!("sym.AnalyzerGetRewriteSimplifyStats")
            .call_tuple((self,))?
            .try_into()
    }

    /// Zero the counters of the rewrite simplifier (`sym.AnalyzerResetRewriteSimplifyStats`).
    pub fn reset_rewrite_simplify_stats(&self) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.AnalyzerResetRewriteSimplifyStats")
            .call_tuple((self,))?;
        Ok(())
    }

    /// Cap the rewrites of one rewrite simplification at `maximum`, or 0 for no cap
    /// (`sym.AnalyzerSetMaximumRewriteSteps`).
    pub fn set_maximum_rewrite_steps(&self, maximum: i64) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.AnalyzerSetMaximumRewriteSteps")
            .call_tuple((self, maximum))?;
        Ok(())
    }

    /// Return whether `expr` is provably true (`sym.AnalyzerCanProve`).
    pub fn can_prove(
        &self,
        expr: impl IntoExpr<PrimExpr>,
        strength: ProofStrength,
    ) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.AnalyzerCanProve")
            .call_tuple((self, expr.into_expr()?.borrow(), strength as i32))?
            .try_into()
    }

    /// Return whether `lhs == rhs` is provably true (`sym.AnalyzerCanProveEqual`).
    pub fn can_prove_equal(
        &self,
        lhs: impl IntoExpr<PrimExpr>,
        rhs: impl IntoExpr<PrimExpr>,
    ) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.AnalyzerCanProveEqual")
            .call_tuple((self, lhs.into_expr()?.borrow(), rhs.into_expr()?.borrow()))?
            .try_into()
    }

    /// Return the relation of `lhs` to `rhs` that the known comparisons imply
    /// (`sym.AnalyzerTryCompare`).
    pub fn try_compare(
        &self,
        lhs: impl IntoExpr<PrimExpr>,
        rhs: impl IntoExpr<PrimExpr>,
    ) -> Result<CompareResult> {
        let result: i64 = tvm_ffi::cached_global_func!("sym.AnalyzerTryCompare")
            .call_tuple((self, lhs.into_expr()?.borrow(), rhs.into_expr()?.borrow()))?
            .try_into()?;
        result.try_into()
    }

    /// Return whether TIRx was built with the Z3 prover (`sym.AnalyzerIsZ3Enabled`).
    pub fn is_z3_enabled(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("sym.AnalyzerIsZ3Enabled").call_tuple((self,))?.try_into()
    }

    /// Return the facts the Z3 prover knows, and `expr` to prove if given, as an SMT-LIB2 script
    /// (`sym.AnalyzerGetSMTLIB2`).
    pub fn smtlib2(&self, expr: Option<&PrimExpr>) -> Result<String> {
        tvm_ffi::cached_global_func!("sym.AnalyzerGetSMTLIB2")
            .call_tuple((self, expr.cloned()))?
            .try_into()
    }

    /// Bound each Z3 proof by `timeout_ms` milliseconds (`sym.AnalyzerSetZ3TimeoutMs`).
    pub fn set_z3_timeout_ms(&self, timeout_ms: u32) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.AnalyzerSetZ3TimeoutMs")
            .call_tuple((self, timeout_ms))?;
        Ok(())
    }

    /// Bound each Z3 proof by the resource limit `rlimit` (`sym.AnalyzerSetZ3RLimit`).
    pub fn set_z3_rlimit(&self, rlimit: u32) -> Result<()> {
        tvm_ffi::cached_global_func!("sym.AnalyzerSetZ3RLimit").call_tuple((self, rlimit))?;
        Ok(())
    }

    /// Return the statistics of the Z3 prover (`sym.AnalyzerGetZ3Stats`).
    pub fn z3_stats(&self) -> Result<String> {
        tvm_ffi::cached_global_func!("sym.AnalyzerGetZ3Stats").call_tuple((self,))?.try_into()
    }
}

/// How hard [`Analyzer::can_prove`] tries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ProofStrength {
    /// The default, inexpensive analyses.
    #[default]
    Default = 0,
    /// Also prove through symbolic bounds, which is more expensive.
    SymbolicBound = 1,
}

/// The result of [`Analyzer::try_compare`].
///
/// The relations other than `Inconsistent` and `Unknown` say what is known of `lhs` against `rhs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum CompareResult {
    /// The known facts contradict each other.
    Inconsistent = 0,
    /// `lhs == rhs`.
    Eq = 1,
    /// `lhs < rhs`.
    Lt = 2,
    /// `lhs <= rhs`.
    Le = 3,
    /// `lhs > rhs`.
    Gt = 4,
    /// `lhs >= rhs`.
    Ge = 5,
    /// `lhs != rhs`.
    Ne = 6,
    /// Nothing is known of the relation.
    Unknown = 7,
}

impl TryFrom<i64> for CompareResult {
    type Error = Error;
    fn try_from(value: i64) -> Result<Self> {
        Ok(match value {
            0 => Self::Inconsistent,
            1 => Self::Eq,
            2 => Self::Lt,
            3 => Self::Le,
            4 => Self::Gt,
            5 => Self::Ge,
            6 => Self::Ne,
            7 => Self::Unknown,
            _ => {
                let message = format!("{value} is not a sym::CompareResult");
                return Err(Error::new(VALUE_ERROR, &message));
            }
        })
    }
}

/// A scope in which unsigned scalars are index types, returned by [`allow_uint_as_index`].
///
/// The assumption is a depth counter of the thread, so a scope stays on the thread that opened it.
///
/// ```compile_fail
/// fn send<T: Send>() {}
/// send::<tvm_tirx::sym::AllowUintAsIndexScope>();
/// ```
#[must_use = "the assumption ends when the scope is dropped"]
pub struct AllowUintAsIndexScope {
    _exit: ExitOnDrop,
}

/// Treat `uint32` and `uint64` scalars as non-wrapping index types until the returned scope is
/// dropped (`sym.AnalyzerEnterAllowUintAsIndex`).
///
/// For compile-time layout proofs, not for code whose unsigned arithmetic may wrap around.
pub fn allow_uint_as_index() -> Result<AllowUintAsIndexScope> {
    let exit: tvm_ffi::Function = tvm_ffi::cached_global_func!("sym.AnalyzerEnterAllowUintAsIndex")
        .call_tuple(())?
        .try_into()?;
    Ok(AllowUintAsIndexScope { _exit: ExitOnDrop::new(exit) })
}

// ---------------------------------------------------------------------------------------------
// Results of the analyses

/// The inclusive constant bounds `[min_value, max_value]` of an integer expression, where
/// `i64::MAX` and `-i64::MAX` stand for unbounded ends.
// tvm-ffi-stubgen(begin): object/sym.ConstIntBound
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ConstIntBound {
    data_: ObjectArc<ConstIntBoundObj>,
}

/// The object behind [`ConstIntBound`].
///
/// Complete: reflected fields fill [24, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.ConstIntBound", final, immutable)]
pub struct ConstIntBoundObj {
    base_: Object,
    pub min_value: i64,
    pub max_value: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<ConstIntBoundObj>() == 40);
    assert!(::core::mem::align_of::<ConstIntBoundObj>() == 8);
};

#[allow(missing_docs)]
impl ConstIntBoundObj {
    pub(crate) fn new(min_value: i64, max_value: i64) -> Self {
        let base_ = Object::new();
        Self { base_, min_value, max_value }
    }
}

#[allow(missing_docs)]
impl ConstIntBound {
    /// Lossless complete-field allocation.
    pub fn new(min_value: i64, max_value: i64) -> Self {
        let obj = ConstIntBoundObj::new(min_value, max_value);
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

/// The congruence class `{coeff * x + base}` over the integers `x` that an integer expression lies
/// in: a `coeff` of 0 is the single value `base`, and a `coeff` of 1 any integer.
// tvm-ffi-stubgen(begin): object/sym.ModularSet
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct ModularSet {
    data_: ObjectArc<ModularSetObj>,
}

/// The object behind [`ModularSet`].
///
/// Complete: reflected fields fill [24, 40) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.ModularSet", final, immutable)]
pub struct ModularSetObj {
    base_: Object,
    pub coeff: i64,
    pub base: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<ModularSetObj>() == 40);
    assert!(::core::mem::align_of::<ModularSetObj>() == 8);
};

#[allow(missing_docs)]
impl ModularSetObj {
    pub(crate) fn new(coeff: i64, base: i64) -> Self {
        let base_ = Object::new();
        Self { base_, coeff, base }
    }
}

#[allow(missing_docs)]
impl ModularSet {
    /// Lossless complete-field allocation.
    pub fn new(coeff: i64, base: i64) -> Self {
        let obj = ModularSetObj::new(coeff, base);
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)

/// The counters of an analyzer's rewrite simplifier: the nodes it visited, the constraints it
/// entered, the rewrites it attempted and performed, its deepest recursion and its recursive
/// rewrites.
// tvm-ffi-stubgen(begin): object/sym.RewriteSimplifierStats
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::ObjectRef, Clone)]
pub struct RewriteSimplifierStats {
    data_: ObjectArc<RewriteSimplifierStatsObj>,
}

/// The object behind [`RewriteSimplifierStats`].
///
/// Complete: reflected fields fill [24, 72) exactly.
#[allow(missing_docs)]
#[repr(C)]
#[derive(tvm_ffi::derive::Object)]
#[object(type_key = "sym.RewriteSimplifierStats", final, immutable)]
pub struct RewriteSimplifierStatsObj {
    base_: Object,
    pub nodes_visited: i64,
    pub constraints_entered: i64,
    pub rewrites_attempted: i64,
    pub rewrites_performed: i64,
    pub max_recursive_depth: i64,
    pub num_recursive_rewrites: i64,
}

const _: () = {
    assert!(::core::mem::size_of::<RewriteSimplifierStatsObj>() == 72);
    assert!(::core::mem::align_of::<RewriteSimplifierStatsObj>() == 8);
};

#[allow(missing_docs)]
impl RewriteSimplifierStatsObj {
    pub(crate) fn new(
        nodes_visited: i64,
        constraints_entered: i64,
        rewrites_attempted: i64,
        rewrites_performed: i64,
        max_recursive_depth: i64,
        num_recursive_rewrites: i64,
    ) -> Self {
        let base_ = Object::new();
        Self {
            base_,
            nodes_visited,
            constraints_entered,
            rewrites_attempted,
            rewrites_performed,
            max_recursive_depth,
            num_recursive_rewrites,
        }
    }
}

#[allow(missing_docs)]
impl RewriteSimplifierStats {
    /// Lossless complete-field allocation.
    pub fn new(
        nodes_visited: i64,
        constraints_entered: i64,
        rewrites_attempted: i64,
        rewrites_performed: i64,
        max_recursive_depth: i64,
        num_recursive_rewrites: i64,
    ) -> Self {
        let obj = RewriteSimplifierStatsObj::new(
            nodes_visited,
            constraints_entered,
            rewrites_attempted,
            rewrites_performed,
            max_recursive_depth,
            num_recursive_rewrites,
        );
        Self { data_: ObjectArc::new(obj) }
    }
}
// tvm-ffi-stubgen(end)
