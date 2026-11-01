# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Analyze and simplify symbolic expressions using bounds, modular sets, and constraints."""

import enum
from contextlib import contextmanager

import tvm_ffi

from tvm_tirx import ir
from tvm_tirx.sym import IntSet

from . import _ffi_api
from .constraint import ConstraintContext


class ProofStrength(enum.IntEnum):
    """Proof strength of the analysis"""

    DEFAULT = 0
    SYMBOLIC_BOUND = 1


class CompareResult(enum.IntEnum):
    """Result of a transitive comparison.

    Values must match the C++ ``sym::CompareResult`` enum.
    """

    INCONSISTENT = 0
    EQ = 1
    LT = 2
    LE = 3
    GT = 4
    GE = 5
    NE = 6
    UNKNOWN = 7


@tvm_ffi.register_object("sym.ModularSet")
class ModularSet(tvm_ffi.Object):
    """Represent range of (coeff * x + base) for x in Z"""

    def __init__(self, coeff, base):
        self.__init_handle_by_constructor__(_ffi_api.ModularSet, coeff, base)


@tvm_ffi.register_object("sym.ConstIntBound")
class ConstIntBound(tvm_ffi.Object):
    """Represent constant integer bound

    Parameters
    ----------
    min_value : int
        The minimum value of the bound.

    max_value : int
        The maximum value of the bound.
    """

    POS_INF = (1 << 63) - 1
    NEG_INF = -POS_INF

    def __init__(self, min_value, max_value):
        self.__init_handle_by_constructor__(_ffi_api.ConstIntBound, min_value, max_value)


@tvm_ffi.register_object("sym.Analyzer")
class Analyzer(tvm_ffi.Object):
    """Integer arithmetic analyzer

    This is a stateful analyzer class that can be used to perform
    various symbolic integer analysis. The same analyzer instance can
    be passed to FFI APIs to share accumulated facts across calls.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Analyzer)

    def set_z3_default_fallback(self, enabled: bool) -> None:
        """Set whether DEFAULT-strength queries try Z3 after native analysis fails.

        Enabled by default. This setting is local to this analyzer and preserved
        by clone(). Stronger proof queries and Z3-specific APIs are unaffected.
        Changing it does not clear accepted premises or existing solver state.
        Z3 must also be available in the native build.

        Parameters
        ----------
        enabled : bool
            Whether to allow the DEFAULT-strength Z3 fallback.
        """
        _ffi_api.AnalyzerSetZ3DefaultFallback(self, enabled)

    @property
    def is_z3_enabled(self) -> bool:
        """Whether this build includes the Z3 backend (``USE_Z3=ON``).

        The Z3-specific methods (:py:meth:`get_smtlib2`, :py:meth:`get_z3_stats`,
        :py:meth:`set_z3_timeout_ms`, :py:meth:`set_z3_rlimit`) only work when
        this is ``True``.
        """
        return bool(_ffi_api.AnalyzerIsZ3Enabled(self))

    def _check_z3_enabled(self) -> None:
        if not self.is_z3_enabled:
            raise RuntimeError(
                "The Z3 backend is not available in this build. "
                "Rebuild TVM with USE_Z3=ON to use Z3-specific Analyzer APIs."
            )

    def get_smtlib2(self, expr: ir.Expr | None = None) -> str:
        """Get the current Z3 problem in SMT-LIB2 format.

        Raises
        ------
        RuntimeError
            If TVM was built without Z3 (``USE_Z3=OFF``), since there is no
            solver state to export. Use :py:attr:`is_z3_enabled` to check first.

        Parameters
        ----------
        expr : Optional[Expr]
            The expression to prove. If provided, its negation is added to the problem.
        """
        self._check_z3_enabled()
        return _ffi_api.AnalyzerGetSMTLIB2(self, expr)

    def set_z3_timeout_ms(self, timeout_ms: int) -> None:
        """Set Z3 timeout in milliseconds.

        Raises
        ------
        RuntimeError
            If TVM was built without Z3 (``USE_Z3=OFF``).

        Parameters
        ----------
        timeout_ms : int
            The timeout in milliseconds.
        """
        self._check_z3_enabled()
        _ffi_api.AnalyzerSetZ3TimeoutMs(self, timeout_ms)

    def set_z3_rlimit(self, rlimit: int) -> None:
        """Set Z3 resource limit.

        The resource limit gives deterministic solver budgeting (unlike a wall
        clock timeout). A value of ``0`` disables the limit.

        Raises
        ------
        RuntimeError
            If TVM was built without Z3 (``USE_Z3=OFF``).

        Parameters
        ----------
        rlimit : int
            The resource limit.
        """
        self._check_z3_enabled()
        _ffi_api.AnalyzerSetZ3RLimit(self, rlimit)

    def get_z3_stats(self) -> str:
        """Get Z3 solver statistics.

        Raises
        ------
        RuntimeError
            If TVM was built without Z3 (``USE_Z3=OFF``).

        Returns
        -------
        stats : str
            The Z3 statistics.
        """
        self._check_z3_enabled()
        return _ffi_api.AnalyzerGetZ3Stats(self)

    def const_int_bound(self, expr: ir.Expr) -> ConstIntBound:
        """Find constant integer bound for expr.

        Parameters
        ----------
        expr : Expr
            The expression.

        Returns
        -------
        bound : ConstIntBound
            The result bound
        """
        return _ffi_api.AnalyzerConstIntBound(self, expr)

    def const_int_bound_is_bound(self, var: ir.Var) -> bool:
        """Check if a variable is bound to a range.

        Parameters
        ----------
        var : tvm_tirx.ir.Var
            The variable.

        Returns
        -------
        result : bool
            Whether the variable is bound to a range.
        """
        return _ffi_api.AnalyzerConstIntBoundIsBound(self, var)

    def modular_set(self, expr: ir.Expr) -> ModularSet:
        """Find a modular set that expr belongs to.

        Parameters
        ----------
        expr : Expr
            The expression.

        Returns
        -------
        result : ModularSet
            The result.
        """
        return _ffi_api.AnalyzerModularSet(self, expr)

    def simplify(self, expr: ir.Expr, steps: int = 2) -> ir.Expr:
        """Simplify expression via both rewrite and canonicalization.

        Parameters
        ----------
        expr : Expr
            The expression.
        steps : The simplification runs in the order of
                rewrite_simplify (step 1) -> canonical_simplify (step 2) ->
                rewrite_simplify (step 3) -> canonical_simplify (step 4) -> ...
                param steps controls how many steps to run.
                Default is 2, i.e., rewrite_simplify + canonical_simplify.

        Returns
        -------
        result : Expr
            The result.
        """
        return _ffi_api.AnalyzerSimplify(self, expr, steps)

    def clone(self) -> "Analyzer":
        """Return a deep copy of this analyzer with independent state.

        The returned analyzer carries the same accumulated facts (variable
        bounds, modular sets, bindings, integer-set domains, literal
        constraints and transitive comparisons) as this one, but owns its own
        state: synchronizing or simplifying either analyzer afterwards does not
        affect the other. Unlike copying the handle, this is a true deep copy.

        Returns
        -------
        result : Analyzer
            A new analyzer holding an independent copy of the facts.
        """
        return _ffi_api.AnalyzerClone(self)

    def rewrite_simplify(self, expr: ir.Expr) -> ir.Expr:
        """Simplify expression via rewriting rules.

        Parameters
        ----------
        expr : Expr
            The expression.

        Returns
        -------
        result : Expr
            The result.
        """
        return _ffi_api.AnalyzerRewriteSimplify(self, expr)

    @property
    def rewrite_simplify_stats(self):
        return _ffi_api.AnalyzerGetRewriteSimplifyStats(self)

    def reset_rewrite_simplify_stats(self):
        _ffi_api.AnalyzerResetRewriteSimplifyStats(self)

    def canonical_simplify(self, expr: ir.Expr) -> ir.Expr:
        """Simplify expression via canonicalization.

        Parameters
        ----------
        expr : Expr
            The expression.

        Returns
        -------
        result : Expr
            The result.
        """
        return _ffi_api.AnalyzerCanonicalSimplify(self, expr)

    def int_set(self, expr: ir.Expr, dom_map: dict[ir.Var, IntSet] | None = None) -> IntSet:
        """Compute a symbolic IntSet that covers expr for all values in dom_map.

        Parameters
        ----------
        expr : Expr
            The expression.

        dom_map : Optional[Dict[tvm_tirx.ir.Var, tvm_tirx.sym.IntSet]]
            The domain for variables to be relaxed.  When omitted, the analyzer
            uses the domains of the variables already bound to it.

        Returns
        -------
        result : IntSet
            The result.
        """
        return _ffi_api.AnalyzerIntSet(self, expr, dom_map)

    def can_prove(self, expr: ir.Expr, strength: ProofStrength = ProofStrength.DEFAULT) -> bool:
        """Check whether we can prove expr to be true.

        Parameters
        ----------
        expr : Expr
            The expression.

        strength: ProofStrength
            The proof strength. ``SYMBOLIC_BOUND`` additionally enables symbolic
            bound analysis. When built with Z3 (``TVM_TIRX_USE_Z3=ON``), all
            strengths, including ``DEFAULT``, try Z3 after native analysis fails
            to prove the predicate. :py:meth:`set_z3_default_fallback` can disable
            this fallback for ``DEFAULT`` without affecting stronger queries.
            The existing Z3 timeout and resource limits apply; an unknown solver
            result means the predicate cannot be proved.

        Returns
        -------
        result : bool
            Whether the predicate can be proved.
        """
        return _ffi_api.AnalyzerCanProve(self, expr, strength)

    def set_maximum_rewrite_steps(self, maximum: int) -> None:
        """Set the maximum allowed number of rewrite-simplify steps.

        When a positive limit is set, the simplifier raises an exception once
        it exceeds that number of rewrite steps.  This is useful for guarding
        against performance regressions in tests.

        Parameters
        ----------
        maximum : int
            The maximum number of rewrite steps, or a non-positive value to
            allow an unlimited number of steps.
        """
        _ffi_api.AnalyzerSetMaximumRewriteSteps(self, maximum)

    def sync_constraint_context(self, context: "ConstraintContext") -> None:
        """Synchronize this analyzer with the premises visible in ``context``.

        The analyzer captures the current context position. Later appends become
        visible only after another synchronization. Switching to a parent,
        sibling, or empty context removes facts no longer present there.
        Repeating this call for an unchanged context is inexpensive.

        Parameters
        ----------
        context : ConstraintContext
            Binding, range, and predicate premises for subsequent queries.
            Captured premises and their referenced expressions must not be
            modified through aliases.
        """
        _ffi_api.AnalyzerSyncConstraintContext(self, context)

    def can_prove_equal(self, lhs: ir.Expr, rhs: ir.Expr) -> bool:
        """Whether we can prove that lhs == rhs

        Parameters
        ----------
        lhs: Expr
            The left-hand side of the comparison

        rhs: Expr
            The right-hand side of the comparison

        Returns
        -------
        result: bool
            Whether we can prove that lhs == rhs
        """
        return _ffi_api.AnalyzerCanProveEqual(self, lhs, rhs)

    def try_compare(self, lhs: ir.Expr, rhs: ir.Expr) -> CompareResult:
        """Compare lhs and rhs using directly known comparisons and constant offsets.

        Parameters
        ----------
        lhs : Expr
            The left-hand side of the comparison.

        rhs : Expr
            The right-hand side of the comparison.

        Returns
        -------
        result : CompareResult
            The most specific result that can be proven about the comparison.
            Returns ``CompareResult.UNKNOWN`` when nothing can be proven.
        """
        return CompareResult(_ffi_api.AnalyzerTryCompare(self, lhs, rhs))


@contextmanager
def allow_uint_as_index():
    """Opt-in no-overflow domain for unsigned index arithmetic.

    Within the scope, the analyzer treats uint32/uint64 scalars as index
    types: it assumes their values never approach the type limit (no
    wraparound), so the signed rewrite rule set (floordiv/floormod
    decomposition, modular analysis) applies to them. Every unsigned
    expression admitted under the assumption is logged once per process
    (WARNING) so the assumption stays auditable.

    Off by default. Intended for compile-time layout proofs (offsets that
    provably stay small), NOT for runtime codegen semantics where unsigned
    wraparound is real behavior.
    """
    exit_scope = _ffi_api.AnalyzerEnterAllowUintAsIndex()
    try:
        yield
    finally:
        exit_scope()
