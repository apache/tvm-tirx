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

import gc
import queue
import threading

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import tirx
from tvm_tirx.sym import Analyzer, ProofStrength

# The Z3 prover is only consulted at the kSymbolicBound strength so the common
# default path never pays the prover cost.
SB = ProofStrength.SYMBOLIC_BOUND


def _require_z3(analyzer):
    if not analyzer.is_z3_enabled:
        pytest.skip("Z3 prover is disabled in this build")


def implies(x, y):
    return tirx.Or(tirx.Not(x), y)


# ---------------------------------------------------------------------------
# API availability (works regardless of whether Z3 is built)
# ---------------------------------------------------------------------------


def test_z3_capability_query():
    # `is_z3_enabled` is the supported way to detect the build configuration.
    # The Z3-specific debug/config methods work only when it is True, and raise
    # a clear error otherwise.
    analyzer = Analyzer()
    assert isinstance(analyzer.is_z3_enabled, bool)

    if analyzer.is_z3_enabled:
        assert isinstance(analyzer.get_smtlib2(), str)
        assert isinstance(analyzer.get_z3_stats(), str)
    else:
        with pytest.raises(RuntimeError):
            analyzer.get_smtlib2()
        with pytest.raises(RuntimeError):
            analyzer.get_z3_stats()
        with pytest.raises(RuntimeError):
            analyzer.set_z3_timeout_ms(1000)
        with pytest.raises(RuntimeError):
            analyzer.set_z3_rlimit(0)


def test_z3_context_lifetime_outlives_worker_thread():
    _require_z3(Analyzer())

    result_queue = queue.Queue()

    def worker():
        try:
            analyzer = Analyzer()
            analyzer_context = tvm_tirx.sym.ConstraintContext()
            x = tirx.Var("x", "int32")
            analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 16)))
            analyzer.sync_constraint_context(analyzer_context)
            assert analyzer.can_prove(x >= 0, SB)
            result_queue.put(("analyzer", analyzer))
        except BaseException as err:  # pylint: disable=broad-exception-caught
            result_queue.put(("error", err))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    kind, payload = result_queue.get_nowait()
    if kind == "error":
        raise payload

    del payload
    gc.collect()


def test_z3_context_clone_lifetime():
    _require_z3(Analyzer())

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    expr = ((b - a) // c) * c + a <= b

    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(a, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(b, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(c, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    assert analyzer.can_prove(expr, SB)

    # Materialize a separate context before cloning. The clone must adopt the
    # source Analyzer's context before copying any Z3 handles.
    independent = Analyzer()
    independent_context = tvm_tirx.sym.ConstraintContext()
    independent_context.append(tvm_tirx.sym.RangeConstraint(a, tvm_tirx.ir.Range(-100000, -1)))
    independent.sync_constraint_context(independent_context)
    independent_context.append(tvm_tirx.sym.RangeConstraint(b, tvm_tirx.ir.Range(-100000, 100000)))
    independent.sync_constraint_context(independent_context)
    independent_context.append(tvm_tirx.sym.RangeConstraint(c, tvm_tirx.ir.Range(1, 100000)))
    independent.sync_constraint_context(independent_context)
    assert independent.can_prove(expr, SB)
    assert independent.can_prove(a < 0, SB)
    cloned = analyzer.clone()

    del analyzer, independent
    gc.collect()
    assert cloned.can_prove(expr, SB)
    assert cloned.can_prove(a > 0, SB)


# ---------------------------------------------------------------------------
# Examples the native analyzer cannot prove but Z3 can.
#
# Most cases assert both that the native analyzers (kDefault, Z3 gated off)
# fail and that Z3 (kSymbolicBound) succeeds. This demonstrates the added value
# of the Z3 backend and that it is correctly gated behind kSymbolicBound. Cases
# the native analyzer learns to prove remain here as Z3 translation regressions.
# ---------------------------------------------------------------------------


def test_z3_floor_division_identity_constraint():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")

    expr = ((b - a) // c) * c + a <= b
    parent_context_1 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(tirx.all(a > 0, b > 0, c > 0)))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(expr)
        assert analyzer.can_prove(expr, SB)
    finally:
        analyzer_context = parent_context_1
        analyzer.sync_constraint_context(analyzer_context)


def test_z3_floor_division_identity_via_bind_range():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")

    analyzer_context.append(tvm_tirx.sym.RangeConstraint(a, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(b, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(c, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)

    expr = ((b - a) // c) * c + a <= b
    assert analyzer.can_prove(expr, SB)


def test_z3_multiplication_monotonicity():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    d = tirx.Var("d", "int32")

    expr = implies(tirx.all(a < b, b < c, a * d < b * d), b * d < c * d)
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_nested_floor_division_collapse():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    expr = implies(
        tirx.all(a >= 0, a < 128),
        a // 128 == (a // 64 * 32 + a % 32 // 16 * 8) // 64,
    )
    assert analyzer.can_prove(expr, SB)


def test_z3_deeply_nested_floor_division_identity():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    expr = implies(
        tirx.all(a >= 0, a < 128),
        (
            a % 16 * 64
            + a // 64 * 32
            + a % 8 // 4 * 32
            + (a % 32 // 16 + a % 2) % 2 * 8
            + 16
            - (a // 64 + a % 8 // 4) // 2 * 64
        )
        // 512
        == (
            a % 16 * 64
            + a // 64 * 32
            + a % 8 // 4 * 32
            + (a % 32 // 16 + a % 2) % 2 * 8
            - (a // 64 + a % 8 // 4) // 2 * 64
        )
        // 512,
    )
    assert analyzer.can_prove(expr, SB)


def test_z3_min_max_sum_identity():
    analyzer = Analyzer()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    expr = tirx.max(x, y) + tirx.min(x, y) == x + y
    assert analyzer.can_prove(expr, SB)


def test_z3_select_absolute_value_nonneg():
    analyzer = Analyzer()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    expr = tirx.Select(x >= 0, x, -x) >= 0
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_transitive_inequality():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    expr = implies(tirx.all(a <= b, b <= c), a <= c)
    assert analyzer.can_prove(expr, SB)


def test_z3_square_expansion_nonneg():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    expr = (a + b) * (a + b) >= a * a + b * b
    parent_context_2 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(tirx.all(a >= 0, b >= 0)))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(expr)
        assert analyzer.can_prove(expr, SB)
    finally:
        analyzer_context = parent_context_2
        analyzer.sync_constraint_context(analyzer_context)


def test_z3_square_monotonicity():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    expr = implies(tirx.all(0 <= a, a <= b), a * a <= b * b)
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_strict_multiplication():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    d = tirx.Var("d", "int32")
    expr = implies(tirx.all(a < b, d > 0), a * d < b * d)
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_floor_division_monotonicity():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    expr = implies(tirx.all(a <= b, c > 0), tirx.floordiv(a, c) <= tirx.floordiv(b, c))
    assert not analyzer.can_prove(expr)
    analyzer.set_z3_rlimit(0)
    assert analyzer.can_prove(expr, SB)


def test_z3_floor_division_lower_bound():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    expr = implies(b > 0, tirx.floordiv(a, b) * b <= a)
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_floor_modulo_range():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    expr = implies(b > 0, tirx.all(0 <= tirx.floormod(a, b), tirx.floormod(a, b) < b))
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_flattened_index_bound():
    # Classic index-flattening bound used throughout TVM: for a row index i in
    # [0, m) and a column index j in [0, n), the flattened index i * n + j stays
    # within [0, m * n).
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    m = tirx.Var("m", "int32")
    n = tirx.Var("n", "int32")
    expr = tirx.all(0 <= i * n + j, i * n + j < m * n)
    parent_context_3 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(
        tvm_tirx.sym.PredicateConstraint(tirx.all(0 <= i, i < m, 0 <= j, j < n, m > 0, n > 0))
    )
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(expr)
        analyzer.set_z3_rlimit(0)
        assert analyzer.can_prove(expr, SB)
    finally:
        analyzer_context = parent_context_3
        analyzer.sync_constraint_context(analyzer_context)


def test_z3_modular_combination():
    # Native modular_set tracks single-variable moduli, but combining two
    # independent modular facts to reason about their sum is left to Z3.
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    expr = tirx.floormod(x + y, 2) == 0
    parent_context_4 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(
        tvm_tirx.sym.PredicateConstraint(
            tirx.all(tirx.floormod(x, 6) == 0, tirx.floormod(y, 6) == 0)
        )
    )
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(expr)
        assert analyzer.can_prove(expr, SB)
    finally:
        analyzer_context = parent_context_4
        analyzer.sync_constraint_context(analyzer_context)


def test_z3_square_non_negative():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    assert not analyzer.can_prove(a * a >= 0)
    assert analyzer.can_prove(a * a >= 0, SB)


def test_z3_min_max_average_bounds():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    assert not analyzer.can_prove(tirx.max(a, b) * 2 >= a + b)
    assert analyzer.can_prove(tirx.max(a, b) * 2 >= a + b, SB)
    assert analyzer.can_prove(tirx.min(a, b) * 2 <= a + b, SB)


def test_z3_symbolic_bind_range_with_constraint():
    # Combine a symbolic range binding (x in [0, n)) with a constraint on the
    # extent to derive a concrete bound on x.
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    n = tirx.Var("n", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, n)))
    analyzer.sync_constraint_context(analyzer_context)
    parent_context_5 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(n <= 8))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(x < 8)
        assert analyzer.can_prove(x < 8, SB)
    finally:
        analyzer_context = parent_context_5
        analyzer.sync_constraint_context(analyzer_context)


def test_z3_equality_congruence():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    expr = implies(a == b, a * a == b * b)
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_integer_strict_transitivity():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    # Over the integers, a < b and b < c implies a + 1 < c.
    expr = implies(tirx.all(a < b, b < c), a + 1 < c)
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_if_then_else_absolute_value():
    analyzer = Analyzer()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    expr = tirx.if_then_else(x >= 0, x, -x) >= 0
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_unsigned_non_negative():
    analyzer = Analyzer()
    _require_z3(analyzer)

    u = tirx.Var("u", "uint32")
    assert not analyzer.can_prove(u >= 0)
    assert analyzer.can_prove(u >= 0, SB)


def test_z3_unsigned64_non_negative():
    # Exercises the special-cased uint64 range handling (UINT64_MAX bound).
    analyzer = Analyzer()
    _require_z3(analyzer)

    u = tirx.Var("u", "uint64")
    assert not analyzer.can_prove(u >= 0)
    assert analyzer.can_prove(u >= 0, SB)


def test_z3_int64_square_expansion():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    expr = (a + b) * (a + b) >= a * a + b * b
    parent_context_6 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(tirx.all(a >= 0, b >= 0)))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(expr)
        assert analyzer.can_prove(expr, SB)
    finally:
        analyzer_context = parent_context_6
        analyzer.sync_constraint_context(analyzer_context)


def test_z3_boolean_variable_reasoning():
    analyzer = Analyzer()
    _require_z3(analyzer)

    p = tirx.Var("p", "bool")
    q = tirx.Var("q", "bool")
    expr = implies(tirx.And(p, q), tirx.Or(p, q))
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_not_equal_from_strict_less():
    analyzer = Analyzer()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    expr = implies(x < y, tirx.NE(x, y))
    assert not analyzer.can_prove(expr)
    assert analyzer.can_prove(expr, SB)


def test_z3_let_expression():
    analyzer = Analyzer()
    _require_z3(analyzer)

    y = tirx.Var("y", "int32")
    t = tirx.Var("t", "int32")
    let = tirx.Let(t, y * 2, t)
    assert not analyzer.can_prove(let == y * 2)
    assert analyzer.can_prove(let == y * 2, SB)


def test_z3_cast_preserves_bounds():
    analyzer = Analyzer()
    _require_z3(analyzer)

    s = tirx.Var("s", "int16")
    widened = tirx.Cast("int32", s)
    assert analyzer.can_prove(widened <= 32767, SB)
    assert analyzer.can_prove(widened >= -32768, SB)


def test_z3_bitwise_and_mask_bound():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    assert analyzer.can_prove(tirx.bitwise_and(x, tvm_tirx.ir.IntImm("int32", 7)) < 8, SB)


def test_z3_bitwise_and_le_operand():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(y, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    # Bit-vector reasoning over two variables exceeds the default deterministic
    # rlimit; lift it (0 == unlimited, still deterministic) for this proof.
    analyzer.set_z3_rlimit(0)
    assert analyzer.can_prove(tirx.bitwise_and(x, y) <= x, SB)


def test_z3_bitwise_or_ge_operand():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(y, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer.set_z3_rlimit(0)
    assert analyzer.can_prove(tirx.bitwise_or(x, y) >= x, SB)


def test_z3_bitwise_xor_bound():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(y, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer.set_z3_rlimit(0)
    assert analyzer.can_prove(tirx.bitwise_xor(x, y) < 256, SB)


def test_z3_bitwise_not_identity():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer.set_z3_rlimit(0)
    # Two's complement: ~x == -x - 1.
    assert analyzer.can_prove(tirx.bitwise_not(x) == -x - 1, SB)


def test_z3_shift_right_halves():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 256)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer.set_z3_rlimit(0)
    # For non-negative x, (x >> 1) * 2 <= x.
    assert analyzer.can_prove(tirx.shift_right(x, tvm_tirx.ir.IntImm("int32", 1)) * 2 <= x, SB)


def test_z3_shift_left_lower_bound():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    n = tirx.Var("n", "int32")
    # Keep operands small so the 32-bit left shift cannot overflow; then
    # x << n == x * 2 ** n >= x for x >= 1.
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(1, 16)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(n, tvm_tirx.ir.Range(0, 4)))
    analyzer.sync_constraint_context(analyzer_context)
    # Bit-vector shift reasoning exceeds the default deterministic rlimit.
    analyzer.set_z3_rlimit(0)
    assert analyzer.can_prove(tirx.shift_left(x, n) >= x, SB)


# ---------------------------------------------------------------------------
# Soundness / negative tests (Z3 must NOT prove false predicates)
# ---------------------------------------------------------------------------


def test_z3_negative_unprovable_inequality():
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    # a < b does not hold for arbitrary a, b.
    assert not analyzer.can_prove(a < b, SB)
    # a * a > a is false (e.g. a == 0).
    assert not analyzer.can_prove(a * a > a, SB)


def test_z3_truncmod_can_be_negative():
    # Regression test for truncated div/mod semantics: TVM Div/Mod round toward
    # zero, so truncmod(a, 4) can be negative. A solver that modeled them as
    # Euclidean would unsoundly "prove" truncmod(a, 4) >= 0.
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    assert not analyzer.can_prove(tirx.truncmod(a, 4) >= 0, SB)


def test_z3_truncdiv_truncmod_identity():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    expr = tirx.truncdiv(a, b) * b + tirx.truncmod(a, b) == a
    parent_context_7 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(b != 0))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert analyzer.can_prove(expr, SB)
    finally:
        analyzer_context = parent_context_7
        analyzer.sync_constraint_context(analyzer_context)


def test_z3_floormod_nested_identities():
    # Ported from TileLang's test_divmod. Here `%` is floormod: nested floormod
    # by opposite-sign divisors collapses to the single-divisor result, while
    # the mixed case does not.
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    assert not analyzer.can_prove(a % 2 % -2 - a % 2 == 0, SB)
    assert analyzer.can_prove(a % -2 % 2 - a % 2 == 0, SB)


def test_z3_floormod_nonnegative():
    # In contrast to truncmod, floormod with a positive divisor is always in
    # [0, divisor), which Z3 should be able to prove.
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    assert analyzer.can_prove(tirx.floormod(a, 4) >= 0, SB)
    assert analyzer.can_prove(tirx.floormod(a, 4) < 4, SB)


def test_z3_shift_does_not_poison_solver():
    # Regression test: evaluating a shift expression must not add permanent
    # assertions (such as `b >= 0` / `b < 64`) to the shared solver. Otherwise
    # an unrelated, unbounded `b` would be wrongly provable to be < 100.
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")

    # Touch a shift expression so the prover visits the shift amount `b`.
    analyzer.can_prove(tirx.shift_left(a, b) >= 0, SB)

    # `b` is otherwise unconstrained, so this must remain unprovable.
    assert not analyzer.can_prove(b < 100, SB)
    assert not analyzer.can_prove(b >= 0, SB)


def test_z3_context_removal_clears_solver_assumptions():
    # Returning to the parent context must remove the child solver assumptions.
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    parent_context_8 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(x > 5))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert analyzer.can_prove(x > 0, SB)
    finally:
        analyzer_context = parent_context_8
        analyzer.sync_constraint_context(analyzer_context)
    # The constraint is gone; x is unconstrained again.
    assert not analyzer.can_prove(x > 0, SB)


def test_z3_opaque_call_is_safe():
    # An opaque/unsupported sub-expression is modeled as a fresh free variable.
    # Effectful raw premises are retained as data and cannot establish stable facts.
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    call = tirx.call_extern("foo", x, ty="int32")
    assert not analyzer.can_prove(call > 0, SB)
    parent_context_9 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(call > 0))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(call > 0, SB)
    finally:
        analyzer_context = parent_context_9
        analyzer.sync_constraint_context(analyzer_context)
    assert not analyzer.can_prove(call > 0, SB)


def test_z3_memo_pool_reuse_survives_clone():
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    # TensorLoad is read-state, so these expressions are memoized only for the
    # duration of the query and then erased.  This leaves holes in the Z3 memo
    # pool that subsequent pure expressions must be able to reuse safely.
    buffer = tirx.Var("tensor", tirx.TensorType((16,), "int32"))
    read_expr = tirx.all(*(buffer[i] >= 0 for i in range(16)))
    assert not analyzer.can_prove(read_expr, SB)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(a, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(b, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(c, tvm_tirx.ir.Range(1, 100000)))
    analyzer.sync_constraint_context(analyzer_context)
    expr = ((b - a) // c) * c + a <= b
    assert analyzer.can_prove(expr, SB)

    # Cloning must replay only live memo entries in slot order, even when the
    # source pool contains erased and reused slots.
    cloned = analyzer.clone()
    del analyzer
    gc.collect()
    assert cloned.can_prove(expr, SB)


def test_z3_shift_overflow_is_not_proven():
    # Z3 models fixed-width shifts via bit-vectors, so it correctly refuses to
    # prove `x << n >= x` for an unbounded `x` (a large `x` overflows int32 and
    # wraps to a negative value). This guards against unsound shift modeling.
    analyzer = Analyzer()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    n = tirx.Var("n", "int32")
    analyzer.set_z3_rlimit(0)
    expr = implies(tirx.all(x >= 1, n >= 0, n < 8), tirx.shift_left(x, n) >= x)
    assert not analyzer.can_prove(expr, SB)


def test_z3_analyzers_are_isolated():
    # Analyzers share a thread-local Z3 context but own separate solvers, so
    # constraints and bindings in one must never leak into another.
    analyzer_a = Analyzer()
    analyzer_a_context = tvm_tirx.sym.ConstraintContext()
    analyzer_b = Analyzer()
    _require_z3(analyzer_a)

    x = tirx.Var("x", "int32")
    parent_context_10 = analyzer_a_context
    analyzer_a_context = analyzer_a_context.snapshot()
    analyzer_a_context.append(tvm_tirx.sym.PredicateConstraint(x > 100))
    analyzer_a.sync_constraint_context(analyzer_a_context)
    try:
        assert analyzer_a.can_prove(x > 50, SB)
        assert not analyzer_b.can_prove(x > 50, SB)
    finally:
        analyzer_a_context = parent_context_10
        analyzer_a.sync_constraint_context(analyzer_a_context)

    analyzer_c = Analyzer()
    analyzer_c_context = tvm_tirx.sym.ConstraintContext()
    analyzer_d = Analyzer()
    analyzer_c_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 10)))
    analyzer_c.sync_constraint_context(analyzer_c_context)
    assert analyzer_c.can_prove(x < 10, SB)
    assert not analyzer_d.can_prove(x < 10, SB)


def test_z3_repeated_can_prove_is_consistent():
    # Repeated queries must be stateless: a CanProve call must not pollute the
    # solver and change the result of a subsequent call.
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    x = tirx.Var("x", "int32")
    assert analyzer.can_prove(x > 0, SB) == analyzer.can_prove(x > 0, SB)

    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(5, 10)))
    analyzer.sync_constraint_context(analyzer_context)
    assert analyzer.can_prove(x >= 5, SB)
    assert analyzer.can_prove(x >= 5, SB)


def test_z3_is_gated_behind_symbolic_bound():
    # The Z3 fallback must not run at the default strength.
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    expr = ((b - a) // c) * c + a <= b
    parent_context_11 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(tirx.all(a > 0, b > 0, c > 0)))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert not analyzer.can_prove(expr, ProofStrength.DEFAULT)
        assert analyzer.can_prove(expr, SB)
    finally:
        analyzer_context = parent_context_11
        analyzer.sync_constraint_context(analyzer_context)


# ---------------------------------------------------------------------------
# SMT-LIB2 export
# ---------------------------------------------------------------------------


def test_z3_smtlib2_roundtrip():
    z3 = pytest.importorskip("z3")
    analyzer = Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    expr = ((b - a) // c) * c + a <= b

    solver = z3.Solver()
    parent_context_12 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(tirx.all(a > 0, b > 0, c > 0)))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        solver.from_string(analyzer.get_smtlib2(expr))
    finally:
        analyzer_context = parent_context_12
        analyzer.sync_constraint_context(analyzer_context)
    assert solver.check() == z3.unsat


def test_z3_smtlib2_roundtrip_with_timeout():
    z3 = pytest.importorskip("z3")
    analyzer = Analyzer()
    _require_z3(analyzer)

    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    c = tirx.Var("c", "int32")
    analyzer.set_z3_timeout_ms(1000)

    expr = implies(tirx.all(a > 0, b > 0, c > 0), ((b - a) // c) * c + a <= b)
    solver = z3.Solver()
    solver.from_string(analyzer.get_smtlib2(expr))
    assert solver.check() == z3.unsat


if __name__ == "__main__":
    tvm_tirx.testing.main()
