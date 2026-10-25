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

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import tirx
from tvm_tirx.sym.analyzer import CompareResult


def test_analyzer_is_ffi_object_with_persistent_state():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int64")

    assert isinstance(analyzer, tvm_ffi.Object)

    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 8)))
    analyzer.sync_constraint_context(analyzer_context)
    assert analyzer.const_int_bound_is_bound(x)
    assert analyzer.can_prove(x < 8)
    assert not analyzer.can_prove(x < 4)

    bound = analyzer.const_int_bound(x + 1)
    assert bound.min_value == 1
    assert bound.max_value == 8


def test_analyzer_object_context_switch_and_replacement():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int64")

    parent_context_1 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(x % 3 == 0))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        assert analyzer.modular_set(x).coeff == 3
    finally:
        analyzer_context = parent_context_1
        analyzer.sync_constraint_context(analyzer_context)

    assert analyzer.modular_set(x).coeff != 3

    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    y = tirx.Var("y", "int64")
    analyzer_context.append(tvm_tirx.sym.BindingConstraint(y, tirx.const(4, "int64")))
    analyzer.sync_constraint_context(analyzer_context)
    tvm_tirx.ir.assert_structural_equal(analyzer.simplify(y + 1), tirx.const(5, "int64"))

    analyzer_context = tvm_tirx.sym.ConstraintContext()
    analyzer_context.append(tvm_tirx.sym.BindingConstraint(y, tirx.const(8, "int64")))
    analyzer.sync_constraint_context(analyzer_context)
    tvm_tirx.ir.assert_structural_equal(analyzer.simplify(y + 1), tirx.const(9, "int64"))


def test_analyzer_object_update_const_int_bound():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int64")

    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(x >= 2))
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(x <= 5))
    analyzer.sync_constraint_context(analyzer_context)

    bound = analyzer.const_int_bound(x + 1)
    assert bound.min_value == 3
    assert bound.max_value == 6


def test_analyzer_object_update_modular_set():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int32")

    assert analyzer.modular_set(x).coeff == 1
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(x % 4 == 0))
    analyzer.sync_constraint_context(analyzer_context)

    result = analyzer.modular_set(x)
    assert result.coeff == 4
    assert result.base == 0


def test_analyzer_object_update_int_set():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    y = tirx.Var("y", "int32")

    analyzer_context.append(tvm_tirx.sym.RangeConstraint(y, tvm_tirx.ir.Range(0, (8) + 1)))
    analyzer.sync_constraint_context(analyzer_context)

    int_set = analyzer.int_set(y)
    assert int_set.min_value.value == 0
    assert int_set.max_value.value == 8


def test_analyzer_object_sync_rejects_unknown_context():
    analyzer = tvm_tirx.sym.Analyzer()
    with pytest.raises(TypeError):
        analyzer.sync_constraint_context("not-a-constraint-context")


def test_analyzer_object_can_prove_comparison_predicates():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 8)))
    analyzer.sync_constraint_context(analyzer_context)

    assert analyzer.can_prove(x >= 0)
    assert not analyzer.can_prove(x >= 1)
    assert analyzer.can_prove(x < 8)
    assert not analyzer.can_prove(x < 7)


def test_analyzer_object_update_const_int_bound_half_space():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    n = tirx.Var("n", "int32")

    assert not analyzer.can_prove(n >= 0)
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(n >= 0))
    analyzer.sync_constraint_context(analyzer_context)
    assert analyzer.can_prove(n >= 0)


def test_analyzer_object_int_set_from_bound_vars():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int32")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 8)))
    analyzer.sync_constraint_context(analyzer_context)

    int_set = analyzer.int_set(x + 1)
    assert int_set.min_value.value == 1
    assert int_set.max_value.value == 8


def test_analyzer_object_set_maximum_rewrite_steps():
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    expr = (x + y) * 2 - x * 2 - y * 2 + tirx.max(x, y) - tirx.min(x, y)

    capped = tvm_tirx.sym.Analyzer()
    capped.set_maximum_rewrite_steps(1)
    with pytest.raises(RuntimeError):
        capped.rewrite_simplify(expr)

    # A generous limit must not interfere with normal simplification.
    relaxed = tvm_tirx.sym.Analyzer()
    relaxed.set_maximum_rewrite_steps(1000)
    relaxed.rewrite_simplify(expr)


def test_analyzer_object_try_compare_direct():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    z = tirx.Var("z", "int32")

    assert analyzer.try_compare(x, y) == CompareResult.UNKNOWN

    parent_context_2 = analyzer_context
    analyzer_context = analyzer_context.snapshot()
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(x < y))
    analyzer.sync_constraint_context(analyzer_context)
    try:
        parent_context_3 = analyzer_context
        analyzer_context = analyzer_context.snapshot()
        analyzer_context.append(tvm_tirx.sym.PredicateConstraint(y < z))
        analyzer.sync_constraint_context(analyzer_context)
        try:
            assert analyzer.try_compare(x, y) == CompareResult.LT
            # No direct comparison connects x and z.
            assert analyzer.try_compare(x, z) == CompareResult.UNKNOWN
        finally:
            analyzer_context = parent_context_3
            analyzer.sync_constraint_context(analyzer_context)
    finally:
        analyzer_context = parent_context_2
        analyzer.sync_constraint_context(analyzer_context)


def test_analyzer_object_rewrite_simplify_stats():
    analyzer = tvm_tirx.sym.Analyzer()
    x = tirx.Var("x", "int32")

    analyzer.reset_rewrite_simplify_stats()
    assert analyzer.rewrite_simplify_stats.nodes_visited == 0

    analyzer.rewrite_simplify(x + 0)
    assert analyzer.rewrite_simplify_stats.nodes_visited > 0

    analyzer.reset_rewrite_simplify_stats()
    assert analyzer.rewrite_simplify_stats.nodes_visited == 0


def test_analyzer_object_state_persists_across_ffi_calls():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    tile = tirx.Var("tile", "int32")
    i = tirx.Var("i", "int32")
    analyzer_context.append(tvm_tirx.sym.BindingConstraint(tile, tvm_tirx.tirx.const(8, "int32")))
    analyzer.sync_constraint_context(analyzer_context)

    # The same analyzer object is borrowed by the C++ DetectIterMap entry point;
    # its binding makes the otherwise-undetectable floormod recognizable.
    result = tvm_tirx.sym.detect_iter_map(
        [i % tile], {i: tvm_tirx.ir.Range(0, 32)}, analyzer=analyzer
    )
    assert len(result.indices) == 1

    # The binding still lives in the same stateful object after the FFI call.
    tvm_tirx.ir.assert_structural_equal(analyzer.simplify(tile), tvm_tirx.tirx.const(8, "int32"))


def test_analyzer_object_clone_is_independent():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int64")
    y = tirx.Var("y", "int64")
    z = tirx.Var("z", "int64")

    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 8)))
    analyzer.sync_constraint_context(analyzer_context)

    clone = analyzer.clone()
    clone_context = analyzer_context.snapshot()
    assert clone is not analyzer
    assert clone.can_prove(x < 8)

    clone_context.append(tvm_tirx.sym.RangeConstraint(y, tvm_tirx.ir.Range(0, 4)))
    clone.sync_constraint_context(clone_context)
    assert clone.can_prove(y < 4)
    assert not analyzer.can_prove(y < 4)

    analyzer_context.append(tvm_tirx.sym.RangeConstraint(z, tvm_tirx.ir.Range(0, 4)))
    analyzer.sync_constraint_context(analyzer_context)
    assert analyzer.can_prove(z < 4)
    assert not clone.can_prove(z < 4)

    assert analyzer.can_prove(x < 8)
    assert clone.can_prove(x < 8)


def test_analyzer_object_clone_copies_every_sub_analyzer():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int64")
    w = tirx.Var("w", "int64")
    v = tirx.Var("v", "int64")

    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 8)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.PredicateConstraint(x % 4 == 0))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.BindingConstraint(w, tirx.const(4, "int64")))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(v, tvm_tirx.ir.Range(2, (9) + 1)))
    analyzer.sync_constraint_context(analyzer_context)

    clone = analyzer.clone()
    clone_context = analyzer_context.snapshot()

    assert clone.can_prove(x < 8)
    assert clone.modular_set(x).coeff == 4
    tvm_tirx.ir.assert_structural_equal(clone.simplify(w + 1), tirx.const(5, "int64"))
    assert clone.int_set(v).max_value.value == 9
    assert clone.try_compare(x, tirx.const(0, "int64")) == CompareResult.GE

    t = tirx.Var("t", "int64")
    clone_context = tvm_tirx.sym.ConstraintContext()
    clone_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 8)))
    clone_context.append(tvm_tirx.sym.PredicateConstraint(x % 8 == 0))
    clone.sync_constraint_context(clone_context)
    clone_context.append(tvm_tirx.sym.RangeConstraint(v, tvm_tirx.ir.Range(0, (3) + 1)))
    clone.sync_constraint_context(clone_context)
    clone_context.append(tvm_tirx.sym.BindingConstraint(w, tirx.const(8, "int64")))
    clone.sync_constraint_context(clone_context)
    clone_context.append(tvm_tirx.sym.RangeConstraint(t, tvm_tirx.ir.Range(0, 4)))
    clone.sync_constraint_context(clone_context)

    assert analyzer.modular_set(x).coeff == 4
    # Coordinated ingestion also uses 0 <= x < 8 when applying x % 8 == 0.
    assert clone.modular_set(x).coeff == 0
    assert clone.modular_set(x).base == 0
    assert analyzer.int_set(v).max_value.value == 9
    assert clone.int_set(v).max_value.value == 3
    tvm_tirx.ir.assert_structural_equal(analyzer.simplify(w + 1), tirx.const(5, "int64"))
    tvm_tirx.ir.assert_structural_equal(clone.simplify(w + 1), tirx.const(9, "int64"))
    assert clone.try_compare(t, tirx.const(0, "int64")) == CompareResult.GE
    assert analyzer.try_compare(t, tirx.const(0, "int64")) == CompareResult.UNKNOWN


def test_analyzer_object_clone_resets_rewrite_stats():
    analyzer = tvm_tirx.sym.Analyzer()
    analyzer_context = tvm_tirx.sym.ConstraintContext()
    x = tirx.Var("x", "int64")
    y = tirx.Var("y", "int64")
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(x, tvm_tirx.ir.Range(0, 8)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer_context.append(tvm_tirx.sym.RangeConstraint(y, tvm_tirx.ir.Range(0, 8)))
    analyzer.sync_constraint_context(analyzer_context)
    analyzer.simplify((x + y) * 2 - x - y)
    source_attempts = analyzer.rewrite_simplify_stats.rewrites_attempted
    assert source_attempts > 0

    clone = analyzer.clone()
    assert clone.rewrite_simplify_stats.rewrites_attempted == 0
    assert analyzer.rewrite_simplify_stats.rewrites_attempted == source_attempts


def test_failed_iter_map_has_absent_padding_predicate():
    i = tvm_tirx.tirx.Var("i", "int32")
    result = tvm_tirx.sym.detect_iter_map([i], {i: tvm_tirx.ir.Range(0, i)})
    assert result.errors
    assert result.padding_predicate is None
    restored = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(result))
    assert restored.padding_predicate is None


if __name__ == "__main__":
    tvm_tirx.testing.main()
