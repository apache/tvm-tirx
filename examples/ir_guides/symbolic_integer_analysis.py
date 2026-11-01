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
"""Simplify and prove index expressions with the symbolic integer analyzer."""

# Imports stay next to the snippets that use them in the docs.
# ruff: noqa: E402

# [add_one_2d.begin]
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one_2d(A: T.Tensor((n, 4), T.float32), B: T.Tensor((n, 4), T.float32)):
    for i in range(n * 4):
        B[i // 4, i % 4] = A[i // 4, i % 4] + 1


# [add_one_2d.end]


# [add_one_1d.begin]
@T.function
def add_one_1d(A: T.Tensor((n, 4), T.float32), B: T.Tensor((n, 4), T.float32)):
    A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor((n * 4,), T.float32)])
    B_flat = T.decl_tensor(B.data, ty_args=[T.Tensor((n * 4,), T.float32)])
    for i in range(n * 4):
        B_flat[i // 4 * 4 + i % 4] = A_flat[i // 4 * 4 + i % 4] + 1


# [add_one_1d.end]

# [simplify.begin]
from tvm_tirx import ir, sym

analyzer = sym.Analyzer()
loop_1d = add_one_1d.body.seq[-1]  # follows the two decl_tensor statements
index = loop_1d.body.seq[0].indices[0]  # i // 4 * 4 + i % 4
print(analyzer.simplify(index))
# [simplify.end]
assert isinstance(loop_1d, ir.For)
assert str(index) == "i // 4 * 4 + i % 4"
assert analyzer.simplify(index).same_as(loop_1d.var)

# [prove.begin]
loop = add_one_2d.body.seq[0]  # for i in range(n * 4)
i = loop.var
print(analyzer.can_prove(i // 4 < n))
# [prove.end]
assert isinstance(loop, ir.For)
assert not analyzer.can_prove(i // 4 < n)

# [loop_range.begin]
loop_context = sym.ConstraintContext()
loop_context.append(sym.RangeConstraint(i, ir.Range(loop.begin, loop.end)))
analyzer.sync_constraint_context(loop_context)
print(analyzer.can_prove(i // 4 < n))
# [loop_range.end]
assert not analyzer.can_prove(i // 4 < n)

# [loop_runs.begin]
loop_context.append(sym.PredicateConstraint(loop.end > 0))
analyzer.sync_constraint_context(loop_context)
print(analyzer.can_prove(i // 4 < n))
# [loop_runs.end]
assert analyzer.can_prove(i // 4 < n)

# [branch.begin]
branch_context = loop_context.snapshot()
branch_context.append(sym.PredicateConstraint(i % 4 < 2))
analyzer.sync_constraint_context(branch_context)
print(analyzer.can_prove(i % 4 < 3))
analyzer.sync_constraint_context(loop_context)
print(analyzer.can_prove(i % 4 < 3))
# [branch.end]
assert not analyzer.can_prove(i % 4 < 3)
analyzer.sync_constraint_context(branch_context)
assert analyzer.can_prove(i % 4 < 3)
# The snapshot leaves the loop context unchanged.
analyzer.sync_constraint_context(loop_context)
assert not analyzer.can_prove(i % 4 < 3)
assert analyzer.can_prove(i // 4 < n)


# [add_one_guarded.begin]
@T.function
def add_one_guarded(A: T.Tensor((n, 4), T.float32), B: T.Tensor((n, 4), T.float32)):
    for i in range(n * 4):
        if i // 4 < n:
            B[i // 4, i % 4] = A[i // 4, i % 4] + 1


# [add_one_guarded.end]

# [walk.begin]
from tvm_tirx import tirx

guard_analyzer = sym.Analyzer()
proven = []


def check_guard(value, constraints):
    if isinstance(value, ir.If):
        guard_analyzer.sync_constraint_context(constraints)
        proven.append(guard_analyzer.can_prove(value.condition))


tirx.structural_walk_with_constraint_context(add_one_guarded, check_guard, order="pre")
print(proven)
# [walk.end]
assert proven == [True]
# The traversal supplies the loop range and the fact that the loop runs.
facts = []


def record_facts(value, constraints):
    if isinstance(value, ir.If):
        facts.extend(str(constraint) for constraint in constraints)


tirx.structural_walk_with_constraint_context(add_one_guarded, record_facts, order="pre")
assert "sym.RangeConstraint(var=i, range=T.Range(0, n * 4))" in facts
assert any("condition=n * 4 > 0" in fact for fact in facts)
# Without facts, the guard is not provable.
assert not sym.Analyzer().can_prove(add_one_guarded.body.seq[0].body.seq[0].condition)


# [map.begin]
def remove_redundant_guard(value, constraints):
    if isinstance(value, ir.If):
        guard_analyzer.sync_constraint_context(constraints)
        if guard_analyzer.can_prove(value.condition):
            return value.then_case
    return value


simplified = tirx.structural_map_with_constraint_context(
    add_one_guarded, remove_redundant_guard, order="pre"
)
print(simplified.script())
# [map.end]
assert "if " not in simplified.script()
assert simplified.script().endswith(
    "    for i in range(n * 4):\n        B[i // 4, i % 4] = A[i // 4, i % 4] + T.float32(1.0)"
)
assert "if i // 4 < n:" in add_one_guarded.script()  # the input is unchanged
