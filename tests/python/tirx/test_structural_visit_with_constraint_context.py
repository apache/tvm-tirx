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

from tvm_tirx import ir, sym
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import structural


def test_walk_collects_binding_and_loop_range():
    @T.function
    def before(A: T.Tensor((4,), "int32")):
        x: T.let = A[0]
        for i in T.serial(4):
            T.evaluate(x + i)

    collected = []

    def observe(value, constraints):
        if isinstance(value, ir.Evaluate):
            collected.append([fact for fact in constraints])

    structural.structural_walk_with_constraint_context(before.body, observe)
    (facts,) = collected
    binding, loop_range, _end_after_begin, _positive_extent = facts
    assert isinstance(binding, sym.BindingConstraint)
    assert binding.var.same_as(before.body[0].var)
    assert binding.value.same_as(before.body[0].value)
    assert isinstance(loop_range, sym.RangeConstraint)
    assert loop_range.var.same_as(before.body[1].var)
    assert int(loop_range.range.begin) == 0 and int(loop_range.range.extent) == 4


def test_walk_omits_zero_trip_loop_constraints():
    @T.function
    def before():
        for i in T.serial(0):
            T.evaluate(T.truncmod(1, i))

    collected = []

    def observe(value, constraints):
        if isinstance(value, ir.Evaluate):
            collected.append(list(constraints))

    structural.structural_walk_with_constraint_context(before.body, observe)
    assert collected == [[]]


def test_visit_collects_branch_predicates():
    @T.function
    def before(x: T.int32):
        if x > 0:
            T.evaluate(1)
        else:
            T.evaluate(2)
        T.evaluate(3)

    collected = {}

    def visit(value, visitor):
        if isinstance(value, ir.Evaluate):
            collected[int(value.value)] = [fact for fact in visitor.constraints]
        return visitor.default_visit(value)

    structural.structural_visit_with_constraint_context(before.body, visit)
    (positive,) = collected[1]
    (negative,) = collected[2]
    assert isinstance(positive, sym.PredicateConstraint) and not positive.assume_always
    assert positive.condition.same_as(before.body[0].condition)
    assert isinstance(negative.condition, ir.prim.Not)
    assert negative.condition.a.same_as(positive.condition)
    assert collected[3] == []
