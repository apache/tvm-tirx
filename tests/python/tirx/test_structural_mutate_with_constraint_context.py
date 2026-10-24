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


def test_map_collects_transformed_guard_and_range():
    @T.function
    def before(x: T.int32):
        if x > 0:
            for i in T.serial(4):
                T.evaluate(i)

    x = before.params[0]

    # This replacement fragment refers to the original function's parameter.
    @T.function(check_well_formed=False)
    def replacement():
        T.evaluate(x > 2)
        T.evaluate(T.int32(8))

    guard, bound = replacement.body[0].value, replacement.body[1].value
    collected = []

    def rewrite(value, constraints):
        if isinstance(value, ir.prim.GT):
            assert [fact for fact in constraints] == []
            return guard
        if isinstance(value, ir.IntImm) and int(value) == 4:
            (predicate,) = [fact for fact in constraints]
            assert predicate.condition.same_as(guard)
            return bound
        if isinstance(value, ir.Evaluate):
            collected.append([fact for fact in constraints])
        return value

    structural.structural_map_with_constraint_context(before.body, rewrite)
    (facts,) = collected
    predicate, loop_range = facts
    assert isinstance(predicate, sym.PredicateConstraint)
    assert predicate.condition.same_as(guard)
    assert isinstance(loop_range, sym.RangeConstraint)
    assert int(loop_range.range.extent) == int(bound)


def test_mutate_collects_replacement_binding():
    @T.function
    def before():
        x: T.let = 1
        T.evaluate(x)

    @T.function
    def replacement():
        y: T.let = 5
        T.evaluate(y)

    binding = replacement.body[0]
    collected = []

    def rewrite(value, mutator):
        if isinstance(value, ir.Bind):
            return binding
        if isinstance(value, ir.Evaluate):
            collected.append([fact for fact in mutator.constraints])
        return mutator.default_mutate(value)

    structural.structural_mutate_with_constraint_context(before.body, rewrite)
    ((fact,),) = collected
    assert isinstance(fact, sym.BindingConstraint)
    assert fact.var.same_as(binding.var) and fact.value.same_as(binding.value)


def test_map_collects_transformed_let_binding():
    @T.function
    def before():
        x = T.int32()
        T.evaluate(T.Let(x, where={x: 1}))

    @T.function
    def replacement():
        T.evaluate(T.int64(2))

    value = replacement.body[0].value
    collected = []

    def rewrite(node, constraints):
        if isinstance(node, ir.IntImm) and int(node) == 1:
            return value
        if isinstance(node, ir.Var):
            facts = [fact for fact in constraints]
            if facts:
                collected.append(facts)
        return node

    result = structural.structural_map_with_constraint_context(before.body[0].value, rewrite)
    ((fact,),) = collected
    assert isinstance(fact, sym.BindingConstraint)
    assert fact.var.same_as(result.var)
    assert fact.value.same_as(value) and fact.var.ty == value.ty
