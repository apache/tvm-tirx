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
#include <gtest/gtest.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/structural_visit_with_constraint_context.h>

namespace tvm::tirx {

TEST(ConstraintTraversal, VisitCollectsScopedFacts) {
  PrimVar x("x", PrimType::Int(32)), i("i", PrimType::Int(32));
  OpDef("test.read_loop_end").set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState);
  PrimExpr zero = IntImm::Int32(0), one = IntImm::Int32(1);
  PrimExpr end = Call(PrimType::Int(32), Op::Get("test.read_loop_end"), {}).as_or_throw<PrimExpr>();
  PrimExpr condition = likely(x <= end);
  Evaluate body(i), after(x);
  SeqStmt root(
      {Bind(x, end),
       If(condition, SeqStmt({For(i, zero, end, one, ForKind::kDefault, SeqStmt({body}))})),
       after});
  ffi::Array<sym::Constraint> body_facts, after_facts;
  StructuralVisitWithConstraintContext(
      root, [&](const EvaluateObj* node, StructuralVisitorWithConstraintContextObj* visitor) {
        (node == body.get() ? body_facts : after_facts) = visitor->constraints->ToArray();
        return visitor->DefaultVisitExpected(node);
      });

  ASSERT_EQ(body_facts.size(), 5);
  const auto* binding = body_facts[0].as<sym::BindingConstraintObj>();
  const auto* predicate = body_facts[1].as<sym::PredicateConstraintObj>();
  const auto* range = body_facts[2].as<sym::RangeConstraintObj>();
  ASSERT_NE(binding, nullptr);
  ASSERT_NE(predicate, nullptr);
  ASSERT_NE(range, nullptr);
  EXPECT_TRUE(binding->var.same_as(x));
  EXPECT_TRUE(binding->value.same_as(end));
  EXPECT_TRUE(predicate->condition.same_as(condition));
  EXPECT_FALSE(predicate->assume_always);
  EXPECT_TRUE(range->var.same_as(i));
  EXPECT_EQ(range->range->begin.as<IntImmObj>()->value, 0);
  EXPECT_TRUE(range->range->extent.same_as(end));
  ASSERT_EQ(after_facts.size(), 1);
  EXPECT_TRUE(after_facts[0].same_as(body_facts[0]));
}

TEST(ConstraintTraversal, VisitAssumptionsAndAssertions) {
  PrimVar condition("condition", PrimType::Bool());
  OpDef("test.read_assertion").set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState);
  PrimExpr read =
      Call(PrimType::Bool(), Op::Get("test.read_assertion"), {}).as_or_throw<PrimExpr>();
  SeqStmt root({AssertStmt(read, StringImm("ValueError"), {}),
                AssertStmt(condition, StringImm("ValueError"), {}),
                Evaluate(Call(PrimType::Void(), assume_always_op(), {read})),
                Evaluate(IntImm::Int32(1))});
  ffi::Array<sym::Constraint> facts;
  StructuralVisitWithConstraintContext(
      root, [&](const IntImmObj*, StructuralVisitorWithConstraintContextObj* visitor) {
        facts = visitor->constraints->ToArray();
        return std::nullopt;
      });
  ASSERT_EQ(facts.size(), 2);
  const auto* assertion = facts[0].as<sym::PredicateConstraintObj>();
  const auto* assumption = facts[1].as<sym::PredicateConstraintObj>();
  EXPECT_TRUE(assertion->condition.same_as(condition));
  EXPECT_FALSE(assertion->assume_always);
  EXPECT_TRUE(assumption->condition.same_as(read));
  EXPECT_TRUE(assumption->assume_always);
}

}  // namespace tvm::tirx
