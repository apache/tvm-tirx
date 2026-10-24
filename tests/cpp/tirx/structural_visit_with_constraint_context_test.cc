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
#include <tvm/tirx/structural_visit_with_constraint_context.h>

namespace tvm::tirx {

TEST(ConstraintTraversal, VisitCollectsScopedFacts) {
  PrimVar x("x", PrimType::Int(32)), i("i", PrimType::Int(32));
  PrimExpr zero = IntImm::Int32(0), one = IntImm::Int32(1), four = IntImm::Int32(4);
  PrimExpr condition = x <= four;
  Evaluate body(i), after(x);
  SeqStmt root(
      {Bind(x, four),
       If(condition, SeqStmt({For(i, zero, four, one, ForKind::kDefault, SeqStmt({body}))})),
       after});
  ffi::Array<sym::Constraint> body_facts, after_facts;
  StructuralVisitWithConstraintContext(
      root, [&](const EvaluateObj* node, StructuralVisitorWithConstraintContextObj* visitor) {
        (node == body.get() ? body_facts : after_facts) = visitor->constraints->ToArray();
        return visitor->DefaultVisitExpected(node);
      });

  ASSERT_EQ(body_facts.size(), 3);
  const auto* binding = body_facts[0].as<sym::BindingConstraintObj>();
  const auto* predicate = body_facts[1].as<sym::PredicateConstraintObj>();
  const auto* range = body_facts[2].as<sym::RangeConstraintObj>();
  ASSERT_NE(binding, nullptr);
  ASSERT_NE(predicate, nullptr);
  ASSERT_NE(range, nullptr);
  EXPECT_TRUE(binding->var.same_as(x));
  EXPECT_TRUE(binding->value.same_as(four));
  EXPECT_TRUE(predicate->condition.same_as(condition));
  EXPECT_TRUE(range->var.same_as(i));
  EXPECT_EQ(range->range->begin.as<IntImmObj>()->value, 0);
  EXPECT_EQ(range->range->extent.as<IntImmObj>()->value, 4);
  ASSERT_EQ(after_facts.size(), 1);
  EXPECT_TRUE(after_facts[0].same_as(body_facts[0]));
}

}  // namespace tvm::tirx
