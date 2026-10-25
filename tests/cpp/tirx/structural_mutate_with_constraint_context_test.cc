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
#include <tvm/tirx/structural_mutate_with_constraint_context.h>

namespace tvm::tirx {

TEST(ConstraintTraversal, MutateCollectsRewrittenFacts) {
  PrimVar x("x", PrimType::Int(32)), i("i", PrimType::Int(32));
  OpDef("test.read_loop_begin").set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState);
  PrimExpr begin =
      Call(PrimType::Int(32), Op::Get("test.read_loop_begin"), {}).as_or_throw<PrimExpr>();
  PrimExpr one = IntImm::Int32(1), four = IntImm::Int32(4);
  Evaluate body(i), after(x);
  SeqStmt root({Bind(x, four),
                If(likely(x <= four),
                   SeqStmt({For(i, begin, four, one, ForKind::kDefault, SeqStmt({body}))})),
                after});
  ffi::Array<sym::Constraint> body_facts, after_facts;
  auto result = StructuralMutateWithConstraintContext(
                    root,
                    [](const IntImmObj* node,
                       StructuralMutatorWithConstraintContextObj*) -> ffi::UnchangedOr<PrimExpr> {
                      if (node->value == 4) return IntImm::Int32(8);
                      return ffi::Unchanged();
                    },
                    [&](const EvaluateObj* node, StructuralMutatorWithConstraintContextObj* mutator,
                        ffi::InplaceMode mode) {
                      (node == body.get() ? body_facts : after_facts) =
                          mutator->constraints->ToArray();
                      return mutator->DefaultMutateExpected(node, mode);
                    })
                    .cast<SeqStmt>();

  ASSERT_EQ(body_facts.size(), 5);
  const auto* binding = body_facts[0].as<sym::BindingConstraintObj>();
  const auto* predicate = body_facts[1].as<sym::PredicateConstraintObj>();
  const auto* range = body_facts[2].as<sym::RangeConstraintObj>();
  ASSERT_NE(binding, nullptr);
  ASSERT_NE(predicate, nullptr);
  ASSERT_NE(range, nullptr);
  const auto* result_binding = result->seq[0].as<BindObj>();
  const auto* result_branch = result->seq[1].as<IfObj>();
  const auto* result_loop = result_branch->then_case->seq[0].as<ForObj>();
  EXPECT_TRUE(binding->var.same_as(result_binding->var));
  EXPECT_TRUE(binding->value.same_as(result_binding->value));
  EXPECT_EQ(binding->value.as<IntImmObj>()->value, 8);
  EXPECT_TRUE(predicate->condition.same_as(result_branch->condition));
  EXPECT_EQ(predicate->condition.as<CallObj>()->args[0].as<prim::LEObj>()->b.as<IntImmObj>()->value,
            8);
  EXPECT_TRUE(range->var.same_as(result_loop->var));
  EXPECT_EQ(result_loop->end.as<IntImmObj>()->value, 8);
  EXPECT_TRUE(range->range->begin.same_as(begin));
  const auto* extent = range->range->extent.as<prim::SubObj>();
  ASSERT_NE(extent, nullptr);
  EXPECT_TRUE(extent->a.same_as(result_loop->end));
  EXPECT_TRUE(extent->b.same_as(begin));
  ASSERT_EQ(after_facts.size(), 1);
  EXPECT_TRUE(after_facts[0].same_as(body_facts[0]));
}

TEST(ConstraintTraversal, MutateConditionalUsesRewrittenPredicate) {
  PrimVar condition("condition", PrimType::Bool()), replacement("replacement", PrimType::Bool());
  PrimExpr rewritten = likely(replacement);
  Call root(PrimType::Int(32), prim::if_then_else_op(),
            {condition, IntImm::Int32(1), IntImm::Int32(2)});
  ffi::Array<sym::Constraint> true_facts, false_facts;
  auto rewrite = [&](const VarObj*, StructuralMutatorWithConstraintContextObj*) {
    return rewritten;
  };
  auto observe = [&](const IntImmObj* node, StructuralMutatorWithConstraintContextObj* mutator) {
    (node->value == 1 ? true_facts : false_facts) = mutator->constraints->ToArray();
    return ffi::Unchanged();
  };
  auto copied = StructuralMutateWithConstraintContext(root, rewrite, observe).cast<Call>();
  EXPECT_TRUE(root->args[0].same_as(condition));
  EXPECT_TRUE(copied->args[0].same_as(rewritten));
  ASSERT_EQ(true_facts.size(), 1);
  ASSERT_EQ(false_facts.size(), 1);
  EXPECT_TRUE(true_facts[0].as<sym::PredicateConstraintObj>()->condition.same_as(rewritten));
  EXPECT_TRUE(
      false_facts[0].as<sym::PredicateConstraintObj>()->condition.as<prim::NotObj>()->a.same_as(
          rewritten));
}

TEST(ConstraintTraversal, MutateAssumptionsAndAssertionEffects) {
  PrimVar condition("condition", PrimType::Bool());
  OpDef("test.rewritten_assertion")
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState);
  PrimExpr read =
      Call(PrimType::Bool(), Op::Get("test.rewritten_assertion"), {}).as_or_throw<PrimExpr>();
  SeqStmt root({AssertStmt(condition, StringImm("ValueError"), {}),
                AssertStmt(read, StringImm("ValueError"), {}),
                Evaluate(Call(PrimType::Void(), assume_always_op(), {condition})),
                Evaluate(IntImm::Int32(1))});
  ffi::Array<sym::Constraint> facts;
  StructuralMutateWithConstraintContext(
      root, [&](const VarObj*, StructuralMutatorWithConstraintContextObj*) { return read; },
      [&](const CallObj* node, StructuralMutatorWithConstraintContextObj* mutator,
          ffi::InplaceMode mode) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (node == read.get()) return ffi::Any(condition);
        return mutator->DefaultMutateExpected(node, mode);
      },
      [&](const IntImmObj*, StructuralMutatorWithConstraintContextObj* mutator) {
        facts = mutator->constraints->ToArray();
        return ffi::Unchanged();
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
