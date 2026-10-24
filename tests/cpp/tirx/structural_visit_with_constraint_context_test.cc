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
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/structural_mutate_with_constraint_context.h>

#include <cstdlib>

namespace tvm::tirx {
namespace {
using Facts = sym::ConstraintContext;
using Snapshot = ffi::Array<sym::Constraint>;
using Result = ffi::Optional<ffi::VisitInterrupt>;
PrimExpr I(int n) { return IntImm::Int32(n); }

TEST(ConstraintTraversal, CapturedContextPrefixesAndIndependentReads) {
  Facts root;
  sym::Constraint a = sym::PredicateConstraint(I(1)), b = sym::PredicateConstraint(I(2));
  sym::Constraint c = sym::PredicateConstraint(I(3)), d = sym::PredicateConstraint(I(4), true);
  EXPECT_FALSE(a.as<sym::PredicateConstraintObj>()->assume_always);
  EXPECT_TRUE(d.as<sym::PredicateConstraintObj>()->assume_always);
  EXPECT_FALSE(ffi::make_object<sym::PredicateConstraintObj>(ffi::UnsafeInit{})->assume_always);
  root->Append(a);
  Facts child(root, root->LocalSize());
  child->Append(b);
  root->Append(c);
  Facts grandchild(child, child->LocalSize());
  child->Append(c);
  grandchild->Append(d);
  Snapshot observed;
  grandchild->ForEach([&](const sym::Constraint& fact) { observed.push_back(fact); });
  ASSERT_EQ(observed.size(), 3);
  EXPECT_TRUE(observed[0].same_as(a));
  EXPECT_TRUE(observed[1].same_as(b));
  EXPECT_TRUE(observed[2].same_as(d));
  EXPECT_EQ(grandchild->LocalSize(), 1);
  EXPECT_EQ(Facts(child, 1)->ToArray().size(), 2);
  EXPECT_EQ(child->ToArray().size(), 3);
  EXPECT_EQ(Facts(grandchild, 0)->ToArray().size(), 2);
  EXPECT_THROW(Facts(child, -1), ffi::Error);
  EXPECT_THROW(Facts(grandchild, 2), ffi::Error);  // 2 is effective, not local size.
  EXPECT_THROW(Facts(child, 3), ffi::Error);
  Snapshot flat = grandchild->ToArray();
  Facts independent;
  for (const auto& fact : flat) independent->Append(fact);
  flat.Set(0, d);
  flat.push_back(c);
  root->Append(d);
  grandchild->Append(c);
  EXPECT_EQ(independent->LocalSize(), 3);
  EXPECT_TRUE(independent->ToArray()[0].same_as(a));
  EXPECT_EQ(Facts(grandchild, 1)->ToArray().size(), 3);
  // Iteration captures its initial local prefix, even if a callback appends.
  int calls = 0;
  independent->ForEach([&](const sym::Constraint&) {
    ++calls;
    independent->Append(d);
  });
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(independent->LocalSize(), 6);
  // Iteration also owns the receiver when a callback replaces its last handle.
  calls = 0;
  independent->ForEach([&](const sym::Constraint&) {
    if (++calls == 1) independent = Facts();
  });
  EXPECT_EQ(calls, 6);
  EXPECT_EQ(independent->LocalSize(), 0);
}

ffi::Array<Var> NoBodyParams(const CallObj*) { return {}; }

TEST(ConstraintTraversal, OperatorVisitHooksReceiveWholeNodes) {
  OpDef def("test.constraint_visit_scope");
  def.set_native_fn_attr<&NoBodyParams>(tvm::op_attr::kRegionGetBodyParams);
  for (bool region : {false, true}) {
    ffi::Any root =
        region ? ffi::Any(RegionStmt(def.op(), {}, {}, DictAttrs(), SeqStmt({Evaluate(I(7))})))
               : ffi::Any(Call(PrimType::Int(32), def.op(), {I(7)}));
    const auto* identity = root.cast<ffi::ObjectRef>().get();
    int visits = 0;
    ffi::Error expected_error("ValueError", "operator hook failure", "");
    bool fail = false;
    ffi::TypedFunction<Result(StructuralVisitorWithConstraintContextObj*, ffi::AnyView)> visit =
        [&](StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) -> Result {
      EXPECT_EQ(value.cast<ffi::ObjectRef>().get(), identity);
      ++visits;
      if (fail) throw expected_error;
      return visitor
          ->WithConstraintScope([&]() -> ffi::Expected<Result> {
            visitor->AppendConstraint(sym::PredicateConstraint(IntImm::Bool(true)));
            return visitor->VisitExpected(region ? ffi::AnyView(value.cast<RegionStmt>()->body)
                                                 : ffi::AnyView(value.cast<Call>()->args));
          })
          .value();
    };
    def.set_attr(op_attr::kStructuralVisitWithConstraintContext, visit);
    auto visitor = ffi::make_object<StructuralVisitorWithConstraintContextObj>();
    EXPECT_FALSE(visitor->VisitExpected(root).value().has_value());
    EXPECT_EQ(visits, 1);
    EXPECT_TRUE(visitor->constraints->ToArray().empty());
    auto stop = StructuralWalkWithConstraintContext<ffi::WalkOrder::kPreOrder>(
        root, [](const IntImmObj*, const Facts& facts) {
          EXPECT_EQ(facts->ToArray().size(), 1);
          return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(I(9)));
        });
    EXPECT_TRUE(stop.has_value());
    fail = true;
    auto visit_error =
        ffi::make_object<StructuralVisitorWithConstraintContextObj>()->VisitExpected(root);
    ASSERT_TRUE(visit_error.is_err());
    EXPECT_TRUE(visit_error.error().same_as(expected_error));
    def.reset_attr(op_attr::kStructuralVisitWithConstraintContext);
  }
}

TEST(ConstraintTraversal, UnregisteredRegionVisitScopesAndLaunchCalls) {
  auto make_root = [] {
    PrimVar x("x", PrimType::Int(32));
    // launch_thread is also a Call before region construction: it has no body range yet.
    return SeqStmt({Evaluate(Call(PrimType::Int(32), Op::Get("tirx.launch_thread"),
                                  {StringImm("threadIdx.x"), I(8)})),
                    RegionStmt(Op::Get("tirx.device_scope"), {}, {}, DictAttrs(),
                               SeqStmt({Bind(x, I(1)), Evaluate(I(2))})),
                    Evaluate(I(3))});
  };
  auto check = [](const EvaluateObj* node, const Facts& facts) {
    const auto* number = node->value.as<IntImmObj>();
    EXPECT_EQ(facts->ToArray().size(), number && number->value == 2 ? 2 : 0);
  };
  auto result = StructuralVisitWithConstraintContextExpected(
      make_root(), [&](ffi::AnyView value, StructuralVisitorWithConstraintContextObj* visitor) {
        if (const auto* node = value.as<SeqStmtObj>(); node && node->seq.size() == 2) {
          visitor->AppendConstraint(sym::PredicateConstraint(IntImm::Bool(true)));
        }
        if (const auto* node = value.as<EvaluateObj>()) check(node, visitor->constraints);
        return visitor->DefaultVisitExpected(value);
      });
  EXPECT_FALSE(result.is_err());
}

TEST(ConstraintTraversalDeathTest, RuntimeVisitHooks) {
  // Type attributes are single-assignment. Isolate registration from other tests.
  ASSERT_EXIT(([] {
                int visits = 0;
                ffi::reflection::TypeAttrDef<prim::AddObj>().def(
                    type_attr::kStructuralVisitWithConstraintContext.name,
                    [&](StructuralVisitorWithConstraintContextObj*, ffi::AnyView) -> Result {
                      ++visits;
                      return std::nullopt;
                    });
                auto make_expr = [] { return prim::Add(I(1), I(2)); };
                auto aware = ffi::make_object<StructuralVisitorWithConstraintContextObj>();
                if (aware->VisitExpected(make_expr()).is_err() || visits != 1) std::_Exit(2);
                ffi::reflection::TypeAttrDef<prim::SubObj>().attr(
                    type_attr::kStructuralVisitWithConstraintContext.name, 42);
                ffi::reflection::TypeAttrDef<prim::MulObj>().attr(
                    type_attr::kStructuralVisitWithConstraintContext.name,
                    static_cast<void*>(nullptr));
                for (bool null_hook : {false, true}) {
                  PrimExpr root =
                      null_hook ? PrimExpr(prim::Mul(I(1), I(2))) : PrimExpr(prim::Sub(I(1), I(2)));
                  auto visitor = ffi::make_object<StructuralVisitorWithConstraintContextObj>();
                  auto result = visitor->VisitExpected(root);
                  if (!result.is_err() || result.error().kind() != "TypeError") std::_Exit(3);
                }
                std::_Exit(0);
              }()),
              testing::ExitedWithCode(0), "");
}

TEST(ConstraintTraversal, LazyScopesPreserveIncomingContexts) {
  auto check = [](auto traversal) {
    Facts incoming = traversal->constraints;
    auto result = traversal->WithConstraintScope([&]() -> ffi::Expected<void> {
      EXPECT_TRUE(traversal->constraints.same_as(incoming));
      auto nested = traversal->WithConstraintScope([&]() -> ffi::Expected<void> {
        EXPECT_TRUE(traversal->constraints.same_as(incoming));
        traversal->AppendConstraint(sym::PredicateConstraint(IntImm::Bool(true)));
        EXPECT_FALSE(traversal->constraints.same_as(incoming));
        return {};
      });
      EXPECT_TRUE(nested.has_value());
      EXPECT_TRUE(traversal->constraints.same_as(incoming));
      // A nested append must not consume the outer scope's pending child.
      traversal->AppendConstraint(sym::PredicateConstraint(IntImm::Bool(false)));
      EXPECT_FALSE(traversal->constraints.same_as(incoming));
      EXPECT_EQ(traversal->constraints->LocalSize(), 1);
      return {};
    });
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(traversal->constraints.same_as(incoming));
    EXPECT_EQ(incoming->LocalSize(), 0);
  };
  check(ffi::make_object<StructuralVisitorWithConstraintContextObj>());
  check(ffi::make_object<StructuralMutatorWithConstraintContextObj>());
}

TEST(ConstraintTraversal, VisitInterruptAndErrorPropagation) {
  PrimVar x("x", PrimType::Int(32));
  SeqStmt root({Bind(x, I(1)), If(x > I(0), SeqStmt({Evaluate(I(2))})), Evaluate(I(3))});
  int visited = 0;
  ffi::Optional<std::pair<Facts, int64_t>> position;
  ffi::Optional<StructuralVisitorWithConstraintContext> active;
  auto stopped = StructuralVisitWithConstraintContextExpected(
      root, [&](const EvaluateObj*, StructuralVisitorWithConstraintContextObj* visitor) -> Result {
        ++visited;
        active = ffi::GetRef<StructuralVisitorWithConstraintContext>(visitor);
        position = std::make_pair(visitor->constraints, visitor->constraints->LocalSize());
        EXPECT_EQ(visitor->constraints->ToArray().size(), 2);
        return ffi::VisitInterrupt(I(42));
      });
  ASSERT_TRUE(stopped.value().has_value());
  EXPECT_EQ(visited, 1);
  EXPECT_TRUE(active.value()->constraints->ToArray().empty());
  EXPECT_EQ(Facts(position.value().first, position.value().second)->ToArray().size(), 2);

  ffi::Error expected_error("ValueError", "expected failure", "");
  auto walk_error = StructuralWalkWithConstraintContextExpected<ffi::WalkOrder::kPreOrder>(
      root, [&](const EvaluateObj*, const Facts&) -> ffi::Expected<ffi::WalkResult> {
        return ffi::Unexpected(expected_error);
      });
  ASSERT_TRUE(walk_error.is_err());
  EXPECT_TRUE(walk_error.error().same_as(expected_error));
}

}  // namespace
}  // namespace tvm::tirx
