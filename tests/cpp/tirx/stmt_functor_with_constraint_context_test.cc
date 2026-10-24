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
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor_with_constraint_context.h>

#include <cstdlib>

namespace tvm::tirx {
namespace {
TEST(ConstraintFunctorDeathTest, RuntimeVisitExpressionSpecializations) {
  // Type attributes are single-assignment; keep registration local to the child process.
  ASSERT_EXIT(([] {
                int visits = 0;
                auto expr = prim::Add(IntImm::Int32(1), IntImm::Int32(2));
                auto visitor = ffi::make_object<StmtExprVisitorWithConstraintContext>();
                EXPECT_FALSE(visitor->VisitExpected(expr).is_err());
                ffi::reflection::TypeAttrDef<prim::AddObj>().def(
                    type_attr::kStructuralVisitWithConstraintContext.name,
                    [&](StructuralVisitorWithConstraintContextObj*,
                        ffi::AnyView) -> ffi::Optional<ffi::VisitInterrupt> {
                      ++visits;
                      return std::nullopt;
                    });
                EXPECT_FALSE(visitor->VisitExpected(expr).is_err());
                EXPECT_EQ(visits, 1);
                std::_Exit(testing::Test::HasFailure() ? 1 : 0);
              }()),
              testing::ExitedWithCode(0), "");
}

TEST(ConstraintFunctorDeathTest, RuntimeMutationExpressionSpecializations) {
  // Type attributes are single-assignment; keep registration local to the child process.
  ASSERT_EXIT(
      ([] {
        auto expr = prim::Add(IntImm::Int32(1), IntImm::Int32(2));
        auto mutator = ffi::make_object<StmtExprMutatorWithConstraintContext>();
        EXPECT_TRUE(mutator->Mutate(expr).IsUnchanged());
        ffi::reflection::TypeAttrDef<prim::AddObj>().def(
            type_attr::kStructuralMutateWithConstraintContext.name,
            [](StructuralMutatorWithConstraintContextObj*, ffi::AnyView) -> ffi::Any {
              return IntImm::Int32(9);
            });
        // In-place permission falls back to the copying hook when it is the only hook.
        EXPECT_EQ(
            mutator->Mutate(expr, ffi::InplaceMode::kAllow).ValueUnchecked().as<IntImmObj>()->value,
            9);
        ffi::reflection::TypeAttrDef<prim::AddObj>().def(
            type_attr::kStructuralMaybeInplaceMutateWithConstraintContext.name,
            [](StructuralMutatorWithConstraintContextObj*, ffi::AnyView) -> ffi::Any {
              return IntImm::Int32(10);
            });
        EXPECT_EQ(mutator->Mutate(expr).ValueUnchecked().as<IntImmObj>()->value, 9);
        EXPECT_EQ(
            mutator->Mutate(expr, ffi::InplaceMode::kAllow).ValueUnchecked().as<IntImmObj>()->value,
            10);
        std::_Exit(testing::Test::HasFailure() ? 1 : 0);
      }()),
      testing::ExitedWithCode(0), "");
}

TEST(ConstraintFunctor, NativeVisitorScopesAndFallback) {
  PrimVar x("x", PrimType::Int(32));
  SeqStmt sequence(
      {Evaluate(Call(PrimType::Void(), assume_always_op(), {x > IntImm::Int32(0)})), Evaluate(x)});
  struct Visitor : ObjectVisitorWithConstraintContext {
    ffi::Array<sym::Constraint> facts;
    ffi::Optional<ffi::VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<EvaluateObj>()) facts = constraints->ToArray();
      return ObjectVisitorWithConstraintContext::Visit(value);
    }
  };
  auto visitor = ffi::make_object<Visitor>();
  OpDef("test.read_thread_extent")
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kReadState);
  PrimExpr extent =
      Call(PrimType::Int(32), Op::Get("test.read_thread_extent"), {}).as_or_throw<PrimExpr>();
  RegionStmt region(launch_thread_op(), {StringImm("threadIdx.x"), extent},
                    {PrimVar("thread", PrimType::Int(32))}, DictAttrs(), sequence);
  // Array fallback reaches both thread-scope and assumption Op hooks.
  visitor->Visit(ffi::Array<Stmt>{region});
  ASSERT_EQ(visitor->facts.size(), 2);
  EXPECT_TRUE(visitor->facts[0].as<sym::RangeConstraintObj>()->range->extent.same_as(extent));
  EXPECT_TRUE(visitor->facts[1].as<sym::PredicateConstraintObj>()->assume_always);
  EXPECT_TRUE(visitor->constraints->ToArray().empty());
}

TEST(ConstraintFunctor, NativeVisitorInterruptRestoresScope) {
  PrimVar x("x", PrimType::Int(32));
  Call root(PrimType::Int(32), prim::if_then_else_op(),
            {likely(x > IntImm::Int32(0)), IntImm::Int32(1), IntImm::Int32(2)});
  struct Stop : StmtExprVisitorWithConstraintContext {
    int visited = 0;
    using StmtExprVisitorWithConstraintContext::Visit_;
    ffi::Optional<ffi::VisitInterrupt> Visit_(const IntImmObj* node) override {
      if (node->value == 0) return std::nullopt;
      ++visited;
      EXPECT_EQ(constraints->ToArray().size(), 1);
      return ffi::VisitInterrupt(IntImm::Int32(42));
    }
  };
  auto visitor = ffi::make_object<Stop>();
  EXPECT_TRUE(visitor->VisitExpected(root).value().has_value());
  EXPECT_EQ(visitor->visited, 1);
  EXPECT_TRUE(visitor->constraints->ToArray().empty());
}

TEST(ConstraintFunctor, NativeTypedEntriesAllowCategoryReplacements) {
  struct Rewrite : StmtExprMutatorWithConstraintContext {
    using StmtExprMutatorWithConstraintContext::Mutate_;
    ffi::UnchangedOr<PrimExpr> Mutate_(const prim::AddObj* node, ffi::InplaceMode) override {
      return node->a;
    }
    ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj* node, ffi::InplaceMode) override {
      return Return(node->value);
    }
    ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj*, ffi::InplaceMode) override {
      return Evaluate(IntImm::Int32(3));
    }
  };
  auto mutator = ffi::make_object<Rewrite>();
  auto expression = mutator->Mutate(prim::Add(IntImm::Int32(1), IntImm::Int32(2))).ValueUnchecked();
  EXPECT_EQ(expression.as<IntImmObj>()->value, 1);
  auto statement = mutator->Mutate(Evaluate(IntImm::Int32(2))).ValueUnchecked();
  EXPECT_NE(statement.as<ReturnObj>(), nullptr);
  auto sequence = mutator->Mutate(SeqStmt({Evaluate(IntImm::Int32(1))})).ValueUnchecked();
  ASSERT_EQ(sequence->seq.size(), 1);
  EXPECT_EQ(sequence->seq[0].as<EvaluateObj>()->value.as<IntImmObj>()->value, 3);
}

// An extension with only a copying specialization must retain it under in-place permission.
class CopyOnlyScopeObj : public OpaqueExprObj {
 public:
  Stmt body;
  explicit CopyOnlyScopeObj(Stmt body) : body(std::move(body)) { ty = PrimType::Int(32); }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.NativeCopyOnlyConstraintScope", CopyOnlyScopeObj,
                                    OpaqueExprObj);
};
ffi::Expected<ffi::UnchangedOr<ffi::Any>> CopyOnlyScope(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  return mutator->WithConstraintScope([&] {
    mutator->AppendConstraint(sym::PredicateConstraint(IntImm::Bool(true)));
    return mutator->ffi::StructuralMutatorObj::DefaultMutateExpected(value);
  });
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::ObjectDef<CopyOnlyScopeObj>().def_ro("body", &CopyOnlyScopeObj::body);
  ffi::reflection::TypeAttrDef<CopyOnlyScopeObj>().attr(
      type_attr::kStructuralMutateWithConstraintContext,
      FStructuralMutateWithConstraintContext::FromNative<&CopyOnlyScope>());
}

TEST(ConstraintFunctor, CopyingSpecializationUnderInplacePermission) {
  ffi::Any root = ffi::make_object<CopyOnlyScopeObj>(Evaluate(IntImm::Int32(1)));
  struct Rewrite : StmtExprMutatorWithConstraintContext {
    int calls = 0;
    using StmtExprMutatorWithConstraintContext::Mutate_;
    ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj*, ffi::InplaceMode) override {
      ++calls;
      EXPECT_EQ(constraints->ToArray().size(), 1);
      return Evaluate(IntImm::Int32(2));
    }
  };
  auto mutator = ffi::make_object<Rewrite>();
  auto mapped = mutator->Mutate(root, ffi::InplaceMode::kAllow);
  auto result = std::move(mapped).ValueOrUnchanged(std::move(root));
  EXPECT_EQ(mutator->calls, 1);
  EXPECT_TRUE(mutator->constraints->ToArray().empty());
  EXPECT_EQ(
      result.cast<const CopyOnlyScopeObj*>()->body.as<EvaluateObj>()->value.as<IntImmObj>()->value,
      2);
}

TEST(ConstraintFunctor, NativeVisitorPreservesExtensionDescent) {
  struct Visitor : StmtExprVisitorWithConstraintContext {
    int visited = 0;
    Visitor() : StmtExprVisitorWithConstraintContext(Table()) {}
    using StmtExprVisitorWithConstraintContext::Visit_;
    static const VTable* Table() {
      static const VTable table = [] {
        VTable table;
        InitVTable(&table);
        SetDispatch<Visitor, CopyOnlyScopeObj>(&table);
        table.Finalize();
        return table;
      }();
      return &table;
    }
    ffi::Optional<ffi::VisitInterrupt> Visit_(const CopyOnlyScopeObj* node) {
      return StmtExprVisitorWithConstraintContext::Visit_(static_cast<const OpaqueExprObj*>(node));
    }
    ffi::Optional<ffi::VisitInterrupt> Visit_(const EvaluateObj*) override {
      ++visited;
      return std::nullopt;
    }
  };
  auto visitor = ffi::make_object<Visitor>();
  auto root = ffi::make_object<CopyOnlyScopeObj>(Evaluate(IntImm::Int32(1)));
  EXPECT_FALSE(visitor->VisitExpected(root).is_err());
  EXPECT_EQ(visitor->visited, 1);
}

TEST(ConstraintFunctor, NativeMutationRemapsDefinitions) {
  PrimVar original("x", PrimType::Int(32)), renamed("renamed", PrimType::Int(32));
  RegionStmt root(launch_thread_op(), {StringImm("threadIdx.x"), IntImm::Int32(7)}, {original},
                  DictAttrs(), SeqStmt({Evaluate(original)}));
  struct Rewrite : StmtExprMutatorWithConstraintContext {
    Var renamed;
    explicit Rewrite(Var renamed) : renamed(renamed) {}
    using StmtExprMutatorWithConstraintContext::Mutate_;
    ffi::UnchangedOr<Expr> Mutate_(const VarObj* node, ffi::InplaceMode mode) override {
      if (def_region_kind() == kTVMFFIDefRegionKindSimple) return renamed;
      return StmtExprMutatorWithConstraintContext::Mutate_(node, mode);
    }
    ffi::UnchangedOr<PrimExpr> Mutate_(const IntImmObj*, ffi::InplaceMode) override {
      return IntImm::Int32(9);
    }
    ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj* node, ffi::InplaceMode mode) override {
      EXPECT_EQ(constraints->ToArray().size(), 1);
      const auto* fact = constraints->ToArray()[0].as<sym::RangeConstraintObj>();
      EXPECT_TRUE(fact->var.same_as(renamed));
      EXPECT_EQ(fact->range->extent.as<IntImmObj>()->value, 9);
      return StmtExprMutatorWithConstraintContext::Mutate_(node, mode);
    }
  };
  auto mutator = ffi::make_object<Rewrite>(renamed);
  RegionStmt mapped = mutator->Mutate(root).ValueUnchecked().as_or_throw<RegionStmt>();
  EXPECT_TRUE(root->body_params[0].same_as(original));
  EXPECT_TRUE(mapped->body_params[0].same_as(renamed));
  EXPECT_TRUE(mapped->body->seq[0].as<EvaluateObj>()->value.same_as(renamed));
  EXPECT_TRUE(mutator->constraints->ToArray().empty());
}

TEST(ConstraintFunctor, NativeInplaceAssumptionUsesRewrittenPredicate) {
  PrimVar condition("condition", PrimType::Bool()), replacement("replacement", PrimType::Bool());
  SeqStmt root({Evaluate(Call(PrimType::Void(), assume_always_op(), {condition})),
                Evaluate(IntImm::Int32(1))});
  const auto* original_call = root->seq[0].as<EvaluateObj>()->value.as<CallObj>();
  struct Rewrite : StmtExprMutatorWithConstraintContext {
    PrimExpr replacement;
    ffi::Array<sym::Constraint> facts;
    bool inplace_call = false;
    explicit Rewrite(PrimExpr replacement) : replacement(std::move(replacement)) {}
    using StmtExprMutatorWithConstraintContext::Mutate_;
    ffi::UnchangedOr<Expr> Mutate_(const VarObj*, ffi::InplaceMode) override { return replacement; }
    ffi::UnchangedOr<Expr> Mutate_(const CallObj* node, ffi::InplaceMode mode) override {
      inplace_call = mode == ffi::InplaceMode::kAllow;
      return StmtExprMutatorWithConstraintContext::Mutate_(node, mode);
    }
    ffi::UnchangedOr<PrimExpr> Mutate_(const IntImmObj*, ffi::InplaceMode) override {
      facts = constraints->ToArray();
      return ffi::Unchanged();
    }
  };
  auto mutator = ffi::make_object<Rewrite>(replacement);
  SeqStmt result = mutator->Mutate(root, ffi::InplaceMode::kAllow).ValueOrUnchanged(root);
  EXPECT_TRUE(mutator->inplace_call);
  EXPECT_EQ(result->seq[0].as<EvaluateObj>()->value.as<CallObj>(), original_call);
  ASSERT_EQ(mutator->facts.size(), 1);
  const auto* fact = mutator->facts[0].as<sym::PredicateConstraintObj>();
  EXPECT_TRUE(fact->condition.same_as(replacement));
  EXPECT_TRUE(fact->assume_always);
  EXPECT_TRUE(mutator->constraints->ToArray().empty());
}

TEST(ConstraintFunctor, NativeMutationErrorPropagation) {
  struct Failure : StmtExprMutatorWithConstraintContext {
    using StmtExprMutatorWithConstraintContext::Mutate_;
    ffi::UnchangedOr<PrimExpr> Mutate_(const IntImmObj*, ffi::InplaceMode) override {
      throw ffi::Error("ValueError", "native mutation failure", "");
    }
  };
  auto mutator = ffi::make_object<Failure>();
  Evaluate root(prim::Add(IntImm::Int32(1), IntImm::Int32(2)));
  auto result = mutator->MutateExpected(root);
  ASSERT_TRUE(result.is_err());
  auto context = ffi::VisitErrorContext::TryGetFromError(result.error());
  ASSERT_TRUE(context.has_value());
  ASSERT_EQ(context.value()->reverse_visit_pattern.size(), 3);
  EXPECT_TRUE(context.value()->reverse_visit_pattern[2].same_as(root));
}

TEST(ConstraintFunctor, NativeVisitorErrorPropagation) {
  struct Failure : StmtExprVisitorWithConstraintContext {
    ffi::Error error{"ValueError", "native visit failure", ""};
    using StmtExprVisitorWithConstraintContext::Visit_;
    ffi::Optional<ffi::VisitInterrupt> Visit_(const IntImmObj*) override { throw error; }
  };
  auto visitor = ffi::make_object<Failure>();
  PrimExpr root = prim::Add(IntImm::Int32(1), IntImm::Int32(2));
  auto result = visitor->VisitExpected(root);
  ASSERT_TRUE(result.is_err());
  EXPECT_TRUE(result.error().same_as(visitor->error));
  auto context = ffi::VisitErrorContext::TryGetFromError(result.error());
  ASSERT_TRUE(context.has_value());
  ASSERT_EQ(context.value()->reverse_visit_pattern.size(), 2);
  EXPECT_TRUE(context.value()->reverse_visit_pattern[1].same_as(root));
}

ffi::Array<Var> NoBodyParams(const CallObj*) { return {}; }

TEST(ConstraintFunctor, NativeOpCopyingHookReceivesContainingNodes) {
  OpDef("test.native_constraint_op")
      .set_native_fn_attr<&NoBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_attr(op_attr::kStructuralMutateWithConstraintContext.name,
                ffi::Function::FromTyped(
                    [](StructuralMutatorWithConstraintContextObj*, ffi::AnyView value) -> ffi::Any {
                      if (const auto* call = value.as<CallObj>()) return call->args[0];
                      return value.cast<RegionStmt>()->body;
                    }));
  Call call(PrimType::Int(32), Op::Get("test.native_constraint_op"), {IntImm::Int32(1)});
  RegionStmt region(Op::Get("test.native_constraint_op"), {}, {}, DictAttrs(),
                    SeqStmt({Evaluate(IntImm::Int32(2))}));
  auto mutator = ffi::make_object<StmtExprMutatorWithConstraintContext>();
  // Both unique containing nodes must use the Op's copying hook when no in-place hook exists.
  ASSERT_TRUE(call.unique());
  EXPECT_TRUE(
      mutator->Mutate(call, ffi::InplaceMode::kAllow).ValueUnchecked().same_as(call->args[0]));
  ASSERT_TRUE(region.unique());
  EXPECT_TRUE(
      mutator->Mutate(region, ffi::InplaceMode::kAllow).ValueUnchecked().same_as(region->body));
}

}  // namespace
}  // namespace tvm::tirx
