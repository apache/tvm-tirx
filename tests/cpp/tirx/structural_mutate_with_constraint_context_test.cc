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
using Mode = ffi::InplaceMode;
PrimExpr I(int n) { return IntImm::Int32(n); }

template <typename... Callbacks>
auto MakeMutator(Callbacks... callbacks) {
  using Engine =
      ffi::StructuralMutateEngine<StructuralMutatorWithConstraintContextObj, Callbacks...>;
  return ffi::make_object<Engine>(std::move(callbacks)...);
}

// An extension with only a copying specialization must retain it under in-place permission.
class CopyOnlyScopeObj : public ffi::Object {
 public:
  Stmt body;
  explicit CopyOnlyScopeObj(Stmt body) : body(std::move(body)) {}
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.CopyOnlyConstraintScope", CopyOnlyScopeObj, ffi::Object);
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

TEST(ConstraintTraversal, CopyingSpecializationUnderInplacePermission) {
  ffi::Any root = ffi::make_object<CopyOnlyScopeObj>(Evaluate(I(1)));
  int calls = 0;
  auto mutator = MakeMutator(
      [&](const EvaluateObj*, StructuralMutatorWithConstraintContextObj* mutator) -> Stmt {
        ++calls;
        EXPECT_EQ(mutator->constraints->ToArray().size(), 1);
        return Evaluate(I(2));
      });
  auto mapped = mutator->MutateExpected(root, Mode::kAllow).value();
  auto result = std::move(mapped).ValueOrUnchanged(std::move(root));
  EXPECT_EQ(calls, 1);
  // Continuing after the custom scope must not reintroduce its predicate.
  EXPECT_TRUE(mutator->constraints->ToArray().empty());
  mutator->constraints->Append(sym::PredicateConstraint(IntImm::Bool(false)));
  EXPECT_EQ(mutator->constraints->ToArray().size(), 1);
  EXPECT_EQ(
      result.cast<const CopyOnlyScopeObj*>()->body.as<EvaluateObj>()->value.as<IntImmObj>()->value,
      2);
}

ffi::Array<Var> NoBodyParams(const CallObj*) { return {}; }

TEST(ConstraintTraversal, OperatorMutationHooksReceiveWholeNodesAndPreserveCopyingFallback) {
  OpDef def("test.constraint_mutate_scope");
  def.set_native_fn_attr<&NoBodyParams>(tvm::op_attr::kRegionGetBodyParams);
  for (bool region : {false, true}) {
    for (int ownership : {0, 1, 2}) {
      ffi::Any root =
          region ? ffi::Any(RegionStmt(def.op(), {}, {}, DictAttrs(), SeqStmt({Evaluate(I(7))})))
                 : ffi::Any(Call(PrimType::Int(32), def.op(), {I(7)}));
      const auto* identity = root.cast<ffi::ObjectRef>().get();
      ffi::Any alias = ownership == 1 ? root : ffi::Any();
      int copies = 0, inplace = 0;
      ffi::Error expected_error("ValueError", "operator hook failure", "");
      bool fail = false;
      ffi::TypedFunction<ffi::UnchangedOr<ffi::Any>(StructuralMutatorWithConstraintContextObj*,
                                                    ffi::AnyView)>
          copy = [&](StructuralMutatorWithConstraintContextObj* mutator,
                     ffi::AnyView value) -> ffi::UnchangedOr<ffi::Any> {
        EXPECT_EQ(value.cast<ffi::ObjectRef>().get(), identity);
        ++copies;
        if (fail) throw expected_error;
        auto mapped = mutator
                          ->WithConstraintScope([&] {
                            mutator->AppendConstraint(sym::PredicateConstraint(IntImm::Bool(true)));
                            return mutator->MutateExpected(
                                region ? ffi::AnyView(value.cast<RegionStmt>()->body)
                                       : ffi::AnyView(value.cast<Call>()->args));
                          })
                          .value();
        if (mapped.IsUnchanged()) return ffi::Unchanged();
        if (region) {
          auto result = ffi::make_object<RegionStmtObj>(*value.cast<const RegionStmtObj*>());
          result->body = ffi::AnyView(mapped).cast<SeqStmt>();
          return ffi::Any(std::move(result));
        }
        auto result = ffi::make_object<CallObj>(*value.cast<const CallObj*>());
        result->args = ffi::AnyView(mapped).cast<ffi::Array<Expr>>();
        return ffi::Any(std::move(result));
      };
      def.set_attr(op_attr::kStructuralMutateWithConstraintContext, copy);
      auto rewrite = [](const IntImmObj*, StructuralMutatorWithConstraintContextObj* mutator,
                        Mode mode) -> PrimExpr {
        EXPECT_EQ(mode, Mode::kDisallow);
        EXPECT_EQ(mutator->constraints->ToArray().size(), 1);
        return I(8);
      };
      auto mutator = MakeMutator(rewrite);
      auto mapped = mutator->MutateExpected(root, ownership == 2 ? Mode::kDisallow : Mode::kAllow);
      ASSERT_FALSE(mapped.is_err());
      EXPECT_FALSE(mapped.value().IsUnchanged());
      EXPECT_EQ(copies, 1);
      EXPECT_TRUE(mutator->constraints->ToArray().empty());
      auto original = region ? root.cast<RegionStmt>()->body->seq[0].as<EvaluateObj>()->value
                             : root.cast<Call>()->args[0];
      EXPECT_EQ(original.as<IntImmObj>()->value, 7);
      // Cached Op maps observe late registration; only unique, permitted roots use it.
      ffi::TypedFunction<ffi::UnchangedOr<ffi::Any>(StructuralMutatorWithConstraintContextObj*,
                                                    ffi::AnyView)>
          update = [&](StructuralMutatorWithConstraintContextObj*, ffi::AnyView value) {
            EXPECT_EQ(value.cast<ffi::ObjectRef>().get(), identity);
            ++inplace;
            return ffi::UnchangedOr<ffi::Any>(ffi::Unchanged());
          };
      def.set_attr(op_attr::kStructuralMaybeInplaceMutateWithConstraintContext, update);
      auto second = MakeMutator(rewrite);
      EXPECT_FALSE(
          second->MutateExpected(root, ownership == 2 ? Mode::kDisallow : Mode::kAllow).is_err());
      EXPECT_EQ(inplace, ownership == 0 ? 1 : 0);
      def.reset_attr(op_attr::kStructuralMaybeInplaceMutateWithConstraintContext);
      fail = true;
      auto mutate_error = MakeMutator(rewrite)->MutateExpected(root);
      ASSERT_TRUE(mutate_error.is_err());
      EXPECT_TRUE(mutate_error.error().same_as(expected_error));
      def.reset_attr(op_attr::kStructuralMutateWithConstraintContext);
    }
  }
}

TEST(ConstraintTraversal, UnregisteredRegionMutationScopesAndLaunchCalls) {
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
  for (Mode mode : {Mode::kDisallow, Mode::kAllow}) {
    auto mutator = MakeMutator(
        [&](ffi::AnyView value, StructuralMutatorWithConstraintContextObj* mutator, Mode mode) {
          if (const auto* node = value.as<SeqStmtObj>(); node && node->seq.size() == 2) {
            mutator->AppendConstraint(sym::PredicateConstraint(IntImm::Bool(true)));
          }
          if (const auto* node = value.as<EvaluateObj>()) check(node, mutator->constraints);
          return mutator->DefaultMutateExpected(value, mode);
        });
    SeqStmt root = make_root();
    EXPECT_FALSE(mutator->MutateExpected(root, mode).is_err());
    EXPECT_TRUE(mutator->constraints->ToArray().empty());
  }
}

TEST(ConstraintTraversalDeathTest, RuntimeMutationHooks) {
  // Type attributes are single-assignment. Isolate registration so other tests
  // keep the default mutation hooks, including when tests are shuffled or repeated.
  ASSERT_EXIT(([] {
                int copies = 0, inplace = 0;
                ffi::reflection::TypeAttrDef<prim::AddObj>()
                    .def(type_attr::kStructuralMutateWithConstraintContext.name,
                         [&](StructuralMutatorWithConstraintContextObj*, ffi::AnyView) -> ffi::Any {
                           ++copies;
                           return I(9);
                         })
                    .def(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext.name,
                         [&](StructuralMutatorWithConstraintContextObj*, ffi::AnyView) -> ffi::Any {
                           ++inplace;
                           return I(10);
                         });
                auto make_expr = [] { return prim::Add(I(1), I(2)); };
                for (Mode mode : {Mode::kDisallow, Mode::kAllow}) {
                  auto mutator = ffi::make_object<StructuralMutatorWithConstraintContextObj>();
                  PrimExpr root = make_expr();
                  auto result = mutator->MutateExpected(root, mode).value();
                  if (result.IsUnchanged()) std::_Exit(3);
                  if (ffi::AnyView(result).cast<PrimExpr>().as<IntImmObj>()->value !=
                      (mode == Mode::kAllow ? 10 : 9)) {
                    std::_Exit(4);
                  }
                }
                if (copies != 1 || inplace != 1) std::_Exit(5);
                // Present, non-callable attributes must return errors through noexcept dispatch.
                ffi::reflection::TypeAttrDef<prim::SubObj>()
                    .attr(type_attr::kStructuralMutateWithConstraintContext.name, 42)
                    .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext.name, 42);
                ffi::reflection::TypeAttrDef<prim::MulObj>()
                    .attr(type_attr::kStructuralMutateWithConstraintContext.name,
                          static_cast<void*>(nullptr))
                    .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext.name,
                          static_cast<void*>(nullptr));
                for (Mode mode : {Mode::kDisallow, Mode::kAllow}) {
                  for (bool null_hook : {false, true}) {
                    auto mutator = ffi::make_object<StructuralMutatorWithConstraintContextObj>();
                    PrimExpr root = null_hook ? PrimExpr(prim::Mul(I(1), I(2)))
                                              : PrimExpr(prim::Sub(I(1), I(2)));
                    auto result = mutator->MutateExpected(root, mode);
                    if (!result.is_err() || result.error().kind() != "TypeError") std::_Exit(7);
                  }
                }
                std::_Exit(0);
              }()),
              testing::ExitedWithCode(0), "");
}

TEST(ConstraintTraversal, RegionDefinitionsRetainRemapsUnderInplaceMutation) {
  PrimVar thread("thread", PrimType::Int(32));
  SeqStmt root({RegionStmt(Op::Get("tirx.launch_thread"), {StringImm("threadIdx.x"), I(8)},
                           {thread}, DictAttrs(), SeqStmt({Evaluate(thread)}))});
  const auto* identity = root.get();
  auto mutator = MakeMutator(
      [](const VarObj* node, StructuralMutatorWithConstraintContextObj* mutator,
         Mode mode) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (mutator->def_region_kind() == kTVMFFIDefRegionKindSimple) {
          return ffi::Any(Var(node->name + "_renamed", node->ty));
        }
        return mutator->DefaultMutateExpected(node, mode);
      },
      [](const EvaluateObj* node, StructuralMutatorWithConstraintContextObj* mutator, Mode mode) {
        auto result = mutator->DefaultMutateExpected(node, mode).value();
        if (!mutator->constraints->ToArray().empty()) {
          Stmt stmt =
              result.IsUnchanged() ? ffi::GetRef<Stmt>(node) : ffi::AnyView(result).cast<Stmt>();
          EXPECT_TRUE(stmt.as<EvaluateObj>()->value.same_as(
              mutator->constraints->ToArray()[0].as<sym::RangeConstraintObj>()->var));
        }
        return result;
      });
  auto mapped = mutator->MutateExpected(root, Mode::kAllow).value();
  SeqStmt result = std::move(mapped).ValueOrUnchanged(root).cast<SeqStmt>();
  EXPECT_EQ(result.get(), identity);
  const auto* region = result->seq[0].as<RegionStmtObj>();
  EXPECT_FALSE(region->body_params[0].same_as(thread));
  EXPECT_TRUE(region->body->seq[0].as<EvaluateObj>()->value.same_as(region->body_params[0]));
}

TEST(ConstraintTraversal, PostReplacementAndExpandedSequence) {
  PrimVar x("x", PrimType::Int(32)), y("y", PrimType::Int(32));
  SeqStmt root({Bind(x, I(1)), Evaluate(x)});
  auto renamed =
      StructuralMapWithConstraintContext<ffi::WalkOrder::kPostOrder>(
          root, [&](const BindObj*, const Facts&) -> Stmt { return Bind(y, I(2)); },
          [&](const EvaluateObj*, const Facts& facts) {
            EXPECT_EQ(facts->ToArray().size(), 1);
            EXPECT_TRUE(facts->ToArray()[0].as<sym::BindingConstraintObj>()->var.same_as(y));
            return ffi::Unchanged();
          })
          .cast<SeqStmt>();
  EXPECT_TRUE(renamed->seq[1].as<EvaluateObj>()->value.same_as(y));
  auto expanded =
      StructuralMapWithConstraintContext<ffi::WalkOrder::kPostOrder>(
          root,
          [&](const BindObj*, const Facts&) -> Stmt {
            return SeqStmt({Bind(x, I(3)), AssertStmt(x > I(0), StringImm("ValueError"), {})});
          },
          [&](const EvaluateObj*, const Facts& facts) {
            EXPECT_EQ(facts->ToArray().size(), 2);
            EXPECT_EQ(
                facts->ToArray()[0].as<sym::BindingConstraintObj>()->value.as<IntImmObj>()->value,
                3);
            return ffi::Unchanged();
          })
          .cast<SeqStmt>();
  EXPECT_EQ(expanded->seq.size(), 3);
}

TEST(ConstraintTraversal, LazyArgumentsRespectArrayOwnership) {
  for (int ownership : {0, 1, 2}) {
    Call root(PrimType::Int(32), prim::if_then_else_op(), {IntImm::Bool(true), I(1), I(2)});
    ffi::Optional<ffi::Array<Expr>> shared_args;
    if (ownership == 1) shared_args = root->args;
    std::vector<Mode> modes;
    auto mutator =
        MakeMutator([&](const IntImmObj* node, StructuralMutatorWithConstraintContextObj*,
                        Mode mode) -> ffi::UnchangedOr<PrimExpr> {
          modes.push_back(mode);
          if (mode == Mode::kAllow) {
            const_cast<IntImmObj*>(node)->value = 0;
            return ffi::Unchanged();
          }
          return IntImm(node->ty.as_or_throw<PrimType>(), 0);
        });
    auto mapped = mutator->MutateExpected(root, ownership == 2 ? Mode::kDisallow : Mode::kAllow);
    ASSERT_FALSE(mapped.is_err());
    auto result = std::move(mapped).value().ValueOrUnchanged(root).cast<Call>();
    ASSERT_EQ(modes.size(), 3);
    for (Mode mode : modes) {
      EXPECT_EQ(mode, ownership == 0 ? Mode::kAllow : Mode::kDisallow);
    }
    EXPECT_EQ(result->args[2].as<IntImmObj>()->value, 0);
    if (shared_args) {
      EXPECT_EQ(shared_args.value()[2].as<IntImmObj>()->value, 2);
    }
    if (ownership == 2) {
      EXPECT_EQ(root->args[2].as<IntImmObj>()->value, 2);
    }
  }
}

TEST(ConstraintTraversal, InvalidLazyConditionsReturnErrors) {
  for (Mode mode : {Mode::kDisallow, Mode::kAllow}) {
    for (Expr replacement : ffi::Array<Expr>{
             StringImm("invalid"), I(1),
             Call(PrimType::Bool(), prim::likely_op(), {StringImm("invalid")}),
             PrimVar("vector", PrimType::Bool(4)),
             Call(PrimType::Bool(), prim::likely_op(), {PrimVar("vector", PrimType::Bool(4))})}) {
      PrimVar condition("condition", PrimType::Bool());
      PrimExpr root = if_then_else(condition, I(1), I(2));
      Mode call_mode = Mode::kDisallow;
      int branch_visits = 0;
      auto mutator = MakeMutator(
          [&](const VarObj*, StructuralMutatorWithConstraintContextObj*) { return replacement; },
          [&](const CallObj* node, StructuralMutatorWithConstraintContextObj* mutator, Mode mode) {
            call_mode = mode;
            return mutator->DefaultMutateExpected(node, mode);
          },
          [&](const IntImmObj*, StructuralMutatorWithConstraintContextObj*) {
            ++branch_visits;
            return ffi::Unchanged();
          });
      auto result = mutator->MutateExpected(root, mode);
      ASSERT_TRUE(result.is_err());
      EXPECT_EQ(result.error().kind(), "TypeError");
      EXPECT_NE(result.error().message().find("boolean primitive expression"), std::string::npos);
      EXPECT_EQ(call_mode, mode);
      EXPECT_EQ(branch_visits, 0);
    }
  }
}

TEST(ConstraintTraversal, MutationErrorPropagation) {
  PrimVar x("x", PrimType::Int(32));
  SeqStmt root(
      {Bind(x, I(1)), If(x > I(0), SeqStmt({Evaluate(prim::Add(I(2), I(4)))})), Evaluate(I(3))});
  ffi::Error expected_error("ValueError", "expected failure", "");
  auto map_error = StructuralMapWithConstraintContextExpected<ffi::WalkOrder::kPostOrder>(
      root, [&](const EvaluateObj*, const Facts&) -> ffi::Expected<ffi::Any> {
        return ffi::Unexpected(expected_error);
      });
  ASSERT_TRUE(map_error.is_err());
  EXPECT_TRUE(map_error.error().same_as(expected_error));

  auto mutator =
      MakeMutator([](const IntImmObj* node,
                     StructuralMutatorWithConstraintContextObj*) -> ffi::UnchangedOr<PrimExpr> {
        if (node->value == 2) throw ffi::Error("ValueError", "callback failure", "");
        return ffi::Unchanged();
      });
  auto failure = mutator->MutateExpected(root);
  ASSERT_TRUE(failure.is_err());
  auto context = ffi::VisitErrorContext::TryGetFromError(failure.error());
  ASSERT_TRUE(context.has_value());
  EXPECT_EQ(context.value()->reverse_visit_pattern.size(), 6);
}
}  // namespace
}  // namespace tvm::tirx
