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

#include <tvm/ir/prim/op.h>
#include <tvm/sym/iter_affine_map.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor_with_constraint_context.h>

#include "../../ir/expr_functor_impl.h"

namespace tvm {
namespace tirx {
void StmtExprMutatorWithConstraintContext::AddParamConstraints(const Function& func) {
  for (const Var& param : func->params) {
    if (!param->ty.as<TensorTypeObj>()) continue;
    for (PrimExpr shape : param.as_or_throw<TensorVar>().shape()) {
      constraints->Append(sym::PredicateConstraint(shape >= prim::MakeConst(shape.ty(), 0)));
    }
  }
}

ffi::Array<PrimExpr> StmtExprMutatorWithConstraintContext::IterMapSimplifyWithContext(
    const ffi::Array<PrimExpr>& indices, bool non_trivial_only) {
  // The collector is the single source of scope truth. Reconstruct this optional
  // iterator view only when layout analysis needs it, rather than retaining a
  // second map whose entries could outlive a branch or a remapped definition.
  ffi::Map<PrimVar, Range> iter_vars;
  PrimExpr predicate = IntImm::Bool(true);
  constraints->ForEach([&](const sym::Constraint& constraint) {
    if (const auto* range = constraint.as<sym::RangeConstraintObj>()) {
      if (SideEffect(range->range->begin) > CallEffectKind::kPure ||
          SideEffect(range->range->extent) > CallEffectKind::kPure)
        return;
      if (auto var = range->var.as<PrimVar>()) iter_vars.Set(*var, range->range);
    }
  });
  constraints->ForEach([&](const sym::Constraint& constraint) {
    if (const auto* fact = constraint.as<sym::PredicateConstraintObj>()) {
      if (SideEffect(fact->condition) > CallEffectKind::kPure) return;
      PrimExpr condition = fact->condition;
      PrimType type = condition.ty();
      if (!type.IsScalar()) return;
      if (!type.MatchesCode(DLDataTypeCode::kDLBool)) {
        if (!(type.MatchesCode(DLDataTypeCode::kDLInt) ||
              type.MatchesCode(DLDataTypeCode::kDLUInt)))
          return;
        // The collector keeps scalar integer guards in their original form;
        // this boolean iterator predicate must express their nonzero truth.
        condition = condition != IntImm(type, 0);
      }
      auto uses_iterator = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
          condition, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
            if (auto prim_var = var.as<PrimVar>(); prim_var && iter_vars.count(*prim_var)) {
              return ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var));
            }
            return ffi::WalkResult::Advance();
          });
      if (uses_iterator.has_value()) predicate = predicate && condition;
    }
  });
  auto analyzer = ffi::GetRef<sym::Analyzer>(GetSyncedAnalyzer());
  auto simplified =
      sym::IterMapSimplify(indices, iter_vars, predicate, sym::IterMapLevel::Surjective, analyzer);
  if (non_trivial_only) {
    for (size_t i = 0; i < indices.size(); ++i) {
      if (simplified[i].as<IntImmObj>() && indices[i].as<PrimVar>()) {
        simplified.Set(i, indices[i]);
      }
    }
  }
  return simplified;
}

ffi::Optional<ffi::VisitInterrupt> ObjectVisitorWithConstraintContext::Visit(ffi::AnyView value) {
  try {
    if (native_vtable_->CanDispatch(value))
      return (*native_vtable_)(value.cast<const ffi::Object*>(), this);
    return DefaultVisitExpected(value).value();
  } catch (ffi::Error& error) {
    ffi::detail::UpdateVisitErrorContext(error, value);
    throw;
  }
}

ffi::UnchangedOr<ffi::Any> ObjectMutatorWithConstraintContext::Mutate(ffi::AnyView value,
                                                                      ffi::InplaceMode mode) {
  const auto* object = value.as<ffi::Object>();
  if (!object || !object->unique()) mode = ffi::InplaceMode::kDisallow;
  if (native_vtable_->CanDispatch(value)) {
    try {
      return (*native_vtable_)(object, this, mode);
    } catch (ffi::Error& error) {
      // Default mutation already names the current node. A user override may throw
      // before delegating; attach its frame without duplicating default descent's frame.
      auto context = ffi::VisitErrorContext::TryGetFromError(error);
      if (!context.has_value() || context.value()->reverse_visit_pattern.empty() ||
          context.value()->reverse_visit_pattern.back().get() != object) {
        ffi::detail::UpdateVisitErrorContext(error, value);
      }
      throw;
    }
  }
  return DefaultMutateExpected(value, mode).value();
}

// Apply F(Node, Result) to nodes whose scope, definition, or metadata rules require
// structural descent instead of the shared expression defaults.
#define TVM_TIRX_CONSTRAINT_STRUCTURAL_NODES(F) \
  F(LambdaExprObj, Expr)                        \
  F(VarObj, Expr)                               \
  F(CallObj, Expr)                              \
  F(DataTypeImmObj, Expr)                       \
  F(StringImmObj, Expr)                         \
  F(prim::LetObj, PrimExpr)                     \
  F(BindObj, Stmt)                              \
  F(RegionStmtObj, Stmt)                        \
  F(IfObj, Stmt)                                \
  F(ForObj, Stmt)                               \
  F(WhileObj, Stmt)                             \
  F(ReturnObj, Stmt)                            \
  F(BreakObj, Stmt)                             \
  F(ContinueObj, Stmt)                          \
  F(TensorStoreObj, Stmt)                       \
  F(AssertStmtObj, Stmt)                        \
  F(SeqStmtObj, Stmt)                           \
  F(EvaluateObj, Stmt)

// Register both groups in each native dispatch table; their default bodies differ below.
#define TVM_TIRX_CONSTRAINT_NATIVE_NODES(F) \
  TVM_IR_SHARED_EXPR_NODES(F)               \
  TVM_TIRX_CONSTRAINT_STRUCTURAL_NODES(F)

namespace {
bool HasConstraintVisitHook(int32_t type_index) {
  static ffi::reflection::TypeAttrColumn column(type_attr::kStructuralVisitWithConstraintContext);
  return column[type_index] != nullptr;
}

bool HasConstraintMutateHook(int32_t type_index, ffi::InplaceMode mode) {
  static ffi::reflection::TypeAttrColumn copying(type_attr::kStructuralMutateWithConstraintContext);
  static ffi::reflection::TypeAttrColumn inplace(
      type_attr::kStructuralMaybeInplaceMutateWithConstraintContext);
  return copying[type_index] != nullptr ||
         (mode == ffi::InplaceMode::kAllow && inplace[type_index] != nullptr);
}
}  // namespace

// Generate native hooks that use shared expression defaults only for exact builtin types.
// Late constraint-hook registrations and derived types must retain structural dispatch.
#define SHARED_DEFAULTS(Node, Result)                                                             \
  ffi::Optional<ffi::VisitInterrupt> StmtExprVisitorWithConstraintContext::Visit_(                \
      const Node* node) {                                                                         \
    if (node->type_index() != Node::RuntimeTypeIndex() ||                                         \
        HasConstraintVisitHook(node->type_index())) {                                             \
      return DefaultVisitExpected(node).value();                                                  \
    }                                                                                             \
    return tvm::details::VisitExprDefault(this, node);                                            \
  }                                                                                               \
  ffi::UnchangedOr<Result> StmtExprMutatorWithConstraintContext::Mutate_(const Node* node,        \
                                                                         ffi::InplaceMode mode) { \
    if (node->type_index() != Node::RuntimeTypeIndex() ||                                         \
        HasConstraintMutateHook(node->type_index(), mode)) {                                      \
      return DefaultMutateExpected(node, mode)                                                    \
          .value()                                                                                \
          .template as_or_throw<ffi::UnchangedOr<Result>>();                                      \
    }                                                                                             \
    return tvm::details::MutateExprDefault(this, node, mode);                                     \
  }
TVM_IR_SHARED_EXPR_NODES(SHARED_DEFAULTS)
#undef SHARED_DEFAULTS

// Delegate scope-sensitive traversal, including Call/Region Op hooks, to the structural engine.
#define DEFAULT_HOOKS(Node, Result)                                                               \
  ffi::Optional<ffi::VisitInterrupt> StmtExprVisitorWithConstraintContext::Visit_(                \
      const Node* node) {                                                                         \
    return DefaultVisitExpected(node).value();                                                    \
  }                                                                                               \
  ffi::UnchangedOr<Result> StmtExprMutatorWithConstraintContext::Mutate_(const Node* node,        \
                                                                         ffi::InplaceMode mode) { \
    return DefaultMutateExpected(node, mode)                                                      \
        .value()                                                                                  \
        .template as_or_throw<ffi::UnchangedOr<Result>>();                                        \
  }
TVM_TIRX_CONSTRAINT_STRUCTURAL_NODES(DEFAULT_HOOKS)
#undef DEFAULT_HOOKS

void StmtExprVisitorWithConstraintContext::InitVTable(VTable* table) {
  ObjectVisitorWithConstraintContext::InitVTable(table);
#define REGISTER_NODE(Node, Result) SetDispatch<StmtExprVisitorWithConstraintContext, Node>(table);
  TVM_TIRX_CONSTRAINT_NATIVE_NODES(REGISTER_NODE)
#undef REGISTER_NODE
}

const StmtExprVisitorWithConstraintContext::VTable*
StmtExprVisitorWithConstraintContext::GlobalVTable() {
  static const VTable table = [] {
    VTable table;
    InitVTable(&table);
    table.Finalize();
    return table;
  }();
  return &table;
}

const ObjectVisitorWithConstraintContext::VTable*
ObjectVisitorWithConstraintContext::GlobalVTable() {
  static const VTable table = [] {
    VTable table;
    InitVTable(&table);
    table.Finalize();
    return table;
  }();
  return &table;
}

const ffi::StructuralVisitorVTable* ObjectVisitorWithConstraintContext::StructuralVTable() {
  static const ffi::StructuralVisitorVTable table{
      [](ffi::StructuralVisitorObj* self, ffi::AnyView value) noexcept {
        return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
            static_cast<ObjectVisitorWithConstraintContext*>(self)->VisitExpected(value));
      }};
  return &table;
}

void StmtExprMutatorWithConstraintContext::InitVTable(VTable* table) {
  ObjectMutatorWithConstraintContext::InitVTable(table);
#define REGISTER_NODE(Node, Result) SetDispatch<StmtExprMutatorWithConstraintContext, Node>(table);
  TVM_TIRX_CONSTRAINT_NATIVE_NODES(REGISTER_NODE)
#undef REGISTER_NODE
}

const StmtExprMutatorWithConstraintContext::VTable*
StmtExprMutatorWithConstraintContext::GlobalVTable() {
  static const VTable table = [] {
    VTable table;
    InitVTable(&table);
    table.Finalize();
    return table;
  }();
  return &table;
}
#undef TVM_TIRX_CONSTRAINT_NATIVE_NODES
#undef TVM_TIRX_CONSTRAINT_STRUCTURAL_NODES

const ObjectMutatorWithConstraintContext::VTable*
ObjectMutatorWithConstraintContext::GlobalVTable() {
  static const VTable table = [] {
    VTable table;
    InitVTable(&table);
    table.Finalize();
    return table;
  }();
  return &table;
}

const ffi::StructuralMutatorVTable* ObjectMutatorWithConstraintContext::StructuralVTable() {
  static const ffi::StructuralMutatorVTable table{
      [](ffi::StructuralMutatorObj* self, ffi::AnyView value) noexcept {
        return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
            static_cast<ObjectMutatorWithConstraintContext*>(self)->MutateExpected(
                value, ffi::InplaceMode::kDisallow));
      },
      [](ffi::StructuralMutatorObj* self, ffi::AnyView value) noexcept {
        return ffi::detail::ExpectedUnsafe::MoveToTVMFFIAny(
            static_cast<ObjectMutatorWithConstraintContext*>(self)->MutateExpected(
                value, ffi::InplaceMode::kAllow));
      },
      DispatchVarRemapGet, DispatchVarRemapSet};
  return &table;
}
}  // namespace tirx
}  // namespace tvm
