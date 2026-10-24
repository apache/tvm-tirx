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

#include <tvm/tirx/stmt_functor_with_constraint_context.h>

#include "../../ir/expr_functor_impl.h"

namespace tvm {
namespace tirx {
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
