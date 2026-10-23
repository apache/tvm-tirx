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
/*!
 * \file stmt_functor.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "../../ir/seq_stmt_mutate.h"
#include "data_type_rewriter.h"

namespace tvm {
namespace tirx {

void StmtExprVisitor::InitVTable(VTable* vtable) {
  tvm::ExprVisitor::InitVTable(vtable);
  SetDispatch<StmtExprVisitor, BindObj>(vtable);
  SetDispatch<StmtExprVisitor, RegionStmtObj>(vtable);
  SetDispatch<StmtExprVisitor, IfObj>(vtable);
  SetDispatch<StmtExprVisitor, ForObj>(vtable);
  SetDispatch<StmtExprVisitor, WhileObj>(vtable);
  SetDispatch<StmtExprVisitor, ReturnObj>(vtable);
  SetDispatch<StmtExprVisitor, BreakObj>(vtable);
  SetDispatch<StmtExprVisitor, ContinueObj>(vtable);
  SetDispatch<StmtExprVisitor, TensorStoreObj>(vtable);
  SetDispatch<StmtExprVisitor, AssertStmtObj>(vtable);
  SetDispatch<StmtExprVisitor, SeqStmtObj>(vtable);
  SetDispatch<StmtExprVisitor, EvaluateObj>(vtable);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const VarObj* op) {
  // Types belong to definitions; uses retain the identity introduced in that region.
  if (def_region_kind() == kTVMFFIDefRegionKindNone) return std::nullopt;
  return ExprVisitor::Visit_(op);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const OpaqueExprObj* op) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BreakObj* op) { return std::nullopt; }

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ContinueObj* op) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TensorLoadObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->source));
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TupleObj* op) {
  for (const auto& child : op->fields) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TupleGetItemObj* op) {
  return this->Visit(op->tuple);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::LetObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const CallObj* op) {
  if (op->op.as<OpaqueExprObj>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->op));
  }
  for (const auto& child : op->args) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  for (const auto& type : op->ty_args) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(type));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::RampObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->base));
  return this->Visit(op->stride);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::BroadcastObj* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const prim::ShuffleObj* op) {
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  for (const auto& child : op->vectors) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const BindObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
  return this->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                 [&]() { return this->Visit(op->var); });
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const RegionStmtObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->attrs));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->args));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(op->body_params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->body));
  return this->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                 [&]() { return this->Visit(op->result_vars); });
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ForObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->begin));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->end));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->step));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const WhileObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  return this->Visit(op->body);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const ReturnObj* op) {
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const TensorStoreObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->dest));
  for (const auto& child : op->indices) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return this->Visit(op->value);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const IfObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->then_case));
  if (op->else_case) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->else_case.value()));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const AssertStmtObj* op) {
  // Constant message_parts are intentionally skipped.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->condition));
  return this->Visit(op->error_kind);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SeqStmtObj* op) {
  for (const auto& child : op->seq) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(child));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const EvaluateObj* op) {
  return this->Visit(op->value);
}

void StmtExprMutator::InitVTable(VTable* vtable) {
  tvm::ExprMutator::InitVTable(vtable);
  SetDispatch<StmtExprMutator, BindObj>(vtable);
  SetDispatch<StmtExprMutator, RegionStmtObj>(vtable);
  SetDispatch<StmtExprMutator, IfObj>(vtable);
  SetDispatch<StmtExprMutator, ForObj>(vtable);
  SetDispatch<StmtExprMutator, WhileObj>(vtable);
  SetDispatch<StmtExprMutator, ReturnObj>(vtable);
  SetDispatch<StmtExprMutator, BreakObj>(vtable);
  SetDispatch<StmtExprMutator, ContinueObj>(vtable);
  SetDispatch<StmtExprMutator, TensorStoreObj>(vtable);
  SetDispatch<StmtExprMutator, AssertStmtObj>(vtable);
  SetDispatch<StmtExprMutator, SeqStmtObj>(vtable);
  SetDispatch<StmtExprMutator, EvaluateObj>(vtable);
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const BindObj* op, InplaceMode inplace_mode) {
  Type old_value_type = op->value->ty;
  auto value_u = Mutate(op->value, inplace_mode);
  bool value_changed = !value_u.UnchangedOrSameAs(op->value);
  Expr value = std::move(value_u).ValueOrUnchanged(op->value);
  value_changed |= !old_value_type.same_as(value->ty);
  Type old_var_type = op->var->ty;
  auto var_u = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                 return Mutate(op->var, inplace_mode);
               }).as_or_throw<UnchangedOr<Var>>();
  bool var_changed = !var_u.UnchangedOrSameAs(op->var);
  Var var = std::move(var_u).ValueOrUnchanged(op->var);
  var_changed |= !old_var_type.same_as(var->ty);
  if ((value_changed || var_changed) && !var->ty.same_as(value->ty)) {
    var = var.CopyWithType(value->ty);
  }
  if (!var.same_as(op->var)) VarRemapSet(op->var, var);
  if (value.same_as(op->value) && var.same_as(op->var)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<BindObj*>(op);
    writable->value = std::move(value);
    writable->var = std::move(var);
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<BindObj>(*op);
  copy->value = std::move(value);
  copy->var = std::move(var);
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) {
  auto attrs = Mutate(op->attrs, inplace_mode).as_or_throw<UnchangedOr<DictAttrs>>();
  auto args = Mutate(op->args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  auto body_params = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
    return Mutate(op->body_params, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Var>>>();
  });
  auto body = Mutate(op->body, inplace_mode);
  auto result_vars = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
    return Mutate(op->result_vars, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Var>>>();
  });
  if (attrs.UnchangedOrSameAs(op->attrs) && args.UnchangedOrSameAs(op->args) &&
      body_params.UnchangedOrSameAs(op->body_params) && body.UnchangedOrSameAs(op->body) &&
      result_vars.UnchangedOrSameAs(op->result_vars)) {
    return ffi::Unchanged();
  }
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<RegionStmtObj*>(op);
    if (!attrs.IsUnchanged()) writable->attrs = std::move(attrs).ValueUnchecked();
    if (!args.IsUnchanged()) writable->args = std::move(args).ValueUnchecked();
    if (!body_params.IsUnchanged()) writable->body_params = std::move(body_params).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    if (!result_vars.IsUnchanged()) writable->result_vars = std::move(result_vars).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<RegionStmtObj>(*op);
  if (!attrs.IsUnchanged()) copy->attrs = std::move(attrs).ValueUnchecked();
  if (!args.IsUnchanged()) copy->args = std::move(args).ValueUnchecked();
  if (!body_params.IsUnchanged()) copy->body_params = std::move(body_params).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  if (!result_vars.IsUnchanged()) copy->result_vars = std::move(result_vars).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ForObj* op, InplaceMode inplace_mode) {
  auto begin = Mutate(op->begin, inplace_mode);
  auto end = Mutate(op->end, inplace_mode);
  auto step = Mutate(op->step, inplace_mode);
  auto body = Mutate(op->body, inplace_mode);
  if (begin.UnchangedOrSameAs(op->begin) && end.UnchangedOrSameAs(op->end) &&
      step.UnchangedOrSameAs(op->step) && body.UnchangedOrSameAs(op->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<ForObj*>(op);
    if (!begin.IsUnchanged()) writable->begin = std::move(begin).ValueUnchecked();
    if (!end.IsUnchanged()) writable->end = std::move(end).ValueUnchecked();
    if (!step.IsUnchanged()) writable->step = std::move(step).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<ForObj>(*op);
  if (!begin.IsUnchanged()) copy->begin = std::move(begin).ValueUnchecked();
  if (!end.IsUnchanged()) copy->end = std::move(end).ValueUnchecked();
  if (!step.IsUnchanged()) copy->step = std::move(step).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const WhileObj* op, InplaceMode inplace_mode) {
  auto condition = Mutate(op->condition, inplace_mode);
  auto body = Mutate(op->body, inplace_mode);
  if (condition.UnchangedOrSameAs(op->condition) && body.UnchangedOrSameAs(op->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<WhileObj*>(op);
    if (!condition.IsUnchanged()) writable->condition = std::move(condition).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<WhileObj>(*op);
  if (!condition.IsUnchanged()) copy->condition = std::move(condition).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ReturnObj* op, InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  if (value.UnchangedOrSameAs(op->value)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<ReturnObj*>(op);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<ReturnObj>(*op);
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const IfObj* op, InplaceMode inplace_mode) {
  auto condition = Mutate(op->condition, inplace_mode);
  auto then_case = Mutate(op->then_case, inplace_mode);
  auto else_case = Mutate(op->else_case, inplace_mode);
  if (condition.UnchangedOrSameAs(op->condition) && then_case.UnchangedOrSameAs(op->then_case) &&
      else_case.UnchangedOrSameAs(op->else_case))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<IfObj*>(op);
    if (!condition.IsUnchanged()) writable->condition = std::move(condition).ValueUnchecked();
    if (!then_case.IsUnchanged()) writable->then_case = std::move(then_case).ValueUnchecked();
    if (!else_case.IsUnchanged()) writable->else_case = std::move(else_case).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<IfObj>(*op);
  if (!condition.IsUnchanged()) copy->condition = std::move(condition).ValueUnchecked();
  if (!then_case.IsUnchanged()) copy->then_case = std::move(then_case).ValueUnchecked();
  if (!else_case.IsUnchanged()) copy->else_case = std::move(else_case).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const AssertStmtObj* op, InplaceMode inplace_mode) {
  auto condition = Mutate(op->condition, inplace_mode);
  auto error_kind = Mutate(op->error_kind, inplace_mode).as_or_throw<UnchangedOr<StringImm>>();
  if (condition.UnchangedOrSameAs(op->condition) && error_kind.UnchangedOrSameAs(op->error_kind))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<AssertStmtObj*>(op);
    if (!condition.IsUnchanged()) writable->condition = std::move(condition).ValueUnchecked();
    if (!error_kind.IsUnchanged()) writable->error_kind = std::move(error_kind).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<AssertStmtObj>(*op);
  if (!condition.IsUnchanged()) copy->condition = std::move(condition).ValueUnchecked();
  if (!error_kind.IsUnchanged()) copy->error_kind = std::move(error_kind).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const EvaluateObj* op, InplaceMode inplace_mode) {
  auto value = Mutate(op->value, inplace_mode);
  if (value.UnchangedOrSameAs(op->value)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<EvaluateObj*>(op);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<EvaluateObj>(*op);
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const BreakObj* op, InplaceMode inplace_mode) {
  return ffi::Unchanged();
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const ContinueObj* op, InplaceMode inplace_mode) {
  return ffi::Unchanged();
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) {
  auto dest = Mutate(op->dest, inplace_mode).as_or_throw<UnchangedOr<Expr>>();
  auto indices = Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto value = Mutate(op->value, inplace_mode);
  if (dest.UnchangedOrSameAs(op->dest) && value.UnchangedOrSameAs(op->value) &&
      indices.UnchangedOrSameAs(op->indices))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TensorStoreObj*>(op);
    if (!dest.IsUnchanged()) writable->dest = std::move(dest).ValueUnchecked();
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    if (!indices.IsUnchanged()) writable->indices = std::move(indices).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TensorStoreObj>(*op);
  if (!dest.IsUnchanged()) copy->dest = std::move(dest).ValueUnchecked();
  if (!value.IsUnchanged()) copy->value = std::move(value).ValueUnchecked();
  if (!indices.IsUnchanged()) copy->indices = std::move(indices).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SeqStmtObj* op, InplaceMode inplace_mode) {
  return tvm::detail::MutateSeqStmt(op, inplace_mode,
                                    [this](ffi::AnyView element, InplaceMode mode) {
                                      return Mutate(element, mode).as_or_throw<UnchangedOr<Stmt>>();
                                    });
}

class IRSubstituteWithDataTypeLegalization : public DataTypeLegalizer {
 public:
  IRSubstituteWithDataTypeLegalization(ffi::AnyView root,
                                       const std::function<ffi::Optional<Expr>(const Var&)>& vmap) {
    ffi::Map<Var, bool> visited;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        root, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          if (visited.count(var)) return ffi::WalkResult::Skip();
          visited.Set(var, true);
          if (auto replacement = vmap(var)) VarRemapSet(var, replacement.value());
          return ffi::WalkResult::Advance();
        });
  }

  using DataTypeLegalizer::Mutate;
  using DataTypeLegalizer::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    auto result = StmtExprMutator::Mutate_(op, InplaceMode::kDisallow);
    if (result.UnchangedOrSameAs(ffi::GetRef<PrimExpr>(op))) return ffi::Unchanged();
    auto load = std::move(result).ValueUnchecked().as_or_throw<TensorLoad>();
    if (auto tensor = load->source.as<TensorVar>()) {
      return TensorLoad(tensor.value(), load->indices, load->loc);
    }
    return load;
  }
};

Stmt SubstituteWithDataTypeLegalization(
    Stmt stmt, ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  auto general_vmap = [vmap = std::move(vmap)](const Var& var) -> ffi::Optional<Expr> {
    if (auto replacement = vmap(var)) return Expr(replacement.value());
    return std::nullopt;
  };
  return ffi::make_object<IRSubstituteWithDataTypeLegalization>(stmt, general_vmap)
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(stmt));
}

PrimExpr SubstituteWithDataTypeLegalization(
    PrimExpr expr, ffi::TypedFunction<ffi::Optional<PrimExpr>(const Var&)> vmap) {
  auto general_vmap = [vmap = std::move(vmap)](const Var& var) -> ffi::Optional<Expr> {
    if (auto replacement = vmap(var)) return Expr(replacement.value());
    return std::nullopt;
  };
  return ffi::make_object<IRSubstituteWithDataTypeLegalization>(expr, general_vmap)
      ->Mutate(expr, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(expr));
}

}  // namespace tirx
}  // namespace tvm
