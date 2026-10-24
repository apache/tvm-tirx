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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/structural_mutate_with_constraint_context.h>

#include "../../ir/seq_stmt_mutate.h"

namespace tvm {
namespace tirx {
namespace {
template <typename Traversal>
void RecordStatement(Traversal* traversal, ffi::AnyView value) {
  if (const auto* bind = value.as<BindObj>()) {
    if (auto rhs = bind->value.as<PrimExpr>()) {
      traversal->AppendConstraint(sym::BindingConstraint(bind->var, rhs.value()));
    }
  } else if (const auto* assertion = value.as<AssertStmtObj>()) {
    if (SideEffect(assertion->condition) <= CallEffectKind::kPure) {
      traversal->AppendConstraint(sym::PredicateConstraint(assertion->condition));
    }
  } else if (const auto* seq = value.as<SeqStmtObj>()) {
    // A replacement sequence is flattened by the normal sequence helper. Record its output
    // facts without revisiting its already-transformed children.
    for (const auto& stmt : seq->seq) {
      RecordStatement(traversal, stmt);
    }
  }
}

template <typename Traversal>
ffi::Expected<void> AddLoopRange(Traversal* traversal, PrimVar var, PrimExpr begin, PrimExpr end,
                                 PrimExpr step) {
  try {
    const auto* stride = step.as<IntImmObj>();
    if (stride && stride->value > 0) {
      traversal->AppendConstraint(sym::RangeConstraint(var, Range(begin, end)));
    }
    return {};
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> IfVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  const IfObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  auto result_then_case =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        visitor->AppendConstraint(sym::PredicateConstraint(self->condition));
        return visitor->VisitExpected(self->then_case);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result_then_case);
  auto result_else_case =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        visitor->AppendConstraint(sym::PredicateConstraint(prim::Not(self->condition)));
        return visitor->VisitExpected(self->else_case);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result_else_case);
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  const IfObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  PrimExpr condition =
      (mapped_condition.IsUnchanged() ? self->condition
                                      : ffi::AnyView(mapped_condition).cast<PrimExpr>());
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Stmt>, mapped_then_case,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(condition));
        return mutator->MutateExpected(self->then_case);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_else_case,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(prim::Not(condition)));
        return mutator->MutateExpected(self->else_case);
      }));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_then_case.UnchangedOrSameAs(self->then_case) &&
      (mapped_else_case.IsUnchanged() || ffi::AnyView(mapped_else_case).same_as(self->else_case))) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<IfObj> copy = ffi::make_object<IfObj>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  if (!mapped_then_case.IsUnchanged())
    copy->then_case = SeqStmt(std::move(mapped_then_case).ValueUnchecked());
  if (!mapped_else_case.IsUnchanged()) {
    auto replacement = std::move(mapped_else_case).ValueUnchecked();
    copy->else_case = replacement.has_value()
                          ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                          : std::nullopt;
  }
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  IfObj* self = const_cast<IfObj*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_condition,
      mutator->MutateExpected(self->condition, ffi::InplaceMode::kAllow));
  PrimExpr condition =
      (mapped_condition.IsUnchanged() ? self->condition
                                      : ffi::AnyView(mapped_condition).cast<PrimExpr>());
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Stmt>, mapped_then_case,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(condition));
        return mutator->MutateExpected(self->then_case, ffi::InplaceMode::kAllow);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_else_case,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(prim::Not(condition)));
        return mutator->MutateExpected(self->else_case, ffi::InplaceMode::kAllow);
      }));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_then_case.UnchangedOrSameAs(self->then_case) &&
      (mapped_else_case.IsUnchanged() || ffi::AnyView(mapped_else_case).same_as(self->else_case))) {
    return ffi::Unchanged();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  if (!mapped_then_case.IsUnchanged())
    self->then_case = SeqStmt(std::move(mapped_then_case).ValueUnchecked());
  if (!mapped_else_case.IsUnchanged()) {
    auto replacement = std::move(mapped_else_case).ValueUnchecked();
    self->else_case = replacement.has_value()
                          ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                          : std::nullopt;
  }
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ForVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  const ForObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->begin));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->end));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->step));
  auto result_body =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (auto result = AddLoopRange(visitor, self->var, self->begin, self->end, self->step);
            result.is_err()) {
          return ffi::Unexpected(std::move(result).error());
        }
        return visitor->VisitExpected(self->body);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result_body);
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ForMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  const ForObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForObj>(value);
  PrimVar old_var = self->var;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimVar>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var);
                                    }));
  PrimVar var = (mapped_var.IsUnchanged() ? self->var : ffi::AnyView(mapped_var).cast<PrimVar>());
  if (!var.same_as(old_var)) {
    auto remap = mutator->VarRemapSetExpected(old_var, var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_begin,
                                    mutator->MutateExpected(self->begin));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_end,
                                    mutator->MutateExpected(self->end));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_step,
                                    mutator->MutateExpected(self->step));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Stmt>, mapped_body,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto result = AddLoopRange(
                mutator, var,
                (mapped_begin.IsUnchanged() ? self->begin
                                            : ffi::AnyView(mapped_begin).cast<PrimExpr>()),
                (mapped_end.IsUnchanged() ? self->end : ffi::AnyView(mapped_end).cast<PrimExpr>()),
                (mapped_step.IsUnchanged() ? self->step
                                           : ffi::AnyView(mapped_step).cast<PrimExpr>()));
            result.is_err()) {
          return ffi::Unexpected(std::move(result).error());
        }
        return mutator->MutateExpected(self->body);
      }));
  if (mapped_var.UnchangedOrSameAs(self->var) && mapped_begin.UnchangedOrSameAs(self->begin) &&
      mapped_end.UnchangedOrSameAs(self->end) && mapped_body.UnchangedOrSameAs(self->body) &&
      mapped_step.UnchangedOrSameAs(self->step)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<ForObj> copy = ffi::make_object<ForObj>(*self);
  copy->var = std::move(mapped_var).ValueOrUnchanged(std::move(copy->var));
  copy->begin = std::move(mapped_begin).ValueOrUnchanged(std::move(copy->begin));
  copy->end = std::move(mapped_end).ValueOrUnchanged(std::move(copy->end));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  copy->step = std::move(mapped_step).ValueOrUnchanged(std::move(copy->step));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ForMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  // skips: kind and constant annotations; unlike SBlock annotations, these carry no expressions.
  ForObj* self = const_cast<ForObj*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForObj>(value));
  PrimVar old_var = self->var;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimVar>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  PrimVar var = (mapped_var.IsUnchanged() ? self->var : ffi::AnyView(mapped_var).cast<PrimVar>());
  if (!var.same_as(old_var)) {
    auto remap = mutator->VarRemapSetExpected(old_var, var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_begin,
                                    mutator->MutateExpected(self->begin, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_end,
                                    mutator->MutateExpected(self->end, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_step,
                                    mutator->MutateExpected(self->step, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Stmt>, mapped_body,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (auto result = AddLoopRange(
                mutator, var,
                (mapped_begin.IsUnchanged() ? self->begin
                                            : ffi::AnyView(mapped_begin).cast<PrimExpr>()),
                (mapped_end.IsUnchanged() ? self->end : ffi::AnyView(mapped_end).cast<PrimExpr>()),
                (mapped_step.IsUnchanged() ? self->step
                                           : ffi::AnyView(mapped_step).cast<PrimExpr>()));
            result.is_err()) {
          return ffi::Unexpected(std::move(result).error());
        }
        return mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow);
      }));
  if (mapped_var.UnchangedOrSameAs(self->var) && mapped_begin.UnchangedOrSameAs(self->begin) &&
      mapped_end.UnchangedOrSameAs(self->end) && mapped_body.UnchangedOrSameAs(self->body) &&
      mapped_step.UnchangedOrSameAs(self->step)) {
    return ffi::Unchanged();
  }
  if (!mapped_var.IsUnchanged()) self->var = std::move(mapped_var).ValueUnchecked();
  if (!mapped_begin.IsUnchanged()) self->begin = std::move(mapped_begin).ValueUnchecked();
  if (!mapped_end.IsUnchanged()) self->end = std::move(mapped_end).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
  if (!mapped_step.IsUnchanged()) self->step = std::move(mapped_step).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> WhileVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  const WhileObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  auto result_body =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        visitor->AppendConstraint(sym::PredicateConstraint(self->condition));
        return visitor->VisitExpected(self->body);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result_body);
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> WhileMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  const WhileObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Stmt>, mapped_body,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(
            mapped_condition.IsUnchanged() ? self->condition
                                           : ffi::AnyView(mapped_condition).cast<PrimExpr>()));
        return mutator->MutateExpected(self->body);
      }));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<WhileObj> copy = ffi::make_object<WhileObj>(*self);
  copy->condition = std::move(mapped_condition).ValueOrUnchanged(std::move(copy->condition));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> WhileMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  WhileObj* self = const_cast<WhileObj*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const WhileObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_condition,
      mutator->MutateExpected(self->condition, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Stmt>, mapped_body,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(
            mapped_condition.IsUnchanged() ? self->condition
                                           : ffi::AnyView(mapped_condition).cast<PrimExpr>()));
        return mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow);
      }));
  if (mapped_condition.UnchangedOrSameAs(self->condition) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  if (!mapped_condition.IsUnchanged())
    self->condition = std::move(mapped_condition).ValueUnchecked();
  if (!mapped_body.IsUnchanged()) self->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> LetVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const prim::LetObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::LetObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  auto result_body =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        visitor->AppendConstraint(sym::BindingConstraint(self->var, self->value));
        return visitor->VisitExpected(self->body);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result_body);
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> LetMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  const prim::LetObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::LetObj>(value);
  Type old_value_type = self->value->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value));
  bool value_changed = !mapped_value_u.UnchangedOrSameAs(self->value);
  PrimExpr new_value = std::move(mapped_value_u).ValueOrUnchanged(self->value);
  value_changed |= !old_value_type.same_as(new_value->ty);
  Type old_var_type = self->var->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var_u,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var);
                                    }));
  bool var_changed = !mapped_var_u.UnchangedOrSameAs(self->var);
  Var new_var = std::move(mapped_var_u).ValueOrUnchanged(self->var);
  var_changed |= !old_var_type.same_as(new_var->ty);
  if ((value_changed || var_changed) && !new_var->ty.same_as(new_value->ty)) {
    new_var = new_var.CopyWithType(new_value->ty);
  }
  if (!new_var.same_as(self->var)) {
    auto remap = mutator->VarRemapSetExpected(self->var, new_var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_body_u,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::BindingConstraint(new_var, new_value));
        return mutator->MutateExpected(self->body);
      }));
  PrimExpr new_body = std::move(mapped_body_u).ValueOrUnchanged(self->body);
  if (new_var.same_as(self->var) && new_value.same_as(self->value) &&
      new_body.same_as(self->body) && new_body->ty.same_as(self->ty)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<prim::LetObj> copy = ffi::make_object<prim::LetObj>(*self);
  copy->var = std::move(new_var);
  copy->value = std::move(new_value);
  copy->body = std::move(new_body);
  copy->ty = copy->body->ty;
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> LetMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  prim::LetObj* self = const_cast<prim::LetObj*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::LetObj>(value));
  Type old_value_type = self->value->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  bool value_changed = !mapped_value_u.UnchangedOrSameAs(self->value);
  PrimExpr new_value = std::move(mapped_value_u).ValueOrUnchanged(self->value);
  value_changed |= !old_value_type.same_as(new_value->ty);
  Type old_var_type = self->var->ty;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Var>, mapped_var_u,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  bool var_changed = !mapped_var_u.UnchangedOrSameAs(self->var);
  Var new_var = std::move(mapped_var_u).ValueOrUnchanged(self->var);
  var_changed |= !old_var_type.same_as(new_var->ty);
  if ((value_changed || var_changed) && !new_var->ty.same_as(new_value->ty)) {
    new_var = new_var.CopyWithType(new_value->ty);
  }
  if (!new_var.same_as(self->var)) {
    auto remap = mutator->VarRemapSetExpected(self->var, new_var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_body_u,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::BindingConstraint(new_var, new_value));
        return mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow);
      }));
  PrimExpr new_body = std::move(mapped_body_u).ValueOrUnchanged(self->body);
  if (new_var.same_as(self->var) && new_value.same_as(self->value) &&
      new_body.same_as(self->body) && new_body->ty.same_as(self->ty)) {
    return ffi::Unchanged();
  }
  self->var = std::move(new_var);
  self->value = std::move(new_value);
  self->body = std::move(new_body);
  self->ty = self->body->ty;
  return ffi::Unchanged();
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>> InvokeMutateHook(
    FStructuralMutateWithConstraintContext hook, StructuralMutatorWithConstraintContextObj* mutator,
    ffi::AnyView value) noexcept {
  // UnchangedOr uses the structural result ABI; packed callbacks return an Any sentinel/value.
  ffi::AnyView attr(hook);
  TVMFFIAny raw;
  if (attr.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
    raw = reinterpret_cast<FStructuralMutateWithConstraintContext::ABIType>(attr.cast<void*>())(
        mutator, value);
  } else {
    raw = ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        attr.cast<ffi::Function>().CallExpected<ffi::Any>(mutator, value));
  }
  return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::UnchangedOr<ffi::Any>>(raw);
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> CallVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  const CallObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(value);
  try {
    static auto hooks = Op::GetAttrMap(op_attr::kStructuralVisitWithConstraintContext);
    if (auto op = self->op.as<Op>(); op && hooks.count(*op)) {
      return hooks[*op].CallExpected(visitor, value);
    }

    // A PrimType carries only a dtype, so it has nothing to visit.  Broad callbacks do not see this
    // skipped field; dynamically typed Call results still descend through the Type value.
    if (!self->ty.as<PrimTypeObj>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
    }
    // An Op is an interned registry singleton, so it has nothing to visit.  Broad callbacks do not
    // see this skipped field; function-valued Call operators still descend through the Expr value.
    if (!self->op.as<OpObj>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->op));
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
    // An empty ty_args has no element to traverse.  Broad callbacks do not see the empty container;
    // nonempty type arguments retain normal container descent and callback behavior.
    if (!self->ty_args.empty()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty_args));
    }
    return std::nullopt;
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> CallMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  const CallObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(value);
  try {
    static auto copying = Op::GetAttrMap(op_attr::kStructuralMutateWithConstraintContext);
    if (auto op = self->op.as<Op>()) {
      if (copying.count(*op)) return InvokeMutateHook(copying[*op], mutator, value);
    }

    // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
    // this skipped field; dynamically typed Call results still descend through the Type value.
    ffi::UnchangedOr<Type> mapped_ty_u = ffi::Unchanged();
    if (!self->ty.as<PrimTypeObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, descended_ty_u,
                                        mutator->MutateExpected(self->ty));
      mapped_ty_u = std::move(descended_ty_u);
    }
    // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
    // not see this skipped field; function-valued Call operators still descend through the Expr.
    ffi::UnchangedOr<Expr> mapped_op_u = ffi::Unchanged();
    if (!self->op.as<OpObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, descended_op_u,
                                        mutator->MutateExpected(self->op));
      mapped_op_u = std::move(descended_op_u);
    }
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args_u,
                                      mutator->MutateExpected(self->args));
    // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
    // container; nonempty type arguments retain normal container descent and callback behavior.
    ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args_u = ffi::Unchanged();
    if (!self->ty_args.empty()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, descended_ty_args_u,
                                        mutator->MutateExpected(self->ty_args));
      mapped_ty_args_u = std::move(descended_ty_args_u);
    }
    if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_op_u.UnchangedOrSameAs(self->op) &&
        mapped_args_u.UnchangedOrSameAs(self->args) &&
        mapped_ty_args_u.UnchangedOrSameAs(self->ty_args)) {
      return ffi::Unchanged();
    }
    ffi::ObjectPtr<CallObj> copy = ffi::make_object<CallObj>(*self);
    if (!mapped_ty_u.IsUnchanged()) copy->ty = std::move(mapped_ty_u).ValueUnchecked();
    if (!mapped_op_u.IsUnchanged()) copy->op = std::move(mapped_op_u).ValueUnchecked();
    if (!mapped_args_u.IsUnchanged()) copy->args = std::move(mapped_args_u).ValueUnchecked();
    if (!mapped_ty_args_u.IsUnchanged())
      copy->ty_args = std::move(mapped_ty_args_u).ValueUnchecked();
    return ffi::Any(std::move(copy));
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> CallMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs, constant metadata left untouched like the classic Expr functors.
  CallObj* self = const_cast<CallObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(value));
  try {
    static auto copying = Op::GetAttrMap(op_attr::kStructuralMutateWithConstraintContext);
    static auto inplace =
        Op::GetAttrMap(op_attr::kStructuralMaybeInplaceMutateWithConstraintContext);
    if (auto op = self->op.as<Op>()) {
      if (inplace.count(*op)) return InvokeMutateHook(inplace[*op], mutator, value);
      if (copying.count(*op)) return InvokeMutateHook(copying[*op], mutator, value);
    }

    // A PrimType carries only a dtype, so it has nothing to substitute.  Broad callbacks do not see
    // this skipped field; dynamically typed Call results still descend through the Type value.
    ffi::UnchangedOr<Type> mapped_ty_u = ffi::Unchanged();
    if (!self->ty.as<PrimTypeObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
          ffi::UnchangedOr<Type>, descended_ty_u,
          mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
      mapped_ty_u = std::move(descended_ty_u);
    }
    // An Op is an interned registry singleton, so it has nothing to substitute.  Broad callbacks do
    // not see this skipped field; function-valued Call operators still descend through the Expr.
    ffi::UnchangedOr<Expr> mapped_op_u = ffi::Unchanged();
    if (!self->op.as<OpObj>()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
          ffi::UnchangedOr<Expr>, descended_op_u,
          mutator->MutateExpected(self->op, ffi::InplaceMode::kAllow));
      mapped_op_u = std::move(descended_op_u);
    }
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args_u,
        mutator->MutateExpected(self->args, ffi::InplaceMode::kAllow));
    // An empty ty_args has no element to substitute.  Broad callbacks do not see the empty
    // container; nonempty type arguments retain normal container descent and callback behavior.
    ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args_u = ffi::Unchanged();
    if (!self->ty_args.empty()) {
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
          ffi::UnchangedOr<ffi::Array<Type>>, descended_ty_args_u,
          mutator->MutateExpected(self->ty_args, ffi::InplaceMode::kAllow));
      mapped_ty_args_u = std::move(descended_ty_args_u);
    }
    if (mapped_ty_u.UnchangedOrSameAs(self->ty) && mapped_op_u.UnchangedOrSameAs(self->op) &&
        mapped_args_u.UnchangedOrSameAs(self->args) &&
        mapped_ty_args_u.UnchangedOrSameAs(self->ty_args)) {
      return ffi::Unchanged();
    }
    if (!mapped_ty_u.IsUnchanged()) self->ty = std::move(mapped_ty_u).ValueUnchecked();
    if (!mapped_op_u.IsUnchanged()) self->op = std::move(mapped_op_u).ValueUnchecked();
    if (!mapped_args_u.IsUnchanged()) self->args = std::move(mapped_args_u).ValueUnchecked();
    if (!mapped_ty_args_u.IsUnchanged())
      self->ty_args = std::move(mapped_ty_args_u).ValueUnchecked();
    return ffi::Unchanged();
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> RegionStmtVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  const RegionStmtObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtObj>(value);
  try {
    static auto hooks = Op::GetAttrMap(op_attr::kStructuralVisitWithConstraintContext);
    if (auto op = self->op.as<Op>(); op && hooks.count(*op)) {
      return hooks[*op].CallExpected(visitor, value);
    }

    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->attrs));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->body_params); }));
    auto result_body =
        visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          return visitor->VisitExpected(self->body);
        });
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result_body);
    return visitor->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                      [&]() { return visitor->VisitExpected(self->result_vars); });
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> RegionStmtMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  const RegionStmtObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtObj>(value);
  try {
    static auto copying = Op::GetAttrMap(op_attr::kStructuralMutateWithConstraintContext);
    if (auto op = self->op.as<Op>()) {
      if (copying.count(*op)) return InvokeMutateHook(copying[*op], mutator, value);
    }

    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<DictAttrs>, mapped_attrs,
                                      mutator->MutateExpected(self->attrs));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
                                      mutator->MutateExpected(self->args));
    ffi::Array<Var> old_body_params = self->body_params;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_body_params,
                                      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                        return mutator->MutateExpected(self->body_params);
                                      }));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<Stmt>, mapped_body,
        mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          const auto& body_params = mapped_body_params.IsUnchanged()
                                        ? self->body_params
                                        : ffi::AnyView(mapped_body_params).cast<ffi::Array<Var>>();
          if (body_params.size() != old_body_params.size()) {
            return ffi::Unexpected(ffi::Error(
                "ValueError", "A region definition rewrite must preserve the number of parameters",
                ""));
          }
          for (size_t i = 0; i < body_params.size(); ++i) {
            if (!body_params[i].same_as(old_body_params[i])) {
              auto remap = mutator->VarRemapSetExpected(old_body_params[i], body_params[i]);
              TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
            }
          }

          return mutator->MutateExpected(self->body);
        }));
    ffi::Array<Var> old_result_vars = self->result_vars;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_result_vars,
                                      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                        return mutator->MutateExpected(self->result_vars);
                                      }));
    const auto& result_vars = mapped_result_vars.IsUnchanged()
                                  ? self->result_vars
                                  : ffi::AnyView(mapped_result_vars).cast<ffi::Array<Var>>();
    if (result_vars.size() != old_result_vars.size()) {
      return ffi::Unexpected(ffi::Error(
          "ValueError", "A region definition rewrite must preserve the number of results", ""));
    }
    for (size_t i = 0; i < result_vars.size(); ++i) {
      if (!result_vars[i].same_as(old_result_vars[i])) {
        auto remap = mutator->VarRemapSetExpected(old_result_vars[i], result_vars[i]);
        TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
      }
    }
    if (mapped_args.UnchangedOrSameAs(self->args) && mapped_attrs.UnchangedOrSameAs(self->attrs) &&
        mapped_body_params.UnchangedOrSameAs(self->body_params) &&
        mapped_body.UnchangedOrSameAs(self->body) &&
        mapped_result_vars.UnchangedOrSameAs(self->result_vars)) {
      return ffi::Unchanged();
    }
    auto copy = ffi::make_object<RegionStmtObj>(*self);
    if (!mapped_args.IsUnchanged()) copy->args = std::move(mapped_args).ValueUnchecked();
    if (!mapped_attrs.IsUnchanged()) copy->attrs = std::move(mapped_attrs).ValueUnchecked();
    if (!mapped_body_params.IsUnchanged())
      copy->body_params = std::move(mapped_body_params).ValueUnchecked();
    if (!mapped_body.IsUnchanged()) copy->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
    if (!mapped_result_vars.IsUnchanged())
      copy->result_vars = std::move(mapped_result_vars).ValueUnchecked();
    return ffi::Any(std::move(copy));
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> RegionStmtMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  const RegionStmtObj* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtObj>(value);
  try {
    static auto copying = Op::GetAttrMap(op_attr::kStructuralMutateWithConstraintContext);
    static auto inplace =
        Op::GetAttrMap(op_attr::kStructuralMaybeInplaceMutateWithConstraintContext);
    if (auto op = self->op.as<Op>()) {
      if (inplace.count(*op)) return InvokeMutateHook(inplace[*op], mutator, value);
      if (copying.count(*op)) return InvokeMutateHook(copying[*op], mutator, value);
    }

    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<DictAttrs>, mapped_attrs,
        mutator->MutateExpected(self->attrs, ffi::InplaceMode::kAllow));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
        mutator->MutateExpected(self->args, ffi::InplaceMode::kAllow));
    ffi::Array<Var> old_body_params = self->body_params;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_body_params,
                                      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                        return mutator->MutateExpected(self->body_params,
                                                                       ffi::InplaceMode::kAllow);
                                      }));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<Stmt>, mapped_body,
        mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          const auto& body_params = mapped_body_params.IsUnchanged()
                                        ? self->body_params
                                        : ffi::AnyView(mapped_body_params).cast<ffi::Array<Var>>();
          if (body_params.size() != old_body_params.size()) {
            return ffi::Unexpected(ffi::Error(
                "ValueError", "A region definition rewrite must preserve the number of parameters",
                ""));
          }
          for (size_t i = 0; i < body_params.size(); ++i) {
            if (!body_params[i].same_as(old_body_params[i])) {
              auto remap = mutator->VarRemapSetExpected(old_body_params[i], body_params[i]);
              TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
            }
          }

          return mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow);
        }));
    ffi::Array<Var> old_result_vars = self->result_vars;
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_result_vars,
                                      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                        return mutator->MutateExpected(self->result_vars,
                                                                       ffi::InplaceMode::kAllow);
                                      }));
    const auto& result_vars = mapped_result_vars.IsUnchanged()
                                  ? self->result_vars
                                  : ffi::AnyView(mapped_result_vars).cast<ffi::Array<Var>>();
    if (result_vars.size() != old_result_vars.size()) {
      return ffi::Unexpected(ffi::Error(
          "ValueError", "A region definition rewrite must preserve the number of results", ""));
    }
    for (size_t i = 0; i < result_vars.size(); ++i) {
      if (!result_vars[i].same_as(old_result_vars[i])) {
        auto remap = mutator->VarRemapSetExpected(old_result_vars[i], result_vars[i]);
        TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
      }
    }
    if (mapped_args.UnchangedOrSameAs(self->args) && mapped_attrs.UnchangedOrSameAs(self->attrs) &&
        mapped_body_params.UnchangedOrSameAs(self->body_params) &&
        mapped_body.UnchangedOrSameAs(self->body) &&
        mapped_result_vars.UnchangedOrSameAs(self->result_vars)) {
      return ffi::Unchanged();
    }
    auto* copy = const_cast<RegionStmtObj*>(self);
    if (!mapped_args.IsUnchanged()) copy->args = std::move(mapped_args).ValueUnchecked();
    if (!mapped_attrs.IsUnchanged()) copy->attrs = std::move(mapped_attrs).ValueUnchecked();
    if (!mapped_body_params.IsUnchanged())
      copy->body_params = std::move(mapped_body_params).ValueUnchecked();
    if (!mapped_body.IsUnchanged()) copy->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
    if (!mapped_result_vars.IsUnchanged())
      copy->result_vars = std::move(mapped_result_vars).ValueUnchecked();
    return ffi::Unchanged();
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> SeqStmtVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtObj>(value);
  return visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
    for (const auto& stmt : self->seq) {
      auto result = visitor->VisitExpected(stmt);
      if (result.is_err()) return ffi::Unexpected(std::move(result).error());
      if (result.value().has_value()) {
        return result;
      }
      RecordStatement(visitor, stmt);
    }
    return std::nullopt;
  });
}

template <ffi::InplaceMode mode>
TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SeqStmtMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqStmtObj>(value);
  return mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    auto mutate = [mutator](ffi::AnyView stmt,
                            ffi::InplaceMode child_mode) -> ffi::Expected<ffi::UnchangedOr<Stmt>> {
      ffi::Optional<Var> old_var;
      if (const auto* bind = stmt.as<BindObj>()) old_var = bind->var;
      TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped,
                                        mutator->MutateExpected(stmt, child_mode));
      Stmt result = mapped.IsUnchanged() ? stmt.cast<Stmt>() : ffi::AnyView(mapped).cast<Stmt>();
      if (const auto* bind = result.as<BindObj>();
          bind && old_var.has_value() && !old_var.value().same_as(bind->var)) {
        auto remap = mutator->VarRemapSetExpected(old_var.value(), bind->var);
        TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
      }
      RecordStatement(mutator, result);
      return mapped;
    };
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, result,
                                      tvm::detail::MutateSeqStmt(self, mode, mutate));
    if (result.IsUnchanged()) return ffi::Unchanged();
    return ffi::Any(std::move(result).ValueUnchecked());
  });
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<IfObj>()
      .attr(type_attr::kStructuralVisitWithConstraintContext,
            FStructuralVisitWithConstraintContext::FromNative<&IfVisit>())
      .attr(type_attr::kStructuralMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&IfMutate>())
      .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&IfMaybeInplaceMutate>());
  ffi::reflection::TypeAttrDef<ForObj>()
      .attr(type_attr::kStructuralVisitWithConstraintContext,
            FStructuralVisitWithConstraintContext::FromNative<&ForVisit>())
      .attr(type_attr::kStructuralMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&ForMutate>())
      .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&ForMaybeInplaceMutate>());
  ffi::reflection::TypeAttrDef<WhileObj>()
      .attr(type_attr::kStructuralVisitWithConstraintContext,
            FStructuralVisitWithConstraintContext::FromNative<&WhileVisit>())
      .attr(type_attr::kStructuralMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&WhileMutate>())
      .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&WhileMaybeInplaceMutate>());
  ffi::reflection::TypeAttrDef<prim::LetObj>()
      .attr(type_attr::kStructuralVisitWithConstraintContext,
            FStructuralVisitWithConstraintContext::FromNative<&LetVisit>())
      .attr(type_attr::kStructuralMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&LetMutate>())
      .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&LetMaybeInplaceMutate>());
  ffi::reflection::TypeAttrDef<RegionStmtObj>()
      .attr(type_attr::kStructuralVisitWithConstraintContext,
            FStructuralVisitWithConstraintContext::FromNative<&RegionStmtVisit>())
      .attr(type_attr::kStructuralMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&RegionStmtMutate>())
      .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&RegionStmtMaybeInplaceMutate>());
  ffi::reflection::TypeAttrDef<CallObj>()
      .attr(type_attr::kStructuralVisitWithConstraintContext,
            FStructuralVisitWithConstraintContext::FromNative<&CallVisit>())
      .attr(type_attr::kStructuralMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&CallMutate>())
      .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<&CallMaybeInplaceMutate>());
  ffi::reflection::TypeAttrDef<SeqStmtObj>()
      .attr(type_attr::kStructuralVisitWithConstraintContext,
            FStructuralVisitWithConstraintContext::FromNative<&SeqStmtVisit>())
      .attr(type_attr::kStructuralMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<
                &SeqStmtMutate<ffi::InplaceMode::kDisallow>>())
      .attr(type_attr::kStructuralMaybeInplaceMutateWithConstraintContext,
            FStructuralMutateWithConstraintContext::FromNative<
                &SeqStmtMutate<ffi::InplaceMode::kAllow>>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::ObjectDef<StructuralVisitorWithConstraintContextObj>().def_ro(
      "constraints", &StructuralVisitorWithConstraintContextObj::constraints);
  ffi::reflection::ObjectDef<StructuralMutatorWithConstraintContextObj>().def_ro(
      "constraints", &StructuralMutatorWithConstraintContextObj::constraints);
  ffi::reflection::GlobalDef()
      .def("tirx.StructuralVisitorWithConstraintContext",
           [] {
             return StructuralVisitorWithConstraintContext(
                 ffi::make_object<StructuralVisitorWithConstraintContextObj>());
           })
      .def("tirx.StructuralMutatorWithConstraintContext",
           [] {
             return StructuralMutatorWithConstraintContext(
                 ffi::make_object<StructuralMutatorWithConstraintContextObj>());
           })
      .def("tirx.StructuralVisitorWithConstraintContextDefaultVisit",
           [](StructuralVisitorWithConstraintContext visitor, ffi::AnyView value) {
             return visitor->DefaultVisitExpected(value).value();
           })
      .def(
          "tirx.StructuralMutatorWithConstraintContextDefaultMutate",
          [](StructuralMutatorWithConstraintContext mutator, ffi::Any value) {
            return mutator->DefaultMutateExpected(value).value().ValueOrUnchanged(std::move(value));
          })
      .def("tirx.StructuralWalkWithConstraintContext",
           [](ffi::AnyView root, ffi::Function callback, bool pre) {
             auto visit = [&](ffi::AnyView value, const sym::ConstraintContext& constraints) {
               return callback.CallExpected<ffi::WalkResult>(value, constraints);
             };
             return pre ? StructuralWalkWithConstraintContext<ffi::WalkOrder::kPreOrder>(root,
                                                                                         visit)
                        : StructuralWalkWithConstraintContext<ffi::WalkOrder::kPostOrder>(root,
                                                                                          visit);
           })
      .def("tirx.StructuralVisitWithConstraintContext",
           [](ffi::AnyView root, ffi::Function callback) {
             return StructuralVisitWithConstraintContext(
                 root, [&](ffi::AnyView value, StructuralVisitorWithConstraintContextObj* visitor) {
                   return callback.CallExpected<ffi::Optional<ffi::VisitInterrupt>>(
                       value, ffi::GetRef<StructuralVisitorWithConstraintContext>(visitor));
                 });
           })
      .def("tirx.StructuralMapWithConstraintContext",
           [](ffi::Any root, ffi::Function callback, bool pre) {
             auto map = [&](ffi::AnyView value, const sym::ConstraintContext& constraints) {
               return callback.CallExpected<ffi::Any>(value, constraints);
             };
             return pre ? StructuralMapWithConstraintContext<ffi::WalkOrder::kPreOrder>(
                              std::move(root), map)
                        : StructuralMapWithConstraintContext<ffi::WalkOrder::kPostOrder>(
                              std::move(root), map);
           })
      .def("tirx.StructuralMutateWithConstraintContext", [](ffi::Any root, ffi::Function callback) {
        return StructuralMutateWithConstraintContext(
            std::move(root),
            [&](ffi::AnyView value, StructuralMutatorWithConstraintContextObj* mutator) {
              return callback.CallExpected<ffi::Any>(
                  value, ffi::GetRef<StructuralMutatorWithConstraintContext>(mutator));
            });
      });
}
}  // namespace tirx
}  // namespace tvm
