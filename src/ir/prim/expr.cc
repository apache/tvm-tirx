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
 * \file expr.cc
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>

#include <utility>

namespace tvm {
namespace prim {

namespace {

int GetLanesOrVScaleFactor(const PrimType& ty) {
  return ty.IsScalableVector() ? ty.VScaleFactor() : ty.lanes();
}

TVM_FFI_INLINE const PrimTypeObj* GetPrimTypeObj(const PrimExpr& expr) {
  const auto* node = expr.get();
  TVM_FFI_DCHECK(node != nullptr);
  TVM_FFI_DCHECK(!node->ExprObj::ty.as<MissingType>().has_value());
  const auto* prim_ty = node->ExprObj::ty.as<PrimTypeObj>();
  TVM_FFI_DCHECK(prim_ty != nullptr);
  return prim_ty;
}

// Structural traversal hooks

template <typename TNode>
TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> BinaryVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const TNode* self = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->b));
  return std::nullopt;
}

template <typename TNode>
TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BinaryMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const TNode* self = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a_u,
                                    mutator->MutateExpected(self->a));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b_u,
                                    mutator->MutateExpected(self->b));
  if (a_u.UnchangedOrSameAs(self->a) && b_u.UnchangedOrSameAs(self->b)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  // StructuralMap preserves node types. A rewrite that changes operand dtypes must keep the
  // operands compatible and set the result type itself; a generic traversal cannot infer the
  // casts that would require.
  if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();
  if (!b_u.IsUnchanged()) copy->b = std::move(b_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

template <typename TNode>
TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BinaryMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  TNode* self = const_cast<TNode*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, a_u,
                                    mutator->MutateExpected(self->a, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, b_u,
                                    mutator->MutateExpected(self->b, ffi::InplaceMode::kAllow));
  if (a_u.UnchangedOrSameAs(self->a) && b_u.UnchangedOrSameAs(self->b)) {
    return ffi::Unchanged();
  }
  if (!a_u.IsUnchanged()) self->a = std::move(a_u).ValueUnchecked();
  if (!b_u.IsUnchanged()) self->b = std::move(b_u).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> CastVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const CastObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> CastMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const CastObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value));
  if (mapped_value_u.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<CastObj> copy = ffi::make_object<CastObj>(*self);
  if (!mapped_value_u.IsUnchanged()) copy->value = std::move(mapped_value_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> CastMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  CastObj* self = const_cast<CastObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CastObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_value_u,
                                    mutator->MutateExpected(self->value, ffi::InplaceMode::kAllow));
  if (mapped_value_u.UnchangedOrSameAs(self->value)) {
    return ffi::Unchanged();
  }
  if (!mapped_value_u.IsUnchanged()) self->value = std::move(mapped_value_u).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> NotVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const NotObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> NotMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const NotObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<NotObj> copy = ffi::make_object<NotObj>(*self);
  if (!mapped_a_u.IsUnchanged()) copy->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> NotMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  NotObj* self = const_cast<NotObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const NotObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a, ffi::InplaceMode::kAllow));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged();
  }
  if (!mapped_a_u.IsUnchanged()) self->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> BitwiseNotVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const BitwiseNotObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BitwiseNotObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->a));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BitwiseNotMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const BitwiseNotObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BitwiseNotObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<BitwiseNotObj> copy = ffi::make_object<BitwiseNotObj>(*self);
  if (!mapped_a_u.IsUnchanged()) copy->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> BitwiseNotMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  BitwiseNotObj* self = const_cast<BitwiseNotObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const BitwiseNotObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_a_u,
                                    mutator->MutateExpected(self->a, ffi::InplaceMode::kAllow));
  if (mapped_a_u.UnchangedOrSameAs(self->a)) {
    return ffi::Unchanged();
  }
  if (!mapped_a_u.IsUnchanged()) self->a = std::move(mapped_a_u).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> SelectVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const SelectObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->true_value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->false_value));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SelectMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const SelectObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_condition_u,
                                    mutator->MutateExpected(self->condition));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_true_value_u,
                                    mutator->MutateExpected(self->true_value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_false_value_u,
                                    mutator->MutateExpected(self->false_value));
  if (mapped_condition_u.UnchangedOrSameAs(self->condition) &&
      mapped_true_value_u.UnchangedOrSameAs(self->true_value) &&
      mapped_false_value_u.UnchangedOrSameAs(self->false_value)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<SelectObj> copy = ffi::make_object<SelectObj>(*self);
  if (!mapped_condition_u.IsUnchanged())
    copy->condition = std::move(mapped_condition_u).ValueUnchecked();
  if (!mapped_true_value_u.IsUnchanged())
    copy->true_value = std::move(mapped_true_value_u).ValueUnchecked();
  if (!mapped_false_value_u.IsUnchanged())
    copy->false_value = std::move(mapped_false_value_u).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SelectMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  SelectObj* self = const_cast<SelectObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SelectObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_condition_u,
      mutator->MutateExpected(self->condition, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_true_value_u,
      mutator->MutateExpected(self->true_value, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_false_value_u,
      mutator->MutateExpected(self->false_value, ffi::InplaceMode::kAllow));
  if (mapped_condition_u.UnchangedOrSameAs(self->condition) &&
      mapped_true_value_u.UnchangedOrSameAs(self->true_value) &&
      mapped_false_value_u.UnchangedOrSameAs(self->false_value)) {
    return ffi::Unchanged();
  }
  if (!mapped_condition_u.IsUnchanged())
    self->condition = std::move(mapped_condition_u).ValueUnchecked();
  if (!mapped_true_value_u.IsUnchanged())
    self->true_value = std::move(mapped_true_value_u).ValueUnchecked();
  if (!mapped_false_value_u.IsUnchanged())
    self->false_value = std::move(mapped_false_value_u).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> LetVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: PrimExpr types are always PrimType and remain unchanged in normal mutation.
  const LetObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> LetMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const LetObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetObj>(value);
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_body_u,
                                    mutator->MutateExpected(self->body));
  PrimExpr new_body = std::move(mapped_body_u).ValueOrUnchanged(self->body);
  if (new_var.same_as(self->var) && new_value.same_as(self->value) &&
      new_body.same_as(self->body) && new_body->ty.same_as(self->ty)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<LetObj> copy = ffi::make_object<LetObj>(*self);
  copy->var = std::move(new_var);
  copy->value = std::move(new_value);
  copy->body = std::move(new_body);
  copy->ty = copy->body->ty;
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> LetMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  LetObj* self = const_cast<LetObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LetObj>(value));
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
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_body_u,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
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

}  // namespace

/* \brief Convert an object to a PrimExpr
 *
 * All conversions to a PrimExpr are performed as part of the FFI,
 * when calling a function that accepts a PrimExpr as an argument.  If
 * a function must normalize to a PrimExpr (e.g. before accessing the
 * `expr.dtype` field), this function allows the FFI conversions to be
 * explicitly invoked.
 */
#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                                         \
  Name::Name(PrimExpr a, PrimExpr b, Location loc) : PrimExpr(ffi::UnsafeInit{}) { \
    using T = Name::ContainerType;                                                 \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                  \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                  \
    const PrimTypeObj* a_ty = GetPrimTypeObj(a);                                   \
    const PrimTypeObj* b_ty = GetPrimTypeObj(b);                                   \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                           \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n";  \
    ffi::ObjectPtr<T> node = ffi::make_object<T>(a, b);                            \
    node->ExprObj::ty = a.get()->ExprObj::ty;                                      \
    node->loc = loc;                                                               \
    data_ = std::move(node);                                                       \
  }

#define TVM_DEFINE_BITWISE_CONSTRUCTOR(Name, AllowBool)                                     \
  Name::Name(PrimExpr a, PrimExpr b, Location loc) : PrimExpr(ffi::UnsafeInit{}) {          \
    using T = Name::ContainerType;                                                          \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                           \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                           \
    const PrimTypeObj* a_ty = GetPrimTypeObj(a);                                            \
    const PrimTypeObj* b_ty = GetPrimTypeObj(b);                                            \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                                    \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n";           \
    TVM_FFI_CHECK(a.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) ||    \
                      (AllowBool && a.ty().MatchesCode(DLDataTypeCode::kDLBool)),           \
                  TypeError)                                                                \
        << #Name << " requires integer" << (AllowBool ? " or boolean" : "") << " operands"; \
    ffi::ObjectPtr<T> node = ffi::make_object<T>(a, b);                                     \
    node->ExprObj::ty = a.get()->ExprObj::ty;                                               \
    node->loc = loc;                                                                        \
    data_ = std::move(node);                                                                \
  }

#define TVM_DEFINE_CMPOP_CONSTRUCTOR(Name)                                         \
  Name::Name(PrimExpr a, PrimExpr b, Location loc) : PrimExpr(ffi::UnsafeInit{}) { \
    using T = Name::ContainerType;                                                 \
    TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined\n";                  \
    TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined\n";                  \
    const PrimTypeObj* a_ty = GetPrimTypeObj(a);                                   \
    const PrimTypeObj* b_ty = GetPrimTypeObj(b);                                   \
    TVM_FFI_CHECK(a_ty->dtype == b_ty->dtype, TypeError)                           \
        << "mismatched types. " << a_ty->dtype << " vs. " << b_ty->dtype << "\n";  \
    ffi::ObjectPtr<T> node = ffi::make_object<T>(a, b);                            \
    node->ExprObj::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});       \
    node->loc = loc;                                                               \
    data_ = std::move(node);                                                       \
  }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("prim.convert",
                        [](ffi::Variant<PrimExpr, ffi::Array<PrimExpr>> expr) { return expr; });
  // Note: kRepr for VarObj is registered in script/printer/script_printer.cc.
}

// Cast
Cast::Cast(PrimType ty, PrimExpr value, Location loc) : PrimExpr(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(value.defined());
  PrimType value_expr_ty = value.ty();
  TVM_FFI_ICHECK_EQ(ty->dtype.lanes, value_expr_ty->dtype.lanes);
  ffi::ObjectPtr<CastObj> node = ffi::make_object<CastObj>(value);
  node->ExprObj::ty = std::move(ty);
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  CastObj::RegisterReflection();
  refl::TypeAttrDef<CastObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&CastVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&CastMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&CastMaybeInplaceMutate>());

  refl::GlobalDef().def(
      "prim.Cast", [](PrimType ty, PrimExpr value, Location loc) { return Cast(ty, value, loc); });
}

// Add
TVM_DEFINE_BINOP_CONSTRUCTOR(Add);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AddObj::RegisterReflection();
  refl::GlobalDef().def("prim.Add",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Add(a, b, loc); });
  refl::TypeAttrDef<AddObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<AddObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<AddObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<AddObj>>());
}

// LShift
TVM_DEFINE_BITWISE_CONSTRUCTOR(LShift, false);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LShiftObj::RegisterReflection();
  refl::GlobalDef().def("prim.LShift",
                        [](PrimExpr a, PrimExpr b, Location loc) { return LShift(a, b, loc); });
  refl::TypeAttrDef<LShiftObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<LShiftObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<LShiftObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<LShiftObj>>());
}

// RShift
TVM_DEFINE_BITWISE_CONSTRUCTOR(RShift, false);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RShiftObj::RegisterReflection();
  refl::GlobalDef().def("prim.RShift",
                        [](PrimExpr a, PrimExpr b, Location loc) { return RShift(a, b, loc); });
  refl::TypeAttrDef<RShiftObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<RShiftObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<RShiftObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<RShiftObj>>());
}

// BitwiseAnd
TVM_DEFINE_BITWISE_CONSTRUCTOR(BitwiseAnd, true);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseAndObj::RegisterReflection();
  refl::GlobalDef().def("prim.BitwiseAnd",
                        [](PrimExpr a, PrimExpr b, Location loc) { return BitwiseAnd(a, b, loc); });
  refl::TypeAttrDef<BitwiseAndObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<BitwiseAndObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<BitwiseAndObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<BitwiseAndObj>>());
}

// BitwiseOr
TVM_DEFINE_BITWISE_CONSTRUCTOR(BitwiseOr, true);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseOrObj::RegisterReflection();
  refl::GlobalDef().def("prim.BitwiseOr",
                        [](PrimExpr a, PrimExpr b, Location loc) { return BitwiseOr(a, b, loc); });
  refl::TypeAttrDef<BitwiseOrObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<BitwiseOrObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<BitwiseOrObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<BitwiseOrObj>>());
}

// BitwiseXor
TVM_DEFINE_BITWISE_CONSTRUCTOR(BitwiseXor, true);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseXorObj::RegisterReflection();
  refl::GlobalDef().def("prim.BitwiseXor",
                        [](PrimExpr a, PrimExpr b, Location loc) { return BitwiseXor(a, b, loc); });
  refl::TypeAttrDef<BitwiseXorObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<BitwiseXorObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<BitwiseXorObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<BitwiseXorObj>>());
}

// Sub
TVM_DEFINE_BINOP_CONSTRUCTOR(Sub);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SubObj::RegisterReflection();
  refl::GlobalDef().def("prim.Sub",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Sub(a, b, loc); });
  refl::TypeAttrDef<SubObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<SubObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<SubObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<SubObj>>());
}

// Mul
TVM_DEFINE_BINOP_CONSTRUCTOR(Mul);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MulObj::RegisterReflection();
  refl::GlobalDef().def("prim.Mul",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Mul(a, b, loc); });
  refl::TypeAttrDef<MulObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<MulObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<MulObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<MulObj>>());
}

// Div
TVM_DEFINE_BINOP_CONSTRUCTOR(Div);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DivObj::RegisterReflection();
  refl::GlobalDef().def("prim.Div",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Div(a, b, loc); });
  refl::TypeAttrDef<DivObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<DivObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<DivObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<DivObj>>());
}

// Mod
TVM_DEFINE_BINOP_CONSTRUCTOR(Mod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ModObj::RegisterReflection();
  refl::GlobalDef().def("prim.Mod",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Mod(a, b, loc); });
  refl::TypeAttrDef<ModObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<ModObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<ModObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<ModObj>>());
}

// FloorDiv
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorDiv);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FloorDivObj::RegisterReflection();
  refl::GlobalDef().def("prim.FloorDiv",
                        [](PrimExpr a, PrimExpr b, Location loc) { return FloorDiv(a, b, loc); });
  refl::TypeAttrDef<FloorDivObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<FloorDivObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<FloorDivObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<FloorDivObj>>());
}

// FloorMod
TVM_DEFINE_BINOP_CONSTRUCTOR(FloorMod);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FloorModObj::RegisterReflection();
  refl::GlobalDef().def("prim.FloorMod",
                        [](PrimExpr a, PrimExpr b, Location loc) { return FloorMod(a, b, loc); });
  refl::TypeAttrDef<FloorModObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<FloorModObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<FloorModObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<FloorModObj>>());
}

// Min
TVM_DEFINE_BINOP_CONSTRUCTOR(Min);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MinObj::RegisterReflection();
  refl::GlobalDef().def("prim.Min",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Min(a, b, loc); });
  refl::TypeAttrDef<MinObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<MinObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<MinObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<MinObj>>());
}

// Max
TVM_DEFINE_BINOP_CONSTRUCTOR(Max);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MaxObj::RegisterReflection();
  refl::GlobalDef().def("prim.Max",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Max(a, b, loc); });
  refl::TypeAttrDef<MaxObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<MaxObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<MaxObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<MaxObj>>());
}

// EQ
TVM_DEFINE_CMPOP_CONSTRUCTOR(EQ);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  EQObj::RegisterReflection();
  refl::GlobalDef().def("prim.EQ",
                        [](PrimExpr a, PrimExpr b, Location loc) { return EQ(a, b, loc); });
  refl::TypeAttrDef<EQObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<EQObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<EQObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<EQObj>>());
}

// NE
TVM_DEFINE_CMPOP_CONSTRUCTOR(NE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  NEObj::RegisterReflection();
  refl::GlobalDef().def("prim.NE",
                        [](PrimExpr a, PrimExpr b, Location loc) { return NE(a, b, loc); });
  refl::TypeAttrDef<NEObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<NEObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<NEObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<NEObj>>());
}

// LT
TVM_DEFINE_CMPOP_CONSTRUCTOR(LT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LTObj::RegisterReflection();
  refl::GlobalDef().def("prim.LT",
                        [](PrimExpr a, PrimExpr b, Location loc) { return LT(a, b, loc); });
  refl::TypeAttrDef<LTObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<LTObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<LTObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<LTObj>>());
}

// LE
TVM_DEFINE_CMPOP_CONSTRUCTOR(LE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LEObj::RegisterReflection();
  refl::GlobalDef().def("prim.LE",
                        [](PrimExpr a, PrimExpr b, Location loc) { return LE(a, b, loc); });
  refl::TypeAttrDef<LEObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<LEObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<LEObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<LEObj>>());
}

// GT
TVM_DEFINE_CMPOP_CONSTRUCTOR(GT);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  GTObj::RegisterReflection();
  refl::GlobalDef().def("prim.GT",
                        [](PrimExpr a, PrimExpr b, Location loc) { return GT(a, b, loc); });
  refl::TypeAttrDef<GTObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<GTObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<GTObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<GTObj>>());
}

// GE
TVM_DEFINE_CMPOP_CONSTRUCTOR(GE);

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  GEObj::RegisterReflection();
  refl::GlobalDef().def("prim.GE",
                        [](PrimExpr a, PrimExpr b, Location loc) { return GE(a, b, loc); });
  refl::TypeAttrDef<GEObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<GEObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<GEObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<GEObj>>());
}

// And
And::And(PrimExpr a, PrimExpr b, Location loc) : PrimExpr(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<AndObj> node = ffi::make_object<AndObj>(a, b);
  node->ExprObj::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AndObj::RegisterReflection();
  refl::GlobalDef().def("prim.And",
                        [](PrimExpr a, PrimExpr b, Location loc) { return And(a, b, loc); });
  refl::TypeAttrDef<AndObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<AndObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<AndObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<AndObj>>());
}

// Or
Or::Or(PrimExpr a, PrimExpr b, Location loc) : PrimExpr(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  TVM_FFI_CHECK(b.defined(), ValueError) << "b is undefined";
  PrimType a_ty = a.ty();
  PrimType b_ty = b.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(b_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_CHECK(a_ty == b_ty, TypeError) << "mismatched types";

  ffi::ObjectPtr<OrObj> node = ffi::make_object<OrObj>(a, b);
  node->ExprObj::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  OrObj::RegisterReflection();
  refl::GlobalDef().def("prim.Or",
                        [](PrimExpr a, PrimExpr b, Location loc) { return Or(a, b, loc); });
  refl::TypeAttrDef<OrObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&BinaryVisit<OrObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMutate<OrObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BinaryMaybeInplaceMutate<OrObj>>());
}

// Not
Not::Not(PrimExpr a, Location loc) : PrimExpr(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  PrimType a_ty = a.ty();
  TVM_FFI_ICHECK(a_ty.MatchesCode(DLDataTypeCode::kDLBool));

  ffi::ObjectPtr<NotObj> node = ffi::make_object<NotObj>(a);
  node->ExprObj::ty = PrimType(DLDataType{kDLBool, 8, a_ty->dtype.lanes});
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  NotObj::RegisterReflection();
  refl::TypeAttrDef<NotObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&NotVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&NotMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&NotMaybeInplaceMutate>());

  refl::GlobalDef().def("prim.Not", [](PrimExpr a, Location loc) { return Not(a, loc); });
}

// BitwiseNot
BitwiseNot::BitwiseNot(PrimExpr a, Location loc) : PrimExpr(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(a.defined(), ValueError) << "a is undefined";
  PrimType a_ty = a.ty();
  TVM_FFI_CHECK(a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) ||
                    a_ty.MatchesCode(DLDataTypeCode::kDLBool),
                TypeError)
      << "BitwiseNot requires an integer or boolean operand";

  ffi::ObjectPtr<BitwiseNotObj> node = ffi::make_object<BitwiseNotObj>(a);
  node->ExprObj::ty = a_ty;
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BitwiseNotObj::RegisterReflection();
  refl::TypeAttrDef<BitwiseNotObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&BitwiseNotVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&BitwiseNotMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&BitwiseNotMaybeInplaceMutate>());

  refl::GlobalDef().def("prim.BitwiseNot",
                        [](PrimExpr a, Location loc) { return BitwiseNot(a, loc); });
}

// Select
Select::Select(PrimExpr condition, PrimExpr true_value, PrimExpr false_value, Location loc)
    : PrimExpr(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(condition.defined(), ValueError) << "condition is undefined";
  TVM_FFI_CHECK(true_value.defined(), ValueError) << "true_value is undefined";
  TVM_FFI_CHECK(false_value.defined(), ValueError) << "true_value is undefined";
  PrimType condition_ty = condition.ty();
  PrimType true_ty = true_value.ty();
  PrimType false_ty = false_value.ty();
  TVM_FFI_ICHECK(condition_ty.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK(GetLanesOrVScaleFactor(condition_ty) == GetLanesOrVScaleFactor(true_ty) ||
                 condition_ty.IsScalar());
  TVM_FFI_CHECK(false_ty == true_ty, TypeError)
      << "mismatched types. "
      << "False type: " << false_ty->dtype << "; True type: " << true_ty->dtype;

  ffi::ObjectPtr<SelectObj> node = ffi::make_object<SelectObj>(condition, true_value, false_value);
  node->ExprObj::ty = true_ty;
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SelectObj::RegisterReflection();
  refl::TypeAttrDef<SelectObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&SelectVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&SelectMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&SelectMaybeInplaceMutate>());

  refl::GlobalDef().def(
      "prim.Select", [](PrimExpr condition, PrimExpr true_value, PrimExpr false_value,
                        Location loc) { return Select(condition, true_value, false_value, loc); });
}

// Let
Let::Let(Var var, PrimExpr value, PrimExpr body, Location loc) : PrimExpr(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(value.defined());
  TVM_FFI_ICHECK(body.defined());
  TVM_FFI_ICHECK(value.ty() == var->ty.as_or_throw<PrimType>());

  ffi::ObjectPtr<LetObj> node = ffi::make_object<LetObj>(var, value, body);
  node->ExprObj::ty = body.ty();
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LetObj::RegisterReflection();
  refl::TypeAttrDef<LetObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&LetVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&LetMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&LetMaybeInplaceMutate>());

  refl::GlobalDef().def("prim.Let", [](Var var, PrimExpr value, PrimExpr body, Location loc) {
    return Let(var, value, body, loc);
  });
}

}  // namespace prim

}  // namespace tvm
