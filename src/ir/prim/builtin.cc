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
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/structural_mutate_with_constraint_context.h>

namespace tvm {
namespace prim {

#define TVM_DEFINE_CACHED_OP_GETTER(Name, RegisteredName) \
  const Op& Name() {                                      \
    static const Op op = Op::Get(RegisteredName);         \
    return op;                                            \
  }

TVM_DEFINE_CACHED_OP_GETTER(likely_op, "prim.likely")
TVM_DEFINE_CACHED_OP_GETTER(if_then_else_op, "prim.if_then_else")
TVM_DEFINE_CACHED_OP_GETTER(vscale_op, "prim.vscale")
TVM_DEFINE_CACHED_OP_GETTER(ceil_op, "prim.ceil")
TVM_DEFINE_CACHED_OP_GETTER(log2_op, "prim.log2")
TVM_DEFINE_CACHED_OP_GETTER(clz_op, "prim.clz")

#undef TVM_DEFINE_CACHED_OP_GETTER

template <size_t N>
Type InferTypeReturnArgType(const CallObj* call) {
  TVM_FFI_CHECK(call->args.size() > N, TypeError) << "Missing operand " << N;
  return call->args[N]->ty;
}

namespace {
TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> IfThenElseVisit(
    tirx::StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  // Well-formed if_then_else Calls have a boolean condition and two value arms.
  // Skip attrs and the interned Op, matching ordinary Call descent.
  const auto* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(value);
  if (!self->ty.as<PrimTypeObj>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args[0]));
  PrimExpr condition = self->args[0].as_or_throw<PrimExpr>();
  auto true_result =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        visitor->AppendConstraint(sym::PredicateConstraint(condition));
        return visitor->VisitExpected(self->args[1]);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(true_result);

  auto false_result =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        visitor->AppendConstraint(sym::PredicateConstraint(prim::Not(condition)));
        return visitor->VisitExpected(self->args[2]);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(false_result);

  if (!self->ty_args.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty_args));
  }
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfThenElseMutate(
    tirx::StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  // Well-formed if_then_else Calls have a boolean condition and two value arms.
  // Skip attrs and the interned Op, matching ordinary Call descent.
  const auto* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(value);
  ffi::UnchangedOr<Type> mapped_ty = ffi::Unchanged();
  if (!self->ty.as<PrimTypeObj>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<Type>, result,
        mutator->MutateExpected(self->ty, ffi::InplaceMode::kDisallow));
    mapped_ty = std::move(result);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_condition,
                                    mutator->MutateExpected(self->args[0]));
  PrimExpr condition = mapped_condition.IsUnchanged()
                           ? self->args[0].as_or_throw<PrimExpr>()
                           : ffi::AnyView(mapped_condition).cast<PrimExpr>();
  ffi::Array<Expr> args = self->args;
  if (!mapped_condition.UnchangedOrSameAs(self->args[0])) args.Set(0, condition);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Expr>, mapped_true_value,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(condition));
        return mutator->MutateExpected(self->args[1]);
      }));
  if (!mapped_true_value.UnchangedOrSameAs(self->args[1])) {
    args.Set(1, std::move(mapped_true_value).ValueUnchecked());
  }

  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Expr>, mapped_false_value,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(prim::Not(condition)));
        return mutator->MutateExpected(self->args[2]);
      }));
  if (!mapped_false_value.UnchangedOrSameAs(self->args[2])) {
    args.Set(2, std::move(mapped_false_value).ValueUnchecked());
  }

  ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args = ffi::Unchanged();
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<ffi::Array<Type>>, result,
        mutator->MutateExpected(self->ty_args, ffi::InplaceMode::kDisallow));
    mapped_ty_args = std::move(result);
  }
  if (mapped_ty.UnchangedOrSameAs(self->ty) && args.same_as(self->args) &&
      mapped_ty_args.UnchangedOrSameAs(self->ty_args)) {
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<CallObj>(*self);
  if (!mapped_ty.IsUnchanged()) copy->ty = std::move(mapped_ty).ValueUnchecked();
  copy->args = std::move(args);
  if (!mapped_ty_args.IsUnchanged()) copy->ty_args = std::move(mapped_ty_args).ValueUnchecked();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfThenElseMaybeInplaceMutate(
    tirx::StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  // Well-formed if_then_else Calls have a boolean condition and two value arms.
  // Skip attrs and the interned Op, matching ordinary Call descent.
  const auto* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(value);
  ffi::UnchangedOr<Type> mapped_ty = ffi::Unchanged();
  if (!self->ty.as<PrimTypeObj>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, result,
                                      mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
    mapped_ty = std::move(result);
  }
  // Borrow slots: typed indexing copies handles and prevents unique-child reuse.
  const ffi::Any* args_data = self->args.GetArrayObj()->begin();
  // A shared args array may own otherwise unique children. Check it before copying its handle.
  ffi::InplaceMode args_mode =
      self->args.unique() ? ffi::InplaceMode::kAllow : ffi::InplaceMode::kDisallow;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_condition,
                                    mutator->MutateExpected(args_data[0], args_mode));
  PrimExpr condition = mapped_condition.IsUnchanged()
                           ? self->args[0].as_or_throw<PrimExpr>()
                           : ffi::AnyView(mapped_condition).cast<PrimExpr>();
  ffi::Array<Expr> args = self->args;
  if (!mapped_condition.UnchangedOrSameAs(self->args[0])) args.Set(0, condition);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Expr>, mapped_true_value,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(condition));
        return mutator->MutateExpected(args_data[1], args_mode);
      }));
  if (!mapped_true_value.UnchangedOrSameAs(self->args[1])) {
    args.Set(1, std::move(mapped_true_value).ValueUnchecked());
  }

  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Expr>, mapped_false_value,
      mutator->WithConstraintScope([&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        mutator->AppendConstraint(sym::PredicateConstraint(prim::Not(condition)));
        return mutator->MutateExpected(args_data[2], args_mode);
      }));
  if (!mapped_false_value.UnchangedOrSameAs(self->args[2])) {
    args.Set(2, std::move(mapped_false_value).ValueUnchecked());
  }

  ffi::UnchangedOr<ffi::Array<Type>> mapped_ty_args = ffi::Unchanged();
  if (!self->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        ffi::UnchangedOr<ffi::Array<Type>>, result,
        mutator->MutateExpected(self->ty_args, ffi::InplaceMode::kAllow));
    mapped_ty_args = std::move(result);
  }
  if (mapped_ty.UnchangedOrSameAs(self->ty) && args.same_as(self->args) &&
      mapped_ty_args.UnchangedOrSameAs(self->ty_args)) {
    return ffi::Unchanged();
  }
  auto* copy = const_cast<CallObj*>(self);
  if (!mapped_ty.IsUnchanged()) copy->ty = std::move(mapped_ty).ValueUnchecked();
  copy->args = std::move(args);
  if (!mapped_ty_args.IsUnchanged()) copy->ty_args = std::move(mapped_ty_args).ValueUnchecked();
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.likely")
      .signature(sig::arg("x", "The input value."))
      .set_native_fn_attr<&InferTypeReturnArgType<0>>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kExprAnnotation)
      .set_attr(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.if_then_else")
      .signature(sig::arg("condition", "The condition."),
                 sig::arg("true_value", "The value when the condition is true."),
                 sig::arg("false_value", "The value when the condition is false."))
      .set_native_fn_attr<&InferTypeReturnArgType<1>>(tvm::op_attr::kInferType)
      .set_native_fn_attr<&IfThenElseVisit>(tirx::op_attr::kStructuralVisitWithConstraintContext)
      .set_native_fn_attr<&IfThenElseMutate>(tirx::op_attr::kStructuralMutateWithConstraintContext)
      .set_native_fn_attr<&IfThenElseMaybeInplaceMutate>(
          tirx::op_attr::kStructuralMaybeInplaceMutateWithConstraintContext)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);

  OpDef("prim.vscale")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);

  OpDef("prim.ceil")
      .signature(sig::arg("x", "The input value."))
      .set_native_fn_attr<&InferTypeReturnArgType<0>>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.log2")
      .signature(sig::arg("x", "The input value."))
      .set_native_fn_attr<&InferTypeReturnArgType<0>>(tvm::op_attr::kInferType)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr(tvm::tirx::op_attr::kVectorizable, true);

  OpDef("prim.clz")
      .signature(sig::arg("x", "The input value."))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
}

}  // namespace prim
}  // namespace tvm
