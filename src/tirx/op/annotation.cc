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
 * \file tirx/op/annotation.cc
 * \brief TIRx annotation operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/structural_mutate_with_constraint_context.h>

namespace tvm {
namespace tirx {

namespace {
ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> AssumeAlwaysVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  // Ordinary Call descent visits the predicate without dispatching this Op hook again.
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
      visitor->ffi::StructuralVisitorObj::DefaultVisitExpected(value));
  const auto* self = value.cast<const CallObj*>();
  visitor->AppendConstraint(sym::PredicateConstraint(self->args[0].as_or_throw<PrimExpr>(), true));
  return std::nullopt;
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>> AssumeAlwaysMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Call>, mapped,
                                    mutator->ffi::StructuralMutatorObj::DefaultMutateExpected(
                                        value, ffi::InplaceMode::kDisallow));
  const auto* result = mapped.IsUnchanged() ? value.cast<const CallObj*>()
                                            : ffi::AnyView(mapped).cast<const CallObj*>();
  mutator->AppendConstraint(
      sym::PredicateConstraint(result->args[0].as_or_throw<PrimExpr>(), true));
  if (mapped.IsUnchanged()) return ffi::Unchanged();
  return ffi::Any(std::move(mapped).ValueUnchecked());
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>> AssumeAlwaysMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Call>, mapped,
      mutator->ffi::StructuralMutatorObj::DefaultMutateExpected(value, ffi::InplaceMode::kAllow));
  const auto* result = mapped.IsUnchanged() ? value.cast<const CallObj*>()
                                            : ffi::AnyView(mapped).cast<const CallObj*>();
  mutator->AppendConstraint(
      sym::PredicateConstraint(result->args[0].as_or_throw<PrimExpr>(), true));
  if (mapped.IsUnchanged()) return ffi::Unchanged();
  return ffi::Any(std::move(mapped).ValueUnchecked());
}
}  // namespace

const Op& assume_always_op() {
  static const Op op = Op::Get("tirx.assume_always");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.assume_always")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(sig::arg<PrimExpr>("predicate", "The scope-wide predicate."))
      .set_native_fn_attr<&AssumeAlwaysVisit>(op_attr::kStructuralVisitWithConstraintContext)
      .set_native_fn_attr<&AssumeAlwaysMutate>(op_attr::kStructuralMutateWithConstraintContext)
      .set_native_fn_attr<&AssumeAlwaysMaybeInplaceMutate>(
          op_attr::kStructuralMaybeInplaceMutateWithConstraintContext)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kEmbedInfo);
}

const Op& kernel_replace_point_op() {
  static const Op op = Op::Get("tirx.kernel_replace_point");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.kernel_replace_point")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque);
}

const Op& ignore_loop_partition_op() {
  static const Op op = Op::Get("tirx.ignore_loop_partition");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.ignore_loop_partition")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Bool())
      .signature(sig::arg<PrimExpr>("predicate", "The predicate."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure);
}

}  // namespace tirx
}  // namespace tvm
