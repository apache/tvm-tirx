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
 * \file tirx/op/region.cc
 * \brief TIRx region operations.
 */
#include <tvm/backend/cuda/attr.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/structural_mutate_with_constraint_context.h>

namespace tvm {
namespace tirx {

static ffi::Array<Var> RegionNoBodyParams(const CallObj*) { return {}; }

ffi::Array<Var> LaunchThreadBodyParams(const CallObj* call) {
  auto tag = call->args[0].as_or_throw<StringImm>();
  PrimType dtype = call->args[1].as_or_throw<IntExpr>().ty();
  TVM_FFI_CHECK(!tag->value.empty(), ValueError)
      << "launch_thread expects a nonempty StringImm thread tag";
  TVM_FFI_CHECK_GT(dtype.bits(), 1, ValueError)
      << "launch_thread extent must have a scalar integer type wider than one bit";
  return {PrimVar("", dtype)};
}

void ValidateBuiltinRegion(const RegionStmtObj* region) {
  TVM_FFI_CHECK(region->result_vars.empty() && region->attrs->dict.empty(), ValueError)
      << region->op->name << " expects no results or attributes";
}

const Op& launch_thread_op() {
  static const Op op = Op::Get("tirx.launch_thread");
  return op;
}

namespace {
template <typename Traversal>
ffi::Expected<void> AddThreadRange(Traversal* traversal, const ffi::Array<Expr>& args,
                                   const ffi::Array<Var>& body_params) {
  try {
    if (body_params.size() == 1 && args.size() == 2) {
      auto extent = args[1].as<PrimExpr>();
      TVM_FFI_RET_CHECK(extent.has_value(), TypeError)
          << "Thread extent must be a primitive expression";
      traversal->AppendConstraint(sym::RangeConstraint(
          body_params[0], Range::FromMinExtent(IntImm(extent.value().ty(), 0), extent.value())));
    }
    return {};
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> LaunchThreadVisit(
    StructuralVisitorWithConstraintContextObj* visitor, ffi::AnyView value) noexcept {
  if (!value.as<RegionStmtObj>())
    return visitor->ffi::StructuralVisitorObj::DefaultVisitExpected(value);

  const RegionStmtObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->attrs));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->body_params); }));
  auto result_body =
      visitor->WithConstraintScope([&]() -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        if (auto result = AddThreadRange(visitor, self->args, self->body_params); result.is_err()) {
          return ffi::Unexpected(std::move(result).error());
        }

        return visitor->VisitExpected(self->body);
      });
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(result_body);
  return visitor->WithDefRegionKind(kTVMFFIDefRegionKindSimple,
                                    [&]() { return visitor->VisitExpected(self->result_vars); });
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> LaunchThreadMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  if (!value.as<RegionStmtObj>())
    return mutator->ffi::StructuralMutatorObj::DefaultMutateExpected(value,
                                                                     ffi::InplaceMode::kDisallow);

  const RegionStmtObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtObj>(value);
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
          return ffi::Unexpected(
              ffi::Error("ValueError",
                         "A region definition rewrite must preserve the number of parameters", ""));
        }
        for (size_t i = 0; i < body_params.size(); ++i) {
          if (!body_params[i].same_as(old_body_params[i])) {
            auto remap = mutator->VarRemapSetExpected(old_body_params[i], body_params[i]);
            TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
          }
        }

        if (auto result = AddThreadRange(mutator,
                                         mapped_args.IsUnchanged()
                                             ? self->args
                                             : ffi::AnyView(mapped_args).cast<ffi::Array<Expr>>(),
                                         body_params);
            result.is_err()) {
          return ffi::Unexpected(std::move(result).error());
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
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> LaunchThreadMaybeInplaceMutate(
    StructuralMutatorWithConstraintContextObj* mutator, ffi::AnyView value) noexcept {
  if (!value.as<RegionStmtObj>())
    return mutator->ffi::StructuralMutatorObj::DefaultMutateExpected(value,
                                                                     ffi::InplaceMode::kAllow);

  const RegionStmtObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<DictAttrs>, mapped_attrs,
                                    mutator->MutateExpected(self->attrs, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
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
          return ffi::Unexpected(
              ffi::Error("ValueError",
                         "A region definition rewrite must preserve the number of parameters", ""));
        }
        for (size_t i = 0; i < body_params.size(); ++i) {
          if (!body_params[i].same_as(old_body_params[i])) {
            auto remap = mutator->VarRemapSetExpected(old_body_params[i], body_params[i]);
            TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
          }
        }

        if (auto result = AddThreadRange(mutator,
                                         mapped_args.IsUnchanged()
                                             ? self->args
                                             : ffi::AnyView(mapped_args).cast<ffi::Array<Expr>>(),
                                         body_params);
            result.is_err()) {
          return ffi::Unexpected(std::move(result).error());
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
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.launch_thread", "Bind a thread index within a body with a launch extent.")
      .signature(sig::arg<StringImm>("tag"), sig::arg<IntExpr>("extent"))
      .set_native_fn_attr<&LaunchThreadBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&LaunchThreadVisit>(op_attr::kStructuralVisitWithConstraintContext)
      .set_native_fn_attr<&LaunchThreadMutate>(op_attr::kStructuralMutateWithConstraintContext)
      .set_native_fn_attr<&LaunchThreadMaybeInplaceMutate>(
          op_attr::kStructuralMaybeInplaceMutateWithConstraintContext)
      .set_native_fn_attr<&ValidateBuiltinRegion>(tvm::op_attr::kRegionValidate)
      .expose();
}

void ValidateDeviceScopeRegion(const RegionStmtObj* region) {
  TVM_FFI_CHECK(region->result_vars.empty(), ValueError)
      << region->op->name << " expects no results";
}
TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.device_scope", "Internal host/device splitting boundary.")
      .signature(sig::var_args<Expr>("launch_values"), sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&RegionNoBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateDeviceScopeRegion>(tvm::op_attr::kRegionValidate)
      .expose();
}

const Op& device_entry_op() {
  static const Op op = Op::Get("tirx.device_entry");
  return op;
}

void ValidateDeviceEntry(const RegionStmtObj* region) {
  TVM_FFI_CHECK(region->result_vars.empty() && region->body_params.empty(), ValueError)
      << "device_entry expects no results or body parameters";
  if (auto fields = region->attrs->dict.Get(tvm::backend::cuda::attr::kLaunchFields.name)) {
    TVM_FFI_CHECK_EQ(fields->as_or_throw<ffi::Array<ffi::String>>().size(), region->args.size(),
                     ValueError)
        << "device_entry launch fields must describe every configuration operand";
  } else {
    TVM_FFI_CHECK(region->args.empty(), ValueError)
        << "device_entry configuration operands require cuda.launch_fields";
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.device_entry", "Enter a device kernel with independent launch configuration.")
      .signature(sig::var_args("launch_values"), sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&RegionNoBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateDeviceEntry>(tvm::op_attr::kRegionValidate)
      .expose();
}

const Op& device_context_op() {
  static const Op op = Op::Get("tirx.device_context");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.device_context", "Supply the device type and ID within a region.")
      .signature(sig::arg<IntExpr>("device_type"), sig::arg<IntExpr>("device_id"))
      .set_native_fn_attr<&RegionNoBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateBuiltinRegion>(tvm::op_attr::kRegionValidate)
      .expose();
}

const Op& compute_scope_op() {
  static const Op op = Op::Get("tirx.compute_scope");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.compute_scope", "Outline a named CPU compute region.")
      .signature(sig::arg<StringImm>("name"))
      .set_native_fn_attr<&RegionNoBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateBuiltinRegion>(tvm::op_attr::kRegionValidate)
      .expose();
}

const Op& parallel_launch_op() {
  static const Op op = Op::Get("tirx.parallel_launch");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.parallel_launch", "Launch a CPU worker team around a region.")
      .signature()
      .set_native_fn_attr<&RegionNoBodyParams>(tvm::op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateBuiltinRegion>(tvm::op_attr::kRegionValidate)
      .expose();
}

}  // namespace tirx
}  // namespace tvm
