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
 *  Lower device runtime allocation and host runtime builtins.
 * \file tirx/transform/lower_tvm_builtin.cc
 */
#include <tvm/backend/cuda/op/tensormap.h>
#include <tvm/backend/opencl/op/memory.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

TVM_FFI_INLINE int GetVectorBytes(const PrimType& dtype) {
  return static_cast<int>(dtype.StorageBytes());
}

}  // namespace

// Lower device allocations in their lexical scope and translate runtime calls.
class BuiltinLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  static Function Build(Function func) {
    if (func->GetAttr(tvm::attr::kDeclaration).value_or(false)) return func;
    ffi::Optional<PrimExpr> device_type = std::nullopt;
    bool preserve_ffi_kernel = false;
    if (auto target = func->GetAttr(tvm::attr::kTarget)) {
      device_type = IntImm::Int32(target.value()->kind->default_device_type);
      auto host = target.value()->GetHost().value_or(target.value());
      preserve_ffi_kernel = host->kind->name == "cuda_host";
    }

    auto mutator = ffi::make_object<BuiltinLower>(device_type, preserve_ffi_kernel);
    func.CopyOnWrite()->body = SeqStmt(mutator->VisitBody(func->body));
    return func;
  }

  explicit BuiltinLower(ffi::Optional<PrimExpr> device_type = std::nullopt,
                        bool preserve_ffi_kernel = false)
      : device_type_(device_type), preserve_ffi_kernel_(preserve_ffi_kernel) {}

  Stmt VisitBody(const SeqStmt& body) {
    return scope_.WithNewScope(
        [&]() -> Stmt { return AppendPendingFrees(Mutate(body).ValueOrUnchanged(body)); });
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallObj>(); call && call->op.same_as(alloc_tensor_op()))
      return MutateAllocTensor(op, inplace_mode);
    if (const CallObj* call = op->value.as<CallObj>()) {
      if (call->op.same_as(backend::opencl::nd_mem_alloc_with_scope_op())) {
        return MakeNdMemAllocWithScope(op, call);
      }
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindObj* op, InplaceMode inplace_mode) {
    // Lower AllocTensor to device allocate when needed.
    // AllocTensor is flat (no body). Visit tensor fields via base class.
    Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<BindObj>();
    const auto* tensor_call = op->value.as<CallObj>();
    tvm::Tuple shape = tvm::Tuple(tensor_call->ty_args[0].as_or_throw<TensorType>()->shape);
    DLDataType dtype = tensor_call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
    PrimType element_type(dtype);
    auto scope = tensor_call->ty_args[0].as_or_throw<TensorType>()->scope;
    if (element_type.IsScalableVector()) {
      return stmt;
    }
    int64_t nbytes = GetVectorBytes(element_type);
    if (const auto* dev_type = device_type_.as<IntImmObj>();
        dev_type && dev_type->value == kDLCPU) {
      if (scope == runtime::MemoryScope::Global()) {
        auto constant_size = op->var->ty.as_or_throw<TensorType>()->ConstantAllocationSize();
        if (constant_size.has_value() && constant_size.value() > 0 &&
            static_cast<size_t>(constant_size.value()) * nbytes < runtime::kMaxStackAlloca) {
          return stmt;
        }
      }
    }
    PrimExpr total_bytes = IntImm(PrimType::UInt(64), nbytes);
    for (const Expr& extent : shape->fields) {
      total_bytes = total_bytes * extent.as_or_throw<PrimExpr>();
    }
    TVM_FFI_ICHECK(device_type_) << "Unknown device type in current IR";
    TVM_FFI_ICHECK(device_id_) << "Unknown device id in current IR";
    Call throw_last_error(PrimType::Void(), throw_last_error_op(), {});

    Stmt alloc_nullptr_check =
        If(Call(PrimType::Bool(), isnullptr_op(), {op->var.as_or_throw<TensorVar>().data()})
               .as_or_throw<PrimExpr>(),
           Evaluate(throw_last_error));

    static const Op free_workspace_op = Op::Get("tirx.free_workspace");
    static const Op alloc_workspace_op = Op::Get("tirx.alloc_workspace");
    PrimExpr free_op = Call(PrimType::Int(32), free_workspace_op,
                            {prim::cast(device_type_.value(), PrimType::Int(32)),
                             prim::cast(device_id_.value(), PrimType::Int(32)),
                             op->var.as_or_throw<TensorVar>().data()})
                           .as_or_throw<PrimExpr>();
    Stmt free_stmt = If(free_op != IntImm::Int32(0), Evaluate(throw_last_error));

    // Push free to enclosing scope's pending_frees (LIFO ordering preserved).
    scope_.Current().pending_frees.push_back(free_stmt);

    Stmt alloc_bind =
        Bind(op->var.as_or_throw<TensorVar>(),
             Call(op->var.as_or_throw<TensorVar>().ty(), decl_tensor_op(),
                  {Call(op->var.as_or_throw<TensorVar>().ty()->DataPtrType(), alloc_workspace_op,
                        {prim::cast(device_type_.value(), PrimType::Int(32)),
                         prim::cast(device_id_.value(), PrimType::Int(32)), total_bytes,
                         IntImm::Int32(element_type.code()), IntImm::Int32(element_type.bits())})},
                  {}, {op->var.as_or_throw<TensorVar>().ty()}, tensor_call->loc),
             op->loc);

    return SeqStmt({alloc_bind, alloc_nullptr_check});
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(device_context_op())) {
      PrimExpr device_type = this->Mutate(op->args[0], inplace_mode)
                                 .ValueOrUnchanged(op->args[0])
                                 .as_or_throw<PrimExpr>();
      PrimExpr device_id = this->Mutate(op->args[1], inplace_mode)
                               .ValueOrUnchanged(op->args[1])
                               .as_or_throw<PrimExpr>();
      auto saved_type = device_type_;
      auto saved_id = device_id_;
      device_type_ = device_type;
      device_id_ = device_id;
      Stmt body = scope_.WithNewScope([&]() -> Stmt {
        return AppendPendingFrees(this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body));
      });
      device_type_ = saved_type;
      device_id_ = saved_id;
      // Packed tensor descriptors still need the lexical device in LowerFFIABI.
      return RegionStmt(op->op, {device_type, device_id}, op->body_params, op->attrs, SeqStmt(body),
                        op->result_vars, op->loc);
    }
    return scope_.WithNewScope([&]() -> Stmt {
      RegionStmt region = StmtExprMutator::Mutate_(op, inplace_mode)
                              .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                              .as_or_throw<RegionStmt>();
      if (!scope_.Current().pending_frees.empty()) {
        region.CopyOnWrite()->body = AppendPendingFrees(region->body);
      }
      return region;
    });
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    auto begin_result = this->Mutate(op->begin, inplace_mode);
    bool begin_unchanged = begin_result.UnchangedOrSameAs(op->begin);
    PrimExpr begin = std::move(begin_result).ValueOrUnchanged(op->begin);
    auto end_result = this->Mutate(op->end, inplace_mode);
    bool end_unchanged = end_result.UnchangedOrSameAs(op->end);
    PrimExpr end = std::move(end_result).ValueOrUnchanged(op->end);
    PrimExpr step = Mutate(op->step, inplace_mode).ValueOrUnchanged(op->step);
    Stmt body = VisitBody(op->body);

    if (begin_unchanged && end_unchanged && step.same_as(op->step) && body.same_as(op->body)) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* n = const_cast<ForObj*>(op);
        n->begin = std::move(begin);
        n->end = std::move(end);
        n->step = step;
        n->body = std::move(body);
        return ffi::Unchanged();
      }
      auto n = ffi::make_object<ForObj>(*op);
      n->begin = std::move(begin);
      n->end = std::move(end);
      n->step = step;
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode inplace_mode) final {
    auto condition_result = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
    // Each branch gets its own scope to prevent frees from leaking across branches.
    Stmt then_case = scope_.WithNewScope([&]() -> Stmt {
      Stmt visited = this->Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
      return AppendPendingFrees(visited);
    });
    ffi::Optional<SeqStmt> else_case;
    if (op->else_case) {
      else_case = scope_.WithNewScope([&]() -> SeqStmt {
        Stmt visited = this->Mutate(op->else_case.value(), inplace_mode)
                           .ValueOrUnchanged(op->else_case.value());
        return AppendPendingFrees(visited);
      });
    }
    if (condition_unchanged && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    }
    return If(condition, then_case, else_case, op->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    auto rewritten = StmtExprMutator::Mutate_(op, inplace_mode);
    if (preserve_ffi_kernel_ || !(op->op.same_as(call_ffi_kernel_op()) ||
                                  op->op.same_as(backend::cuda::tensormap_encode_tiled_op())))
      return rewritten;
    auto call = std::move(rewritten).ValueOrUnchanged(ffi::GetRef<Expr>(op)).as_or_throw<Call>();
    if (call->op.same_as(call_ffi_kernel_op())) {
      return Call(call->ty, call_packed_op(), call->args, call->attrs, call->ty_args, call->loc);
    }
    op = call.get();
    const auto* attr =
        (op->attrs.has_value() ? op->attrs.value().as<backend::cuda::TensorMapEncodeTiledAttr>()
                               : nullptr);
    TVM_FFI_CHECK(attr && attr->rank >= 1 && attr->rank <= 5 &&
                      op->args.size() == static_cast<size_t>(4 * attr->rank + 1),
                  ValueError)
        << "Invalid tensormap_encode_tiled attributes or operands";
    ffi::Array<Expr> args{StringImm("runtime.cuTensorMapEncodeTiled"), op->args[0],
                          StringImm(ffi::DLDataTypeToString(attr->descriptor_dtype)),
                          IntImm(PrimType::Int(32), attr->rank), op->args[1]};
    for (size_t i = 2; i < op->args.size(); ++i) args.push_back(op->args[i]);
    for (int64_t value : {attr->interleave, attr->swizzle, attr->l2_promotion, attr->oob_fill}) {
      args.push_back(IntImm(PrimType::Int(32), value));
    }
    if (attr->force_cu_dtype != -1) {
      args.push_back(IntImm(PrimType::Int(32), attr->force_cu_dtype));
    }
    return Call(PrimType::Void(), call_packed_op(), args);
  }

  StringImm GetDeviceMethodName(const char* method_name) const {
    TVM_FFI_ICHECK(device_type_) << "Method " << method_name << " requires the device type, "
                                 << "but occurred outside of a \"device_context\" region";

    auto as_int = device_type_.as<IntImmObj>();
    TVM_FFI_ICHECK(as_int) << "Method " << method_name
                           << " requires the device type to be a DLDeviceType enum value, "
                           << "but was instead the expression " << device_type_ << " with type "
                           << device_type_.value()->GetTypeKey();

    ffi::String device_name = runtime::DLDeviceType2Str(as_int->value.as<int>().value());
    return StringImm("device_api." + device_name + "." + method_name);
  }

  Stmt MakeNdMemAllocWithScope(const BindObj* let, const CallObj* call) {
    TVM_FFI_ICHECK(device_type_) << "Unknown device type in current IR";
    TVM_FFI_ICHECK(device_id_) << "Unknown device id in current IR";
    Call throw_last_error(PrimType::Void(), throw_last_error_op(), {});

    const auto* dtype_node = let->var->ty.as<PtrTypeObj>()->element_type.as<PrimTypeObj>();
    TVM_FFI_ICHECK(dtype_node);
    PrimType dtype = ffi::GetRef<PrimType>(dtype_node);

    ffi::Array<Expr> args = {
        GetDeviceMethodName("alloc_nd"), device_type_.value(),        device_id_.value(),
        IntImm::Int32(dtype.code()),     IntImm::Int32(dtype.bits()),
    };

    for (size_t i = 0; i < call->args.size(); ++i) {
      args.push_back(call->args[i]);
    }

    Call call_packed = Call(let->var->ty, call_packed_op(), args);
    Stmt null_check = If(
        Call(PrimType::Bool(), isnullptr_op(), ffi::Array<Expr>{let->var}).as_or_throw<PrimExpr>(),
        Evaluate(throw_last_error));

    // Construct free_nd call and register in current scope.
    // The free will be emitted on scope exit, matching the old LetStmt body semantics.
    Expr storage_scope = call->args[0];
    Call free_op = Call(PrimType::Int(32), call_packed_op(),
                        {GetDeviceMethodName("free_nd"), device_type_.value(), device_id_.value(),
                         storage_scope, let->var});
    Stmt free_stmt =
        If(free_op.as_or_throw<PrimExpr>() != IntImm::Int32(0), Evaluate(throw_last_error));
    scope_.Current().pending_frees.push_back(free_stmt);
    return SeqStmt({Bind(let->var, call_packed), null_check});
  }

 private:
  /*!
   * \brief Scope level for tracking nd_mem_alloc_with_scope deallocations.
   *
   * When a Bind allocates via nd_mem_alloc_with_scope, the corresponding
   * free_nd stmt is pushed to the current scope's pending_frees. Body-carrying
   * stmts (For, If, RegionStmt) create new scopes via
   * WithNewScope. On scope exit, pending_frees are appended after the body.
   * AllocTensor (flat, no body) pushes its free to the enclosing scope.
   */
  struct ScopeLevel {
    std::vector<Stmt> pending_frees;
  };

  /*! \brief Emit any pending frees from the current scope after the given body stmt. */
  Stmt AppendPendingFrees(Stmt body) {
    auto& frees = scope_.Current().pending_frees;
    if (frees.empty()) return body;
    ffi::Array<Stmt> stmts;
    stmts.push_back(body);
    for (auto it = frees.rbegin(); it != frees.rend(); ++it) {
      stmts.push_back(*it);
    }
    frees.clear();
    return SeqStmt(stmts);
  }

  ffi::Optional<PrimExpr> device_type_{std::nullopt};
  ffi::Optional<PrimExpr> device_id_{std::nullopt};

  // CUDA host codegen consumes explicit launches after ordinary builtin lowering.
  bool preserve_ffi_kernel_{false};
  // Scope stack for nd_mem_alloc_with_scope free tracking.
  ScopeStack<ScopeLevel> scope_;
};

namespace transform {

Pass LowerTVMBuiltin() {
  auto pass_func = [](Function func, IRModule m, PassContext ctx) {
    bool is_host = func->HasNonzeroAttr(tvm::tirx::attr::kIsHostFunc.name);
    if (!is_host) {
      if (auto target = func->GetAttr(tvm::attr::kTarget)) {
        is_host = target.value()->HasKey("cpu");
      }
    }
    if (is_host) {
      func = BuiltinLower::Build(func);
      VLOG(2) << "LowerTVMBuiltin: " << func;
    }
    return func;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.LowerTVMBuiltin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerTVMBuiltin", LowerTVMBuiltin);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
