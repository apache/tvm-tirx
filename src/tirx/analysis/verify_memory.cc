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
 * \file verify_memory.cc
 * \brief Pass to check if memory accesses are legal.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tirx {
namespace {

/*!
 * \brief Verify if memory accesses are legal.
 *
 *  In the case that tgt is cuda, if workload is not bound with
 *  threads, CPU code is generated that tries to access GPU memory,
 *  which is illegal.
 *
 *  This pass performs such verification by checking if all
 *  memory accesses are bound with threads when device type is GPU.
 */
class MemoryAccessVerifier final : public StmtExprVisitor {
 public:
  /// Special member functions
  //@{
  explicit MemoryAccessVerifier(Function f, int device_type) : func_(f), dev_type_(device_type) {}
  virtual ~MemoryAccessVerifier() = default;
  MemoryAccessVerifier(const MemoryAccessVerifier&) = delete;
  MemoryAccessVerifier(MemoryAccessVerifier&&) = delete;
  MemoryAccessVerifier& operator=(const MemoryAccessVerifier&) = delete;
  MemoryAccessVerifier& operator=(MemoryAccessVerifier&&) = delete;
  //@}

  /// Interface to perform memory access verification
  void Run() {
    if (!IsGPUDevice(dev_type_)) return;
    StmtExprVisitor::Visit(func_->body);
  }

  /// Verification result
  std::vector<ffi::String> Errors() const { return errs_; }

 protected:
  /// Visitor implementation
  //@{

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    // Book keep definitions
    defs_.insert_or_assign(op->var.get(), op->value);
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    bool is_device_region = op->op.same_as(Op::Get("tirx.device_scope"));
    bool is_thread_region =
        op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) != 0;
    if (!InThreadEnv() && (is_device_region || is_thread_region)) {
      // Launch operands execute in the enclosing environment.
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->args));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->attrs));
      EnterThreadEnv();
      auto result = Visit(op->body);
      ExitThreadEnv();
      return result;
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    HandleLoadStoreToVariable(op->source.as_or_throw<tvm::tirx::TensorVar>());
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    HandleLoadStoreToVariable(op->dest.as_or_throw<TensorVar>());
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if ((op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) &&
        !op->args.empty()) {
      HandleLoadStoreToVariable(op->args[0].as_or_throw<Var>());
    }
    return StmtExprVisitor::Visit_(op);
  }
  //@}

  /// Resolve decl_tensor(tensor_data_ptr(X)) views of a tensor to their base X.
  const VarObj* ResolveTensorView(const VarObj* V) const {
    while (true) {
      auto it = defs_.find(V);
      if (it == defs_.end()) return V;
      const CallObj* decl = it->second.as<const CallObj>();
      if (!decl || !decl->op.same_as(tirx::decl_tensor_op()) || decl->args.empty()) return V;
      const CallObj* data = decl->args[0].as<const CallObj>();
      if (!data || !data->op.same_as(tirx::tensor_data_ptr_op()) || data->args.empty()) return V;
      const VarObj* base = data->args[0].as<VarObj>();
      if (!base) return V;
      V = base;
    }
  }

  /// Check if the value of a Variable comes from function argument.
  bool IsFromFunctionArgs(const VarObj* var) const {
    const VarObj* V = ResolveTensorView(var);
    for (const Var& param : func_->params) {
      if (param->ty.as<TensorTypeObj>() && V == param.get()) return true;
    }

    while (true) {
      // Variable is from function args. Return true.
      if (!func_->params.empty() && V == func_->params[0].get()) return true;

      // The value is expected to come from a abi_field_get Call.
      // Get the first argument of abi_field_get, and continue.
      const auto& iter = defs_.find(V);
      if (iter == defs_.end()) return false;
      const CallObj* C = iter->second.as<const CallObj>();
      if (!C || !C->op.same_as(tirx::abi_field_get_op())) return false;
      V = C->args[0].as<VarObj>();
    }
    return false;
  }

  /// Handle memory access to a Variable
  void HandleLoadStoreToVariable(const Var& var) {
    // We skip the access within thread env.
    if (InThreadEnv()) return;

    // We only handle the variable from function argument.
    // If it does not come from args, then it could be allocated internally,
    // it may possibly be in host or device address space.
    // We do not handle this case, and skip it conservatively.
    if (!IsFromFunctionArgs(var.get())) return;

    // The verification fails in this case.
    std::stringstream s;
    s << "Variable `" << var
      << "` is directly accessed by host memory (it is not contained in a thread environment or in "
         "the function arguments.";
    errs_.push_back(s.str());
  }

  /// Status getter/setter
  //@{
  bool InThreadEnv() const { return in_thread_env_; }
  void EnterThreadEnv() { in_thread_env_ = true; }
  void ExitThreadEnv() { in_thread_env_ = false; }
  //@}

  /// Check if a given DLDeviceType/TVMDeviceExtType value denotes GPU device.
  static bool IsGPUDevice(int dev_type) {
    return kDLCUDA == dev_type || kDLOpenCL == dev_type || kDLVulkan == dev_type ||
           kDLMetal == dev_type || kDLROCM == dev_type;
  }

 private:
  /// Status of visitor
  //@{
  bool in_thread_env_{false};
  std::vector<ffi::String> errs_;
  //@}
  tirx::Function func_;                           ///< Function to be verified.
  int dev_type_{kDLCPU};                          ///< Device type
  std::unordered_map<const VarObj*, Expr> defs_;  ///< Variable definitions
};
}  // namespace

/// Interface of VerifyMemory pass
std::vector<ffi::String> VerifyMemory_(const Function& func) {
  auto target = func->GetAttr(tvm::attr::kTarget);
  // Skip verification for functions without a target attribute, as they are
  // typically host-only helper functions that do not have device-memory constraints.
  if (!target.has_value()) return {};

  VLOG(1) << "verifying memory for target '" << target.value()->str()
          << "' for primitive:" << std::endl
          << func;

  if (func->GetAttr(tvm::attr::kCallingConv, CallingConv::kDefault).value() ==
      CallingConv::kDefault) {
    auto v = ffi::make_object<MemoryAccessVerifier>(func, target.value()->GetTargetDeviceType());
    v->Run();
    return v->Errors();
  } else {
    return {};
  }
}

bool VerifyMemory(const Function& func) { return VerifyMemory_(func).size() == 0; }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.analysis.VerifyMemory", VerifyMemory);
}

namespace transform {

Pass VerifyMemory() {
  auto pass_func = [=](IRModule mod, PassContext ctx) {
    for (auto kv : mod->functions) {
      if (auto func = kv.second.as<Function>()) {
        auto errs = VerifyMemory_(func.value());
        if (errs.size() > 0) {
          std::stringstream s;
          for (auto& err : errs) {
            s << "    " << err << "\n";
          }
          TVM_FFI_THROW(RuntimeError) << "Memory verification failed with the following errors:\n"
                                      << s.str() << "  Did you forget to bind?\n"
                                      << func.value();
        }
      }
    }
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tirx.VerifyMemory");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.VerifyMemory", VerifyMemory);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
