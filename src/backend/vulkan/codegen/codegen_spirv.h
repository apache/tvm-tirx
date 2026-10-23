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
 * \file ir_builder.h
 * \brief Utility for building SPIRV code block
 */
#ifndef TVM_TARGET_VULKAN_CODEGEN_SPIRV_H_
#define TVM_TARGET_VULKAN_CODEGEN_SPIRV_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../runtime/spirv_shader.h"
#include "ir_builder.h"
#include "spirv_support.h"

namespace tvm {
namespace codegen {

using namespace tirx;

/*!
 * \brief Code generator into SPIRV
 */
class CodeGenSPIRV : public tvm::ExprFunctor<spirv::Value(const Expr&)>,
                     public StmtFunctor<void(const Stmt&)> {
 public:
  using tvm::ExprFunctor<spirv::Value(const Expr&)>::Dispatch;
  using StmtFunctor::Dispatch;

  /*!
   * \brief Initialize the codegen based on a specific target.
   *
   * \param target The target for which code should be generated.  The
   * device_type for this target must be kDLVulkan.
   */
  CodeGenSPIRV(Target target);

  /*!
   * \brief Compile and add function f to the current module.
   * \param f The function to be added.
   * \param name The name of the target function.
   * \return The final spirv module.
   */
  virtual runtime::SPIRVShader BuildFunction(const Function& f, const std::string& name);
  /*!
   * \brief Create Value for expression e
   * \param e The expression to be created value for.
   * \return created value.
   */
  spirv::Value MakeValue(const PrimExpr& e) { return Dispatch(e); }
  spirv::Value MakeValue(const Expr& e) { return Dispatch(e); }
  // override codegen
  spirv::Value Dispatch_(const VarObj* op) override;
  spirv::Value Dispatch_(const prim::CastObj* op) override;
  spirv::Value Dispatch_(const IntImmObj* op) override;
  spirv::Value Dispatch_(const FloatImmObj* op) override;
  spirv::Value Dispatch_(const StringImmObj* op) override;
  spirv::Value Dispatch_(const prim::AddObj* op) override;
  spirv::Value Dispatch_(const prim::SubObj* op) override;
  spirv::Value Dispatch_(const prim::MulObj* op) override;
  spirv::Value Dispatch_(const prim::DivObj* op) override;
  spirv::Value Dispatch_(const prim::ModObj* op) override;
  spirv::Value Dispatch_(const prim::MinObj* op) override;
  spirv::Value Dispatch_(const prim::MaxObj* op) override;
  spirv::Value Dispatch_(const prim::LTObj* op) override;
  spirv::Value Dispatch_(const prim::LEObj* op) override;
  spirv::Value Dispatch_(const prim::GTObj* op) override;
  spirv::Value Dispatch_(const prim::GEObj* op) override;
  spirv::Value Dispatch_(const prim::EQObj* op) override;
  spirv::Value Dispatch_(const prim::NEObj* op) override;
  spirv::Value Dispatch_(const prim::AndObj* op) override;
  spirv::Value Dispatch_(const prim::OrObj* op) override;
  spirv::Value Dispatch_(const prim::NotObj* op) override;
  spirv::Value Dispatch_(const prim::LShiftObj* op) override;
  spirv::Value Dispatch_(const prim::RShiftObj* op) override;
  spirv::Value Dispatch_(const prim::BitwiseAndObj* op) override;
  spirv::Value Dispatch_(const prim::BitwiseOrObj* op) override;
  spirv::Value Dispatch_(const prim::BitwiseXorObj* op) override;
  spirv::Value Dispatch_(const prim::BitwiseNotObj* op) override;
  spirv::Value Dispatch_(const prim::SelectObj* op) override;
  spirv::Value Dispatch_(const prim::LetObj* op) override;
  spirv::Value Dispatch_(const CallObj* op) override;
  spirv::Value Dispatch_(const prim::RampObj* op) override;
  spirv::Value Dispatch_(const prim::BroadcastObj* op) override;
  spirv::Value Dispatch_(const TensorLoadObj* op) override;
  spirv::Value Dispatch_(const prim::ShuffleObj* op) override;
  // stmt
  void Dispatch_(const TensorStoreObj* op) override;
  void Dispatch_(const ForObj* op) override;
  void Dispatch_(const WhileObj* op) override;
  void Dispatch_(const IfObj* op) override;
  void DispatchDeclTensor(const BindObj* op, const CallObj* buffer_call);
  void DispatchAllocTensor(const BindObj* op, const CallObj* buffer_call);
  void Dispatch_(const RegionStmtObj* op) override;
  void Dispatch_(const AssertStmtObj* op) override;
  void Dispatch_(const BindObj* op) override;
  void Dispatch_(const SeqStmtObj* op) override;
  void Dispatch_(const EvaluateObj* op) override;

 protected:
  /*! \brief Storage information for a buffer */
  struct StorageInfo {
    /*! \brief The name of the tvm::Var for the buffer
     *
     * Used for error messages.
     */
    std::string name_hint;

    /*! \brief Whether it is volatile */
    bool is_volatile{false};

    /*! \brief Whether the element type of the buffer is known.
     *
     * This value is determined based on the type_annotation of the
     * buffer variable (alloc_tensor binding) or of the parameter (shader
     * arguments).
     */
    bool element_type_known{false};

    /*! \brief The known element type of the buffer.
     *
     * This value is determined based on the type_annotation of the
     * buffer variable (alloc_tensor binding) or of the parameter (shader
     * arguments).
     */
    PrimType element_type{PrimType::Void()};

    /* \brief Check that the access type matches the known type
     *
     * Asserts that the type given is the same as the type previously
     * stored in this array.
     *
     * @param type The data type being stored/loaded in the buffer
     *
     * @param index_lanes The number of lanes of the index.  The
     * number of lanes in the value being stored/loaded should be the
     * product of the number of lanes of the buffer element type and
     * the number of lanes of the index.
     */
    void CheckContentType(PrimType type, int index_lanes = 1) const {
      TVM_FFI_ICHECK(element_type_known) << "Cannot check element type of buffer " << name_hint
                                         << " no previous element type defined";
      PrimType expected_type = element_type.WithLanes(index_lanes * element_type.lanes());
      TVM_FFI_ICHECK_EQ(type, expected_type)
          << "Attempted to access buffer " << name_hint << " as element type " << type
          << " using an index of size " << index_lanes << " when the element type is "
          << element_type;
    }

    // Update content type if it hasn't been updated.
    void SetContentType(PrimType type, std::string name_hint) {
      TVM_FFI_ICHECK(!element_type_known)
          << "Cannot set element type of buffer " << name_hint << " a second time.";
      this->element_type = type;
      this->name_hint = name_hint;
      element_type_known = true;
    }
  };

  struct FragmentInfo {
    std::string shape;
    runtime::MemoryScope scope;
    spirv::SType stype;
    spv::StorageClass sclass;
  };

  // Reset the state so it works for a new function.
  void InitFuncState();
  // Get the thread index
  spirv::Value GetThreadIndex(const PrimVar& var, const ffi::String& thread_tag,
                              const PrimExpr& extent);

  spirv::Value CreateStorageSync(const CallObj* op);
  void Scalarize(const PrimExpr& e, std::function<void(int i, spirv::Value v)> f);

  spirv::SType GetFragmentSType(const VarObj* buffer, const PrimType& dtype);
  PrimType GetElementDataType(const VarObj* buffer);

  // SPIRV-related capabilities of the target
  SPIRVSupport spirv_support_;

  // The builder
  std::unique_ptr<spirv::IRBuilder> builder_;

  // Work group size of three
  uint32_t workgroup_size_[3];

  // Likely branch
  uint32_t weight_likely_branch_{128};

  /* The data type used for the backing array for booleans.
   *
   * Currently matched to the data type used in Buffer::vstore and
   * Buffer::vload.  In the future, this should be the smallest
   * integer type supported by the device, as not all Vulkan
   * implementations support int8.
   */
  PrimType boolean_storage_type_{PrimType::Int(8)};

  // the storage scope of allocation
  std::unordered_map<const VarObj*, StorageInfo> storage_info_;

  // The definition of local variable.
  /*! \brief Extents of physical thread axes in the current function. */
  std::unordered_map<std::string, PrimExpr> thread_extents_;
  std::unordered_map<const VarObj*, spirv::Value> var_map_;

  // The analyzer.
  sym::Analyzer analyzer_;

  // deep comparison of PrimExpr
  prim::ExprDeepEqual deep_equal_;

  // binding of let variables. Enables duplicate var defs that map to same value
  std::unordered_map<Var, const prim::LetObj*> let_binding_;

  // Running total of the number of bytes of shared memory used.
  // Checked against the max_shared_memory_per_group
  size_t shared_memory_bytes_used_{0};

  std::unordered_map<const VarObj*, FragmentInfo> fragment_info_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_VULKAN_CODEGEN_SPIRV_H_
