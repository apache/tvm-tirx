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
 * \file codegen_trn.h
 * \brief Generate Metal device code.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_TRN_H_
#define TVM_TARGET_SOURCE_CODEGEN_TRN_H_

#include <tvm/codegen/codegen.h>
#include <tvm/runtime/memory_scope.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "../../../codegen/c/codegen_c.h"

namespace tvm {
namespace codegen {

struct NKIInstructionCtx {
  std::unordered_set<const VarObj*> tensorized_loop_vars;
  std::unordered_map<const VarObj*, ffi::String> loopvar2dim;
  bool is_matmul_input = false;
  int buffer_index = -1;
  int used_var_cnt = 0;
  PrimType dst_dtype = PrimType::Void();
  ffi::Optional<PrimExpr> mask;
  bool tensorizing = false;
};

class CodeGenTrainium final : public CodeGenC {
 public:
  explicit CodeGenTrainium(Target target);
  using CodeGenC::Dispatch_;
  using CodeGenC::PrintType;
  // override print thread tag.
  void PrintArgUnionDecl();
  void AddFunction(const GlobalVar& gvar, const Function& func) final;
  void InitFuncState(const Function& f) final;
  std::string GetStorageScopeStr(const runtime::MemoryScope& scope);  // NOLINT(*)
  void Dispatch_(const VarObj* op, std::ostream& os) final;           // NOLINT(*)
  void PrintType(const PrimType& t, std::ostream& os) final;          // NOLINT(*)
  void Dispatch_(const BindObj* op) final;
  void DispatchAllocTensor(const BindObj* op, const CallObj* buffer_call);
  void Dispatch_(const RegionStmtObj* op) final;                        // NOLINT(*)
  void Dispatch_(const ForObj* op) final;                               // NOLINT(*)
  void Dispatch_(const TensorStoreObj* op) final;                       // NOLINT(*)=
  void Dispatch_(const EvaluateObj* op) final;                          // NOLINT(*)
  std::string PrintIndices(const ffi::Array<PrimExpr>& indices);        // NOLINT(*)
  void Dispatch_(const TensorLoadObj* op, std::ostream& os) final;      // NOLINT(*)
  void Dispatch_(const CallObj* op, std::ostream& os) final;            // NOLINT(*)
  void Dispatch_(const FloatImmObj* op, std::ostream& os) final;        // NOLINT(*)
  void Dispatch_(const prim::CastObj* op, std::ostream& os) final;      // NOLINT(*)
  void Dispatch_(const prim::FloorDivObj* op, std::ostream& os) final;  // NOLINT(*)
  void Dispatch_(const prim::FloorModObj* op, std::ostream& os) final;  // NOLINT(*)
  void DispatchDeclTensor(const BindObj* op, const CallObj* buffer_call);
  void Dispatch_(const IfObj* op) final;                           // NOLINT(*)
  void Dispatch_(const prim::AndObj* op, std::ostream& os) final;  // NOLINT(*)
  void Dispatch_(const prim::OrObj* op, std::ostream& os) final;   // NOLINT(*)

 private:
  Target target_;
  NKIInstructionCtx ctx_;
  std::unordered_map<std::string, std::string> opcode_map_;
  std::unordered_map<TensorVar, std::string, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> buffer_idmap_;
  std::unordered_map<TensorVar, const VarObj*, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      buffer_data_varmap_;
  std::unordered_map<const VarObj*, std::string> data_buffer_idmap_;
  std::unordered_map<const VarObj*, TensorVar> data_decl_buffer_map_;
  bool is_outermost_loop_ = true;
};
}  // namespace codegen
namespace backend {
namespace trn {
namespace attr {

inline constexpr ffi::reflection::AttrKey<ffi::String> kNkiDim{"nki_dim"};
}  // namespace attr
}  // namespace trn
}  // namespace backend

}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_TRN_H_
