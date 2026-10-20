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

#include <tvm/backend/cuda/op/tcgen05.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/op/attr.h>

namespace tvm {
namespace backend {
namespace cuda {

using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  TCGen05InstrDescriptorAttrs::RegisterReflection();
  TCGen05InstrDescriptorBlockScaledAttrs::RegisterReflection();

  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor")
      .signature(sig::arg("desc"), sig::call_attrs<backend::cuda::TCGen05InstrDescriptorAttrs>())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void());
  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled")
      .signature(sig::arg("desc"),
                 sig::call_attrs<backend::cuda::TCGen05InstrDescriptorBlockScaledAttrs>())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void());
  OpDef("tirx.cuda.tcgen05_encode_matrix_descriptor")
      .signature(sig::arg("desc"), sig::arg("addr"), sig::arg<IntExpr>("ldo"),
                 sig::arg<IntExpr>("sdo"), sig::arg<IntExpr>("swizzle"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void());
}

}  // namespace cuda
}  // namespace backend
}  // namespace tvm
