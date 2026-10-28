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
// Load the add_one_cpu.so exported by add_one_cpu.py and run it from C++.
// [main.begin]
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdlib>
#include <iostream>

namespace ffi = tvm::ffi;

// TIRx kernels require 64-byte aligned tensor data for performance.
struct CPUAllocator {
  void AllocData(DLTensor* tensor) {
    size_t bytes = (tensor->shape[0] * sizeof(float) + 63) / 64 * 64;
    tensor->data = std::aligned_alloc(64, bytes);
  }
  void FreeData(DLTensor* tensor) { std::free(tensor->data); }
};

int main() {
  DLDataType f32{kDLFloat, 32, 1};
  DLDevice cpu{kDLCPU, 0};
  ffi::Tensor a = ffi::Tensor::FromNDAlloc(CPUAllocator(), {10}, f32, cpu);
  ffi::Tensor b = ffi::Tensor::FromNDAlloc(CPUAllocator(), {10}, f32, cpu);
  float* a_data = static_cast<float*>(a.data_ptr());
  for (int i = 0; i < 10; ++i) a_data[i] = static_cast<float>(i);

  ffi::Module mod = ffi::Module::LoadFromFile("./add_one_cpu.so");
  ffi::Function add_one = mod->GetFunction("add_one").value();
  add_one(a, b);

  const float* b_data = static_cast<const float*>(b.data_ptr());
  for (int i = 0; i < 10; ++i) std::cout << b_data[i] << " ";
  std::cout << std::endl;
  return 0;
}
// [main.end]
