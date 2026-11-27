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

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/extra/module.h>

#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  TVM_FFI_ICHECK_EQ(argc, 2);
  auto module = tvm::ffi::Module::LoadFromFile(argv[1]);
  auto run = module->GetFunction("wrapper").value();
  for (int64_t n : {31, 513}) {
    std::vector<float> input(n), output(n);
    std::iota(input.begin(), input.end(), 0.0f);
    size_t bytes = n * sizeof(float);
    float* x_data;
    float* y_data;
    TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&x_data, bytes));
    TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&y_data, bytes));
    TVM_FFI_CHECK_CUDA_ERROR(cudaMemcpy(x_data, input.data(), bytes, cudaMemcpyHostToDevice));
    TVM_FFI_CHECK_CUDA_ERROR(cudaMemset(y_data, 0, bytes));
    int64_t stride = 1;
    DLTensor x{x_data, {kDLCUDA, 0}, 1, {kDLFloat, 32, 1}, &n, &stride, 0};
    DLTensor y = x;
    y.data = y_data;
    run(&x, &y, 64);
    TVM_FFI_CHECK_CUDA_ERROR(cudaMemcpy(output.data(), y.data, bytes, cudaMemcpyDeviceToHost));
    TVM_FFI_CHECK_CUDA_ERROR(cudaFree(x_data));
    TVM_FFI_CHECK_CUDA_ERROR(cudaFree(y_data));
    for (int64_t i = 0; i < n; ++i) {
      TVM_FFI_ICHECK_EQ(output[i], input[i] + 1);
    }
  }
  std::ifstream maps("/proc/self/maps");
  TVM_FFI_ICHECK(maps.is_open());
  for (std::string line; std::getline(maps, line);) {
    TVM_FFI_ICHECK(line.find("/libtvm_runtime") == std::string::npos);
    TVM_FFI_ICHECK(line.find("/libtvm_tirx") == std::string::npos);
  }
  return 0;
}
