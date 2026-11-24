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

#define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 1
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <new>

extern "C" {
extern const unsigned char tirx_cuda_image[] __attribute__((visibility("hidden")));

__attribute__((visibility("hidden"))) int tirx_cuda_set_device(int device) {
  cudaError_t result = cudaSetDevice(device);
  if (result == cudaSuccess) return 0;
  TVMFFIErrorSetRaisedFromCStr("RuntimeError", cudaGetErrorString(result));
  return -1;
}

__attribute__((visibility("hidden"))) int tirx_cuda_prepare_launch(
    const char* name, void* storage, CUfunction* function, int64_t gx, int64_t gy, int64_t gz,
    int64_t bx, int64_t by, int64_t bz, int64_t shared, CUstream stream, int has_stream) {
  TVM_FFI_SAFE_CALL_BEGIN();
  static_assert(sizeof(CUlaunchConfig) <= 8 * sizeof(uint64_t));
  static_assert(alignof(CUlaunchConfig) <= alignof(uint64_t));
  for (int64_t dim : {gx, gy, gz, bx, by, bz}) {
    TVM_FFI_CHECK(dim > 0 && dim <= UINT32_MAX, ValueError) << "Invalid CUDA launch dimension";
  }
  TVM_FFI_CHECK(shared >= 0 && shared <= UINT32_MAX, ValueError) << "Invalid shared memory size";
  int device;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device));
  static tvm::ffi::CubinModule module(tirx_cuda_image);
  auto kernel = module.GetKernel(name);
  TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuKernelGetFunction(function, kernel.GetHandle()));
  if (shared > 0) {
    int static_bytes, default_limit;
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(
        cuFuncGetAttribute(&static_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, *function));
    TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuDeviceGetAttribute(
        &default_limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
    if (shared + static_bytes > default_limit) {
      int optin_limit;
      TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuDeviceGetAttribute(
          &optin_limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
      TVM_FFI_CHECK(shared + static_bytes <= optin_limit, ValueError)
          << "Requested shared memory exceeds the CUDA device limit";
      // Always set the same limit for this function and device, including concurrent launches.
      TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(cuFuncSetAttribute(
          *function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin_limit - static_bytes));
    }
  }
  auto* config = new (storage) CUlaunchConfig{};
  config->gridDimX = gx;
  config->gridDimY = gy;
  config->gridDimZ = gz;
  config->blockDimX = bx;
  config->blockDimY = by;
  config->blockDimZ = bz;
  config->sharedMemBytes = shared;
  config->hStream =
      has_stream ? stream : static_cast<CUstream>(TVMFFIEnvGetStream(kDLCUDA, device));
  TVM_FFI_SAFE_CALL_END();
}
}
