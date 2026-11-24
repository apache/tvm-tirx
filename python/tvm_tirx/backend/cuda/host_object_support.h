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

#ifndef TVM_TIRX_BACKEND_CUDA_HOST_OBJECT_SUPPORT_H_
#define TVM_TIRX_BACKEND_CUDA_HOST_OBJECT_SUPPORT_H_

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#elif defined(__FLT16_MANT_DIG__)
// The generic C emitter spells the IEEE binary16 type "half".
typedef _Float16 half;
#endif

int tirx_cuda_set_device(int device);
int tirx_cuda_prepare_launch(const char* name, void* storage, CUfunction* function, int64_t gx,
                             int64_t gy, int64_t gz, int64_t bx, int64_t by, int64_t bz,
                             int64_t shared, CUstream stream, int has_stream);

#ifdef __cplusplus
}
#endif
#endif  // TVM_TIRX_BACKEND_CUDA_HOST_OBJECT_SUPPORT_H_
