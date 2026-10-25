#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Compile the backends excluded from the GPU test wheel without running them.
set -euxo pipefail

# Keep these libraries separate from the published GPU wheel artifacts.
unset PYTHONPATH TVM_LIBRARY_PATH TVM_FFI_LIBRARY_PATH TVM_TIRX_LIBRARY_PATH
export PYTHONNOUSERSITE=1
source .venv/bin/activate
uv sync --only-group build
uv pip install --no-deps --offline --reinstall dist/apache_tvm_ffi-*.whl
uv run --no-sync cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTVM_TIRX_USE_OPENCL=ON -DTVM_TIRX_USE_VULKAN=ON -DTVM_TIRX_USE_ROCM=ON \
    -DCMAKE_CXX_FLAGS=-Werror
uv run --no-sync cmake --build build --target tvm_tirx_compiler tvm_runtime_opencl tvm_runtime_vulkan tvm_runtime_rocm
