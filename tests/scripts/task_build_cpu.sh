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

set -euxo pipefail

source ./tests/scripts/task_install_rust.sh

export CMAKE_ARGS="
    -DTVM_TIRX_BUILD_TESTS=ON
    -DTVM_TIRX_USE_RUST=ON
    -DCMAKE_CXX_FLAGS='-Wno-error=range-loop-construct -Wno-error=comment'
"
uv build --wheel --no-build-isolation --offline --out-dir dist .

./build/tirx_cpptest

# build/rust/bin holds the tvm-tirx-config and tvm-ffi-config of this build, which print build/lib.
PATH="$PWD/build/rust/bin:$PATH" cargo test --manifest-path rust/Cargo.toml

# The bindings that tvm-ffi-stubgen generates in rust/ must match the types of this build.
uv run --no-sync python -m tvm_ffi.stub.cli rust/src --target rust --dlls build/lib/libtvm_tirx_compiler.so --check
