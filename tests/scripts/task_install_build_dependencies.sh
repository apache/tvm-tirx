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

# Use the same bundled interpreter and toolchain for manylinux builds and consumers.
if [ -d /opt/python ]; then
    export PATH="/opt/python/cp312-cp312/bin:/opt/rh/gcc-toolset-14/root/usr/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
    test -x /opt/python/cp312-cp312/bin/python3
    export CC=gcc CXX=g++
    python3 -m pip install --user uv
fi

rm -rf dist
unset PYTHONPATH TVM_LIBRARY_PATH TVM_FFI_LIBRARY_PATH TVM_TIRX_LIBRARY_PATH
export PYTHONNOUSERSITE=1
uv venv --clear --python "$(command -v python3)"
source .venv/bin/activate
uv sync --only-group build
uv build --wheel --no-build-isolation --offline --out-dir dist ./3rdparty/tvm-ffi
# Binding checks import FFI's Python package, including its runtime dependencies.
uv pip install --reinstall dist/apache_tvm_ffi-*.whl
