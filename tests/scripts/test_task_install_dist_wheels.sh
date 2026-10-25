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

unset PYTHONPATH TVM_LIBRARY_PATH TVM_FFI_LIBRARY_PATH TVM_TIRX_LIBRARY_PATH
export PYTHONNOUSERSITE=1
uv venv --clear --python "$(command -v python3)"
source .venv/bin/activate

# Prepare groups before installing the exact artifacts under test.
uv sync --no-default-groups --group test \
    --no-install-project --no-install-package apache-tvm-ffi "$@"
uv pip install --no-deps --offline --reinstall \
    dist/apache_tvm_ffi-*.whl dist/apache_tvm_tirx-*.whl
