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

# The C++ and Rust examples of the IR Guides are functions of a package that builds a C++ and a
# Rust library against the installed TIRx. Install it as the guides do, then run them.
source ./tests/scripts/task_install_rust.sh
uv pip install "scikit-build-core>=0.11,!=1.1.0"
uv pip install --no-build-isolation --no-deps ./examples/ir_guides
uv run --no-sync python examples/ir_guides/working_with_ir_in_cpp.py
uv run --no-sync python examples/ir_guides/working_with_ir_in_rust.py
