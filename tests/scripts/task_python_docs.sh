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

source ./tests/scripts/test_task_install_dist_wheels.sh --group docs
# Keep function-pointer XML compatible with Breathe.
uv pip install mlc-doxygen-bin==1.17.0
# cargo for the Rust API reference; the crate's build script finds the installed wheel's
# libtvm_tirx_compiler through the tvm-tirx-config of the activated .venv.
# Rust 1.84's rustdoc crashes on proc-macro doc links (rust-lang/rust#134715).
source ./tests/scripts/task_install_rust.sh 1.85.1
rm -rf docs/_build docs/reference/python/generated
export DOXYGEN_NUM_PROC_THREADS=0
BUILD_CPP_DOCS=1 BUILD_RUST_DOCS=1 uv run --no-sync python -m sphinx -j auto -E -a -b html -d docs/_build/doctrees -W --keep-going docs docs/_build/html
git rev-parse HEAD > docs/_build/html/commit_hash
tar -czf docs.tgz -C docs/_build/html .
