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

# Install the toolchain given as the first argument, defaulting to the crate's rust-version
# for the build and tests of rust/. Source it from the repository root.
export RUSTUP_HOME="$PWD/.local/rustup" CARGO_HOME="$PWD/.local/cargo"
curl --proto '=https' --tlsv1.2 --retry 3 -sSf https://sh.rustup.rs | sh -s -- -y -q \
    --no-modify-path --profile minimal \
    --default-toolchain "${1:-$(sed -n 's/^rust-version = "\(.*\)"$/\1/p' rust/tvm-tirx/Cargo.toml)}"
export PATH="$CARGO_HOME/bin:$PATH"

# rustc links with the cc first on PATH, which in the CI images is a wrapper of sccache: link with
# the compiler that CC names, as the C++ build does.
rust_host=$(rustc -vV | sed -n 's/^host: //p')
export "CARGO_TARGET_$(echo "$rust_host" | tr 'a-z-' 'A-Z_')_LINKER=${CC:-cc}"
