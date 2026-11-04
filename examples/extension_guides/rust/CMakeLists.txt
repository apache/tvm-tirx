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

# scikit-build-core runs this to build the Rust library with cargo and install it into my_ext.
cmake_minimum_required(VERSION 3.18)
project(my_ext LANGUAGES NONE)

# rustup's cargo is preferred over an older one on PATH.
find_program(CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)
set(cargo_target_dir "${CMAKE_BINARY_DIR}/cargo")
set(lib_name "${CMAKE_SHARED_LIBRARY_PREFIX}my_ext${CMAKE_SHARED_LIBRARY_SUFFIX}")
add_custom_target(my_ext ALL
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${cargo_target_dir}"
    "${CARGO}" build --release --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  USES_TERMINAL VERBATIM)
install(FILES "${cargo_target_dir}/release/${lib_name}" DESTINATION my_ext)
