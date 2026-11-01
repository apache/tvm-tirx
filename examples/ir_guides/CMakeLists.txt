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

# scikit-build-core runs this to build the example libraries and install them into tirx_ir_guide_examples:
# the C++ library always, and the Rust library when cargo is available.
cmake_minimum_required(VERSION 3.20)
project(tirx_ir_guide_examples LANGUAGES CXX)

# Locate the CMake packages with the TIRx and TVM-FFI config tools.
find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_tirx.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_tirx_ROOT
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE tvm_ffi_ROOT
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(tvm_ffi CONFIG REQUIRED)
find_package(tvm_tirx CONFIG REQUIRED)

# The C++ examples, linked against the TIRx compiler library, which contains the IR.
add_library(tirx_ir_guide_examples_cpp SHARED cpp/working_with_ir.cc)
target_link_libraries(tirx_ir_guide_examples_cpp PRIVATE tvm_tirx_compiler::shared tvm_ffi::shared)
install(TARGETS tirx_ir_guide_examples_cpp LIBRARY DESTINATION tirx_ir_guide_examples)

# The Rust examples, built with cargo when it is available. rustup's cargo is preferred over an
# older one on PATH.
find_program(CARGO cargo HINTS "$ENV{HOME}/.cargo/bin")
if(NOT CARGO)
  message(STATUS "cargo not found: skipping the Rust examples")
  return()
endif()
set(cargo_target_dir "${CMAKE_BINARY_DIR}/cargo")
set(rust_lib_name
    "${CMAKE_SHARED_LIBRARY_PREFIX}tirx_ir_guide_examples_rust${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
add_custom_target(
  tirx_ir_guide_examples_rust ALL
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${cargo_target_dir}" "${CARGO}" build
          --release --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml"
  USES_TERMINAL VERBATIM
)
install(FILES "${cargo_target_dir}/release/${rust_lib_name}" DESTINATION tirx_ir_guide_examples)
