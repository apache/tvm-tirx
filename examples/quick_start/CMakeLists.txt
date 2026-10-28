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
cmake_minimum_required(VERSION 3.20)
project(tirx_quick_start LANGUAGES CXX)

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

# Load add_one_cpu.so from C++ using only the runtime libraries.
add_executable(load_cpp load_cpp.cc)
# load_cpp never calls the TIRx runtime directly; keep it linked so that it is
# loaded at startup and can run the exported module.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(load_cpp PRIVATE "LINKER:--no-as-needed")
endif ()
target_link_libraries(load_cpp PRIVATE tvm_runtime::shared tvm_ffi::shared)
set_target_properties(load_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
