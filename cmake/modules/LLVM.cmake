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

# LLVM rules
# Due to LLVM debug symbols you can sometimes face linking issues on
# certain compiler, platform combinations if you don't set NDEBUG.
#
# See https://github.com/imageworks/OpenShadingLanguage/issues/1069
# for more discussion.
add_definitions(-DNDEBUG=1)
# TODO(@jroesch, @tkonolige): if we actually use targets we can do this.
# target_compile_definitions(tvm_compiler PRIVATE NDEBUG=1)

# Target schema and canonicalizer dispatch do not depend on LLVM libraries.
list(APPEND COMPILER_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/llvm/target_kind.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/llvm/canonicalizer/canonicalize.cc")

set(TVM_LLVM_BACKEND_LIB "")

# Test if ${TVM_TIRX_USE_LLVM} is not an explicit boolean false
# It may be a boolean or a string
if(NOT ${TVM_TIRX_USE_LLVM} MATCHES ${IS_FALSE_PATTERN})
  set(TVM_LLVM_BACKEND_LIB tvm_tirx_llvm_backend)
  if(TVM_TIRX_LLVM_NATIVE_ONLY)
    set(TVM_LLVM_BACKEND_LIB tvm_tirx_llvm_backend_native)
  endif()
  find_llvm(${TVM_TIRX_USE_LLVM})
  if (${TVM_LLVM_VERSION} LESS 170)
    message(FATAL_ERROR "LLVM version 17.0 or greater is required.")
  endif()
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  add_library(tvm_llvm_header INTERFACE)
  if(MSVC)
    # MSVC treats GCC-style -isystem operands as source files.
    target_include_directories(tvm_llvm_header SYSTEM INTERFACE ${LLVM_INCLUDE_DIRS})
    target_compile_options(tvm_llvm_header INTERFACE ${LLVM_DEFINITIONS})
  else()
    set(TVM_LLVM_INCLUDE_FLAGS "")
    foreach(__llvm_include_dir IN LISTS LLVM_INCLUDE_DIRS)
      string(STRIP "${__llvm_include_dir}" __llvm_include_dir)
      list(APPEND TVM_LLVM_INCLUDE_FLAGS "-isystem" "${__llvm_include_dir}")
    endforeach()
    target_compile_options(tvm_llvm_header INTERFACE ${TVM_LLVM_INCLUDE_FLAGS} ${LLVM_DEFINITIONS})
  endif()
  message(STATUS "Build with LLVM " ${LLVM_PACKAGE_VERSION})
  message(STATUS "Set TVM_LLVM_VERSION=" ${TVM_LLVM_VERSION})
  # Set flags that are only needed for LLVM target
  add_definitions(-DTVM_LLVM_VERSION=${TVM_LLVM_VERSION})
  add_definitions(-DTVM_LLVM_HAS_AARCH64_TARGET=${TVM_LLVM_HAS_AARCH64_TARGET})
  file(GLOB COMPILER_LLVM_SRCS CONFIGURE_DEPENDS src/codegen/llvm/*.cc)
  if(NOT TVM_TIRX_LLVM_NATIVE_ONLY)
    file(GLOB ACCELERATOR_LLVM_SRCS CONFIGURE_DEPENDS
      src/backend/cuda/codegen/llvm/*.cc
      src/backend/rocm/codegen/llvm/*.cc
      src/backend/hexagon/codegen/llvm/*.cc
    )
    list(APPEND COMPILER_LLVM_SRCS ${ACCELERATOR_LLVM_SRCS})
  endif()
  list(REMOVE_ITEM COMPILER_LLVM_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/llvm/target_kind.cc")
  if(NOT MSVC)
    set_source_files_properties(${COMPILER_LLVM_SRCS}
      PROPERTIES COMPILE_FLAGS "-fno-rtti")
  endif()
  # Keep the Arm canonicalizer with its LLVM implementation. With LLVM OFF,
  # it remains in the core compiler for target parsing without LLVM.
  list(APPEND COMPILER_LLVM_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/llvm/canonicalizer/arm_aprofile.cc")

  # Native initialization references only the host target, so static LLVM
  # archives contribute native code; otherwise initialize all configured targets.
  add_library(${TVM_LLVM_BACKEND_LIB} SHARED ${COMPILER_LLVM_SRCS})
  target_link_libraries(${TVM_LLVM_BACKEND_LIB} PRIVATE
    tvm_tirx_compiler tvm_llvm_header ${LLVM_LIBS})
  target_compile_definitions(${TVM_LLVM_BACKEND_LIB} PRIVATE
    TVM_TIRX_LLVM_NATIVE_ONLY=$<BOOL:${TVM_TIRX_LLVM_NATIVE_ONLY}>)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Avoid GNU ld relaxation of LLVM's GOT calls into incorrect direct calls.
    target_link_options(${TVM_LLVM_BACKEND_LIB} PRIVATE "-Wl,--no-relax" "-Wl,--as-needed")
  endif()

  tvm_tirx_configure_target(${TVM_LLVM_BACKEND_LIB})
  tvm_ffi_add_prefix_map(${TVM_LLVM_BACKEND_LIB} "${CMAKE_CURRENT_SOURCE_DIR}")

  # The optional package puts its DLL beside, rather than inside, tvm_tirx.
  if(APPLE)
    set_property(TARGET ${TVM_LLVM_BACKEND_LIB} APPEND PROPERTY INSTALL_RPATH "@loader_path/../../tvm_tirx/lib" "@loader_path/../../tvm_ffi/lib")
  elseif(UNIX)
    set_property(TARGET ${TVM_LLVM_BACKEND_LIB} APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../../tvm_tirx/lib" "$ORIGIN/../../tvm_ffi/lib")
  endif()
  install(TARGETS ${TVM_LLVM_BACKEND_LIB}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx_llvm_backend
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx_llvm_backend
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT tvm_tirx_llvm_backend)
else()
  list(APPEND COMPILER_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/codegen/llvm/canonicalizer/arm_aprofile.cc")
endif()
