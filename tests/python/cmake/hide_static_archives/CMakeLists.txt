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
cmake_minimum_required(VERSION 3.26)
project(hide_static_archives LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../3rdparty/tvm-ffi/cmake/Utils/Library.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/utils/Library.cmake")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_library(direct STATIC direct.cc unused.cc)
add_library(imported_impl STATIC imported.cc)
add_library(transitive STATIC transitive.cc)
add_library(leaf STATIC leaf.cc)
target_link_libraries(transitive PRIVATE leaf)
add_library(test::transitive ALIAS transitive)
add_library(dynamic SHARED dynamic.cc)
add_library(test::imported STATIC IMPORTED)
set_target_properties(
  test::imported PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libimported_impl.a"
                            INTERFACE_LINK_LIBRARIES test::transitive
)
add_dependencies(test::imported imported_impl)
# UNKNOWN models FindZLIB and other config-dependent imported dependencies.
add_library(unknown_impl STATIC unknown.cc)
set_target_properties(unknown_impl PROPERTIES OUTPUT_NAME unknown.impl)
add_library(test::dynamic UNKNOWN IMPORTED)
set_target_properties(
  test::dynamic
  PROPERTIES IMPORTED_CONFIGURATIONS RELEASE
             IMPORTED_LOCATION_RELEASE
             "${CMAKE_BINARY_DIR}/libdynamic${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
add_dependencies(test::dynamic dynamic)
# Exercise use from a nested CMake function.
function (hide_archives target)
  tvm_tirx_hide_static_linked_lib_symbols(${target})
endfunction ()
add_subdirectory(child)
foreach (name protected baseline)
  if (NOT TARGET ${name})
    add_library(${name} SHARED api.cc)
  endif ()
  target_link_libraries(
    ${name} PRIVATE "${CMAKE_BINARY_DIR}/libdirect.a" test::imported carrier
                    "$<$<CONFIG:Release>:test::dynamic>"
  )
  add_dependencies(${name} direct)
endforeach ()
