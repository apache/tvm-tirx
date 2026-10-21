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

# Apply compilation policy only to TIRx targets, including their object libraries.
function(tvm_tirx_configure_target target)
  set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET ${target} PROPERTY CUDA_STANDARD 20)
  set_property(TARGET ${target} PROPERTY CUDA_STANDARD_REQUIRED ON)
  set_property(TARGET ${target} PROPERTY CXX_EXTENSIONS OFF)
  set_property(TARGET ${target} PROPERTY CXX_VISIBILITY_PRESET hidden)
  set_property(TARGET ${target} PROPERTY VISIBILITY_INLINES_HIDDEN ON)

  # Keep shared libraries relocatable within their installation directory.
  if(APPLE)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
  elseif(UNIX)
    set_target_properties(${target} PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
  endif()
endfunction()

# Hide archive dependencies in the final shared library, without changing the archives themselves.
function(tvm_tirx_hide_static_linked_lib_symbols target_name)
  if(NOT APPLE)
    tvm_ffi_hide_static_linked_lib_symbols(${target_name})
    return()
  endif()

  # CMake evaluates link features in the caller's directory scope, after this function returns.
  set(CMAKE_LINK_LIBRARY_USING_tvm_hidden "LINKER:-load_hidden,<LIBRARY>" PARENT_SCOPE)
  set(CMAKE_LINK_LIBRARY_USING_tvm_hidden_SUPPORTED TRUE PARENT_SCOPE)
  get_target_property(_libs ${target_name} LINK_LIBRARIES)
  foreach(_lib IN LISTS _libs)
    if(TARGET ${_lib})
      # Shared, interface and object targets do not provide dependency archives to hide.
      get_target_property(_type ${_lib} TYPE)
      if(_type STREQUAL "STATIC_LIBRARY" OR _type STREQUAL "UNKNOWN_LIBRARY")
        # UNKNOWN imports may resolve to a shared library; inspect the selected configuration's file.
        set(_is_archive "$<STREQUAL:$<PATH:GET_EXTENSION,LAST_ONLY,$<TARGET_FILE:${_lib}>>,.a>")
        set_property(TARGET ${target_name} PROPERTY LINK_LIBRARY_OVERRIDE_${_lib}
          "$<IF:${_is_archive},tvm_hidden,DEFAULT>")
      endif()
    elseif(_lib MATCHES "[.]a$")
      # llvm-config also supplies archive paths directly, without imported CMake targets.
      set_property(TARGET ${target_name} PROPERTY LINK_LIBRARY_OVERRIDE_${_lib} tvm_hidden)
    endif()
  endforeach()
endfunction()
