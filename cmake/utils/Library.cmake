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

# ~~~
# tvm_tirx_hide_static_linked_lib_symbols(target_name)
# Prevent symbols from static archives linked into a shared library from being exported by that
# shared library.
#
# This matters when a toolchain links helper archives into a shared target, for example
# libstdc++_nonshared.a. Without this guard, symbols pulled from those archives can become dynamic
# exports of TIRx libraries and unexpectedly interpose with symbols from downstream libraries.
#
# Linux, Android, and BSD targets generally use GNU-compatible linkers that support this through
# `--exclude-libs,ALL`. Darwin uses -load_hidden for archives, preserving dynamic links.
# On Darwin, Ninja and Makefile generators use a target-local link launcher for resolved .a
# inputs, including imported and transitive dependencies. Other generators retain their previous
# behavior with a warning. Unresolved -l choices and compiler-injected archives are not covered;
# explicit archive-loading flags cannot be combined with this policy.
#
# Parameters:
#   target_name: CMake target to modify
# ~~~
function (tvm_tirx_hide_static_linked_lib_symbols target_name)
  if (NOT APPLE)
    tvm_ffi_hide_static_linked_lib_symbols(${target_name})
  else ()
    if (NOT CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
      message(
        WARNING "Static archive symbols are not hidden for ${target_name}: Darwin archive hiding "
                "supports Ninja and Makefiles generators; ${CMAKE_GENERATOR} is unchanged."
      )
      return()
    endif ()
    get_target_property(configured ${target_name} _TVM_TIRX_HIDE_DARWIN_ARCHIVES)
    if (NOT configured)
      foreach (lang C CXX)
        get_target_property(launcher ${target_name} ${lang}_LINKER_LAUNCHER)
        if (NOT launcher)
          set(launcher "")
        endif ()
        list(LENGTH launcher launcher_size)
        set(archive_launcher
            "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HideDarwinArchives.cmake" --
            "${launcher_size}" ${launcher}
        )
        set_property(TARGET ${target_name} PROPERTY ${lang}_LINKER_LAUNCHER "${archive_launcher}")
      endforeach ()
      set_property(
        TARGET ${target_name}
        APPEND
        PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HideDarwinArchives.cmake"
      )
      set_property(TARGET ${target_name} PROPERTY _TVM_TIRX_HIDE_DARWIN_ARCHIVES TRUE)
    endif ()
  endif ()
endfunction ()
