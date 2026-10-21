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
# A target-local link launcher sees resolved archive paths after CMake has evaluated dependency
# graphs and configurations. Preserve driver response files so large links stay within ARG_MAX.
function (tvm_tirx_hide_archive_arguments output cleanup depth)
  if (depth GREATER 16)
    message(FATAL_ERROR "Too many nested linker response files")
  endif ()
  set(transformed "")
  set(temporary_files "")
  foreach (argument IN LISTS ARGN)
    if (argument MATCHES "^@(.+)$")
      file(READ "${CMAKE_MATCH_1}" content)
      separate_arguments(arguments UNIX_COMMAND "${content}")
      math(EXPR next_depth "${depth} + 1")
      tvm_tirx_hide_archive_arguments(nested nested_files ${next_depth} ${arguments})
      set(content "")
      foreach (value IN LISTS nested)
        string(REPLACE "\\" "\\\\" value "${value}")
        string(REPLACE "\"" "\\\"" value "${value}")
        string(APPEND content "\"${value}\"\n")
      endforeach ()
      string(RANDOM LENGTH 16 nonce)
      set(response "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/tvm_tirx_hidden_${nonce}.rsp")
      file(WRITE "${response}" "${content}")
      set_property(GLOBAL APPEND PROPERTY TVM_TIRX_HIDDEN_RESPONSES "${response}")
      list(APPEND transformed "@${response}")
      list(APPEND temporary_files "${response}" ${nested_files})
    elseif (argument MATCHES "(^|,)-(force_load|all_load|load_hidden)(,|$)")
      get_property(created GLOBAL PROPERTY TVM_TIRX_HIDDEN_RESPONSES)
      if (created)
        file(REMOVE ${created})
      endif ()
      message(
        FATAL_ERROR "Darwin archive hiding cannot combine with explicit archive loading flags"
      )
    elseif (NOT argument MATCHES "^-" AND argument MATCHES "[.]a$")
      # An explicit -Xlinker archive input already has its driver escape prefix.
      if (transformed)
        list(GET transformed -1 previous)
        if (previous STREQUAL "-Xlinker")
          list(POP_BACK transformed)
        endif ()
      endif ()
      list(APPEND transformed -Xlinker -load_hidden -Xlinker "${argument}")
    else ()
      list(APPEND transformed "${argument}")
    endif ()
  endforeach ()
  set(${output}
      "${transformed}"
      PARENT_SCOPE
  )
  set(${cleanup}
      "${temporary_files}"
      PARENT_SCOPE
  )
endfunction ()

# CMAKE_ARGV4 is the original launcher prefix length. Leave that prefix and the compiler untouched.
set(_command "")
set(_arguments "")
math(EXPR _first_argument "5 + ${CMAKE_ARGV4} + 1")
math(EXPR _last_prefix "${_first_argument} - 1")
math(EXPR _last_argument "${CMAKE_ARGC} - 1")
foreach (index RANGE 5 ${_last_prefix} 1)
  list(APPEND _command "${CMAKE_ARGV${index}}")
endforeach ()
if (_first_argument LESS_EQUAL _last_argument)
  foreach (index RANGE ${_first_argument} ${_last_argument} 1)
    list(APPEND _arguments "${CMAKE_ARGV${index}}")
  endforeach ()
endif ()
tvm_tirx_hide_archive_arguments(_arguments _temporary_files 0 ${_arguments})
execute_process(COMMAND ${_command} ${_arguments} RESULT_VARIABLE _result)
if (_temporary_files)
  file(REMOVE ${_temporary_files})
endif ()
if (NOT _result EQUAL 0)
  message(FATAL_ERROR "Darwin archive link failed: ${_result}")
endif ()
