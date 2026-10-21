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
# A directory-local imported dependency must still be hidden by the consuming parent.
add_library(local_unknown UNKNOWN IMPORTED)
set_target_properties(
  local_unknown PROPERTIES IMPORTED_CONFIGURATIONS RELEASE IMPORTED_LOCATION_RELEASE
                                                           "${CMAKE_BINARY_DIR}/libunknown.impl.a"
)
add_dependencies(local_unknown unknown_impl)
add_library(carrier INTERFACE)
target_link_libraries(carrier INTERFACE local_unknown)
# The parent adds all link dependencies after this directory finishes.
add_library(protected SHARED ../api.cc)
hide_archives(protected)
