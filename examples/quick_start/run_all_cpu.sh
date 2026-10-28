#!/bin/bash
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
set -ex
cd "$(dirname "$0")"

# Write, compile, and run add_one; export it to add_one_cpu.so
python add_one_cpu.py

# Load add_one_cpu.so in a new Python process
python load_add_one_cpu.py

# Inspect add_one and specialize it to 128 elements
python inspect_and_transform.py

# Load add_one_cpu.so from C++ using only the runtime libraries
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo
build/load_cpp
