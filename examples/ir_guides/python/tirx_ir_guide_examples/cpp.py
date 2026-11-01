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
"""The functions of the C++ library, registered as tirx_ir_guide_examples.cpp.<name>."""

from pathlib import Path

import tvm_ffi
from tvm_ffi.libinfo import load_lib_module

import tvm_tirx  # noqa: F401  # loads the TIRx compiler library that the C++ library links

# The library registers its functions when it loads, so load it before collecting them.
_LIB = load_lib_module(
    "tirx-ir-guide-examples", "tirx_ir_guide_examples_cpp", extra_lib_paths=[Path(__file__).parent]
)
tvm_ffi.init_ffi_api("tirx_ir_guide_examples.cpp", __name__)
