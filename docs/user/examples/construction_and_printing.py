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
# TVMScript resolves parameter references inside tensor annotations.
# ruff: noqa: F821
"""Construct IR with TVMScript and builder frames, then print and parse it."""

# begin construction imports
from __future__ import annotations

from tvm_tirx import ir
from tvm_tirx.script import from_source
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

# end script imports
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as B


# begin script function
@T.function
def main(
    n: T.int64,
    A: T.Tensor((2 * n, 128), "int32"),
    C: T.Tensor((2 * n, 128), "int32"),
) -> T.void:
    for i, j in T.grid(2 * n, T.int64(128)):
        value = A[i, j] + 1
        if i < n:
            C[i, j] = value
        else:
            C[i, j] = value + 2


# end script function definition


# begin script module
@I.ir_module
class ScriptModule:
    main = main


# end script module definition

# begin grid pair
with IRBuilder() as loop_builder:
    with B.grid((B.int64(2), B.int32(3))) as index:
        B.evaluate(index)
loop = loop_builder.get()
# end loop construction

# begin function source round trip
function_text = main.script()
restored_function = from_source(function_text, extra_vars={"T": T})
ir.assert_structural_equal(main, restored_function)
# end function source round trip

assert ScriptModule["main"].same_as(main)
assert int(loop.begin) == 2 and int(loop.end) == 5
print("Construction and printing examples passed.")
