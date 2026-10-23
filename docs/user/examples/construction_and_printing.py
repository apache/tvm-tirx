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
"""Construct the same increment function with TVMScript and builder frames."""

# begin construction imports
from tvm_tirx import ir
from tvm_tirx.script import from_source
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

# end script imports
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as B


# begin script function
@T.function
def main(x: T.int32) -> T.int32:
    result = x + 1
    return result


# end script function definition


# begin script module
@I.ir_module
class ScriptModule:
    main = main


# end script module definition

# begin builder function
with IRBuilder() as builder:
    with B.function_():
        B.func_name_("main")
        x = B.arg_("x", B.int32)
        B.func_ret_type_(ir.PrimType("int32"))
        result = B.bind_(x + B.int32(1), name="result")
        B.return_(result)
built = builder.get()
# end builder function
ir.assert_structural_equal(main, built)

# begin function source round trip
function_text = main.script()
restored_function = from_source(function_text, extra_vars={"T": T})
ir.assert_structural_equal(main, restored_function)
# end function source round trip

assert ScriptModule["main"].same_as(main)
print("Construction and printing examples passed.")
