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
"""Construct IR through shared builder scopes and Python program factories."""

# begin construction imports
from __future__ import annotations

from tvm_tirx import ir, tirx
from tvm_tirx.script import from_source
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

# end script imports
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as B

# end construction imports


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


assert isinstance(main, tirx.Function)
assert main.attrs["global_symbol"] == "main"
# end script function


# begin script module
@I.ir_module
class ScriptModule:
    main = main


# end script module definition


assert isinstance(ScriptModule, ir.IRModule)
assert ScriptModule["main"].same_as(main)
assert tirx.analysis.check_well_formed(ScriptModule)
# end script module

# begin imperative construction
with IRBuilder() as builder:
    with B.ir_module():
        with B.function_():
            B.func_name_("main")
            n = B.arg_("n", B.int64)
            A = B.arg_("A", B.Tensor((2 * n, 128), "int32"))
            C = B.arg_("C", A.ty)
            B.func_ret_type_(ir.PrimType("void"))
            with B.grid(2 * n, B.int64(128)) as (i, j):
                value = B.bind_(A[i, j] + B.int32(1), name="value")
                with B.if_(i < n):
                    with B.then_():
                        B.tensor_store(C, [i, j], value)
                    with B.else_():
                        B.tensor_store(C, [i, j], value + B.int32(2))
built_module = builder.get()
ir.assert_structural_equal(ScriptModule, built_module)
assert tirx.analysis.check_well_formed(built_module)
# end imperative construction

# begin grid pair
with IRBuilder() as loop_builder:
    with B.grid((B.int64(2), B.int32(3))) as index:
        B.evaluate(index)
loop = loop_builder.get()
# end loop construction
assert loop.var.ty == ir.PrimType("int64")
assert int(loop.begin) == 2
assert int(loop.end) == 5
# end grid pair

# begin parse source
source = """
@T.function(private=True)
def increment(value: T.int32) -> T.int32:
    return value + 1
"""
parsed = from_source(source, extra_vars={"T": T}, filename="increment.py")
assert isinstance(parsed, tirx.Function)
assert parsed.ty.ret_type == ir.PrimType("int32")
assert "global_symbol" not in parsed.attrs
# end parse source

# begin missing namespace
try:
    from_source(source)
except SyntaxError:
    pass
else:
    raise AssertionError("Parsing source text requires its construction namespace")
# end missing namespace

# begin script text
function_text = main.script()
module_text = ScriptModule.script()
assert isinstance(function_text, str)
assert isinstance(module_text, str)
# end script text

# begin source round trip
# begin function source round trip
restored_function = from_source(function_text, extra_vars={"T": T})
ir.assert_structural_equal(main, restored_function)
# end function source round trip
restored_module = from_source(module_text, extra_vars={"T": T})
ir.assert_structural_equal(ScriptModule, restored_module)
# end source round trip

# begin explicit imports
standalone_source = ScriptModule.script(extra_config={"ir.comment_imports": False})
ir.assert_structural_equal(ScriptModule, from_source(standalone_source))
# end explicit imports


# begin program factory
@T.inline
def increment(value):
    return value + T.int32(1)


def make_increment(width):
    if not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive Python integer")
    n = T.dynamic("n", "int64")

    @T.function
    def generated(
        A: T.Tensor((2 * n, width), "int32"),
        C: T.Tensor((2 * n, width), "int32"),
    ):
        for i, j in T.grid(2 * n, width):
            C[i, j] = increment(A[i, j])

    return generated


wide = make_increment(128)
narrow = make_increment(64)
assert isinstance(wide, tirx.Function)
assert int(wide.params[0].ty.shape[1]) == 128
assert int(narrow.params[0].ty.shape[1]) == 64
# end program factory

restored = ir.load_json(ir.save_json(wide))
ir.assert_structural_equal(wide, restored, map_free_vars=True)
assert isinstance(wide.script(), str)
try:
    make_increment(0)
except ValueError as error:
    assert str(error) == "width must be a positive Python integer"
else:
    raise AssertionError("The factory should reject a zero width")

print("IR construction and metaprogramming checks passed.")
