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
"""Print, parse, and build add_one."""

# Imports stay next to the snippets that use them in the docs.
# ruff: noqa: E402

# [function.begin]
from tvm_tirx import ir
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    for i in range(n):
        B[i] = A[i] + 1


# [function.end]

# [print.begin]
print(add_one.script())
# [print.end]

# [print_module.begin]
module = ir.IRModule({"add_one": add_one})
print(module.script())
# [print_module.end]

# [parse.begin]
from add_one_printed import add_one as printed_add_one

ir.assert_structural_equal(add_one, printed_add_one)
# [parse.end]


# [metaprogramming.begin]
def make_add(value, dtype):
    @T.function
    def add(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        for i in range(n):
            B[i] = A[i] + value

    return add


add_two = make_add(2, T.int32)
add_three = make_add(3, T.float32)
# [metaprogramming.end]
print(add_two.script())
print(add_three.script())

# [builder.begin]
from tvm_tirx.script.ir_builder import IRBuilder

with IRBuilder() as builder:
    with T.function_():
        T.func_name_("add_one")
        A = T.arg_("A", T.Tensor((n,), T.float32))
        B = T.arg_("B", T.Tensor((n,), T.float32))
        with T.serial(n) as i:
            T.assign(B[i], A[i] + 1)
built = builder.get()

ir.assert_structural_equal(built, add_one)
# [builder.end]

# [highlight.begin]
loop = add_one.body.seq[0]
addition = loop.body.seq[0].value
print(
    add_one.script(
        obj_to_underline=[addition],
        obj_to_annotate={addition: "result dtype: float32"},
    )
)
# [highlight.end]
