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
"""Functions, modules, and global variables in a two-function module."""

# [module.begin]
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.ir_module
class Module:
    @T.function
    def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        for i in range(n):
            B[i] = A[i] + 1

    @T.function
    def add_two(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        for i in range(n):
            B[i] = A[i] + 2


# [module.end]

add_one = Module["add_one"]
assert [p.name for p in add_one.params] == ["A", "B"]
assert len(add_one.ret_type.fields) == 0
print(add_one.ty)
assert add_one.attrs["global_symbol"] == "add_one"


def private_example():
    @T.function(private=True)
    def helper(A: T.Tensor((n,), T.float32)):
        A[0] = T.float32(0)

    return helper


assert "global_symbol" not in private_example().attrs


def attrs_example():
    # [func_attr.begin]
    @T.function
    def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        T.func_attr({"note": "example"})
        for i in range(n):
            B[i] = A[i] + 1

    # [func_attr.end]
    return add_one


noted = attrs_example()
assert noted.attrs["note"] == "example"
print(noted.script())

# [with_attr.begin]
add_one = Module["add_one"]
marked = add_one.with_attr("note", "example")
assert "note" not in add_one.attrs  # the original is unchanged
# [with_attr.end]
assert marked.attrs["note"] == "example"

# [global_var.begin]
add_one_ref = Module.get_global_var("add_one")
add_one = Module[add_one_ref]
# [global_var.end]
assert add_one_ref.name_hint == "add_one"
assert add_one.same_as(Module["add_one"])
print("functions and modules: OK")
