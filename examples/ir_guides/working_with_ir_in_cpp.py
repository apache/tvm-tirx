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
"""Call the C++ functions of tirx_ir_guide_examples on add_one.

Install the package first, building against the installed TIRx:

    pip install scikit-build-core
    pip install --no-build-isolation -e examples/ir_guides
"""

import tirx_ir_guide_examples.cpp

from tvm_tirx import ir
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    for i in range(n):
        B[i] = A[i] + 1


summary = tirx_ir_guide_examples.cpp.inspect(add_one)
print(summary)
assert summary.splitlines() == [
    "param A",
    "param B",
    "loop over i up to n",
    "  store to B: A[i] + T.float32(1.0)",
]

doubled = tirx_ir_guide_examples.cpp.double_store(add_one)
print(doubled.script())
assert isinstance(doubled, ir.TensorStore)
assert not doubled.same_as(add_one.body.seq[0].body.seq[0])  # add_one is unchanged

writes = tirx_ir_guide_examples.cpp.tensors_written(add_one)
print(list(writes))
assert list(writes) == ["B"]

trace = tirx_ir_guide_examples.cpp.trace_statements(add_one)
print(list(trace))
assert list(trace) == ["loop i", "store B"]

found = tirx_ir_guide_examples.cpp.find_write(add_one)
print(found.dest.name)
assert isinstance(found, ir.TensorStore)

sites = tirx_ir_guide_examples.cpp.definitions(add_one)
print("\n".join(sites))
assert list(sites) == ["A pattern", "n pattern", "B pattern", "n pattern", "i simple"]

add_one_128 = tirx_ir_guide_examples.cpp.specialize(add_one, n)
print(add_one_128.script())
assert "for i in range(128):" in add_one_128.script()
assert "for i in range(n):" in add_one.script()  # the input is unchanged
# The mapper creates new A and B with the specialized type and remaps their uses.
new_A, new_B = add_one_128.params
new_store = add_one_128.body.seq[0].body.seq[0]
assert not new_A.same_as(add_one.params[0])
assert new_store.dest.same_as(new_B)
assert new_store.value.a.source.same_as(new_A)


def make_add(value):
    n = T.dynamic("n")

    @T.function
    def add(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        for i in range(n):
            B[i] = A[i] + value

    return add


comparison = tirx_ir_guide_examples.cpp.compare(make_add(1), make_add(1))
print(list(comparison))
assert list(comparison) == [False, True, True]
assert list(tirx_ir_guide_examples.cpp.compare(make_add(1), make_add(2))) == [False, False, False]

loops = tirx_ir_guide_examples.cpp.loops_around_writes(add_one)
print(list(loops))
assert list(loops) == ["B in i"]

add_one_reversed = tirx_ir_guide_examples.cpp.reverse_loop(add_one)
print(add_one_reversed.script())
assert "B[n - 1 - i] = A[n - 1 - i] + T.float32(1.0)" in add_one_reversed.script()
assert add_one_reversed.body.seq[0].var.same_as(add_one.body.seq[0].var)
