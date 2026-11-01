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
"""Call the Rust functions of tirx_ir_guide_examples on add_one.

Install the package with cargo available first, building against the installed TIRx:

    pip install scikit-build-core
    pip install --no-build-isolation -e examples/ir_guides
"""

import tirx_ir_guide_examples.rust

from tvm_tirx import ir
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    for i in range(n):
        B[i] = A[i] + 1


summary = tirx_ir_guide_examples.rust.inspect(add_one)
print(summary)
assert summary.splitlines() == [
    "param A",
    "param B",
    "loop over i up to n",
    "  store to B: A[i] + T.float32(1.0)",
]

doubled = tirx_ir_guide_examples.rust.double_store(add_one)
print(doubled.script())
assert isinstance(doubled, ir.TensorStore)
assert not doubled.same_as(add_one.body.seq[0].body.seq[0])  # add_one is unchanged

print(list(tirx_ir_guide_examples.rust.written_tensors(add_one)))
assert list(tirx_ir_guide_examples.rust.written_tensors(add_one)) == ["B"]

found = tirx_ir_guide_examples.rust.first_write(add_one)
print(found.dest.name)
assert isinstance(found, ir.TensorStore)

add_one_128 = tirx_ir_guide_examples.rust.specialize(add_one, n, 128)
print(add_one_128.script())
assert "for i in range(128):" in add_one_128.script()
assert "for i in range(n):" in add_one.script()  # the input is unchanged
# Mapping n changes the type of A and B, so the mapper creates new variables and remaps their uses.
new_A, new_B = add_one_128.params
new_store = add_one_128.body.seq[0].body.seq[0]
assert not new_A.same_as(add_one.params[0])
assert new_store.dest.same_as(new_B)
assert new_store.value.a.source.same_as(new_A)

print("\n".join(tirx_ir_guide_examples.rust.definitions(add_one)))
assert list(tirx_ir_guide_examples.rust.definitions(add_one)) == [
    "A Pattern",
    "n Pattern",
    "B Pattern",
    "n Pattern",
    "i Simple",
]

print(list(tirx_ir_guide_examples.rust.compare_specialized(add_one, n)))
assert list(tirx_ir_guide_examples.rust.compare_specialized(add_one, n)) == [False, True]

print(list(tirx_ir_guide_examples.rust.trace_statements(add_one, True)))
print(list(tirx_ir_guide_examples.rust.trace_statements(add_one, False)))
assert list(tirx_ir_guide_examples.rust.trace_statements(add_one, True)) == ["loop i", "write B"]
assert list(tirx_ir_guide_examples.rust.trace_statements(add_one, False)) == ["write B", "loop i"]

print(list(tirx_ir_guide_examples.rust.loops_around_writes(add_one)))
assert list(tirx_ir_guide_examples.rust.loops_around_writes(add_one)) == ["B inside [i]"]

add_one_reversed = tirx_ir_guide_examples.rust.reverse_loops(add_one)
print(add_one_reversed.script())
assert "B[n - 1 - i] = A[n - 1 - i] + T.float32(1.0)" in add_one_reversed.script()
assert add_one_reversed.body.seq[0].var.same_as(add_one.body.seq[0].var)
