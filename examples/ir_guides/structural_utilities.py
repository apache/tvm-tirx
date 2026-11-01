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
"""Walk, map, visit, and mutate add_one with the TVM-FFI structural utilities."""

# Imports stay next to the snippets that use them in the docs.
# ruff: noqa: E402

from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    for i in range(n):
        B[i] = A[i] + 1


# [tree.begin]
import tvm_ffi

from tvm_tirx import ir

print(ir.format_tree(add_one))
# [tree.end]

# [walk.begin]
writes = []


def on_store(store):
    writes.append(store.dest.name)


tvm_ffi.structural_walk(add_one, (ir.TensorStore, on_store))
print(writes)
# [walk.end]
assert writes == ["B"]

# [order.begin]
for order in ("pre", "post"):
    trace = []
    tvm_ffi.structural_walk(
        add_one, (ir.Stmt, lambda stmt: trace.append(type(stmt).__name__)), order=order
    )
    print(order, trace)
# [order.end]
assert trace == ["TensorStore", "SeqStmt", "For", "SeqStmt"]  # the post-order trace

# [skip.begin]
seen = []


def skip_loops(stmt):
    seen.append(type(stmt).__name__)
    if isinstance(stmt, ir.For):
        return tvm_ffi.WalkResult.SKIP
    return None


tvm_ffi.structural_walk(add_one, (ir.Stmt, skip_loops), order="pre")
print(seen)
# [skip.end]
assert seen == ["SeqStmt", "For"]


# [stop.begin]
def find_write(store):
    return tvm_ffi.VisitInterrupt(store)


found = tvm_ffi.structural_walk(add_one, (ir.TensorStore, find_write))
print(found.value.dest.name)
# [stop.end]
assert isinstance(found.value, ir.TensorStore)
assert found.value.dest.name == "B"


# [specialize.begin]
def specialize(var):
    if var.same_as(n):
        return ir.const(128, var.ty)
    return var


add_one_128 = tvm_ffi.structural_map(add_one, (ir.Var, specialize))
print(add_one_128.script())
# [specialize.end]
assert "for i in range(128):" in add_one_128.script()
assert "for i in range(n):" in add_one.script()  # the input is unchanged


# A map rebuilds only the path from a replacement up to the root and shares
# every unchanged part with the original.
add_two = tvm_ffi.structural_map(add_one, (ir.FloatImm, lambda const: ir.const(2.0, const.ty)))
assert "B[i] = A[i] + T.float32(2.0)" in add_two.script()
assert "B[i] = A[i] + T.float32(1.0)" in add_one.script()
loop, new_loop = add_one.body.seq[0], add_two.body.seq[0]
assert add_two.params.same_as(add_one.params)
assert new_loop.var.same_as(loop.var)
assert new_loop.body.seq[0].value.a.same_as(loop.body.seq[0].value.a)
assert not new_loop.same_as(loop)


# [equal.begin]
def make_add(value):
    n = T.dynamic("n")

    @T.function
    def add(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        for i in range(n):
            B[i] = A[i] + value

    return add


first, second = make_add(1), make_add(1)
print(first.same_as(second))
print(tvm_ffi.structural_equal(first, second))
print(tvm_ffi.structural_hash(first) == tvm_ffi.structural_hash(second))
# [equal.end]
assert not first.same_as(second)
assert tvm_ffi.structural_equal(first, second)
assert not first.params[0].same_as(second.params[0])

# [free_vars.begin]
x, y = ir.Var("x", "int32"), ir.Var("y", "int32")
print(tvm_ffi.structural_equal(x + 1, y + 1))
print(tvm_ffi.structural_equal(x + 1, y + 1, map_free_vars=True))
# [free_vars.end]

try:
    # [assert_equal.begin]
    ir.assert_structural_equal(first, make_add(2))
    # [assert_equal.end]
    raise AssertionError("first and make_add(2) should differ")
except ValueError as err:
    message = str(err).splitlines()
# Print the part of the error shown in the docs: the first line, and the
# underlined line of each program.
excerpt = [message[0]]
for index, line in enumerate(message):
    if line.startswith("and rhs at"):
        excerpt += ["...", line]
    if line.strip() and set(line.strip()) == {"^"}:
        excerpt += ["...", message[index - 1], line]
print("\n".join(excerpt))
assert "value.b.value" in message[0]

# [dedup.begin]
programs = [make_add(1), make_add(1), make_add(2)]
unique = {tvm_ffi.StructuralKey(program) for program in programs}
print(len(unique))
# [dedup.end]
assert len(unique) == 2


# [definitions.begin]
def show_definition(var, kind):
    if kind != tvm_ffi.DefRegionKind.NONE:
        print(var.name, kind.name)


tvm_ffi.structural_walk(add_one, with_def_region_kind=(ir.Var, show_definition), order="pre")
# [definitions.end]
sites = []
tvm_ffi.structural_walk(
    add_one,
    with_def_region_kind=(ir.Var, lambda var, kind: sites.append((var.name, kind.name))),
    order="pre",
)
assert [site for site in sites if site[1] != "NONE"] == [
    ("A", "DEF_PATTERN"),
    ("n", "DEF_PATTERN"),
    ("B", "DEF_PATTERN"),
    ("n", "DEF_PATTERN"),
    ("i", "DEF_SIMPLE"),
]

# A later appearance of n in a pattern is a constraint checked at runtime.
import tvm_tirx
from tvm_tirx import runtime

add_one_compiled = tvm_tirx.compile(add_one, target="llvm")
a = runtime.empty((100,), "float32", runtime.cpu())
b = runtime.empty((50,), "float32", runtime.cpu())
try:
    add_one_compiled(a, b)
    raise AssertionError("a B of length 50 should be rejected")
except ValueError as err:
    assert "expected to match A.shape[0]" in str(err)

# Mapping n changes the type of A and B, so the mapper creates new variables
# and remaps their uses; the loop variable i keeps its type and stays shared.
new_A, new_B = add_one_128.params
new_loop = add_one_128.body.seq[0]
new_store = new_loop.body.seq[0]
assert not new_A.same_as(add_one.params[0])
assert new_store.value.a.source.same_as(new_A)
assert new_store.dest.same_as(new_B)
assert new_loop.var.same_as(add_one.body.seq[0].var)

# [rename.begin]
new_vars = {}


def rename(var, kind):
    if kind != tvm_ffi.DefRegionKind.NONE and var not in new_vars:
        new_vars[var] = ir.Var(var.name + "1", var.ty)
    return new_vars.get(var, var)


renamed = tvm_ffi.structural_map(add_one, with_def_region_kind=(ir.Var, rename), order="pre")
print(renamed.script())
# [rename.end]
new_A, new_B = renamed.params
new_store = renamed.body.seq[0].body.seq[0]
assert new_store.dest.same_as(new_B)
assert new_store.value.a.source.same_as(new_A)
assert new_store.indices[0].same_as(renamed.body.seq[0].var)
assert [var.name for var in renamed.params] == ["A1", "B1"]
ir.assert_structural_equal(renamed, add_one)

# [visit.begin]
loops = []
writes = []


def visit_for(loop, visitor):
    loops.append(loop.var.name)
    result = visitor.default_visit(loop)
    loops.pop()
    return result


def visit_store(store, visitor):
    writes.append((store.dest.name, list(loops)))


tvm_ffi.structural_visit(add_one, [(ir.For, visit_for), (ir.TensorStore, visit_store)])
print(writes)
# [visit.end]
assert writes == [("B", ["i"])]
assert loops == []


# [mutate.begin]
reversed_index = {}


def reverse_loop(loop, mutator):
    reversed_index[loop.var] = loop.end - 1 - loop.var
    return mutator.default_mutate(loop)


def reverse_uses(var, mutator):
    is_use = mutator.def_region_kind() == tvm_ffi.DefRegionKind.NONE
    if is_use and var in reversed_index:
        return reversed_index[var]
    return var


add_one_reversed = tvm_ffi.structural_mutate(
    add_one, [(ir.For, reverse_loop), (ir.Var, reverse_uses)]
)
print(add_one_reversed.script())
# [mutate.end]
assert "B[n - 1 - i] = A[n - 1 - i] + T.float32(1.0)" in add_one_reversed.script()
assert add_one_reversed.body.seq[0].var.same_as(add_one.body.seq[0].var)
