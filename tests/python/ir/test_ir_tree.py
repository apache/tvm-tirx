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
"""Test reflected IR tree inspection."""

from pathlib import Path
from typing import Any

import pytest
import tvm_ffi
from tvm_ffi.dataclasses import field, py_class

import tvm_tirx.testing
from tvm_tirx import ir
from tvm_tirx.script import from_source
from tvm_tirx.script import tirx as T


@py_class("testing.tree.Base")
class TreeBase(tvm_ffi.Object):
    label: str


@py_class("testing.tree.Extension")
class TreeExtension(TreeBase):
    child: Any
    count: int = 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quick_start(device):
    # Parse the documentation's actual definitions, without importing torch or
    # executing compilation, device allocation, or library export.
    root = Path(__file__).resolve().parents[3]
    source = (root / f"examples/quick_start/add_one_{device}.py").read_text()
    source = source.split("# [run.begin]")[0]
    source = source[source.index("from tvm_tirx.script") :]
    function = from_source(source)
    text = ir.format_tree(function, inline_width=0)
    for name in (
        "Function",
        "params",
        "name='A'",
        "name='B'",
        "TensorType",
        "SeqStmt",
        "TensorStore",
        "TensorLoad",
        "Add",
        "FloatImm(value=1.0)",
    ):
        assert name in text
    assert "loc:" not in text
    assert "global_symbol" in text
    if device == "cpu":
        assert "For(" in text and "kind=0" in text
    else:
        assert "RegionStmt" in text
        assert "If" in text
        assert "cuda" in text
    # The inspection path must not change the IR or its script printer.
    ir.assert_structural_equal(function, from_source(source))
    assert "def add_one" in function.script()


def test_nested_control_flow():
    @T.function
    def nested(A: T.Tensor((4, 4), T.float32)):
        for i in range(4):
            for j in range(4):
                if i < j:
                    A[i, j] = A[i, j] + 1.0

    text = ir.format_tree(nested, inline_width=0)
    assert text.count("For(kind=0)") == 2
    assert "If" in text
    assert "TensorStore" in text
    assert "SeqStmt" in text  # Singleton wrappers retain the actual IR shape.


def test_inline_leaf_and_small_snapshot():
    var = ir.Var("i", "int32")
    assert ir.format_tree(var) == "Var(ty=PrimType(dtype=int32), name='i')"
    assert ir.format_tree(var, inline_width=0) == ("Var(name='i')\n└── ty: PrimType(dtype=int32)")
    # Width controls compaction rather than truncating long scalar contents.
    assert "a long name" in ir.format_tree(ir.Var("a long name", "int32"), inline_width=1)


def test_extension_fields_and_defaults():
    extension = TreeExtension(label="extension", child=ir.Var("x", "int32"))
    text = ir.format_tree(extension)
    assert "label='extension'" in text  # Inherited field, no registered formatter.
    assert "child=x" in text
    assert "count=" not in text
    assert "count=0" in ir.format_tree(extension, show_defaults=True)
    extension.count = 2
    assert "count=2" in ir.format_tree(extension)
    # No reflected default means that a meaningful zero must stay visible.
    assert "value=0" in ir.format_tree(ir.IntImm("int32", 0))


def test_locations_and_containers():
    var = ir.Var("x", "int32")
    assert "loc" in ir.format_tree(var, hide_fields=(), show_defaults=True)
    assert "ty" not in ir.format_tree(var, hide_fields=("loc", "ty"))
    assert ir.format_tree({"z": 1, "a": 2}) == "{'a': 2, 'z': 1}"
    assert ir.format_tree(tvm_ffi.convert({"z": 1, "a": 2})) == "{'a': 2, 'z': 1}"
    assert ir.format_tree([0, False, None, "x\ny"]) == "[0, False, None, 'x\\ny']"
    # Map keys are not affected by the reflected-field filter.
    assert "'loc'" in ir.format_tree({"loc": var})
    assert ir.format_tree([]) == "[]"
    assert ir.format_tree({}) == "{}"
    assert ir.format_tree(object()) == "<object>"


def test_object_map_keys_and_shared_references():
    var = ir.Var("shared", "int32")
    text = ir.format_tree(tvm_ffi.convert({var: 7}), inline_width=0)
    assert "shared" in text
    assert "[1]: 7" in text
    text = ir.format_tree([var, var], inline_width=0)
    assert text.count("shared") == 2
    assert "<cycle>" not in text


@pytest.mark.parametrize("inline_width", [0, 80])
def test_depth_and_cycles(inline_width):
    assert ir.format_tree([[[1]]], max_depth=0, inline_width=inline_width) == "list(...)"
    text = ir.format_tree([[[1]]], max_depth=1, inline_width=inline_width)
    assert "list(...)" in text
    assert "1" not in text
    assert ir.format_tree(1, max_depth=0, inline_width=inline_width) == "1"
    cycle = []
    cycle.append(cycle)
    assert "<cycle>" in ir.format_tree(cycle, inline_width=inline_width)
    extension = TreeExtension(label="cycle", child=None)
    extension.child = extension
    try:
        assert "<cycle>" in ir.format_tree(extension, inline_width=inline_width)
    finally:
        extension.child = None


@pytest.mark.parametrize(
    "options", [{"max_depth": -1}, {"max_depth": 1.5}, {"inline_width": -1}, {"inline_width": True}]
)
def test_invalid_limits(options):
    with pytest.raises(ValueError):
        ir.format_tree(None, **options)


@py_class("testing.tree.Binding")
class TreeBinding(tvm_ffi.Object):
    declaration: Any = field(structural_eq="def-simple")
    value: Any


def test_reflected_binding_sites_and_indexed_edges():
    var = ir.Var("item", "int32")
    # A downstream binder participates through field metadata, not a node table
    # or a field-name heuristic. Containers preserve the declaration role.
    text = ir.format_tree(TreeBinding(declaration=[var], value=var), inline_width=0)
    assert "declaration[0]: Var(name='item')" in text
    assert "ty: PrimType(dtype=int32)" in text
    assert "value=item" in text
    assert "Array" not in text
    seq = ir.SeqStmt([ir.Evaluate(var), ir.Evaluate(var)])
    text = ir.format_tree(seq, inline_width=0)
    assert "seq[0]: Evaluate(value=item)" in text
    assert "seq[1]: Evaluate(value=item)" in text


def test_loop_header_and_variable_uses():
    @T.function
    def copy(A: T.Tensor((4,), T.float32), B: T.Tensor((4,), T.float32)):
        for i in range(4):
            B[i] = A[i]

    text = ir.format_tree(copy.body, inline_width=80)
    header = next(line for line in text.splitlines() if "For(" in line)
    for field_name in (
        "var=Var(",
        "ty=PrimType(dtype=int32)",
        "begin=",
        "end=",
        "step=",
        "annotations={}",
    ):
        assert field_name in header
    assert "seq[0]: For(" in header
    assert "TensorStore(dest=B, indices=[i])" in text
    assert "source=A" in text
    assert "name='A'" not in text
    # Parameter definitions retain the full variable object and its type.
    full = ir.format_tree(copy, inline_width=0)
    assert "params[0]: Var(name='A')" in full
    assert "params[1]: Var(name='B')" in full
    assert "ty: TensorType" in full


def test_invalid_field_filter():
    with pytest.raises(TypeError, match="field names"):
        ir.format_tree(None, hide_fields="loc")


if __name__ == "__main__":
    tvm_tirx.testing.main()
