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
from tvm_tirx import ir, tirx
from tvm_tirx.script import from_source


@py_class("testing.tree.Base")
class TreeBase(tvm_ffi.Object):
    label: str


@py_class("testing.tree.Extension")
class TreeExtension(TreeBase):
    child: Any
    count: Any = 0


@py_class("testing.tree.Binding")
class TreeBinding(tvm_ffi.Object):
    declaration: Any = field(structural_eq="def-simple")
    value: Any


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quick_start(device):
    # Parse the actual definitions without compilation, device allocation, or export.
    root = Path(__file__).resolve().parents[3]
    source = (root / f"examples/quick_start/add_one_{device}.py").read_text()
    source = source.split("# [run.begin]")[0]
    source = source[source.index("from tvm_tirx.script") :]
    function = from_source(source)
    text = ir.format_tree(function)
    assert "params[0]: A = Var(ty=TensorType" in text
    assert "params[1]: B = Var(ty=TensorType" in text
    assert text.count("shape=[n]") == 2  # Folded definitions use names too.
    expanded = ir.format_tree(function, expand=[(ir.Var, "ty")])
    assert expanded.count("shape[0]: n = Var") == 2  # Repeated expanded pattern occurrences.
    assert "SeqStmt" in text
    assert "TensorStore(dest=B, indices=[i])" in text
    assert "├── a: TensorLoad(source=A, indices=[i])" in text
    assert "└── b: FloatImm(value=1.0)" in text
    assert "loc:" not in text
    assert "global_symbol" not in text
    if device == "cpu":
        assert "For(begin=IntImm(value=0), end=n)" in text
        assert "var: i = Var(ty=int32)" in text
        assert "var=i" not in text
        assert "step=" not in text and "kind=" not in text and "annotations" not in text
    else:
        assert "RegionStmt(op=tirx.device_entry)" in text and "If" in text and "cuda" in text
        assert "index=" not in text and "args_info" not in text
        assert "op: Op(index=" in ir.format_tree(function, expand=["op"])
    ir.assert_structural_equal(function, from_source(source))
    assert "def add_one" in function.script()


def test_var_definition_and_type_expansion():
    var = ir.Var("i", "int32")
    assert ir.format_tree(var) == "i = Var(ty=int32)"
    assert ir.format_tree(var, expand=[(ir.Var, "ty")]) == (
        "i = Var\n└── ty: PrimType(dtype=int32)"
    )
    assert ir.format_tree(var, hide=[(ir.Var, "ty")]) == "i = Var()"
    assert ir.format_tree(ir.Var("a long name", "int32")) == "'a long name' = Var(ty=int32)"


def test_extension_fields_and_defaults():
    extension = TreeExtension(label="extension", child=ir.Var("x", "int32"))
    text = ir.format_tree(extension)
    assert "Extension(label='extension')" in text  # Includes inherited fields.
    assert "child: x" in text
    assert "count=" not in text
    assert "count=0" in ir.format_tree(extension, show_defaults=True)
    assert "count: 0" in ir.format_tree(extension, expand=["count"])
    extension.count = 2
    assert "count=2" in ir.format_tree(extension)
    extension.count = False
    assert "count=False" in ir.format_tree(extension)  # 0 is not structurally False.
    assert "value=0" in ir.format_tree(ir.IntImm("int32", 0))  # No declared default.


def test_rule_overlay_specificity_and_nonmutation():
    var = ir.Var("i", "int32")
    defaults = (ir.DEFAULT_FOLD, ir.DEFAULT_EXPAND, ir.DEFAULT_HIDE)
    assert all(isinstance(selectors, tuple) for selectors in defaults)
    formatters = dict(ir.DEFAULT_FOLD_FORMATTERS)
    baseline = ir.format_tree(var)
    assert ir.format_tree(var, fold=None, expand=None, hide=None, fold_formatters=None) == baseline
    assert ir.format_tree(var, fold=[], expand=[], hide=[], fold_formatters={}) == baseline
    # A broad override cannot erase the more-specific default Var.ty rule.
    assert ir.format_tree(var, expand=[(ir.Expr, "ty")]) == baseline
    assert ir.format_tree(var, hide=["ty"]) == baseline
    assert "ty: PrimType" in ir.format_tree(var, expand=[(ir.Var, "ty")])
    extension = TreeExtension(label="extension", child=ir.PrimType("int32"))
    fold, expand, hide = ["child"], [(TreeBase, "child")], [(TreeExtension, "child")]
    assert "child" not in ir.format_tree(extension, fold=fold, expand=expand, hide=hide)
    assert (fold, expand, hide) == (["child"], [(TreeBase, "child")], [(TreeExtension, "child")])
    # Bare and Object forms normalize to one key before caller precedence.
    text = ir.format_tree(
        extension, hide=["child"], fold=[(tvm_ffi.Object, "child")], expand=["child"]
    )
    assert "child: PrimType(dtype=int32)" in text
    assert "child=int32" in ir.format_tree(extension, hide=["child"], fold=["child"])
    assert (ir.DEFAULT_FOLD, ir.DEFAULT_EXPAND, ir.DEFAULT_HIDE) == defaults
    assert ir.DEFAULT_FOLD_FORMATTERS == formatters


def test_bare_selector_overrides_default_fold():
    load = ir.TensorLoad(ir.Var("A", tirx.TensorType([4], "float32")), [ir.IntImm("int32", 0)])
    assert "source=A" in ir.format_tree(load)
    assert "source: A = Var(ty=TensorType" in ir.format_tree(load, expand=["source"])
    assert "source: A = Var(ty=TensorType" in ir.format_tree(
        load, expand=[(ir.TensorLoad, "source")]
    )
    assert "source: A = Var()" in ir.format_tree(load, expand=["source"], hide=[(ir.Var, "ty")])
    expanded = ir.format_tree(load, expand=["source", (ir.Var, "ty")])
    assert "ty: TensorType" in expanded and "shape[0]" in expanded
    assert "dtype: PrimType(dtype=float32)" in expanded
    extension = TreeExtension(label="load", child=load)
    assert "child=TensorLoad(source=A," in ir.format_tree(
        extension, fold=["child"], expand=["source"]
    )
    assert "source" not in ir.format_tree(load, hide=["source"])


def test_recursive_fold_is_sticky():
    inner = TreeExtension(label="inner", child=ir.PrimType("int32"))
    outer = TreeExtension(label="outer", child=inner)
    text = ir.format_tree(outer, fold=[(TreeExtension, "child")], expand=[(TreeBase, "label")])
    assert "child=Extension(label='inner', child=int32)" in text
    assert "label: 'outer'" in text
    assert "inner" in text and "..." not in text
    assert "child=Extension(child=int32)" in ir.format_tree(outer, fold=["child"], hide=["label"])
    seq = ir.SeqStmt([ir.Evaluate(ir.IntImm("int32", 1))])
    assert ir.format_tree(seq, fold=["seq"], expand=[(ir.Evaluate, "value")]) == (
        "SeqStmt(seq=[Evaluate(value=IntImm(value=1))])"
    )


def test_fold_formatters_inheritance_and_expansion():
    inner = TreeExtension(label="inner", child=None)
    outer = TreeExtension(label="outer", child=inner)
    callbacks = {TreeBase: lambda obj: obj.label.upper()}
    assert "child=INNER" in ir.format_tree(outer, fold=["child"], fold_formatters=callbacks)
    assert "child: Extension(label='inner')\n    └── child: None" in ir.format_tree(
        outer, expand=["child"], fold_formatters=callbacks
    )
    callbacks[TreeExtension] = lambda obj: "specific"
    assert "child=specific" in ir.format_tree(outer, fold=["child"], fold_formatters=callbacks)
    assert len(callbacks) == 2
    # Most-specific standard PrimType callback wins over a broad caller one.
    var = ir.Var("i", "int32")
    assert ir.format_tree(var, fold_formatters={ir.Type: lambda obj: "type"}) == "i = Var(ty=int32)"
    assert (
        ir.format_tree(var, fold_formatters={ir.PrimType: lambda obj: "custom"})
        == "i = Var(ty=custom)"
    )
    assert ir.format_tree(
        ir.PrimType("int32"), fold_formatters={ir.PrimType: lambda obj: "custom"}
    ) == ("PrimType(dtype=int32)")


def test_reflected_binding_sites_and_indexed_edges():
    var = ir.Var("item", "int32")
    # Binding roles come from reflection, including on downstream arbitrary names.
    binding = TreeBinding(declaration=[var, var], value=var)
    text = ir.format_tree(binding)
    assert "declaration[0]: item = Var(ty=int32)" in text
    assert "declaration[1]: item = Var(ty=int32)" in text
    assert "value: item" in text
    assert "declaration=[item, item]" in ir.format_tree(binding, fold=["declaration"])
    seq = ir.SeqStmt([ir.Evaluate(var), ir.Evaluate(var)])
    text = ir.format_tree(seq)
    assert "seq[0]: Evaluate" in text and "seq[1]: Evaluate" in text
    assert text.count("value: item") == 2


def make_loop(dtype="int32", step=1, kind=ir.ForKind.DEFAULT, annotations=None):
    return ir.For(
        ir.Var("i", dtype),
        ir.IntImm(dtype, 0),
        ir.IntImm(dtype, 4),
        ir.IntImm(dtype, step),
        kind,
        [],
        annotations=annotations,
    )


def test_native_loop_defaults_and_structural_comparison():
    fields = {f.name: f for f in ir.For.__tvm_ffi_type_info__.fields}
    loop = make_loop()
    for name in ("kind", "step", "annotations"):
        assert fields[name].c_has_default
        assert tvm_ffi.structural_equal(
            getattr(loop, name), fields[name].c_default, map_free_vars=False
        )
    assert not loop.step.same_as(fields["step"].c_default)  # Equal, separately allocated.
    assert not loop.annotations.same_as(fields["annotations"].c_default)
    text = ir.format_tree(loop)
    assert "var: i = Var(ty=int32)" in text
    assert "For(var=i," in ir.format_tree(loop, fold=[(ir.For, "var")])
    assert "step=" not in text and "kind=" not in text and "annotations" not in text
    assert "step=IntImm(value=1)" in ir.format_tree(loop, show_defaults=True)
    assert "kind=ForKind(DEFAULT)" in ir.format_tree(loop, show_defaults=True)
    assert "annotations: {}" in ir.format_tree(loop, expand=[(ir.For, "annotations")])
    assert "step=IntImm(value=1)" in ir.format_tree(loop, fold=[(ir.For, "step")])
    assert "step=IntImm(value=2)" in ir.format_tree(make_loop(step=2))
    assert "step=IntImm(value=1)" in ir.format_tree(make_loop(dtype="int64"))
    assert "kind=ForKind(PARALLEL)" in ir.format_tree(make_loop(kind=ir.ForKind.PARALLEL))
    assert "'tag': 1" in ir.format_tree(make_loop(annotations={"tag": 1}))
    # Metadata defaults also participate in generic reflected construction.
    constructed = ir.make_node(
        "ir.For", var=loop.var, begin=loop.begin, end=loop.end, body=loop.body, loc=loop.loc
    )
    ir.assert_structural_equal(loop, constructed)
    ir.assert_structural_equal(loop, ir.load_json(ir.save_json(loop)))
    ir.assert_structural_equal(
        make_loop(dtype="int64"), ir.load_json(ir.save_json(make_loop(dtype="int64")))
    )


@py_class("testing.tree.DefaultVar")
class TreeDefaultVar(tvm_ffi.Object):
    value: Any = ir.Var("default", "int32")


def test_default_free_variables_are_not_remapped():
    assert "value" not in ir.format_tree(TreeDefaultVar())
    other = ir.Var("default", "int32")
    assert "value: default" in ir.format_tree(TreeDefaultVar(value=other))


def test_explicit_expansion_through_indexed_edges():
    var = ir.Var("item", "int32")
    extension = TreeExtension(label="uses", child=[var, var])
    assert "child[0]: item" in ir.format_tree(extension)
    expanded = ir.format_tree(extension, expand=["child"])
    assert expanded.count("item = Var(ty=int32)") == 2
    assert "child[1]: item = Var" in expanded


def test_locations_and_containers():
    var = ir.Var("x", "int32")
    assert "loc: UnknownLoc()" in ir.format_tree(var, expand=["loc"])
    assert "loc" not in ir.format_tree(var, show_defaults=True)
    assert "loc" not in ir.format_tree(var, hide=[])
    assert ir.format_tree({"z": 1, "a": 2}) == "dict\n├── 'a': 2\n└── 'z': 1"
    assert ir.format_tree(tvm_ffi.convert({"z": 1, "a": 2})) == "Map\n├── 'a': 2\n└── 'z': 1"
    assert "'loc': x" in ir.format_tree({"loc": var}, hide=["loc"])
    assert ir.format_tree([]) == "[]" and ir.format_tree({}) == "{}"
    assert ir.format_tree(object()) == "<object>"
    assert "'x\\ny'" in ir.format_tree([0, False, None, "x\ny"])
    text = ir.format_tree(tvm_ffi.convert({var: 7}))
    assert "[0][0]: x" in text and "[0][1]: 7" in text
    text = ir.format_tree([var, var])
    assert text.count("x") == 2 and "<cycle>" not in text


@pytest.mark.parametrize("folded", [False, True])
def test_depth_and_cycles(folded):
    options = {"fold": ["child"]} if folded else {}
    nested = TreeExtension(label="outer", child=[[[1]]])
    assert ir.format_tree([[[1]]], max_depth=0) == "list(...)"
    text = ir.format_tree(nested, max_depth=1, **options)
    assert "Array(...)" in text and "1" not in text
    assert ir.format_tree(1, max_depth=0) == "1"
    assert ir.format_tree(ir.Var("i", "int32"), max_depth=0) == "i = Var(...)"
    cycle = []
    cycle.append(cycle)
    assert "<cycle>" in ir.format_tree(cycle)
    extension = TreeExtension(label="cycle", child=None)
    extension.child = extension
    try:
        assert "<cycle>" in ir.format_tree(extension, **options)
    finally:
        extension.child = None


@pytest.mark.parametrize("depth", [-1, 1.5, True])
def test_invalid_depth(depth):
    with pytest.raises(ValueError, match="max_depth"):
        ir.format_tree(None, max_depth=depth)


@pytest.mark.parametrize(
    "option,selectors",
    [
        ("fold", "ty"),
        ("expand", [""]),
        ("hide", [(ir.Var,)]),
        ("fold", [(int, "ty")]),
        ("expand", [(ir.Var, 1)]),
        ("hide", [1]),
    ],
)
def test_invalid_selectors(option, selectors):
    with pytest.raises(TypeError):
        ir.format_tree(None, **{option: selectors})


@pytest.mark.parametrize("callbacks", [[], {int: str}, {ir.Type: None}])
def test_invalid_callbacks(callbacks):
    with pytest.raises(TypeError, match="fold_formatters"):
        ir.format_tree(None, fold_formatters=callbacks)


@pytest.mark.parametrize("result", [1, "line\nline"])
def test_invalid_callback_output(result):
    with pytest.raises(TypeError, match="single-line text"):
        ir.format_tree(ir.Var("i", "int32"), fold_formatters={ir.PrimType: lambda obj: result})


def test_ffi_enum_renders_as_type_and_name():
    tensor_type = tirx.TensorType((4,), "float32", scope="shared")
    assert ir.format_tree(tensor_type).startswith("TensorType(scope=MemoryScope(shared))")


if __name__ == "__main__":
    tvm_tirx.testing.main()
