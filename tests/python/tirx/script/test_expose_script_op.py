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
"""TVMScript builders generated for registered operators, and classes for registered types."""

import pytest
import tvm_ffi
from tvm_ffi.dataclasses import py_class

import tvm_tirx
from tvm_tirx.ir import Call, Op, PrimType, Type, assert_structural_equal, register_op_attr
from tvm_tirx.script import parser
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.script import ir_builder as builder


def define(name, args, result):
    """Register an operator once per process, as the registry keeps operators for good."""
    if name not in Op.list_op_names():
        tvm_ffi.get_global_func("ir.RegisterOp")(name, f"Test operator {name}.")
        Op.get(name).set_signature(args)
        register_op_attr(name, "TFixedReturnType", PrimType(result))
    return Op.get(name)


@py_class("tirx.TestScriptPairType")
class ScriptPair(Type):
    """A pair of types, printed as ``T.TestScriptPair(first, second)``."""

    first: Type
    second: Type


@py_class("test_script_types.BoxType")
class ScriptBox(Type):
    """A type and a count, printed as ``test_script_types.Box(elem, count)``."""

    elem: Type
    count: int


@py_class("tirx.evaluateType")
class ScriptEvaluate(Type):
    """A type whose printed name, ``T.evaluate``, is taken by a builder."""


@py_class("tirx.TestScriptConstantType")
class ScriptConstant(Type):
    """A type whose printed name, ``T.TestScriptConstant``, a test takes by a non-callable."""


@pytest.fixture
def exposed():
    """The names a test exposes on T, which are removed after it."""
    names = []
    yield names
    for name in names:
        vars(builder).pop(name, None)
        parser._NAMESPACES.pop(name, None)


def test_builders_round_trip_calls_under_every_root(exposed):
    exposed += ["test_script_scale", "test_script_mark", "test_script_ns", "test_script_twice"]
    exposed += ["test_script_root", "test_script_bare"]
    scale = define("tirx.test_script_scale", ["x", "factor"], "int32")
    mark = define("tirx.test_script_mark", ["tag"], "void")
    touch = define("tirx.test_script_ns.touch", ["x"], "void")
    twice = define("ir.test_script_twice", ["x"], "int32")
    other = define("test_script_root.sub.op", ["x"], "void")
    bare = define("test_script_bare", ["x"], "void")
    for op in (scale, mark, touch, twice, other, bare):
        T.expose_script_op(op.name)
    test_script_root, test_script_bare = parser.test_script_root, parser.test_script_bare
    assert T.expose_script_op(scale.name) is T.test_script_scale
    assert T.test_script_scale.__doc__ == scale.doc
    assert_structural_equal(T.test_script_scale(x=1, factor=2), Call(scale, [1, 2]))

    @T.function
    def func(i: T.int32):
        T.evaluate(T.test_script_scale(i, 2))
        T.test_script_mark(i)
        T.test_script_ns.touch(i)
        T.evaluate(T.test_script_twice(i))
        test_script_root.sub.op(i)
        test_script_bare(i)

    text = func.script()
    assert "T.test_script_ns.touch(i)" in text, text
    assert "T.test_script_twice(i)" in text, text
    assert "test_script_root.sub.op(i)" in text, text
    assert "\n    test_script_bare(i)" in text, text
    assert_structural_equal(tvm_tirx.script.parse(text, extra_vars={"T": T}), func)


def test_existing_names_keep_their_values(exposed):
    exposed += ["test_script_written"]
    written = define("tirx.test_script_written", ["x"], "int32")

    def test_script_written(x):
        return Call(written, [x])

    builder.test_script_written = test_script_written
    assert T.expose_script_op(written.name) is test_script_written
    define("tirx.supports_mutable_declarations", ["x"], "int32")
    with pytest.raises(ValueError, match="conflicts with a non-callable"):
        T.expose_script_op("tirx.supports_mutable_declarations")


def test_type_classes_round_trip_types_under_every_root(exposed):
    exposed += ["TestScriptPair", "test_script_types"]
    assert T.expose_script_type("tirx.TestScriptPairType") is ScriptPair
    assert T.expose_script_type("test_script_types.BoxType") is ScriptBox
    assert T.expose_script_type("tirx.TestScriptPairType") is T.TestScriptPair
    test_script_types = parser.test_script_types

    @T.function
    def func(p: T.TestScriptPair(T.int32, T.float32), b: test_script_types.Box(T.int8, 4)):
        T.evaluate(p)
        T.evaluate(b)

    text = func.script()
    assert "p: T.TestScriptPair(T.int32, T.float32)" in text, text
    assert "b: test_script_types.Box(T.int8, 4)" in text, text
    assert_structural_equal(tvm_tirx.script.parse(text, extra_vars={"T": T}), func)


def test_existing_names_keep_their_values_over_types(exposed):
    exposed += ["TestScriptConstant"]
    assert T.expose_script_type("tirx.evaluateType") is builder.evaluate
    builder.TestScriptConstant = 1
    with pytest.raises(ValueError, match="conflicts with a non-callable"):
        T.expose_script_type("tirx.TestScriptConstantType")
