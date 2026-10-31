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
"""Python construction semantics, shared declarations and source diagnostics."""

import linecache
from types import SimpleNamespace

import pytest

from tvm_tirx import ir
from tvm_tirx.script import from_source
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder


def test_builder_python_semantics_and_sugar_round_trip(monkeypatch):
    visited = []
    counter = 0
    monkeypatch.setitem(globals(), "_builder_global", 0)

    def read_counter():
        return counter

    @T.function.builder(private=True)
    def built(A: T.Tensor((4,), T.int32)):
        global _builder_global
        nonlocal counter
        counter += 1
        _builder_global += 1

        def bump():
            nonlocal counter
            counter += 1

        bump()
        assert read_counter() == 2 and globals()["_builder_global"] == 1
        _host0 = _L_0 = 7
        assert _host0 == _L_0 == 7
        I = SimpleNamespace(constexpr=lambda value: visited.append(value) or value)  # noqa: E741
        assert I.constexpr(7) == 7 and visited == [7]
        assert not (False and visited.append("and"))
        assert True or visited.append("or")
        assert (3 if True else visited.append("if")) == 3
        I.value = 7
        match 7:
            case I.value if visited.append("guard") is None:
                assert visited == [7, "guard"]
            case _:
                raise AssertionError("unreachable")
        visited.clear()
        indices = [0, 1, 2]
        indices[1:2] = [3]
        visited.append(first := indices[0])
        local_values = locals
        assert first == 0 and local_values()["first"] == 0
        visited.clear()
        for offset in indices:
            if offset == 3:
                continue
            visited.append(offset)
            T.assign(A[0], offset)
        count = 0
        while count < 2:
            count += 1
            if count == 2:
                break
        assert count == 2 and visited == [0, 2]
        with T.device_entry():
            with T.grid(2, 2) as (i, j):
                value = T.bind_(i + j, name="value")
                with T.if_(value > 0):
                    with T.then_():
                        T.assign(A[value], value + 1)
                    with T.else_():
                        T.assign(A[value], 0)
        return
        T.assign(A[0], 99)

    @T.function(private=True)
    def expected(A: T.Tensor((4,), T.int32)):
        A[0] = 0
        A[0] = 2
        with T.device_entry():
            for i, j in T.grid(2, 2):
                value = i + j
                if value > 0:
                    T.assign(A[value], value + 1)
                else:
                    T.assign(A[value], 0)

    assert visited == [0, 2] and counter == 2 and _builder_global == 1
    ir.assert_structural_equal(built, expected)
    printed = built.script()
    assert "@T.function(private=True)" in printed and ".builder" not in printed
    ir.assert_structural_equal(built, from_source(printed, extra_vars={"T": T}))


@pytest.mark.parametrize("namespace", [I, T])
def test_builder_mixed_module_forward_declarations(namespace):
    helper = SimpleNamespace(leaf=lambda value: value + 2)

    @namespace.ir_module
    class Mixed:
        @T.function.builder
        def built(A: T.Tensor((4,), T.int32)):
            def leaf(value):
                return value + 1

            assert leaf(4) == 5
            module = Mixed
            with T.serial(4) as i:
                T.assign(A[i], T.call_global_var_(module.sugar, [i]))
            T.call_global_var_(Mixed.store, [A])
            for module in [helper]:
                assert module.leaf(4) == 6
            from math import floor as leaf

            assert leaf(4.1) == 4

        @T.function
        def sugar(value: T.int32) -> T.int32:
            return Mixed.leaf(value)

        @T.function.builder(private=True)
        def leaf(value: T.int32) -> T.int32:
            T.return_(value + 1)

        @T.function.builder
        def store(A: T.Tensor((4,), T.int32)):
            T.assign(A[0], 1)

    call = Mixed["built"].body[0].body[0].value
    assert call.op.same_as(Mixed.get_global_var("sugar"))
    assert Mixed["sugar"].body[0].value.op.same_as(Mixed.get_global_var("leaf"))
    assert len(Mixed["built"].body) == 2
    assert Mixed["built"].body[1].value.op.same_as(Mixed.get_global_var("store"))
    ir.assert_structural_equal(Mixed, from_source(Mixed.script(), extra_vars={"T": T, "I": I}))


def test_builder_annotations_options_and_locations(monkeypatch):
    checked = []
    monkeypatch.setattr(T, "check_well_formed_", checked.append)
    dtype = T.int32
    tensor = T.Tensor((4,), dtype)

    def helper(A):
        T.assign(A[0], A[0] + 1)

    @T.function.builder(private=True, persistent=True)
    def built(A: "tensor", /, value: "dtype", *, step: "dtype") -> "dtype":
        with T.serial(4) as i:
            T.assign(A[i], (value + step) * (i + 1))
        helper(A)
        T.return_(value)

    assert checked == [built]
    assert "global_symbol" not in built.attrs and built.attrs["tirx.persistent_kernel"] == 1
    assert [parameter.name for parameter in built.params] == ["A", "value", "step"]
    assert built.params[0].ty.same_as(tensor) and built.ret_type == ir.PrimType("int32")
    parameter = built.params[0].loc
    assert (
        linecache.getline(__file__, parameter.start_line)[
            parameter.start_column - 1 : parameter.end_column - 1
        ]
        == 'A: "tensor"'
    )
    loop, helper_store, _ = built.body.seq
    store = loop.body[0]
    expression = store.value.a.loc
    assert (
        linecache.getline(__file__, expression.start_line)[
            expression.start_column - 1 : expression.end_column - 1
        ]
        == "value + step"
    )
    assert "T.assign(A[i]" in linecache.getline(__file__, store.loc.start_line)
    assert "helper(A)" in linecache.getline(__file__, helper_store.loc.start_line)
    assert isinstance(helper_store.value.loc, ir.UnknownLoc)

    source = (
        "from __future__ import annotations\n"
        "@T.function.builder(check_well_formed=False)\n"
        "def future(n: T.int32, A: T.Tensor((n,), T.int32)): T.assign(A[0], n)\n"
    )
    future = from_source(source, extra_vars={"T": T})
    assert future.params[0].ty == ir.PrimType("int32") and checked == [built]
    with pytest.raises(RuntimeError, match="body failed"):

        @T.function.builder
        def failed():
            T.evaluate(0)
            raise RuntimeError("body failed")

    assert not IRBuilder.is_in_scope()


@pytest.mark.parametrize(
    "source, message",
    [
        ("def f(value): pass", "requires an annotation"),
        ("def f(value: T.int32 = 1): pass", "cannot have defaults"),
        ("def f(*values: T.int32): pass", "ordinary named parameters"),
        ("def f(): yield 1", "Unsupported expression: Yield"),
    ],
)
def test_builder_source_errors_and_cleanup(source, message):
    with pytest.raises(SyntaxError, match=message) as error:
        from_source("@T.function.builder\n" + source, extra_vars={"T": T})
    assert (error.value.filename, error.value.lineno) == ("<str>", 2)
    assert not IRBuilder.is_in_scope()


@pytest.mark.parametrize(
    "context, sugar",
    [
        ("serial(4)", "for"),
        ("parallel(4)", "for"),
        ("unroll(4)", "for"),
        ("vectorized(4)", "for"),
        ("grid(2, 3)", "for"),
        ("for_(range(4))", "for"),
        ("if_(True)", "if"),
        ("while_(True)", "while"),
    ],
)
def test_parser_rejects_explicit_control_frames_at_source(context, sugar):
    source = (
        f"@X.function\ndef f():\n    with X.device_entry(), X.{context}:\n        X.evaluate(0)\n"
    )
    with pytest.raises(
        SyntaxError, match=f"use a {sugar} statement or @T.function.builder"
    ) as error:
        from_source(source, extra_vars={"X": T})
    assert (error.value.filename, error.value.lineno) == ("<str>", 3)
    assert error.value.offset == source.splitlines()[2].index(f"X.{context}") + 1
    assert error.value.end_offset == len(source.splitlines()[2])
