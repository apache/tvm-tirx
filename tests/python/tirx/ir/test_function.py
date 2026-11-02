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
"""Test TIRx function structural operations."""

import pytest
import tvm_ffi

from tvm_tirx import ir, tirx


@pytest.mark.parametrize("as_map_key", [False, True])
def test_parameter_attr_json_roundtrip(as_map_key):
    x = ir.Var("x", "int32")
    func = tirx.Function([x], [ir.Return(x + 1)], ir.PrimType("int32"))
    func = func.with_attr("parameter", {x: 7} if as_map_key else x)

    restored = ir.load_json(ir.save_json(func))
    attr = restored.attrs["parameter"]
    restored_ref = next(iter(attr)) if as_map_key else attr
    assert restored_ref.same_as(restored.params[0])
    assert not restored.params[0].same_as(x)
    ir.assert_structural_equal(func, restored)
    ir.assert_structural_equal(restored, func)
    assert tvm_ffi.structural_hash(func) == tvm_ffi.structural_hash(restored)


@pytest.mark.parametrize("as_map_key", [False, True])
def test_parameter_attr_alpha_equivalence(as_map_key):
    def make_function(name):
        x = ir.Var(name, "int32")
        func = tirx.Function([x], [ir.Return(x + 1)], ir.PrimType("int32"))
        return func.with_attr("parameter", {x: 7} if as_map_key else x)

    lhs = make_function("x")
    rhs = make_function("renamed")
    ir.assert_structural_equal(lhs, rhs)
    ir.assert_structural_equal(rhs, lhs)
    assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)


@pytest.mark.parametrize("as_map_key", [False, True])
def test_parameter_attr_distinguishes_parameters(as_map_key):
    x = ir.Var("x", "int32")
    y = ir.Var("x", "int32")
    func = tirx.Function([x, y], [ir.Return(x + y)], ir.PrimType("int32"))
    lhs = func.with_attr("parameter", {x: 7} if as_map_key else x)
    rhs = func.with_attr("parameter", {y: 7} if as_map_key else y)

    assert not tvm_ffi.structural_equal(lhs, rhs)
    assert not tvm_ffi.structural_equal(rhs, lhs)


@pytest.mark.parametrize("as_map_key", [False, True])
def test_function_attr_preserves_free_variable_identity(as_map_key):
    def make_function(free_var):
        x = ir.Var("x", "int32")
        func = tirx.Function([x], [ir.Return(x + 1)], ir.PrimType("int32"))
        return func.with_attr("free_var", {free_var: 7} if as_map_key else free_var)

    free_var = ir.Var("x", "int32")
    lhs = make_function(free_var)
    rhs = make_function(free_var)
    other = make_function(ir.Var("x", "int32"))
    ir.assert_structural_equal(lhs, rhs)
    assert tvm_ffi.structural_hash(lhs) == tvm_ffi.structural_hash(rhs)
    assert not tvm_ffi.structural_equal(lhs, other)
    assert not tvm_ffi.structural_equal(other, lhs)


def test_structural_map_specializes_function_signature():
    from tvm_tirx.ir import Evaluate, FuncType, IntImm, Return, Var, assert_structural_equal
    from tvm_tirx.tirx import Function, TensorType

    n = Var("n", "int32")
    scale = Var("scale", "int32")
    tensor = Var("tensor", TensorType((n,), "float32"))
    original = Function([tensor, scale], [Evaluate(scale), Return(tensor)], tensor.ty)
    bindings = {n: IntImm("int32", 16), scale: IntImm("int32", 2)}

    # Removing runtime parameters is explicit. structural_map handles the shape
    # definitions embedded in tensor types and remaps the retained tensor's uses.
    signature = Function([tensor], original.body, original.ret_type)
    specialized = tvm_ffi.structural_map(
        signature, (Var, lambda var: bindings.get(var, var)), order="pre"
    )
    expected_tensor = Var("tensor", TensorType((16,), "float32"))
    expected = Function(
        [expected_tensor],
        [Evaluate(IntImm("int32", 2)), Return(expected_tensor)],
        expected_tensor.ty,
    )
    assert_structural_equal(specialized, expected)
    assert specialized.body.seq[1].value.same_as(specialized.params[0])
    assert_structural_equal(
        specialized.ty, FuncType([p.ty for p in specialized.params], specialized.ret_type)
    )
    assert original.params[0].ty.shape[0].same_as(n)
    assert original.ty.arg_types[0].shape[0].same_as(n)

    # A unique declaration allows the parameter array and its Var to remain in
    # place while a nested TensorType changes. The cached FuncType must follow.
    def declaration():
        return Function(
            [Var("tensor", TensorType((n,), "float32"))], [], TensorType((n,), "float32")
        ).with_attr("declaration", True)

    inplace = declaration()
    inplace = tvm_ffi.structural_map(
        inplace._move(), (Var, lambda var: bindings.get(var, var)), order="pre"
    )
    assert int(inplace.params[0].ty.shape[0]) == 16
    assert int(inplace.ret_type.shape[0]) == 16
    assert_structural_equal(inplace.ty, FuncType([inplace.params[0].ty], inplace.ret_type))
