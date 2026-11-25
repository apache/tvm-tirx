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

"""Compilation through the native TIRx pipeline."""

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize("pipeline", [None, "default", "tirx"])
def test_default_pipeline_allocations(pipeline):
    @T.function
    def add_one(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        temp = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(16):
            temp[i] = A[i] + T.float32(1)
        for i in range(16):
            B[i] = temp[i]

    module = tvm_tirx.compile(add_one, target="llvm", pipeline=pipeline)
    source = np.arange(16, dtype="float32")
    a = tvm_tirx.runtime.tensor(source)
    b = tvm_tirx.runtime.tensor(np.zeros(16, dtype="float32"))
    module(a, b)
    np.testing.assert_equal(b.numpy(), source + 1)


@pytest.mark.parametrize("print_after_dispatch", [None, "", "1"])
def test_explicit_tile_pass_order_and_diagnostic(monkeypatch, capfd, print_after_dispatch):
    from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

    if print_after_dispatch is None:
        monkeypatch.delenv("TVM_TIRX_PRINT_AFTER_DISPATCH_OPS", raising=False)
    else:
        monkeypatch.setenv("TVM_TIRX_PRINT_AFTER_DISPATCH_OPS", print_after_dispatch)

    @T.function
    def kernel(A: T.Tensor((2, 4), "float32", layout="default")):
        for i, j in T.grid(2, 4):
            A[i, j] = T.float32(1)

    visited = []
    layouts = {}

    @tvm_tirx.ir.transform.pass_instrument
    class RecordPasses:
        def run_after_pass(self, mod, info):
            visited.append(info.name)
            if info.name in ("tirx.TileDispatch", "tirx.LowerTensorLayouts"):
                layouts[info.name] = mod.script()

    pipeline, _, _ = tirx_pipeline(prepare_only=True, target=tvm_tirx.target.Target("llvm"))
    with tvm_tirx.ir.transform.PassContext(instruments=[RecordPasses()]):
        pipeline(tvm_tirx.IRModule({"main": kernel}))

    start = visited.index("tirx.TileDispatch")
    expected = ["tirx.TileDispatch"]
    if print_after_dispatch is not None:
        expected.append("PrintIR")
    expected.extend(["tirx.LowerTensorLayouts", "tirx.StmtSimplify", "tirx.FlattenTensor"])
    assert visited[start : start + len(expected)] == expected
    assert "TileLayout" in layouts["tirx.TileDispatch"]
    assert "TileLayout" not in layouts["tirx.LowerTensorLayouts"]
    output = capfd.readouterr().err
    assert ("PrintIR(" in output) == (print_after_dispatch is not None)
    if print_after_dispatch is not None:
        assert "TileLayout" in output


def _prepare_for_cuda(func):
    from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"})
    pipeline, _, _ = tirx_pipeline(prepare_only=True, target=target)
    return pipeline(tvm_tirx.IRModule({"main": func}))


def test_verify_memory_rejects_host_store_to_gpu_tensor():
    @T.function
    def kernel(A: T.Tensor((1,), "float32")):
        A[0] = T.float32(1)

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(kernel)


def test_verify_memory_rejects_host_store_to_gpu_tensor_view():
    @T.function
    def kernel(A: T.Tensor((1, 1), "float32")):
        A_view = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), "float32")])
        A_view[0] = T.float32(1)

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(kernel)


def test_verify_memory_accepts_host_address_of_gpu_tensor_views():
    @T.function
    def kernel(A: T.Tensor((4, 4), "float32"), index: T.int32):
        A_decl = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32")])
        A_view = A_decl.view(16)
        T.call_extern("consume_pointer", T.address_of(A_view[index]), ty="void")

    _prepare_for_cuda(kernel)


def test_pipeline_rejects_unsupported_managed_ownership():
    @T.function
    def parallel(arg: T.Any) -> T.int32:
        for i in T.parallel(4):
            x: T.let[T.Any] = T.call_packed("f", arg, i, ty=T.Any)
            T.call_packed("g", x, ty="void")
        return 0

    with pytest.raises(ValueError, match="Managed owners require sequential loops"):
        tvm_tirx.compile(parallel, target="c")


@pytest.mark.parametrize("packed", [T.call_packed, T.call_cpacked])
def test_packed_call_placement(packed):
    @T.function
    def nested_packed(arg: T.int32) -> T.int32:
        value: T.let[T.int32] = T.call_packed("outer", arg, packed("inner", ty="int32"), ty="int32")
        return value

    @T.function
    def nested_ordinary(arg: T.int32) -> T.int32:
        return T.call_extern("outer", arg, packed("inner", ty="int32"), ty="int32")

    @T.function
    def direct() -> T.int32:
        return packed("value", ty="int32")

    @T.function
    def direct_managed() -> T.Any:
        return packed("value", ty=T.Any)

    for func in (nested_packed, nested_ordinary):
        with pytest.raises(ValueError, match="A packed call cannot be used inside a call argument"):
            tvm_tirx.compile(func, target="c")
    for func in (direct, direct_managed):
        with pytest.raises(ValueError, match="A packed call cannot be returned directly"):
            tvm_tirx.compile(func, target="c")
        with pytest.raises(ValueError, match="A packed call cannot be returned directly"):
            tvm_tirx.tirx.transform.LowerFFIABI()(tvm_tirx.IRModule({"main": func}))


def test_expression_setup_diagnostics():
    @T.function
    def true_arm(c: T.bool) -> T.int32:
        return T.if_then_else(c, T.call_packed("value", ty="int32"), 0)

    @T.function
    def false_arm(c: T.bool) -> T.int32:
        return T.if_then_else(c, 0, T.call_packed("value", ty="int32"))

    @T.function
    def while_condition():
        while T.call_packed("again", ty="bool"):
            T.call_packed("iteration", ty="void")

    @T.function
    def let_body(input_value: T.int32) -> T.int32:
        x = T.int32()
        return T.Let(x + T.call_packed("value", ty="int32"), where={x: input_value})

    # The script builder emits standalone calls for explicit void expressions.
    # Construct a discarded conditional with the calls still inside its arms.
    condition = tvm_tirx.ir.Var("condition", "bool")
    void_call = tvm_tirx.tirx.call_packed("mark", ty="void")
    discarded = tvm_tirx.ir.Call("prim.if_then_else", [condition, void_call, void_call], ty="void")
    void_arms = tvm_tirx.tirx.Function(
        [condition], [tvm_tirx.ir.Evaluate(discarded)], tvm_tirx.ir.PrimType("void")
    )

    for func, diagnostic in (
        (void_arms, "true and false values of `if_then_else`"),
        (true_arm, "true and false values of `if_then_else`"),
        (false_arm, "true and false values of `if_then_else`"),
        (while_condition, "The `while` condition requires setup statements"),
        (let_body, "The body of this `let` expression requires setup statements"),
    ):
        with pytest.raises(ValueError, match=diagnostic):
            tvm_tirx.tirx.transform.LowerFFIABI()(tvm_tirx.IRModule({"main": func}))


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_ordered_packed_calls_and_statement_alternatives(target):
    if target == "llvm" and not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("LLVM is not enabled")

    @T.function
    def ordered(a: T.Tensor((1,), "int32")) -> T.float64:
        prior: T.let[T.float64] = T.float64(a[0])
        exponent: T.let[T.int32] = T.call_packed("test.operand_update", ty="int32")
        return T.call_extern("ldexp", prior, exponent, ty="float64")

    @T.function
    def branch(c: T.bool) -> T.int32:
        if c:
            yes: T.let[T.int32] = T.call_packed("test.record_operand", 1, ty="int32")
            return yes
        else:
            no: T.let[T.int32] = T.call_packed("test.record_operand", 2, ty="int32")
            return no

    @T.function
    def loop():
        while True:
            value: T.let[T.int32] = T.call_packed("test.record_operand", 3, ty="int32")
            if value >= 3:
                break
            if value == 1:
                continue
            T.call_packed("test.record_operand", 4, ty="void")

    @T.function
    def condition() -> T.int32:
        return T.if_then_else(T.call_packed("test.record_operand", 5, ty="bool"), 7, 11)

    @T.function
    def initializer() -> T.int32:
        x = T.int32()
        return T.Let(x + 1, where={x: T.call_packed("test.record_operand", 6, ty="int32")})

    values = tvm_tirx.runtime.tensor(np.array([3], dtype="int32"))
    calls = []

    @tvm_ffi.register_global_func("test.operand_update")
    def update():
        values.copyfrom(np.array([11], dtype="int32"))
        return 0

    @tvm_ffi.register_global_func("test.record_operand")
    def record(kind):
        calls.append(kind)
        if kind == 3:
            return calls.count(3)
        if kind == 5:
            return True
        return kind

    try:
        assert tvm_tirx.compile(ordered, target=target)(values) == 3.0
        assert values.numpy()[0] == 11
        invoke = tvm_tirx.compile(branch, target=target)
        assert invoke(True) == 1
        assert calls == [1]
        assert invoke(False) == 2
        assert calls == [1, 2]
        tvm_tirx.compile(loop, target=target)()
        assert calls == [1, 2, 3, 3, 4, 3]
        assert tvm_tirx.compile(condition, target=target)() == 7
        assert tvm_tirx.compile(initializer, target=target)() == 7
        assert calls[-2:] == [5, 6]
    finally:
        tvm_ffi.registry.remove_global_func("test.record_operand")
        tvm_ffi.registry.remove_global_func("test.operand_update")


if __name__ == "__main__":
    tvm_tirx.testing.main()
