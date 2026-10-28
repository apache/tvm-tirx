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

# The parser consumes named IR bindings that Python sees as unused.
# ruff: noqa: F841
import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

simple_target = tvm_tirx.target.Target({"kind": "llvm", "mtriple": "x86_64-linux-gnu"})
sve_target = tvm_tirx.target.Target(
    {
        "kind": "llvm",
        "device": "arm_cpu",
        "mtriple": "aarch64-linux-gnu",
        "mattr": ["+v8.2a", "+sve"],
    }
)


def _assert_scalar_fallback(mod):
    after = tvm_tirx.tirx.transform.VectorizeLoop()(mod)
    tvm_tirx.ir.assert_structural_equal(after, tvm_tirx.tirx.transform.VectorizeLoop(False)(mod))
    return after


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_loop(extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": target})
            for j in T.vectorized(0, extent):
                A[j] = 1

    @I.ir_module
    class After:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": target})
            A[T.Ramp(0, 1, extent)] = T.Broadcast(1, extent)

    mod = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, After)


def test_fixed_vector_elements_keep_scalar_loop():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((4,), "float32x4"), n: T.int32):
            for i in range(n):
                for j in T.vectorized(4):
                    A[j] = T.Broadcast(T.float32(1), 4)

    _assert_scalar_fallback(Module)


def test_fixed_vector_access_in_scalable_loop_keeps_scalar_loop():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((25,), "float32")):
            T.func_attr({"target": sve_target})
            for j in T.vectorized(T.vscale() * 4):
                A[T.ramp(j * 4, 1, 4)] = T.Broadcast(T.float32(1), 4)

    _assert_scalar_fallback(Module)


def test_scalable_vector_elements_in_fixed_loop_keep_scalar_loop():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((25,), "float32xvscalex4")):
            for j in T.vectorized(4):
                A[j] = T.Broadcast(T.float32(1), T.vscale() * 4)

    _assert_scalar_fallback(Module)


def test_scalable_vector_access_in_fixed_loop_keeps_scalar_loop():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((25,), "float32")):
            T.func_attr({"target": sve_target})
            for j in T.vectorized(4):
                A[T.ramp(j * T.vscale() * 4, 1, T.vscale() * 4)] = T.Broadcast(
                    T.float32(1), T.vscale() * 4
                )

    _assert_scalar_fallback(Module)


def test_scalable_vector_access_in_scalable_loop_keeps_scalar_loop():
    @I.ir_module
    class Module:
        @T.function(private=True)
        def main(A: T.Tensor((25,), "float32")):
            T.func_attr({"target": sve_target})
            for j in T.vectorized(T.vscale() * 4):
                A[T.ramp(j * T.vscale() * 4, 1, T.vscale() * 4)] = T.Broadcast(
                    T.float32(1), T.vscale() * 4
                )

    _assert_scalar_fallback(Module)


def test_if_in_fixed_loop_keeps_scalar_loop():
    extent = 4
    target = simple_target

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32"), n: T.int32, x: T.int32):
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                if x < n:
                    A[i] = A[i] + T.float32(1)
                else:
                    if i < n:
                        A[i] = T.float32(2)

    _assert_scalar_fallback(Before)


def test_if_in_scalable_loop_keeps_scalar_loop():
    extent = T.vscale() * 4
    target = sve_target

    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32"), n: T.int32, x: T.int32):
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                if x < n:
                    A[i] = A[i] + T.float32(1)
                else:
                    if i < n:
                        A[i] = T.float32(2)

    _assert_scalar_fallback(Before)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_let(extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32")):
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                v: T.let = A[i] + T.float32(1)
                A[i] = v + T.float32(2)

    @I.ir_module
    class After:
        @T.function
        def main(A: T.Tensor((25,), "float32")):
            T.func_attr({"target": target})
            v: T.let = A[T.Ramp(0, 1, extent)] + T.Broadcast(T.float32(1), extent)
            A[T.Ramp(0, 1, extent)] = v + T.Broadcast(T.float32(2), extent)

    mod = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
@pytest.mark.parametrize("comparison", ["le", "ge"])
def test_vectorize_with_bound_condition(extent, target, comparison):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "float32"), n: T.int32):
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                if T.constexpr(comparison == "le"):
                    if i <= n:
                        A[i] = A[i] + T.float32(1)
                else:
                    if i >= n:
                        A[i] = A[i] + T.float32(1)

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor((16,), "float32"), n: T.int32):
            T.func_attr({"target": target})
            for i in T.serial(extent):
                if T.constexpr(comparison == "le"):
                    if i <= n:
                        A[i] = A[i] + T.float32(1)
                else:
                    if i >= n:
                        A[i] = A[i] + T.float32(1)

    After = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_varying_if_then_else_keeps_scalar_loop(extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32")):
            T.func_attr({"target": target})
            for i in T.vectorized(extent):
                A[i] = T.if_then_else(i > 0, A[i] + T.float32(1), A[i])

    @I.ir_module
    class After:
        @T.function
        def main(A: T.Tensor((25,), "float32")):
            T.func_attr({"target": target})
            for i_s in range(extent):
                A[i_s] = T.if_then_else(i_s > 0, A[i_s] + T.float32(1), A[i_s])

    mod = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_uniform_if_then_else_keeps_scalar_loop(extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32"), n: T.int32):
            T.func_attr({"target": target})
            for i in range(n):
                for j in T.vectorized(extent):
                    A[i * extent + j] = T.if_then_else(i > 0, A[i * extent + j], 0)

    _assert_scalar_fallback(Before)


def test_vectorize_let_if_then_else():
    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"target": simple_target})
            for i in T.vectorized(4):
                if i < 2:
                    result: T.let[T.int32] = T.if_then_else(i < 1, 1, 2)

    @I.ir_module
    class After:
        @T.function
        def main():
            T.func_attr({"target": simple_target})
            for i_s in range(4):
                if i_s < 2:
                    result: T.let[T.int32] = T.if_then_else(i_s < 1, 1, 2)
                    T.evaluate(0)

    mod = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, After)


def test_vectorize_while_fallback():
    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((64,), "float32"),
            B: T.Tensor((64,), "float32"),
            C: T.Tensor((64,), "float32"),
        ):
            # Initialize C to 0
            for j in range(64):
                C[j] = T.float32(0)

            # While loop keeps scalar execution
            i = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
            i[0] = 0
            for j in T.vectorized(64):
                while i[0] < 10:
                    C[j] = C[j] + A[j] + B[j]
                    i[0] = i[0] + 1

    after = _assert_scalar_fallback(Module)
    tvm_tirx.compile(after, target="llvm")


@pytest.mark.parametrize(
    "extent, target",
    [(16, simple_target), (T.vscale() * 8, sve_target)],
)
def test_reinterpret_keeps_scalar_loop(extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "float32")):
            T.func_attr({"target": target, "tirx.noalias": True})
            for i in T.vectorized(0, extent):
                B[i] = T.reinterpret(A[i], ty="float32")

    _assert_scalar_fallback(Before)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
@pytest.mark.parametrize(
    "op",
    (
        T.Mul,
        T.Add,
        T.Sub,
        T.Div,
        T.Mod,
        T.FloorDiv,
        T.FloorMod,
        T.Min,
        T.Max,
        T.EQ,
        T.LT,
        T.LE,
        T.GE,
        T.GT,
        T.NE,
    ),
)
def test_vectorize_binary(op, extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32"), B: T.Tensor((25,), "float32")):
            T.func_attr({"target": target, "tirx.noalias": True})
            for j in T.vectorized(extent):
                A[j] = op(T.float32(3), B[j])

    @I.ir_module
    class After:
        @T.function
        def main(A: T.Tensor((25,), "float32"), B: T.Tensor((25,), "float32")):
            T.func_attr({"target": target, "tirx.noalias": True})
            A[T.Ramp(0, 1, extent)] = op(T.Broadcast(T.float32(3), extent), B[T.Ramp(0, 1, extent)])

    mod = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
@pytest.mark.parametrize("op", (T.And, T.Or))
def test_boolean_storage_keeps_scalar_loop(op, extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "bool"), B: T.Tensor((25,), "bool")):
            T.func_attr({"target": target, "tirx.noalias": True})
            for j in T.vectorized(extent):
                A[j] = op(T.bool(1), B[j])

    _assert_scalar_fallback(Before)


@pytest.mark.parametrize("extent, target", [(4, simple_target), (T.vscale() * 4, sve_target)])
def test_vectorize_select(extent, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32"), B: T.Tensor((25,), "float32")):
            T.func_attr({"target": target, "tirx.noalias": True})
            for j in T.vectorized(extent):
                A[j] = T.Select(T.bool(True), A[j], B[j])

    @I.ir_module
    class After:
        @T.function
        def main(A: T.Tensor((25,), "float32"), B: T.Tensor((25,), "float32")):
            T.func_attr({"target": target, "tirx.noalias": True})
            A[T.Ramp(0, 1, extent)] = T.Select(
                T.Broadcast(T.bool(True), extent),
                A[T.Ramp(0, 1, extent)],
                B[T.Ramp(0, 1, extent)],
            )

    mod = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, After)


@pytest.mark.parametrize(
    "extent, vec_str, target",
    [(4, "int32x4", simple_target), (T.vscale() * 4, "int32xvscalex4", sve_target)],
)
def test_vectorize_cast(extent, vec_str, target):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "int32"), B: T.Tensor((25,), "float32")):
            T.func_attr({"target": target, "tirx.noalias": True})
            for j in T.vectorized(extent):
                A[j] = T.Cast("int32", B[j])

    @I.ir_module
    class After:
        @T.function
        def main(A: T.Tensor((25,), "int32"), B: T.Tensor((25,), "float32")):
            T.func_attr({"target": target, "tirx.noalias": True})
            A[T.Ramp(0, 1, extent)] = T.Cast(vec_str, B[T.Ramp(0, 1, extent)])

    mod = tvm_tirx.tirx.transform.VectorizeLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, After)


def test_dynamic_extent_fallback():
    n = T.dynamic("n", "int32")

    @I.ir_module(check_well_formed=False)
    class Mod:
        @T.function
        def main(A: T.Tensor((25,), "int32")):
            for j in T.vectorized(n):
                A[j] = 3

    _assert_scalar_fallback(Mod)


def test_vscale_in_non_sve_vectorization_fallback():
    @I.ir_module
    class Mod:
        @T.function
        def main(A: T.Tensor((16,), "float32")):
            T.func_attr({"target": simple_target})
            for j in T.vectorized(0, 4 * T.vscale()):
                A[j] = 13

    _assert_scalar_fallback(Mod)


def test_conditional_load_store_keeps_scalar_loop():
    @T.function
    def before(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + 1.0

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    _assert_scalar_fallback(InputModule)


def test_conditional_uniform_load_keeps_scalar_loop():
    @T.function
    def before(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    B[i_0 * 4 + i_1] = A[i_0] + 1.0

    @T.function
    def expected(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i_0, i_1_s in T.grid(4, 4):
            if i_0 * 4 + i_1_s < 14:
                B[i_0 * 4 + i_1_s] = A[i_0] + T.float32(1)

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    mod = InputModule
    after = tvm_tirx.tirx.transform.VectorizeLoop()(mod)["main"]
    tvm_tirx.ir.assert_structural_equal(after, expected)


def test_conditional_multiple_stores_keep_scalar_loop():
    @T.function
    def before(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    A[i_0 * 4 + i_1] = 2.0
                    B[i_0 * 4 + i_1] = 1.0

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    _assert_scalar_fallback(InputModule)


def test_nested_conditionals_keep_scalar_loop():
    rvv_target = tvm_tirx.target.Target(
        {"kind": "llvm", "mtriple": "riscv64-unknown-linux-gnu", "mattr": ["+v"]}
    )

    @T.function
    def before(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"target": rvv_target})
        for i_0 in T.serial(4):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 15:
                    if i_0 * 4 + i_1 < 14:
                        A[i_0 * 4 + i_1] = T.float32(1)
                    B[i_0 * 4 + i_1] = T.float32(2)

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        before = input_func

    _assert_scalar_fallback(InputModule)


def test_multiple_conditions_keep_scalar_loop():
    @T.function
    def before(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 > 14:
                    A[i_0 * 4 + i_1] = 2.0
                if 14 < i_0 * 4 + i_1:
                    A[i_0 * 4 + i_1] = 2.0
                if i_0 * 4 + i_1 < i_0 * 4 + i_1:
                    A[i_0 * 4 + i_1] = 2.0

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    _assert_scalar_fallback(InputModule)


def test_conditional_load_store_on_sve_keeps_scalar_loop():
    @T.function
    def before(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "target": sve_target})
        for i_0 in T.serial(T.ceildiv(14, 4)):
            for i_1 in T.vectorized(4):
                if i_0 * 4 + i_1 < 14:
                    B[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + 1.0

    input_func = before

    @I.ir_module(check_well_formed=False)
    class InputModule:
        main = input_func

    _assert_scalar_fallback(InputModule)


def test_llvm_pure_intrin_keeps_scalar_loop():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((25,), "float32"), B: T.Tensor((25,), "float32")):
            T.func_attr({"target": simple_target, "tirx.noalias": True})
            for j in T.vectorized(4):
                A[j] = T.call_llvm_pure_intrin("llvm.sqrt", B[j], ty="float32")

    after = _assert_scalar_fallback(Before)
    tvm_tirx.compile(after, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need LLVM")
def test_vectorize_assert_preserves_scalar_failure_order():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        for i in T.vectorized(4):
            value: T.let = i + 1
            A[i] = value
            assert i < 2, ("RuntimeError", ["stop"])
            A[i] = value + 10

    before = tvm_tirx.IRModule({"main": main})
    _assert_scalar_fallback(before)
    data = tvm_tirx.runtime.tensor(np.full(4, -1, dtype="int32"))
    with pytest.raises(RuntimeError, match="stop"):
        tvm_tirx.compile(main, target="llvm")(data)
    np.testing.assert_array_equal(data.numpy(), [11, 12, 3, -1])


def test_vectorize_unsupported_scalable_factor_keeps_scalar_loop():
    @T.function
    def main(A: T.Tensor((64,), "int32")):
        T.func_attr({"target": sve_target})
        for i in T.vectorized(T.Mul(T.vscale(), 1)):
            value: T.let = i + 1
            A[i] = value

    before = tvm_tirx.IRModule({"main": main})
    _assert_scalar_fallback(before)
    if env.has_llvm():
        tvm_tirx.compile(before, target=sve_target)


if __name__ == "__main__":
    tvm_tirx.testing.main()
