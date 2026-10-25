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
# ruff: noqa: E501, E741
from __future__ import annotations

import math
import re

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.codegen import llvm
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import utils
from tvm_tirx.testing import env


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_bool_comparisons():
    @T.function
    def compare(A: T.Tensor((4,), "bool"), B: T.Tensor((4,), "bool"), C: T.Tensor((6, 4), "bool")):
        for i in range(4):
            C[0, i] = A[i] == B[i]
            C[1, i] = A[i] != B[i]
            C[2, i] = A[i] < B[i]
            C[3, i] = A[i] <= B[i]
            C[4, i] = A[i] > B[i]
            C[5, i] = A[i] >= B[i]

    a = np.array([False, False, True, True])
    b = np.array([False, True, False, True])
    output = tvm_tirx.runtime.empty((6, 4), "bool")
    compiled = tvm_tirx.tirx.compile(compare, target="llvm")
    compiled(tvm_tirx.runtime.tensor(a), tvm_tirx.runtime.tensor(b), output)
    np.testing.assert_array_equal(output.numpy(), [a == b, a != b, a < b, a <= b, a > b, a >= b])


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_duplicate_function_global_symbol_diagnostic():
    @I.ir_module
    class Module:
        @T.function
        def first_unique_key(A: T.Tensor((1,), "float32")):
            T.func_attr({"global_symbol": "dup_symbol", "tirx.noalias": True})
            A[0] = T.float32(1)

        @T.function
        def second_unique_key(A: T.Tensor((1,), "float32")):
            T.func_attr({"global_symbol": "dup_symbol", "tirx.noalias": True})
            A[0] = T.float32(2)

    with pytest.raises(RuntimeError, match="Duplicate Function global_symbol 'dup_symbol'") as err:
        tvm_tirx.tirx.compile(Module, target="llvm")
    assert "first_unique_key" in str(err.value)
    assert "second_unique_key" in str(err.value)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_unique_function_global_symbols_compile():
    @I.ir_module
    class Module:
        @T.function
        def first_unique_key(A: T.Tensor((1,), "float32")):
            T.func_attr({"global_symbol": "dup_symbol_a", "tirx.noalias": True})
            A[0] = T.float32(1)

        @T.function
        def second_unique_key(A: T.Tensor((1,), "float32")):
            T.func_attr({"global_symbol": "dup_symbol_b", "tirx.noalias": True})
            A[0] = T.float32(2)

    tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_intrin():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.handle("float32")):
            A_buf = T.decl_tensor(A, ty_args=[T.Tensor((4,), "float32", data=A)])
            T.evaluate(
                T.Call(
                    "tirx.prefetch",
                    [T.address_of(A_buf[0]), 0, 3, 1],
                    ty="void",
                )
            )

    tvm_tirx.tirx.compile(Module)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_void_intrin():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.handle("uint8")):
            # Create an intrinsic that returns void.
            T.call_llvm_intrin("llvm.assume", T.bool(True), ty="void")

    tvm_tirx.tirx.compile(Module)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_intrinsic_id():
    orig_name = "llvm.x86.sse2.pmadd.wd"
    intrin_id = llvm.lookup_intrinsic_id(orig_name)
    name = llvm.get_intrinsic_name(intrin_id)
    assert orig_name == name


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_overloaded_intrin():
    # Name lookup for overloaded intrinsics in LLVM 4- requires a name
    # that includes the overloaded types.
    if tvm_tirx.codegen.llvm.version_major() < 5:
        return

    # int1 is the type for the is_zero_undef parameter
    int1_zero = tvm_tirx.tirx.const(0, "int1")

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((1, 1), "int32"), C: T.Tensor((1, 1), "int32")):
            C[0, 0] = T.call_llvm_pure_intrin("llvm.ctlz", A[0, 0], int1_zero, ty="int32")

    tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_lookup_intrin():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.handle("uint8x8")):
            A_buf = T.decl_tensor(A, ty_args=[T.Tensor((1,), "uint8x8", data=A)])
            T.evaluate(T.call_llvm_pure_intrin("llvm.ctpop.v8i8", A_buf[0], ty="uint8x8"))

    tvm_tirx.tirx.compile(Module, None)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_large_uintimm():
    value = (1 << 63) + 123
    large_val = tvm_tirx.tirx.const(value, "uint64")

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((), "uint64")):
            T.func_attr({"tirx.noalias": True})
            A[()] = large_val + T.uint64(3)

    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.empty((), dtype="uint64", device=dev)
    f(a)
    assert a.numpy() == value + 3


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_multi_parallel():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
            T.func_attr({"tirx.noalias": True})
            B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
            for i0_0_0 in T.parallel(1):
                for ax0 in range(128):
                    B[ax0] = A[ax0] + T.float32(1.0)
                for i0_0_1 in range(16):
                    for i0_1 in T.parallel(8):
                        C[i0_0_0 * 128 + i0_0_1 * 8 + i0_1] = T.sqrt(
                            B[i0_0_0 * 128 + i0_0_1 * 8 + i0_1]
                        ) * T.float32(2.0) + T.float32(2.0)

    n = 128
    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    c = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)
    f(a, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), np.sqrt(a.numpy() + 1) * 2 + 2, rtol=1e-5)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_flip_pipeline():
    def check_llvm(nn, base):
        @I.ir_module
        class Module:
            @T.function
            def main(A: T.Tensor((nn + base,), "float32"), C: T.Tensor((nn,), "float32")):
                T.func_attr({"tirx.noalias": True})
                for i_0 in T.parallel((nn + 3) // 4):
                    for i_1 in T.vectorized(4):
                        if i_0 * 4 + i_1 < nn:
                            C[i_0 * 4 + i_1] = A[nn + base - 1 - (i_0 * 4 + i_1)]

        f = tvm_tirx.tirx.compile(Module, target="llvm")
        dev = tvm_tirx.cpu(0)
        a = tvm_tirx.runtime.tensor(np.random.uniform(size=(nn + base)).astype("float32"), dev)
        c = tvm_tirx.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
        f(a, c)
        tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy()[::-1][:nn])

    check_llvm(4, 0)
    check_llvm(128, 8)
    check_llvm(3, 0)
    check_llvm(128, 1)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_vadd_pipeline():
    n = T.dynamic("n", "int32")

    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")
        ):
            T.func_attr({"tirx.noalias": True})

            for i_0 in range((n + 3) // 4):
                for i_1 in T.vectorized(4):
                    if i_0 * 4 + i_1 < n:
                        C[i_0 * 4 + i_1] = A[i_0 * 4 + i_1] + B[i_0 * 4 + i_1]

    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    n = 128
    a = tvm_tirx.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    b = tvm_tirx.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    c = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)
    f(a, b, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_add_linear_shape():
    n = T.dynamic("n", "int64")

    @T.function
    def add(
        A: T.Tensor((4 * n + 3,), "float32"),
        B: T.Tensor((4 * n + 3,), "float32"),
        C: T.Tensor((4 * n + 3,), "float32"),
    ):
        for i in range(4 * n + 3):
            C[i] = A[i] + B[i]

    lib = tvm_tirx.tirx.compile(add, target="llvm")
    a = tvm_tirx.runtime.tensor(np.arange(19, dtype="float32"))
    b = tvm_tirx.runtime.tensor(np.ones(19, dtype="float32"))
    c = tvm_tirx.runtime.tensor(np.zeros(19, dtype="float32"))
    lib(a, b, c)
    np.testing.assert_equal(c.numpy(), a.numpy() + b.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_strides_and_linear_layout_offset():
    s = T.dynamic("s", "int64")
    m = T.dynamic("m", "int64")

    @T.function
    def copy(
        A: T.Tensor((3, 4), "float32", layout=T.TileLayout(T.S[(3, 4) : (s, 1)] + 4 * m), align=4),
        B: T.Tensor((3, 4), "float32"),
    ):
        for i, j in T.grid(3, 4):
            B[i, j] = A[i, j] + T.Cast("float32", m + s)

    lib = tvm_tirx.tirx.compile(copy, target="llvm")
    data = np.arange(64, dtype="float32")
    allocation = tvm_tirx.runtime.tensor(data)
    result = tvm_tirx.runtime.tensor(np.zeros((3, 4), dtype="float32"))
    for offset in (4, 8):
        # CreateView keeps the allocation pointer and sets DLTensor.byte_offset.
        view = allocation._create_view((3, 4), relative_byte_offset=offset * 4)
        lib(view, result)
        expected = data[offset : offset + 12].reshape(3, 4) + offset // 4 + 4
        np.testing.assert_equal(result.numpy(), expected)
    strided = data.reshape(8, 8)[:3, :4]
    lib(strided, result)
    np.testing.assert_equal(result.numpy(), strided + 8)
    invalid = allocation._create_view((3, 4), relative_byte_offset=4)
    with pytest.raises(ValueError, match="divisible"):
        lib(invalid, result)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_madd_pipeline():
    def check_llvm(nn, base, stride):
        @I.ir_module
        class Module:
            @T.function
            def main(
                A: T.Tensor((nn + base, stride), "float32"),
                C: T.Tensor((nn, stride), "float32"),
            ):
                T.func_attr({"tirx.noalias": True})
                for i_0 in T.parallel((nn + 3) // 4):
                    for i_1 in T.vectorized(4):
                        for j in range(stride):
                            if i_0 * 4 + i_1 < nn:
                                C[i_0 * 4 + i_1, j] = A[i_0 * 4 + i_1 + base, j] + T.float32(1.0)

        f = tvm_tirx.tirx.compile(Module, target="llvm")
        dev = tvm_tirx.cpu(0)
        a = tvm_tirx.runtime.tensor(
            np.random.uniform(size=(nn + base, stride)).astype("float32"), dev
        )
        c = tvm_tirx.runtime.tensor(np.zeros((nn, stride), dtype="float32"), dev)
        f(a, c)
        tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy()[base:] + 1)

    check_llvm(64, 0, 2)
    check_llvm(4, 0, 1)

    with tvm_tirx.ir.transform.PassContext(config={"tirx.noalias": False}):
        check_llvm(4, 0, 3)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_temp_space():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((1024,), "float32"), C: T.Tensor((1024,), "float32")):
            T.func_attr({"tirx.noalias": True})
            B = T.alloc_tensor(ty_args=[T.Tensor((1024,), "float32")])
            for i in range(1024):
                B[i] = A[i] + T.float32(1.0)
            for i in range(1024):
                C[i] = B[i] + T.float32(1.0)

    nn = 1024
    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.tensor(np.random.uniform(size=nn).astype("float32"), dev)
    c = tvm_tirx.runtime.tensor(np.zeros(nn, dtype="float32"), dev)
    f(a, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy() + 1 + 1)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_multiple_func():
    fadd1_n = T.dynamic("n", "int32")
    fadd2_n = T.dynamic("n", "int32")

    @I.ir_module
    class Module:
        @T.function
        def fadd1(
            A: T.Tensor((fadd1_n,), "float32"),
            B: T.Tensor((fadd1_n,), "float32"),
            C: T.Tensor((fadd1_n,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for i in range(fadd1_n):
                C[i] = A[i] + B[i]

        @T.function
        def fadd2(
            A: T.Tensor((fadd2_n,), "float32"),
            B: T.Tensor((fadd2_n,), "float32"),
            C: T.Tensor((fadd2_n,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for i in range(fadd2_n):
                C[i] = A[i] + B[i]

    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    n = 10
    a = tvm_tirx.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    b = tvm_tirx.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
    c = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)

    f["fadd1"](a, b, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())
    f["fadd2"](a, b, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_condition():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i in range(64):
                C[i] = T.if_then_else(8 <= i, A[i], T.float32(0.0))

    n = 64
    offset = 8
    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.tensor(np.random.uniform(size=(n,)).astype("float32"), dev)
    c = tvm_tirx.runtime.empty((n,), "float32", dev)
    f(a, c)
    c_np = a.numpy()
    c_np[:offset] = 0
    tvm_tirx.testing.assert_allclose(c.numpy(), c_np)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_bool():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((64,), "int32"), C: T.Tensor((64,), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i in range(64):
                C[i] = T.Cast("float32", A[i] == 1)

    n = 64
    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.tensor(np.random.randint(0, 2, size=(n,)).astype("int32"), dev)
    c = tvm_tirx.runtime.empty((n,), "float32", dev)
    f(a, c)
    c_np = a.numpy() == 1
    tvm_tirx.testing.assert_allclose(c.numpy(), c_np)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_cast_float_to_bool():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((4,), "float32"), C: T.Tensor((4,), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i in range(4):
                C[i] = T.Cast("bool", A[i])

    n = 4
    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.tensor(np.array([0.0, 1.0, np.nan, np.inf], dtype="float32"), dev)
    c = tvm_tirx.runtime.empty((n,), dtype="bool", device=dev)
    f(a, c)
    c_np = np.array([False, True, True, True], dtype="bool")
    tvm_tirx.testing.assert_allclose(c.numpy(), c_np)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_rank_zero():
    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((64,), "float32"),
            scale: T.Tensor((), "float32"),
            compute: T.Tensor((), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            C = T.alloc_tensor(ty_args=[T.Tensor((), "float32")])
            C[()] = T.float32(0.0)
            for k in range(64):
                C[()] = C[()] + A[k] * scale[()]
            compute[()] = C[()] + T.float32(1.0)

    n = 64
    f = tvm_tirx.tirx.compile(Module, target="llvm")
    dev = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.tensor(np.random.randint(0, 2, size=(n,)).astype("float32"), dev)
    sc = tvm_tirx.runtime.tensor(np.random.randint(0, 2, size=()).astype("float32"), dev)
    d = tvm_tirx.runtime.empty((), "float32", dev)
    f(a, sc, d)
    d_np = np.sum(a.numpy()) * sc.numpy() + 1
    tvm_tirx.testing.assert_allclose(d.numpy(), d_np)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_alignment():
    @I.ir_module
    class Module:
        @T.function
        def test_alignment(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i_0 in range(128):
                for i_1 in T.vectorized(8):
                    B[i_0 * 8 + i_1] = A[i_0 * 8 + i_1] * T.float32(3.0)

    f = tvm_tirx.tirx.compile(Module, target="llvm").mod

    lines = f.inspect_source().split("\n")

    # Check alignment on load/store.
    for l in lines:
        if "align" in l and "4 x float" in l:
            assert "align 32" in l

    # Check parameter alignment. This looks for the definition of the
    # outlined "compute_" function to see if there is an "align" attribute
    # listed there.
    def has_param_alignment():
        for l in lines:
            if re.search(r"test_alignment_compute_\([^(]*align [0-9]", l):
                return True
        return False

    if tvm_tirx.codegen.llvm.version_major() >= 5:
        assert has_param_alignment()

    # Check for assume intrinsics. This isn't 100% accurate, since it just
    # checks if the llvm.assume is there, but detailed check would require
    # a much more detailed analysis of the LLVM IR.
    def has_call_to_assume():
        for l in lines:
            if re.search(r"call.*llvm.assume", l):
                return True
        return False

    assert has_call_to_assume()


def _llvm_div_cases():
    dividend_ranges = [
        (-12, -12),
        (-11, -1),
        (-11, 0),
        (0, 0),
        (12, 12),
        (1, 11),
        (0, 11),
        (-11, 11),
    ]
    divisor_ranges = [
        (-11, -1),
        (-11, 1),
        (-4, -4),
        (-2, -2),
        (1, 11),
        (0, 11),
        (4, 4),
        (2, 2),
        (-11, 11),
    ]

    for start, end in dividend_ranges:
        for dstart, dend in divisor_ranges:
            for dtype in ["int32", "int8"]:
                for floor_div in [False, True]:
                    yield pytest.param(
                        start,
                        end,
                        dstart,
                        dend,
                        dtype,
                        floor_div,
                        id=f"{dtype}-{'floor' if floor_div else 'trunc'}-{start}:{end}-{dstart}:{dend}",
                    )
            if start >= 0 and dstart >= 0:
                for floor_div in [False, True]:
                    yield pytest.param(
                        start,
                        end,
                        dstart,
                        dend,
                        "uint32",
                        floor_div,
                        id=f"uint32-{'floor' if floor_div else 'trunc'}-{start}:{end}-{dstart}:{dend}",
                    )

    for dstart, dend in [(0, 11), (1, 11), (2, 2), (4, 4)]:
        for start, end in [(123, 133), (0, 255)]:
            for floor_div in [False, True]:
                yield pytest.param(
                    start,
                    end,
                    dstart,
                    dend,
                    "uint8",
                    floor_div,
                    id=f"uint8-{'floor' if floor_div else 'trunc'}-{start}:{end}-{dstart}:{dend}",
                )


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize(
    "start,end,dstart,dend,dtype,floor_div",
    list(_llvm_div_cases()),
)
def test_llvm_div(start, end, dstart, dend, dtype, floor_div):
    """Check that the semantics of div and mod is correct"""

    def check(start, end, dstart, dend, dtype, floor_div=False):
        a_size = end - start + 1
        b_size = dend - dstart + 1

        div_fn = tvm_tirx.tirx.floordiv if floor_div else tvm_tirx.tirx.truncdiv
        mod_fn = tvm_tirx.tirx.floormod if floor_div else tvm_tirx.tirx.truncmod

        # Build clipping helpers — capture TIR const values from env
        _start = tvm_tirx.tirx.const(start, dtype)
        _end = tvm_tirx.tirx.const(end, dtype)
        _dstart = tvm_tirx.tirx.const(dstart, dtype)
        _dend = tvm_tirx.tirx.const(dend, dtype)

        if start == end:

            def clipa(x):
                return _start
        else:

            def clipa(x):
                return T.min(_end, T.max(_start, x))

        if dstart == dend:

            def clipb(x):
                return _dstart
        else:

            def clipb(x):
                return T.min(_dend, T.max(_dstart, x))

        @I.ir_module
        class Module:
            @T.function
            def main(
                A: T.Tensor((a_size,), dtype),
                B: T.Tensor((b_size,), dtype),
                D: T.Tensor((a_size, b_size), dtype),
                M: T.Tensor((a_size, b_size), dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                for i, j in T.grid(a_size, b_size):
                    D[i, j] = div_fn(clipa(A[i]), clipb(B[j]))
                    M[i, j] = mod_fn(clipa(A[i]), clipb(B[j]))

        f = tvm_tirx.tirx.compile(Module, target="llvm")

        # Fill input arrays with values
        A_arr = tvm_tirx.runtime.empty((a_size,), dtype)
        B_arr = tvm_tirx.runtime.empty((b_size,), dtype)
        A_arr.copyfrom(np.arange(start, end + 1, dtype=dtype))
        B_np = np.arange(dstart, dend + 1, dtype=dtype)
        # If the range of the divisor contains 0, replace it with 1 to avoid division by zero
        if dend >= 0 and dstart <= 0:
            B_np[-dstart] = 1
        B_arr.copyfrom(B_np)
        D_arr = tvm_tirx.runtime.empty((a_size, b_size), dtype)
        M_arr = tvm_tirx.runtime.empty((a_size, b_size), dtype)

        # Run the function and convert the results to numpy
        f(A_arr, B_arr, D_arr, M_arr)
        D_arr = D_arr.numpy()
        M_arr = M_arr.numpy()

        # This helper just prints additional info on failure
        def _show_info():
            print(f"dtype: {dtype}")
            print(f"dividend range: [{start}, {end}]")
            print(f"divisor range: [{dstart}, {dend}]")

        # Check that the computed values are correct
        for i in range(start, end + 1):
            for j in range(dstart, dend + 1):
                if j == 0:
                    continue

                if floor_div:
                    dref = i // j
                    mref = i % j
                else:
                    dref = int(float(i) / j)
                    mref = int(math.fmod(i, j))

                if D_arr[i - start, j - dstart] != dref:
                    _show_info()
                    raise AssertionError(
                        f"Incorrect division result: {div_fn.__name__}({i}, {j}) is {D_arr[i - start, j - dstart]} "
                        f"but should be {dref}"
                    )
                if M_arr[i - start, j - dstart] != mref:
                    _show_info()
                    raise AssertionError(
                        f"Incorrect modulo result: {mod_fn.__name__}({i}, {j}) is {M_arr[i - start, j - dstart]} "
                        f"but should be {mref}"
                    )

    check(start, end, dstart, dend, dtype, floor_div)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_fp_math():
    n = T.dynamic("n", "int32")

    @I.ir_module
    class RecipModule:
        @T.function
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})

            for i in range(n):
                B[i] = T.float32(1.0) / (T.float32(1e37) * A[i])

    f_recip = tvm_tirx.tirx.compile(RecipModule, target="llvm")

    for n in [4, 8, 16]:
        a = tvm_tirx.runtime.tensor(np.full((n,), 100, "float32"))
        b = tvm_tirx.runtime.empty((n,), "float32")
        f_recip(a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), np.zeros((n,), "float32"))

    n = T.dynamic("n", "int32")

    @I.ir_module
    class SigmoidModule:
        @T.function
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            T.func_attr({"tirx.noalias": True})

            for i in range(n):
                B[i] = T.sigmoid(A[i])

    f_sigmoid = tvm_tirx.tirx.compile(SigmoidModule, target="llvm")

    for n in [4, 8, 16]:
        a = tvm_tirx.runtime.tensor(np.full((n,), -1000, "float32"))
        b = tvm_tirx.runtime.empty((n,), "float32")
        f_sigmoid(a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), np.zeros((n,), "float32"))


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_dwarf_debug_information():
    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((1024,), "float32"),
            B: T.Tensor((1024,), "float32"),
            C: T.Tensor((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0_0 in T.parallel(256):
                for i0_1 in T.vectorized(4):
                    C[i0_0 * 4 + i0_1] = A[i0_0 * 4 + i0_1] + B[i0_0 * 4 + i0_1]

    def check_llvm_object():
        if tvm_tirx.codegen.llvm.version_major() < 5:
            return
        if tvm_tirx.codegen.llvm.version_major() > 6:
            return
        # build two functions
        mod = tvm_tirx.IRModule(
            {
                "fadd1": Module["main"].with_attr("global_symbol", "fadd1"),
                "fadd2": Module["main"].with_attr("global_symbol", "fadd2"),
            }
        )
        m = tvm_tirx.tirx.compile(mod, target="llvm")
        temp = utils.tempdir()
        o_path = temp.relpath("temp.o")
        m.write_to_file(o_path)
        import shutil
        import subprocess
        import sys

        # Try the dwarfdump utility (OS X)
        if shutil.which("dwarfdump"):
            output = subprocess.check_output(["dwarfdump", o_path])
            assert re.search(r"""DW_AT_name\\t\("fadd1"\)""", str(output))
            assert re.search(r"""DW_AT_name\\t\("fadd2"\)""", str(output))

        # Try gobjdump (OS X)
        if shutil.which("gobjdump"):
            output = subprocess.check_output(["gobjdump", "--dwarf", o_path])
            assert re.search(r"""DW_AT_name.*fadd1""", str(output))
            assert re.search(r"""DW_AT_name.*fadd2""", str(output))

        # Try objdump (Linux) - Darwin objdump has different DWARF syntax.
        if shutil.which("objdump") and sys.platform != "darwin":
            output = subprocess.check_output(["objdump", "--dwarf", o_path])
            assert re.search(r"""DW_AT_name.*fadd1""", str(output))
            assert re.search(r"""DW_AT_name.*fadd2""", str(output))

    def check_llvm_ir():
        if tvm_tirx.codegen.llvm.version_major() < 5:
            return
        if tvm_tirx.codegen.llvm.version_major() > 6:
            return
        # build two functions
        mod = tvm_tirx.IRModule(
            {
                "fadd1": Module["main"].with_attr("global_symbol", "fadd1"),
                "fadd2": Module["main"].with_attr("global_symbol", "fadd2"),
            }
        )
        m = tvm_tirx.tirx.compile(mod, target={"kind": "llvm", "mtriple": "aarch64-linux-gnu"}).mod
        ll = m.inspect_source("ll")

        # On non-Darwin OS, don't explicitly specify DWARF version.
        import re

        assert not re.search(r""""Dwarf Version""" "", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

        # Try Darwin, require DWARF-2
        m = tvm_tirx.tirx.compile(
            mod, target={"kind": "llvm", "mtriple": "x86_64-apple-darwin-macho"}
        ).mod
        ll = m.inspect_source("ll")
        assert re.search(r"""i32 4, !"Dwarf Version", i32 2""", ll)
        assert re.search(r"""llvm.dbg.value""", ll)

    check_llvm_object()
    check_llvm_ir()


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_bf16():
    def dotest(do_vectorize):
        loop_kind = T.vectorized if do_vectorize else T.serial

        @I.ir_module
        class Module:
            @T.function
            def main(
                A: T.Tensor((32,), "bfloat16"),
                B: T.Tensor((32,), "bfloat16"),
                D: T.Tensor((32,), "bfloat16"),
            ):
                T.func_attr({"tirx.noalias": True})
                for x in loop_kind(32):
                    D[x] = A[x] + B[x]

        np.random.seed(122)
        module = tvm_tirx.tirx.compile(Module, target="llvm")
        npa = np.random.rand(32).astype("bfloat16")
        npb = np.random.rand(32).astype("bfloat16")
        res = npa + npb
        a_ = tvm_tirx.runtime.tensor(npa)
        b_ = tvm_tirx.runtime.tensor(npb)
        c_ = tvm_tirx.runtime.empty((32,), "bfloat16")
        module(a_, b_, c_)
        # Note: directly compare without casting to float32 should work with the
        # latest numpy version.
        tvm_tirx.testing.assert_allclose(c_.numpy().astype("float32"), res.astype("float32"))

    dotest(True)
    dotest(False)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_crt_static_lib():
    @I.ir_module
    class Module:
        @T.function
        def main(
            A: T.Tensor((32,), "bfloat16"),
            B: T.Tensor((32,), "bfloat16"),
            C: T.Tensor((32,), "bfloat16"),
        ):
            T.func_attr({"tirx.noalias": True})
            for x in range(32):
                C[x] = A[x] + B[x]

    module = tvm_tirx.tirx.compile(
        Module.with_attr("system_lib_prefix", ""),
        target=tvm_tirx.target.Target("llvm"),
    ).mod
    module.inspect_source()
    with utils.tempdir() as temp:
        module.write_to_file(temp.relpath("test.o"))


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_order_functions():
    """Check that functions in the LLVM module are ordered alphabetically."""

    # Note: the order is alphabetical because that's a predictable ordering. Any predictable
    # ordering will work fine, but if the ordering changes, this test will need to be updated.
    @I.ir_module
    class Module:
        @T.function
        def Danny(v: T.float32) -> T.float32:
            return T.call_extern("Dave", v, ty="float32")

        @T.function
        def Sammy(v: T.float32) -> T.float32:
            return T.call_extern("Eve", v, ty="float32")

        @T.function
        def Kirby(v: T.float32) -> T.float32:
            return T.call_extern("Fred", v, ty="float32")

    ir_text = tvm_tirx.tirx.compile(Module, target="llvm").mod.inspect_source("ll")
    # Skip functions whose names start with _.
    matches = re.findall(r"^define[^@]*@([a-zA-Z][a-zA-Z0-9_]*)", ir_text, re.MULTILINE)
    assert matches == sorted(matches)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_scalar_concat():
    @I.ir_module
    class Module:
        @T.function
        def main(x: T.int32, y: T.int32, buffer: T.Tensor((1,), "int32x2")):
            buffer[0] = T.Shuffle([x, y], [0, 1])

    # This will crash in LLVM codegen if CodeGenLLVM::CreateVecConcat doesn't convert
    # scalars to single-lane LLVM vectors.
    with tvm_tirx.ir.transform.PassContext(config={"tirx.disable_assert": True}):
        tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_raise_exception_during_codegen():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((4, 4), "float32"), B: T.Tensor((4, 4), "float32")) -> None:
            T.func_attr({"tirx.noalias": True})
            for i in T.parallel(4):
                for j in T.parallel(4):
                    B[i, j] = A[i, j] * 2.0

    with pytest.raises(RuntimeError) as e:
        tvm_tirx.tirx.compile(Module, target="llvm")
    msg = str(e)
    assert msg.find("Nested parallel loop is not supported") != -1


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_target_attributes():
    """Check that when LLVM codegen creates new functions, they get the same target
    attributes as the original function.
    """

    @I.ir_module
    class Module:
        @T.function
        def test_func(
            A: T.Tensor((tindex,), "float32"),  # noqa: F821
            B: T.Tensor((tindex,), "float32"),  # noqa: F821
            C: T.Tensor((tindex,), "float32"),  # noqa: F821
            tindex: T.int32,
        ):
            T.func_attr({"tirx.noalias": True})

            for i in range(tindex):
                B[i] = A[i]
            for i_0 in T.parallel(2):
                for i_1 in range((tindex + 1) // 2):
                    if i_0 * ((tindex + 1) // 2) + i_1 < tindex:
                        C[i_0 * ((tindex + 1) // 2) + i_1] = B[
                            i_0 * ((tindex + 1) // 2) + i_1
                        ] + T.float32(1.0)

    target_llvm = {
        "kind": "llvm",
        "mtriple": "x86_64-linux-gnu",
        "mcpu": "skylake",
        "mattr": ["+avx512f"],
    }
    target = tvm_tirx.target.Target(target_llvm, host=target_llvm)
    module = tvm_tirx.tirx.compile(Module, target=target).mod

    llvm_ir = module.inspect_source()
    llvm_ir_lines = llvm_ir.split("\n")

    attribute_definitions = dict()
    function_attr_map = dict()

    for line in llvm_ir_lines:
        func_def = re.match(
            "define.* @(?P<func_name>[^(]*)[(].* #(?P<attr_num>[0-9]+) (!.* |){$", line
        )
        if func_def:
            function_attr_map[func_def.group("func_name")] = func_def.group("attr_num")
            continue
        attr_def = re.match("attributes #(?P<attr_num>[0-9]+) = {(?P<attr_list>.*)}", line)
        if attr_def:
            attribute_definitions[attr_def.group("attr_num")] = attr_def.group("attr_list")

    expected_functions = [
        "__tvm_ffi_test_func",
        "__tvm_parallel_lambda",
    ]
    for n in expected_functions:
        assert n in function_attr_map, f"Expected function {n} not found in LLVM IR"
        attr_num = function_attr_map[n]
        assert re.match('.*"target-cpu"="skylake".*', attribute_definitions[attr_num])
        assert re.match('.*"target-features"=".*[+]avx512f.*".*', attribute_definitions[attr_num])


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_assume():
    """
    Check that LLVM does not error out when generating code with tirx.assume.
    Verifying for llvm.assume being generated is not easy as the intrinsic and its
    related instructions get removed during optimizations
    """

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((4, 4), "int32"), B: T.Tensor((14,), "int32")):
            T.func_attr({"tirx.noalias": True})
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "int32", data=A.data)])
            for axis0, axis1 in T.grid(4, 4):
                T.assume(axis0 < 3 or axis1 < 2 or A_1[axis0 * 4 + axis1] == 0)
            for i in range(14):
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((14,), "int32", data=B.data)])
                B_1[i] = A_1[i] * 2

    tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_debug_symbol_for_float64():
    """Check that LLVM can define DWARF debug type for float64

    In previous versions, only specific data types could exist in the
    function signature.  In this test, the "calling_conv" attribute
    prevents lowering to the PackedFunc API.
    """

    @I.ir_module
    class Module:
        @T.function
        def main(a: T.handle("float64"), b: T.handle("float64"), n: T.int64):
            T.func_attr({"calling_conv": 2})
            A = T.decl_tensor(a, ty_args=[T.Tensor(16, "float64", data=a)])
            B = T.decl_tensor(b, ty_args=[T.Tensor(16, "float64", data=b)])
            for i in range(n):
                B[i] = A[i]

    tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_debug_symbol_for_buffer_var():
    """TensorVars use their physical data pointer type in LLVM debug info."""

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
            C = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
            for i in T.parallel(16):
                C[i] = A[i]
                B[i] = C[i]

    tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_subroutine_call():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor(1, dtype="float32")):
            Module.subroutine(A.data)

        @T.function
        def subroutine(A_data: T.handle("float32")):
            # The calling_conv parameter is to prevent MakePackedAPI
            # from changing the call signature of the subroutine.
            T.func_attr({"calling_conv": -1})
            A = T.decl_tensor(A_data, ty_args=[T.Tensor(1, "float32", data=A_data)])
            A[0] = 42.0

    dev = tvm_tirx.cpu()

    built = tvm_tirx.tirx.compile(Module)

    arr = tvm_tirx.runtime.tensor(np.zeros([1], "float32"), device=dev)
    built["main"](arr)
    assert arr.numpy()[0] == 42.0


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_call_packed_returning_void():
    """Allow codegen of PackedFunc calls returning void

    The LLVM codegen uses the CallObj's dtype to cast the return type
    of the PackedFunc into the appropriate LLVM output type.  However,
    there is no runtime dtype value for a void return.  When the return type
    of a PackedFunc is void, the generated code should not attempt to
    read the return value.

    While `T.call_packed()` will produce a CallObj with an output
    dtype of "int32", the use of other return types is valid in TIR.
    This test case uses `T.Call` directly to allow an explicit dtype
    for the packed function call.
    """

    @I.ir_module
    class Module:
        @T.function
        def main():
            T.Call(
                tvm_tirx.ir.Op.get("tirx.call_packed"),
                ["dummy_function_name"],
                ty="void",
            )

    # Error occurred during build, as part of
    # CodeGenCPU::MakeCallPackedLowered.
    tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_call_packed_without_string_arg():
    """The first argument to call_packed must be a string

    Even if the invalid TIR is constructed, this should throw an
    exception to exit cleanly.  Previously, use of
    `args[0].as<StringImmObj>()` without a null check resulted in
    a segfault during codegen.
    """

    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor(1, "float32")):
            T.Call(
                tvm_tirx.ir.Op.get("tirx.call_packed"),
                [A.data],
                ty="int32",
            )

    with pytest.raises(RuntimeError):
        tvm_tirx.tirx.compile(Module, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_call_extern_returning_void():
    """Like test_call_packed_returning_void, but for call_extern"""

    @I.ir_module
    class Module:
        @T.function
        def main():
            T.Call(
                tvm_tirx.ir.Op.get("tirx.call_extern"),
                ["dummy_function_name"],
                ty="void",
            )

    tvm_tirx.tirx.compile(Module, target="llvm")


def test_invalid_volatile_masked_buffer_load():
    @I.ir_module
    class Module:
        @T.function
        def main(B: T.Tensor([4], "float32")):
            A = T.alloc_tensor(
                ty_args=[T.Tensor((4,), "float32")], annotations={"tirx.volatile": True}
            )
            B[0:4] = T.call_intrin(
                "tirx.masked_load", A, T.Ramp(0, 1, 4), T.Broadcast(T.bool(True), 4), ty="float32x4"
            )

    err_msg = "The masked load intrinsic does not support declaring load as volatile."
    with pytest.raises(RuntimeError, match=err_msg):
        with tvm_tirx.target.Target("llvm"):
            tvm_tirx.tirx.compile(Module)


def test_invalid_volatile_masked_decl_buffer_load():
    @I.ir_module
    class Module:
        @T.function
        def main(B: T.Tensor([4], "float32")):
            A = T.alloc_tensor(
                ty_args=[T.Tensor((4,), "float32")], annotations={"tirx.volatile": True}
            )
            A_alias = T.decl_tensor(A.data, ty_args=[T.Tensor((4,), "float32", data=A.data)])
            B[0:4] = T.call_intrin(
                "tirx.masked_load",
                A_alias,
                T.Ramp(0, 1, 4),
                T.Broadcast(T.bool(True), 4),
                ty="float32x4",
            )

    err_msg = "The masked load intrinsic does not support declaring load as volatile."
    with pytest.raises(RuntimeError, match=err_msg):
        with tvm_tirx.target.Target("llvm"):
            tvm_tirx.tirx.compile(Module)


def test_invalid_volatile_masked_tensor_store():
    @I.ir_module
    class Module:
        @T.function
        def main():
            A = T.alloc_tensor(
                ty_args=[T.Tensor((4,), "float32")], annotations={"tirx.volatile": True}
            )
            T.evaluate(
                T.call_intrin(
                    "tirx.masked_store",
                    A,
                    T.Broadcast(0.0, 4),
                    T.Ramp(0, 1, 4),
                    T.Broadcast(T.bool(True), 4),
                    ty="void",
                )
            )

    err_msg = "The masked store intrinsic does not support declaring store as volatile."
    with pytest.raises(RuntimeError, match=err_msg):
        with tvm_tirx.target.Target("llvm"):
            tvm_tirx.tirx.compile(Module)


def test_int_parameter():
    """Boolean may be passed to functions accepting int"""

    @I.ir_module
    class Module:
        @T.function
        def main(arg: T.int32) -> T.int32:
            T.func_attr({"target": T.target("llvm")})
            if arg > 0:
                return 10
            else:
                return 20

    built = tvm_tirx.tirx.compile(Module)
    output = built(True)
    assert output == 10

    output = built(False)
    assert output == 20


def test_bool_parameter():
    """Integers may be passed to functions accepting bool"""

    @I.ir_module
    class Module:
        @T.function
        def main(arg: T.bool) -> T.int32:
            T.func_attr({"target": T.target("llvm")})
            if arg:
                return 10
            else:
                return 20

    built = tvm_tirx.tirx.compile(Module)
    output = built(1)
    assert output == 10

    output = built(2)
    assert output == 10

    output = built(0)
    assert output == 20


def test_bool_return_value():
    """Booleans may be returned from a Function"""

    @I.ir_module
    class Module:
        @T.function
        def main(value: T.int32) -> T.bool:
            T.func_attr({"target": T.target("llvm")})
            return value < 10

    built = tvm_tirx.tirx.compile(Module)
    assert isinstance(built(0), bool)
    assert built(0)

    assert isinstance(built(15), bool)
    assert not built(15)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_module_properties_and_formats(tmp_path):
    @T.function
    def main(A: T.Tensor((10,), "int64")):
        for i in range(9):
            A[i + 1] = A[i] + T.int64(1)

    module = tvm_tirx.tirx.compile(main, target="llvm").mod
    assert not module.is_binary_serializable()
    assert module.is_runnable()
    assert module.is_compilation_exportable()
    for extension in ("o", "ll", "bc"):
        path = tmp_path / ("module." + extension)
        module.write_to_file(str(path))
        assert path.is_file()
        if extension == "ll":
            loaded = tvm_tirx.runtime.load_module(str(path))
            data = tvm_tirx.runtime.tensor(np.zeros(10, dtype="int64"))
            loaded(data)
            np.testing.assert_equal(data.numpy(), np.arange(10))


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize(
    "dtype,lanes",
    [(dtype, lanes) for dtype in ("float16", "float32", "float64", "bfloat16") for lanes in (1, 4)]
    + [("float32", "scalable")],
)
def test_llvm_min_max_nan_preserving(dtype, lanes):
    scalable = lanes == "scalable"
    extent = 4 * tvm_tirx.tirx.vscale() if scalable else lanes
    n = 128 if scalable else 12
    nan = float("nan")
    infinity = float("inf")

    @T.function
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((14, n), dtype)):
        for outer in range(n // extent):
            i = outer if T.constexpr(lanes == 1) else T.ramp(outer * extent, 1, extent)
            # Rows 0-4: min(a, b), min(0, a), min(a, 0), min(NaN, a), min(a, NaN).
            C[0, i] = T.min(A[i], B[i])
            C[1, i] = T.min(T.cast(0, dtype), A[i])
            C[2, i] = T.min(A[i], T.cast(0, dtype))
            C[3, i] = T.min(T.cast(nan, dtype), A[i])
            C[4, i] = T.min(A[i], T.cast(nan, dtype))
            # Rows 5-9: max with the same operand pairs.
            C[5, i] = T.max(A[i], B[i])
            C[6, i] = T.max(T.cast(0, dtype), A[i])
            C[7, i] = T.max(A[i], T.cast(0, dtype))
            C[8, i] = T.max(T.cast(nan, dtype), A[i])
            C[9, i] = T.max(A[i], T.cast(nan, dtype))
            # Rows 10-11: use max results inside arithmetic.
            C[10, i] = T.max(A[i], B[i]) + T.cast(1, dtype)
            C[11, i] = T.max(T.cast(0, dtype), A[i]) + T.cast(1, dtype)
            # Rows 12-13: positive and negative infinity constants.
            C[12, i] = T.min(A[i], T.cast(infinity, dtype))
            C[13, i] = T.max(A[i], T.cast(-infinity, dtype))

    target = {"kind": "llvm", "opt-level": 0}
    if scalable:
        target.update(mtriple="aarch64-linux-gnu", mattr=["+sve"])
    compiled = tvm_tirx.tirx.compile(main, target=target)
    # BF16 arithmetic is legalized to float32 before LLVM codegen.
    llvm_type = {"float16": "half", "float32": "float", "float64": "double", "bfloat16": "float"}[
        dtype
    ]
    if scalable:
        llvm_type = f"<vscale x 4 x {llvm_type}>"
    elif lanes != 1:
        llvm_type = f"<{lanes} x {llvm_type}>"
    llvm_ir = compiled.mod.inspect_source("ll")
    for intrinsic in ("minimum", "maximum"):
        assert f"call {llvm_type} @llvm.{intrinsic}." in llvm_ir
    if scalable:
        return  # Cross-compile SVE; execution requires SVE hardware.

    np_dtype = pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bfloat16" else dtype
    a = np.array([np.nan, 1, np.nan, 0.0, -0.0, 3, 2, -5, np.inf, -np.inf, 1, -1], dtype=np_dtype)
    b = np.array([1, np.nan, np.nan, -0.0, 0.0, 2, 2, -4, 1, 1, np.inf, -np.inf], dtype=np_dtype)
    out = tvm_tirx.runtime.empty((14, n), dtype)
    compiled(tvm_tirx.runtime.tensor(a), tvm_tirx.runtime.tensor(b), out)

    zero, nan_array = np.zeros_like(a), np.full_like(a, np.nan)
    operand_pairs = ((a, b), (zero, a), (a, zero), (nan_array, a), (a, nan_array))
    expected_by_op = {}
    for name, compare in (("min", np.less), ("max", np.greater)):
        values = []
        for lhs, rhs in operand_pairs:
            take_lhs = compare(lhs, rhs) | np.isnan(lhs)
            # IEEE minimum selects -0 and maximum selects +0 on a zero tie.
            both_zero = (lhs == 0) & (rhs == 0)
            preferred_zero = np.signbit(lhs) if name == "min" else ~np.signbit(lhs)
            take_lhs |= both_zero & preferred_zero
            values.append(np.where(take_lhs, lhs, rhs))
        expected_by_op[name] = values
    expected = expected_by_op["min"] + expected_by_op["max"]
    max_ab, max_zero_a = expected_by_op["max"][:2]
    expected.extend([max_ab + np.ones_like(a), max_zero_a + np.ones_like(a)])
    # min(a, +Inf) and max(a, -Inf) both preserve a, including NaNs and signed zero.
    expected.extend([a, a])
    expected = np.asarray(expected, dtype=np_dtype)
    got = out.numpy()
    nan_mask = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(got), nan_mask)
    # NaN payloads may be canonicalized; preserve exact non-NaN values and signed zero.
    uint_dtype = f"uint{a.dtype.itemsize * 8}"
    np.testing.assert_array_equal(
        got.view(uint_dtype)[~nan_mask], expected.view(uint_dtype)[~nan_mask]
    )


def _compile_zero_stride_ramp(dtype, lanes):
    """Compile the fuzz reproducer: a zero-stride ramp whose lane count exceeds the dtype."""
    source = f"""from tvm_tirx.script import tirx as T

@T.function
def f():
    T.evaluate(T.Ramp(T.Cast("{dtype}", 0), T.Cast("{dtype}", 0), {lanes}))
"""
    func = tvm_tirx.script.from_source(source)
    tvm_tirx.tirx.analysis.check_well_formed(func, assert_mode=True)
    tvm_tirx.ir.assert_structural_equal(
        func, tvm_tirx.script.from_source(func.script(), extra_vars={"T": T})
    )
    target = tvm_tirx.target.Target.from_device(tvm_tirx.cpu())
    func = func.with_attr("target", target).with_attr("global_symbol", "f")
    module = tvm_tirx.IRModule({"f": func})
    with tvm_tirx.ir.transform.PassContext(opt_level=0):
        tvm_tirx.tirx.compile(module, target=target, pipeline="tirx")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_uint1x3_zero_stride_ramp_opt_level_0():
    """A three-lane uint1 ramp is well-formed and must compile at opt_level 0.

    Regression for apache/tvm-tirx#34. Codegen used to build each lane as
    ``base + stride * MakeConst(uint1, i)`` and reject lane 2.
    """
    _compile_zero_stride_ramp("uint1", 3)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_uint2x5_zero_stride_ramp_opt_level_0():
    """A five-lane uint2 ramp compiles; four lanes fit in uint2 and the fifth does not."""
    _compile_zero_stride_ramp("uint2", 5)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_llvm_narrow_integer_ramp_wraparound():
    """Nonzero-stride narrow ramps wrap in the element width, including lane ordinals.

    uint2 lane 4 is not a uint2 literal. The stored lanes are still
    ``(base + stride * i) mod 4``. int2 uses the same modular arithmetic and
    sign-extends when widened to int8.
    """

    @I.ir_module
    class Unsigned:
        @T.function
        def main(out: T.Tensor((5,), "uint8")):
            T.func_attr({"tirx.noalias": True})
            out[T.Ramp(0, 1, 5)] = T.Cast(
                "uint8x5", T.Ramp(T.Cast("uint2", 1), T.Cast("uint2", 3), 5)
            )

    @I.ir_module
    class Signed:
        @T.function
        def main(out: T.Tensor((5,), "int8")):
            T.func_attr({"tirx.noalias": True})
            out[T.Ramp(0, 1, 5)] = T.Cast("int8x5", T.Ramp(T.Cast("int2", 1), T.Cast("int2", 1), 5))

    target = "llvm"
    with tvm_tirx.ir.transform.PassContext(opt_level=0):
        unsigned = tvm_tirx.tirx.compile(Unsigned, target=target, pipeline="tirx")
        signed = tvm_tirx.tirx.compile(Signed, target=target, pipeline="tirx")

    dev = tvm_tirx.cpu(0)
    out_u = tvm_tirx.runtime.empty((5,), "uint8", dev)
    unsigned(out_u)
    # (1 + 3*i) mod 4 for i in 0..4
    np.testing.assert_array_equal(out_u.numpy(), np.array([1, 0, 3, 2, 1], dtype="uint8"))

    out_s = tvm_tirx.runtime.empty((5,), "int8", dev)
    signed(out_s)
    # (1 + i) mod 4, then interpreted as int2: 1, -2, -1, 0, 1
    np.testing.assert_array_equal(out_s.numpy(), np.array([1, -2, -1, 0, 1], dtype="int8"))


if __name__ == "__main__":
    tvm_tirx.testing.main()
