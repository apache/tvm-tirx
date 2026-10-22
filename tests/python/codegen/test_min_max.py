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
"""NaN propagation, constant operands, and evaluation of host min/max."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _build(module, target, tmp_path, **kwargs):
    compiled = tvm_tirx.tirx.compile(module, target=target)
    library = str(tmp_path / "min_max.so")
    compiled.export_library(library, **kwargs)
    return tvm_tirx.runtime.load_module(library)


@pytest.mark.parametrize(
    "target,dtype,lanes",
    [("c", dtype, 1) for dtype in ("float32", "float64")]
    + [
        ("llvm", dtype, lanes)
        for dtype in ("float16", "float32", "float64", "bfloat16")
        for lanes in (1, 4)
    ],
)
def test_min_max_nan_preserving(target, dtype, lanes, tmp_path):
    if target == "llvm" and not env.has_llvm():
        pytest.skip("need llvm")
    np_dtype = pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bfloat16" else dtype
    a = np.array([np.nan, 1, np.nan, 0.0, -0.0, 3, 2, -5, np.inf, -np.inf, 1, -1], dtype=np_dtype)
    b = np.array([1, np.nan, np.nan, -0.0, 0.0, 2, 2, -4, 1, 1, np.inf, -np.inf], dtype=np_dtype)
    n = len(a)
    nan = float("nan")
    infinity = float("inf")

    @T.function
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((15, n), dtype)):
        for i in range(n):
            C[12, i] = T.cast(0, dtype)
        for outer in range(n // lanes):
            i = outer if T.constexpr(lanes == 1) else T.ramp(outer * lanes, 1, lanes)
            C[0, i] = T.min(A[i], B[i])
            C[1, i] = T.min(T.cast(0, dtype), A[i])
            C[2, i] = T.min(A[i], T.cast(0, dtype))
            C[3, i] = T.min(T.cast(nan, dtype), A[i])
            C[4, i] = T.min(A[i], T.cast(nan, dtype))
            C[5, i] = T.max(A[i], B[i])
            C[6, i] = T.max(T.cast(0, dtype), A[i])
            C[7, i] = T.max(A[i], T.cast(0, dtype))
            C[8, i] = T.max(T.cast(nan, dtype), A[i])
            C[9, i] = T.max(A[i], T.cast(nan, dtype))
            # Parenthesization of both the general and constant ternaries.
            C[10, i] = T.max(A[i], B[i]) + T.cast(1, dtype)
            C[11, i] = T.max(T.cast(0, dtype), A[i]) + T.cast(1, dtype)
            # A write must invalidate the previous statement's cached read.
            C[12, i] = T.max(C[12, i], T.cast(3, dtype))
            C[12, i] = T.max(C[12, i], T.cast(2, dtype))
            C[13, i] = T.min(A[i], T.cast(infinity, dtype))
            C[14, i] = T.max(A[i], T.cast(-infinity, dtype))

    compiled = _build(main, target, tmp_path)
    out = tvm_tirx.runtime.empty((15, n), dtype)
    compiled(tvm_tirx.runtime.tensor(a), tvm_tirx.runtime.tensor(b), out)

    zero, nan_array = np.zeros_like(a), np.full_like(a, np.nan)
    expected = []
    for compare in (np.less, np.greater):
        for lhs, rhs in ((a, b), (zero, a), (a, zero), (nan_array, a), (a, nan_array)):
            expected.append(np.where(compare(lhs, rhs) | np.isnan(lhs), lhs, rhs))
    expected.extend(
        [expected[5] + np.ones_like(a), expected[6] + np.ones_like(a), np.full_like(a, 3)]
    )
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


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("lanes", [1, 4, "scalable"])
def test_llvm_min_max_constant_conditions(lanes):
    scalable = lanes == "scalable"
    target = {"kind": "llvm", "opt-level": 0}
    if scalable:
        target.update(mtriple="aarch64-linux-gnu", mattr=["+sve"])
    extent = 4 * tvm_tirx.tirx.vscale() if scalable else lanes
    cases = {
        "dynamic": {"olt", "ogt", "uno"},
        "lhs_zero": {"olt", "ogt"},
        "rhs_zero": {"ult", "ugt"},
        "lhs_nan": set(),
        "rhs_nan": {"uno"},
    }
    functions = {}
    for case in cases:
        const_value = 0.0 if case.endswith("zero") else float("nan")

        @T.function
        def main(
            A: T.Tensor((128,), "float32"),
            B: T.Tensor((128,), "float32"),
            C: T.Tensor((2, 128), "float32"),
        ):
            index = 0 if T.constexpr(lanes == 1) else T.ramp(0, 1, extent)
            lhs = T.float32(const_value) if T.constexpr(case.startswith("lhs")) else A[index]
            rhs = T.float32(const_value) if T.constexpr(case.startswith("rhs")) else B[index]
            C[0, index] = T.min(lhs, rhs)
            C[1, index] = T.max(lhs, rhs)

        functions[case] = main.with_attr("global_symbol", case)

    source = tvm_tirx.tirx.compile(tvm_tirx.IRModule(functions), target=target).mod.inspect_source(
        "ll"
    )
    llvm_type = "<vscale x 4 x float>" if scalable else "<4 x float>" if lanes == 4 else "float"
    for case, expected in cases.items():
        # Inspect each compute function separately, excluding its packed ABI wrapper.
        body = next(
            part for part in source.split("\ndefine ") if f"@{case}_" in part.splitlines()[0]
        )
        comparisons = [line for line in body.splitlines() if " fcmp " in line]
        assert {line.split(" fcmp ")[1].split()[0] for line in comparisons} == expected, case
        assert all(f" {llvm_type} " in line for line in comparisons), case


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_min_max_operand_evaluation(target, tmp_path):
    if target == "llvm" and not env.has_llvm():
        pytest.skip("need llvm")

    helper_source = tmp_path / "increment.cc"
    helper_source.write_text(
        'extern "C" float test_min_max_increment(float* p) { float old = *p; *p += 1; return old; }'
    )

    @T.function
    def main(count: T.Tensor((2,), "float32"), out: T.Tensor((2,), "float32")):
        out[0] = T.max(
            T.call_extern("test_min_max_increment", count.data, ty="float32"),
            T.call_extern("test_min_max_increment", count.data, ty="float32"),
        )
        out[1] = T.max(
            T.float32("nan"),
            T.call_extern("test_min_max_increment", T.address_of(count[1]), ty="float32"),
        )

    options = (
        {"options": ["-include", str(helper_source)]}
        if target == "c"
        else {"addons": [str(helper_source)]}
    )
    compiled = _build(main, target, tmp_path, **options)
    count = tvm_tirx.runtime.tensor(np.zeros(2, dtype="float32"))
    out = tvm_tirx.runtime.empty((2,), "float32")
    compiled(count, out)
    np.testing.assert_array_equal(count.numpy(), [2, 1])
    assert out.numpy()[0] == 1
    assert np.isnan(out.numpy()[1])


if __name__ == "__main__":
    tvm_tirx.testing.main()
