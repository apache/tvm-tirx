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
"""NaN-preserving min/max for C and LLVM."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@pytest.mark.parametrize(
    "target,dtype,lanes",
    [("c", dtype, 1) for dtype in ("float32", "float64")]
    + [
        ("llvm", dtype, lanes)
        for dtype in ("float16", "float32", "float64", "bfloat16")
        for lanes in (1, 4)
    ]
    + [("llvm", "float32", "scalable")],
)
def test_min_max(target, dtype, lanes):
    if target == "llvm" and not env.has_llvm():
        pytest.skip("need llvm")
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
            # Rows 10-11: parentheses for general and constant max inside arithmetic.
            C[10, i] = T.max(A[i], B[i]) + T.cast(1, dtype)
            C[11, i] = T.max(T.cast(0, dtype), A[i]) + T.cast(1, dtype)
            # Rows 12-13: positive and negative infinity constants.
            C[12, i] = T.min(A[i], T.cast(infinity, dtype))
            C[13, i] = T.max(A[i], T.cast(-infinity, dtype))

    compile_target = {"kind": "llvm", "opt-level": 0} if target == "llvm" else target
    if scalable:
        compile_target.update(mtriple="aarch64-linux-gnu", mattr=["+sve"])
    compiled = tvm_tirx.tirx.compile(main, target=compile_target)
    if target == "llvm" and dtype == "float32":
        comparisons = [
            line for line in compiled.mod.inspect_source("ll").splitlines() if " fcmp " in line
        ]
        # Check predicate presence and scalar/vector types across the kernel.
        assert {line.split(" fcmp ")[1].split()[0] for line in comparisons} == {
            "olt",
            "ogt",
            "ult",
            "ugt",
            "uno",
        }
        if scalable:
            llvm_type = "<vscale x 4 x float>"
        elif lanes == 1:
            llvm_type = "float"
        else:
            llvm_type = f"<{lanes} x float>"
        assert all(f" {llvm_type} " in line for line in comparisons)
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
        expected_by_op[name] = [
            np.where(compare(lhs, rhs) | np.isnan(lhs), lhs, rhs) for lhs, rhs in operand_pairs
        ]
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


if __name__ == "__main__":
    tvm_tirx.testing.main()
