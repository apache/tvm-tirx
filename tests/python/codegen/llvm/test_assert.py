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
"""LLVM assertions require every lane of a vector condition to hold."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

pytestmark = pytest.mark.skipif(not env.has_llvm(), reason="need llvm")


def test_vectorized_assert_constant_condition():
    """Regression for #48: a statically false lane must produce a runtime error."""

    @T.function
    def main():
        for i in T.vectorized(4):
            assert i < 3, ("RuntimeError", ["x"])

    with tvm_tirx.ir.transform.PassContext(opt_level=0):
        compiled = tvm_tirx.tirx.compile(main, target="llvm", pipeline="tirx")
    with pytest.raises(RuntimeError, match=r"^x$"):
        compiled()


@pytest.mark.parametrize("condition_kind", ["varying", "uniform"])
@pytest.mark.parametrize("opt_level", [0, 3])
def test_vectorized_assert_runtime_condition(condition_kind, opt_level):
    @T.function
    def main(limit: T.int32):
        for i in T.vectorized(4):
            if T.constexpr(condition_kind == "uniform"):
                assert limit > 0, ("ValueError", ["vectorized ", "assertion failed"])
            else:
                assert i < limit, ("ValueError", ["vectorized ", "assertion failed"])

    with tvm_tirx.ir.transform.PassContext(opt_level=opt_level):
        compiled = tvm_tirx.tirx.compile(main, target={"kind": "llvm", "opt-level": opt_level})
    for limit in (0, 1, 3, 4, 5):
        succeeds = limit > 0 if condition_kind == "uniform" else limit >= 4
        if succeeds:
            compiled(limit)
        else:
            with pytest.raises(ValueError, match="vectorized assertion failed"):
                compiled(limit)


def test_assert_explicit_vector_condition():
    """Any failing lane raises before statements following the assertion execute."""

    @T.function
    def main(A: T.Tensor((4,), "int32"), B: T.Tensor((1,), "int32")):
        assert A[T.Ramp(0, 1, 4)] > T.Broadcast(0, 4), ("RuntimeError", ["non-positive lane"])
        B[0] = 1

    compiled = tvm_tirx.tirx.compile(main, target="llvm")
    for failing_lane in (None, 0, 1, 2, 3):
        values = np.ones(4, dtype="int32")
        if failing_lane is not None:
            values[failing_lane] = 0
        a = tvm_tirx.runtime.tensor(values)
        b = tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32"))
        if failing_lane is None:
            compiled(a, b)
            np.testing.assert_array_equal(b.numpy(), [1])
        else:
            with pytest.raises(RuntimeError, match="non-positive lane"):
                compiled(a, b)
            np.testing.assert_array_equal(b.numpy(), [0])


def test_assert_scalable_vector_condition():
    """Cross-compile an SVE assertion without requiring SVE execution hardware."""

    @T.function
    def main(limit: T.int32):
        for i in T.vectorized(T.vscale() * 4):
            assert i < limit, ("RuntimeError", ["scalable assertion failed"])

    target = {
        "kind": "llvm",
        "mtriple": "aarch64-linux-gnu",
        "mattr": ["+sve"],
        "opt-level": 0,
    }
    compiled = tvm_tirx.tirx.compile(main, target=target)
    llvm_ir = compiled.mod.inspect_source("ll")
    assert "call i1 @llvm.vector.reduce.and.nxv4i1" in llvm_ir


if __name__ == "__main__":
    tvm_tirx.testing.main()
