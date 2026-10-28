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

import numpy as np
import pytest

import tvm_tirx as tx
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need CUDA")
def test_bool_storage_uses_one_byte_per_lane():
    @T.function
    def compare(
        A: T.Tensor((32,), "int32"),
        B: T.Tensor((32,), "int32"),
        C: T.Tensor((40,), "bool"),
    ):
        T.func_attr({"tirx.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 8) as tx:
                C[T.Ramp(4 + tx * 4, 1, 4)] = A[T.Ramp(tx * 4, 1, 4)] < B[T.Ramp(tx * 4, 1, 4)]

    device = tx.cuda()
    a = np.arange(32, dtype="int32") % 3
    b = np.ones(32, dtype="int32")
    expected = np.ones(40, dtype="bool")
    expected[4:-4] = a < b
    result = tx.runtime.tensor(np.ones_like(expected), device)
    compiled = tx.compile(compare, target="cuda")
    compiled(tx.runtime.tensor(a, device), tx.runtime.tensor(b, device), result)
    np.testing.assert_array_equal(result.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need CUDA")
def test_unaligned_vector_store_is_rejected():
    @T.function
    def copy(A: T.Tensor((32,), "int32"), B: T.Tensor((33,), "int32")):
        T.func_attr({"tirx.noalias": True})
        with T.launch_thread("blockIdx.x", 1):
            with T.launch_thread("threadIdx.x", 8) as tx:
                B[T.Ramp(tx * 4 + 1, 1, 4)] = A[T.Ramp(tx * 4, 1, 4)]

    with pytest.raises(ValueError, match="Cannot prove alignment for CUDA vector store"):
        tx.compile(copy, target="cuda")
