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
"""Conditions and loops in CUDA kernels."""

# [imports.begin]
import torch

import tvm_tirx
from tvm_tirx.script import tirx as T

n = T.dynamic("n")
# [imports.end]


# [if.begin]
@T.function
def relu(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
    i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
    if i < n:
        if A[i] > 0.0:
            B[i] = A[i]
        else:
            B[i] = 0.0
    # [if.end]


a = torch.randn(1000, device="cuda")
b = torch.empty_like(a)
tvm_tirx.compile(relu, target="cuda")(a, b)
torch.testing.assert_close(b, torch.relu(a))


# [for.begin]
@T.function
def row_sum(A: T.Tensor((n, 64), T.float32), B: T.Tensor((n,), T.float32)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 128), block=128))
    row = T.cuda.block_idx("x") * 128 + T.cuda.thread_idx("x")
    if row < n:
        total: T.float32 = 0.0
        for j in range(16):
            for k in T.unroll(4):
                total += A[row, j * 4 + k]
        B[row] = total
    # [for.end]


a = torch.randn(300, 64, device="cuda")
b = torch.empty(300, device="cuda")
tvm_tirx.compile(row_sum, target="cuda")(a, b)
torch.testing.assert_close(b, a.sum(dim=1))


# [grid.begin]
@T.function
def transpose(A: T.Tensor((64, 64), T.float32), B: T.Tensor((64, 64), T.float32)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=16, block=16))
    row0 = T.cuda.block_idx("x") * 4
    col0 = T.cuda.thread_idx("x") * 4
    for r, c in T.grid(4, 4):
        B[col0 + c, row0 + r] = A[row0 + r, col0 + c]
    # [grid.end]


a = torch.randn(64, 64, device="cuda")
b = torch.empty_like(a)
tvm_tirx.compile(transpose, target="cuda")(a, b)
torch.testing.assert_close(b, a.T)


# [while.begin]
@T.function
def search_sorted(
    S: T.Tensor((64,), T.float32), Q: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.int32)
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
    i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
    if i < n:
        lo: T.int32 = 0
        hi: T.int32 = 64
        while lo < hi:
            mid = (lo + hi) // 2
            if S[mid] < Q[i]:
                lo = mid + 1
            else:
                hi = mid
        B[i] = lo
    # [while.end]


s = torch.sort(torch.randn(64, device="cuda")).values
q = torch.randn(1000, device="cuda")
b = torch.empty(1000, dtype=torch.int32, device="cuda")
tvm_tirx.compile(search_sorted, target="cuda")(s, q, b)
torch.testing.assert_close(b, torch.searchsorted(s, q).to(torch.int32))


# [return.begin]
@T.function
def scale(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
    i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
    if i >= n:
        T.gpu_thread_return()
    B[i] = A[i] * 2.0
    # [return.end]


a = torch.randn(1000, device="cuda")
b = torch.empty_like(a)
tvm_tirx.compile(scale, target="cuda")(a, b)
torch.testing.assert_close(b, a * 2)
