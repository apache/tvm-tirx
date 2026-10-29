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
"""Generate, reuse, and specialize CUDA kernels with Python."""

# T.jit binds constexpr parameters such as N before the annotations are parsed.
# ruff: noqa: F821

# [imports.begin]
from __future__ import annotations

import torch

import tvm_tirx
from tvm_tirx.script import tirx as T

# [imports.end]


# [generator.begin]
def make_add(dtype, value, block=256):
    n = T.dynamic("n")

    @T.function
    def add(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, block), block=block))
        i = T.cuda.block_idx("x") * block + T.cuda.thread_idx("x")
        if i < n:
            B[i] = A[i] + value

    return add


add_two_f16 = make_add(T.float16, 2, block=128)
add_two_f16_compiled = tvm_tirx.compile(add_two_f16, target="cuda")
a = torch.arange(10, dtype=torch.float16, device="cuda")
b = torch.empty_like(a)
add_two_f16_compiled(a, b)
print(b)
# [generator.end]


# [inline.begin]
@T.inline
def add_one_if_valid(A, B, i, n):
    if i < n:
        B[i] = A[i] + 1


n = T.dynamic("n")


@T.function
def add_one_pairs(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 512), block=256))
    i = (T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")) * 2
    add_one_if_valid(A, B, i, n)
    add_one_if_valid(A, B, i + 1, n)
    # [inline.end]


a = torch.arange(1001, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
tvm_tirx.compile(add_one_pairs, target="cuda")(a, b)
torch.testing.assert_close(b, a + 1)


# [jit.begin]
@T.jit
def add_one_fixed(
    A: T.Tensor((N,), T.float32),
    B: T.Tensor((N,), T.float32),
    *,
    N: T.constexpr,
):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(N, 256), block=256))
    i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
    if i < N:
        B[i] = A[i] + 1


add_one_256 = add_one_fixed.specialize(N=256)
# [jit.end]
a = torch.arange(256, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
tvm_tirx.compile(add_one_256, target="cuda")(a, b)
torch.testing.assert_close(b, a + 1)
