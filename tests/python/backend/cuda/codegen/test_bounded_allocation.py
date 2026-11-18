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

import re

import pytest
import tvm_ffi

import tvm_tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _build_cuda(mod):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"})
    return tvm_ffi.get_global_func("target.build.cuda")(mod, target)


@pytest.mark.parametrize("scope", ["local", "shared"])
@pytest.mark.parametrize("bounded", [True, False])
def test_symbolic_allocation_bound(scope, bounded):
    limit = 64 if bounded else 2147483647

    @I.ir_module
    class Module:
        @T.function
        def allocation_kernel(n: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "allocation_kernel",
                    "tirx.is_global_func": True,
                }
            )
            extent: T.let[T.int32] = T.min(n, limit)
            scratch = T.alloc_tensor(ty_args=[T.Tensor((extent * 2,), "float32", scope=scope)])
            scratch[0] = 1.0

    if bounded:
        source = _build_cuda(Module).inspect_source()
        assert re.search(r"float\s+\w+\[128\]", source)
    else:
        with pytest.raises(RuntimeError, match="finite compile-time upper bound"):
            _build_cuda(Module)


def test_allocation_bound_does_not_substitute_stateful_read():
    @I.ir_module
    class Module:
        @T.function
        def allocation_kernel(n: T.int32):
            T.func_attr(
                {
                    "calling_conv": 2,
                    "global_symbol": "allocation_kernel",
                    "tirx.is_global_func": True,
                }
            )
            state = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
            state[0] = 0
            snapshot: T.let[T.int32] = state[0]
            state[0] = 32
            difference: T.let[T.int32] = state[0] - snapshot
            scratch = T.alloc_tensor(
                ty_args=[T.Tensor((T.min(difference, 64),), "float32", scope="local")]
            )
            scratch[0] = 1.0

    source = _build_cuda(Module).inspect_source()
    assert re.search(r"float\s+\w+\[64\]", source)
