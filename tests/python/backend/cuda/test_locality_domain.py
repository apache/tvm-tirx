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
"""Locality-domain memory scopes of the CUDA device API."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda import locality
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

requires_cuda = pytest.mark.skipif(not env.has_cuda(), reason="CUDA GPU not available")


def test_locality_domain_scopes_registered():
    for k in range(locality.MAX_LOCALITY_DOMAINS):
        assert tvm_tirx.runtime.MemoryScope(locality.locality_domain_scope(k)).name == (
            f"global.locality_domain.{k}"
        )
    with pytest.raises(ValueError):
        locality.locality_domain_scope(locality.MAX_LOCALITY_DOMAINS)
    with pytest.raises(ValueError):
        tvm_tirx.runtime.MemoryScope(f"global.locality_domain.{locality.MAX_LOCALITY_DOMAINS}")


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_topology():
    dev = tvm_tirx.cuda(0)
    count = locality.locality_domain_count(dev)
    per_domain = locality.locality_domain_sm_count(dev)
    assert count >= 1
    assert 0 < per_domain * count <= dev.multi_processor_count


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_allocation_roundtrip():
    """Each scope places memory in its domain; plain `global` memory is not localized."""
    dev = tvm_tirx.cuda(0)
    count = locality.locality_domain_count(dev)
    plain = tvm_tirx.runtime.empty((1024,), "float32", dev)
    assert locality.locality_domain_of(plain) == -1
    for k in range(count):
        data = np.arange(4096, dtype=np.float32) + k
        x = tvm_tirx.runtime.tensor(data, dev, mem_scope=locality.locality_domain_scope(k))
        np.testing.assert_array_equal(x.numpy(), data)
        y = tvm_tirx.runtime.empty((4096,), "float32", dev, mem_scope=f"global.locality_domain.{k}")
        y.copyfrom(x)
        np.testing.assert_array_equal(y.numpy(), data)
        z = plain.copyto(dev, mem_scope=locality.locality_domain_scope(k))
        if count > 1:
            # On a device without domains, domain 0 is the whole device and may report -1.
            assert locality.locality_domain_of(x) == k
            assert locality.locality_domain_of(y) == k
            assert locality.locality_domain_of(z) == k


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_allocation_as_first_cuda_call():
    """A localized allocation works as the very first CUDA operation of a process."""
    import subprocess
    import sys

    code = (
        "import tvm_tirx\n"
        "from tvm_tirx.backend.cuda import locality\n"
        "dev = tvm_tirx.cuda(0)\n"
        "x = tvm_tirx.runtime.empty((256,), 'float32', dev, mem_scope='global.locality_domain.0')\n"
        "print('ok', x.shape)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ok")


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_out_of_range():
    dev = tvm_tirx.cuda(0)
    count = locality.locality_domain_count(dev)
    if count >= locality.MAX_LOCALITY_DOMAINS:
        pytest.skip("every registered scope is a valid domain on this device")
    with pytest.raises(ValueError, match=f"has {count} locality domain"):
        tvm_tirx.runtime.empty(
            (16,), "float32", dev, mem_scope=locality.locality_domain_scope(count)
        )


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_free_returns_memory():
    """Freed localized tensors go back to the device rather than staying in the pool."""
    dev = tvm_tirx.cuda(0)
    nbytes = 1 << 30
    scope = locality.locality_domain_scope(locality.locality_domain_count(dev) - 1)
    for _ in range(3):  # warm up the pool and the allocator
        tvm_tirx.runtime.empty((nbytes // 4,), "float32", dev, mem_scope=scope)
    before = dev.available_global_memory
    for _ in range(8):
        x = tvm_tirx.runtime.empty((nbytes // 4,), "float32", dev, mem_scope=scope)
        del x
    dev.sync()
    assert dev.available_global_memory >= before - nbytes


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_tensor_in_kernel_and_dlpack():
    """Localized tensors are ordinary CUDA tensors for kernels and DLPack consumers."""
    torch = pytest.importorskip("torch")

    @T.function
    def double(A: T.Tensor((1024,), "float32"), B: T.Tensor((1024,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=4, block=256))
        i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
        B[i] = A[i] * T.float32(2)

    mod = tvm_tirx.compile(
        tvm_tirx.IRModule({"main": double}),
        target=tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}),
        pipeline="tirx",
    )
    dev = tvm_tirx.cuda(0)
    last = locality.locality_domain_count(dev) - 1
    data = np.random.default_rng(0).standard_normal(1024).astype(np.float32)

    def run_and_check():
        a = tvm_tirx.runtime.tensor(data, dev, mem_scope=locality.locality_domain_scope(0))
        b = tvm_tirx.runtime.empty(
            (1024,), "float32", dev, mem_scope=locality.locality_domain_scope(last)
        )
        mod(a, b)
        np.testing.assert_array_equal(b.numpy(), data * 2)
        t = torch.from_dlpack(b)
        assert t.is_cuda
        np.testing.assert_array_equal(t.cpu().numpy(), data * 2)
        # The query accepts any DLPack tensor, e.g. torch's own (plain) allocations.
        assert locality.locality_domain_of(torch.zeros(4, device="cuda")) == -1

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_aware_kernel():
    """Each CTA writes the half of the output that lives in the domain it runs on."""
    dev = tvm_tirx.cuda(0)
    count = locality.locality_domain_count(dev)
    if count < 2 or env.cuda_arch() is None or int(env.cuda_arch()[3:6]) < 100:
        pytest.skip("needs an sm_100+ GPU with several locality domains")

    @T.function
    def fill(
        O0: T.Tensor((4096,), "int32"),
        O1: T.Tensor((4096,), "int32"),
        C: T.Tensor((2,), "int32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=64, block=64))
        dom = T.cuda.mov_sreg(32, "locality_domain_id")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            if dom == 0:
                T.cuda.atomic_add(C.ptr_to([0]), T.int32(1))
            else:
                T.cuda.atomic_add(C.ptr_to([1]), T.int32(1))
        for i in T.serial(64):
            if dom == 0:
                O0[i * 64 + tx] = dom
            else:
                O1[i * 64 + tx] = dom

    mod = tvm_tirx.compile(
        tvm_tirx.IRModule({"main": fill}),
        target=tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}),
        pipeline="tirx",
    )

    def run_and_check():
        o = [
            tvm_tirx.runtime.tensor(
                np.full(4096, -1, np.int32), dev, mem_scope=locality.locality_domain_scope(k)
            )
            for k in range(2)
        ]
        c = tvm_tirx.runtime.tensor(np.zeros(2, np.int32), dev)
        mod(o[0], o[1], c)
        ctas = c.numpy()
        assert ctas.sum() == 64
        for k in range(2):
            got = o[k].numpy()
            # A domain that received no CTA leaves its output untouched.
            assert (got == (k if ctas[k] else -1)).all()

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)
