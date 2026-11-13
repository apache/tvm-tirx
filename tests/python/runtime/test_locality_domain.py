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
from tvm_tirx.runtime import locality_domain
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

requires_cuda = pytest.mark.skipif(not env.has_cuda(), reason="CUDA GPU not available")


def test_locality_domain_scopes_registered():
    for k in range(locality_domain.MAX_LOCALITY_DOMAINS):
        assert tvm_tirx.runtime.MemoryScope(locality_domain.locality_domain_scope(k)).name == (
            f"global.locality_domain.{k}"
        )
    with pytest.raises(ValueError):
        locality_domain.locality_domain_scope(locality_domain.MAX_LOCALITY_DOMAINS)
    with pytest.raises(ValueError):
        tvm_tirx.runtime.MemoryScope(
            f"global.locality_domain.{locality_domain.MAX_LOCALITY_DOMAINS}"
        )


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_topology():
    dev = tvm_tirx.cuda(0)
    count = locality_domain.get_num_locality_domains(dev)
    per_domain = locality_domain.get_num_sms_per_locality_domain(dev)
    assert count >= 1
    assert 0 < per_domain * count <= dev.multi_processor_count
    assert locality_domain.is_localization_supported(dev) == (count > 1)
    # A device id works in place of a Device, as in torch.cuda.green_contexts.
    assert locality_domain.get_num_locality_domains(0) == count
    assert locality_domain.get_num_sms_per_locality_domain(0) == per_domain
    assert locality_domain.is_localization_supported(0) == (count > 1)
    with pytest.raises(ValueError, match="CUDA device"):
        locality_domain.get_num_locality_domains(tvm_tirx.cpu(0))


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_allocation_roundtrip():
    """Each scope places memory in its domain; plain `global` memory is not localized."""
    dev = tvm_tirx.cuda(0)
    count = locality_domain.get_num_locality_domains(dev)
    plain = tvm_tirx.runtime.empty((1024,), "float32", dev)
    assert locality_domain.get_locality_domain_id(plain) is None
    for k in range(count):
        data = np.arange(4096, dtype=np.float32) + k
        x = tvm_tirx.runtime.tensor(data, dev, mem_scope=locality_domain.locality_domain_scope(k))
        np.testing.assert_array_equal(x.numpy(), data)
        y = tvm_tirx.runtime.empty((4096,), "float32", dev, mem_scope=f"global.locality_domain.{k}")
        y.copyfrom(x)
        np.testing.assert_array_equal(y.numpy(), data)
        z = plain.copyto(dev, mem_scope=locality_domain.locality_domain_scope(k))
        if count > 1:
            # On a device without domains, domain 0 is the whole device and reports None.
            assert locality_domain.get_locality_domain_id(x) == k
            assert locality_domain.get_locality_domain_id(y) == k
            assert locality_domain.get_locality_domain_id(z) == k


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_allocation_as_first_cuda_call():
    """A localized allocation works as the very first CUDA operation of a process."""
    import subprocess
    import sys

    code = (
        "import tvm_tirx\n"
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
    count = locality_domain.get_num_locality_domains(dev)
    if count >= locality_domain.MAX_LOCALITY_DOMAINS:
        pytest.skip("every registered scope is a valid domain on this device")
    with pytest.raises(ValueError, match=f"has {count} locality domain"):
        tvm_tirx.runtime.empty(
            (16,), "float32", dev, mem_scope=locality_domain.locality_domain_scope(count)
        )


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_free_returns_memory():
    """Freed localized tensors go back to the device rather than staying in the pool."""
    dev = tvm_tirx.cuda(0)
    nbytes = 1 << 30
    scope = locality_domain.locality_domain_scope(locality_domain.get_num_locality_domains(dev) - 1)
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
    last = locality_domain.get_num_locality_domains(dev) - 1
    data = np.random.default_rng(0).standard_normal(1024).astype(np.float32)

    def run_and_check():
        a = tvm_tirx.runtime.tensor(data, dev, mem_scope=locality_domain.locality_domain_scope(0))
        b = tvm_tirx.runtime.empty(
            (1024,), "float32", dev, mem_scope=locality_domain.locality_domain_scope(last)
        )
        mod(a, b)
        np.testing.assert_array_equal(b.numpy(), data * 2)
        t = torch.from_dlpack(b)
        assert t.is_cuda
        np.testing.assert_array_equal(t.cpu().numpy(), data * 2)
        # The query accepts any DLPack tensor, e.g. torch's own (plain) allocations.
        assert locality_domain.get_locality_domain_id(torch.zeros(4, device="cuda")) is None

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@requires_cuda
def test_locality_domain_aware_kernel():
    """Each CTA writes the half of the output that lives in the domain it runs on."""
    dev = tvm_tirx.cuda(0)
    count = locality_domain.get_num_locality_domains(dev)
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
                np.full(4096, -1, np.int32), dev, mem_scope=locality_domain.locality_domain_scope(k)
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


def _domain_queue_reader(n_words, chunk_words, grid, threads=256):
    """A persistent kernel: CTAs read the buffer of their own domain, then help the other.

    B<k> lives in locality domain k and Ctr[k] is its chunk queue. Every word read
    is XORed into Sink[0], so the result proves each chunk was read exactly once.
    """
    steps = chunk_words // (threads * 4)
    n_chunks = n_words // chunk_words

    @T.macro
    def drain(B, Ctr, k, tx, chunk, c, r, acc):
        if tx == 0:
            chunk[0] = T.cuda.atomic_add(Ctr.ptr_to([k]), T.uint32(1))
        T.cuda.cta_sync()
        c[0] = chunk[0]
        T.cuda.cta_sync()
        while c[0] < T.uint32(n_chunks):
            for i in T.unroll(steps):
                idx = T.Cast("int32", c[0]) * chunk_words + (i * threads + tx) * 4
                T.ptx.ld.global_.cs.v4.u32(r[0], r[1], r[2], r[3], B.ptr_to([idx]))
                acc[0] = acc[0] ^ r[0] ^ r[1] ^ r[2] ^ r[3]
            if tx == 0:
                chunk[0] = T.cuda.atomic_add(Ctr.ptr_to([k]), T.uint32(1))
            T.cuda.cta_sync()
            c[0] = chunk[0]
            T.cuda.cta_sync()

    @T.function
    def read(
        B0: T.Tensor((n_words,), "uint32"),
        B1: T.Tensor((n_words,), "uint32"),
        Ctr: T.Tensor((2,), "uint32"),
        Sink: T.Tensor((1,), "uint32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=grid, block=threads))
        dom = T.cuda.mov_sreg(32, "locality_domain_id")
        tx = T.cuda.thread_idx("x")
        chunk = T.alloc_shared([1], "uint32")
        c = T.alloc_local([1], "uint32")
        r = T.alloc_local([4], "uint32")
        acc = T.alloc_local([1], "uint32")
        acc[0] = T.uint32(0)
        # The hardware decides which domain a CTA runs in, so a domain may get
        # few CTAs or none: drain the own queue first, then the other one.
        if dom == 0:
            drain(B0, Ctr, 0, tx, chunk, c, r, acc)
            drain(B1, Ctr, 1, tx, chunk, c, r, acc)
        else:
            drain(B1, Ctr, 1, tx, chunk, c, r, acc)
            drain(B0, Ctr, 0, tx, chunk, c, r, acc)
        T.ptx.red.global_.xor.b32(Sink.ptr_to([0]), acc[0])

    return read


@pytest.mark.gpu
@requires_cuda
@pytest.mark.parametrize("grid", ["full", 1])
def test_locality_domain_work_queue_kernel(grid):
    """A domain-aware persistent kernel reads every chunk exactly once, however CTAs land."""
    dev = tvm_tirx.cuda(0)
    count = locality_domain.get_num_locality_domains(dev)
    if count != 2 or env.cuda_arch() is None or int(env.cuda_arch()[3:6]) < 100:
        pytest.skip("needs an sm_100+ GPU with two locality domains")
    n_words, chunk_words = 1 << 22, 1 << 14
    if grid == "full":
        grid = dev.multi_processor_count * 2
    mod = tvm_tirx.compile(
        tvm_tirx.IRModule({"main": _domain_queue_reader(n_words, chunk_words, grid)}),
        target=tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}),
        pipeline="tirx",
    )
    rng = np.random.default_rng(0)
    data = [rng.integers(0, 1 << 32, n_words, dtype=np.uint32) for _ in range(2)]

    def run_and_check():
        b = [
            tvm_tirx.runtime.tensor(
                data[k], dev, mem_scope=locality_domain.locality_domain_scope(k)
            )
            for k in range(2)
        ]
        ctr = tvm_tirx.runtime.tensor(np.zeros(2, np.uint32), dev)
        sink = tvm_tirx.runtime.tensor(np.zeros(1, np.uint32), dev)
        mod(b[0], b[1], ctr, sink)
        want = np.bitwise_xor.reduce(data[0]) ^ np.bitwise_xor.reduce(data[1])
        assert sink.numpy()[0] == want
        # Each queue hands out all its chunks, then one end marker to every CTA.
        assert (ctr.numpy() == n_words // chunk_words + grid).all()

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)
