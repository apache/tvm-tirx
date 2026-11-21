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

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.support.nvcc
import tvm_tirx.testing
from tvm_tirx.backend.cuda import bind_cubin, build_cuda_host, export_cuda_host, jit_cuda_host
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

_SIGNATURE = {"add_one": ("handle", "handle", "int32")}


def _wrapper():
    n = T.dynamic("n")

    @T.function
    def wrapper(x: T.Tensor((n,), T.float32), y: T.Tensor((n,), T.float32), threads: T.int32):
        if n > 0:
            T.cuda.launch(
                "add_one",
                (x.data, y.data, T.Cast("int32", n)),
                T.cuda.LaunchConfig(grid=T.ceildiv(n, threads), block=threads),
            )

    return wrapper


def test_cubin_binding_diagnostics():
    # Only the ELF header is inspected at binding time. Never load this fixture on a GPU.
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[18:20] = (190).to_bytes(2, "little")
    image = bytes(header)
    wrapper = _wrapper()
    with pytest.raises(ValueError, match="CUDA ELF cubin"):
        bind_cubin(wrapper, b"\x7fELF", kernels=_SIGNATURE)
    with pytest.raises(ValueError, match="CUDA ELF cubin"):
        bind_cubin(wrapper, image[:18] + b"\x3e\x00" + image[20:], kernels=_SIGNATURE)
    with pytest.raises(ValueError, match=r"No explicit CUDA launch.*typo"):
        bind_cubin(wrapper, image, kernels={"typo": ()})
    for signature in [("handle", "handle"), ("handle", "handle", "int64")]:
        with pytest.raises(ValueError, match=r"add_one.*expects.*got"):
            bind_cubin(wrapper, image, kernels={"add_one": signature})

    @T.function
    def two_sites(x: T.Tensor((1,), T.int32)):
        T.cuda.launch("fill", (x.data,), T.cuda.LaunchConfig(grid=1, block=1))
        T.cuda.launch(
            "fill", (x.data,), T.cuda.LaunchConfig(grid=1, block=1, dynamic_smem_bytes=32)
        )

    with pytest.raises(ValueError, match="same launch fields"):
        bind_cubin(two_sites, image, kernels={"fill": ("handle",)})

    @T.function
    def two_entries(x: T.Tensor((1,), T.int32)):
        T.cuda.launch("first", (x.data,), T.cuda.LaunchConfig(grid=1, block=1))
        T.cuda.launch("second", (x.data,), T.cuda.LaunchConfig(grid=1, block=1))

    original = tvm_tirx.IRModule.from_expr(two_entries)
    bound = bind_cubin(original, image, kernels={"first": ("handle",), "second": ("handle",)})
    assert not original.get_attr("external_mods")
    (device,) = bound.get_attr("external_mods")
    data, fmt, names = device["__tvm_cuda_binary"]()
    assert bytes(data) == image and fmt == "cubin" and set(names) == {"first", "second"}
    with pytest.raises(ValueError, match=r"first.*already defined"):
        bind_cubin(bound, image, kernels={"first": ("handle",)})

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100"}, host="cuda_host")
    with pytest.raises(ValueError, match="CUDA ELF cubin"):
        export_cuda_host(tvm_tirx.compile(original, target=target).mod, cubin=b"invalid")
    with pytest.raises(ValueError, match="no device imports"):
        export_cuda_host(tvm_tirx.compile(bound, target=target).mod, cubin=image)

    @T.function
    def mixed(x: T.Tensor((32,), T.int32)):
        T.cuda.launch("mixed_kernel", (x.data,), T.cuda.LaunchConfig(grid=1, block=32))
        with T.launch_thread("threadIdx.x", 32) as i:
            x[i] = i

    # SplitHostDevice introduces mixed_kernel for the thread region.
    program = bind_cubin(mixed, image, kernels={"mixed_kernel": ("handle",)})
    with pytest.raises(ValueError, match=r"mixed_kernel.*also defined by an import"):
        tvm_tirx.compile(program, target=target)


@pytest.fixture(scope="module")
def cubin():
    from tvm_tirx.backend.cuda.compiler import compile_source

    pytest.importorskip("cuda.bindings.nvrtc")
    return compile_source(
        'extern "C" __global__ void add_one(const float* x, float* y, int n) {'
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;"
        "  if (i < n) y[i] = x[i] + 1;"
        "}",
        {"cuda": {"compiler": "nvrtc", "target_format": "cubin"}},
        target=tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}),
    ).binary


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("host", ["llvm", "cuda_host"])
def test_precompiled_cubin(host, cubin, monkeypatch, tmp_path):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("JIT of a bound cubin invoked a device compiler")

    monkeypatch.setattr(tvm_tirx.support.nvcc, "_compile_cuda_nvrtc", unexpected_compile)
    monkeypatch.setattr(tvm_tirx.support.nvcc, "_compile_cuda_nvcc", unexpected_compile)

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}, host=host)
    program = bind_cubin(_wrapper(), cubin, kernels=_SIGNATURE)
    executable = tvm_tirx.compile(program, target=target)
    if host == "llvm":
        # The standard entry point JITs through ORC and keeps its import alive.
        run = executable["wrapper"]
    else:
        assert "cuLaunchKernelEx" in export_cuda_host(executable.mod)
        run = jit_cuda_host(executable.mod)["wrapper"]
    dev = tvm_tirx.cuda(0)
    stream = dev.create_raw_stream()
    try:
        for length in (31, 513):
            values = np.arange(length, dtype="float32")
            x = tvm_tirx.runtime.tensor(values, dev)
            y = tvm_tirx.runtime.empty((length,), "float32", dev)
            with tvm_ffi.use_raw_stream(dev, stream):
                if length == 513:
                    with pytest.raises(RuntimeError, match="at most 1024 threads"):
                        run(x, y, 2048)
                run(x, y, 256)
                dev.sync(stream)
            np.testing.assert_array_equal(y.numpy(), values + 1)
        if host == "llvm":
            path = tmp_path / "with_cubin.so"
            executable.export_library(path)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
import numpy as np
import tvm_tirx
loaded = tvm_tirx.runtime.load_module(sys.argv[1])
values = np.arange(67, dtype="float32")
x = tvm_tirx.runtime.tensor(values, tvm_tirx.cuda())
y = tvm_tirx.runtime.empty(values.shape, "float32", tvm_tirx.cuda())
loaded["wrapper"](x, y, 64)
np.testing.assert_array_equal(y.numpy(), values + 1)
""",
                    str(path),
                ],
                check=True,
            )
    finally:
        dev.free_raw_stream(stream)


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != "linux" or not env.has_cuda(), reason="requires Linux and CUDA")
def test_standalone_raw_cubin(cubin, tmp_path):
    """A native TVM-FFI process can run a single deployed library."""
    from tvm_ffi import libinfo

    from tvm_tirx.support.nvcc import find_cuda_path

    n = T.dynamic("n")

    @T.function
    def wrapper(x: T.Tensor((n,), T.float32), y: T.Tensor((n,), T.float32), threads: T.int32):
        # Two sites with different launch fields for the same cubin entry.
        if n <= threads:
            T.cuda.launch(
                "add_one",
                (x.data, y.data, T.Cast("int32", n)),
                T.cuda.LaunchConfig(grid=1, block=threads),
            )
        else:
            T.cuda.launch(
                "add_one",
                (x.data, y.data, T.Cast("int32", n)),
                T.cuda.LaunchConfig(
                    grid=T.ceildiv(n, threads), block=threads, dynamic_smem_bytes=128
                ),
            )

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}, host="cuda_host")
    host = tvm_tirx.compile(wrapper, target=target).mod
    assert not host.imports
    library = shutil.copyfile(build_cuda_host(host, cubin=cubin), tmp_path / "wrapper.so")

    cuda = Path(find_cuda_path())
    ffi_lib = Path(libinfo.find_libtvm_ffi()).parent
    runner = tmp_path / "runner"
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX") or "c++"),
            "-std=c++17",
            Path(__file__).with_name("standalone_cubin_runner.cc"),
            *[f"-I{inc}" for inc in libinfo.include_paths()],
            f"-I{cuda / 'include'}",
            f"-L{cuda / 'lib64'}",
            "-lcudart",
            f"-L{ffi_lib}",
            f"-Wl,-rpath,{ffi_lib}",
            "-ltvm_ffi",
            "-o",
            runner,
        ],
        check=True,
    )
    subprocess.run([runner, library], cwd=tmp_path, check=True)
