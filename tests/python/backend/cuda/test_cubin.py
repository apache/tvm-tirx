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

import ctypes
import os
import re
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
from tvm_tirx.backend.cuda import (
    bind_cubin,
    build_cuda_host,
    build_cuda_host_object,
    export_cuda_host,
    jit_cuda_host,
)
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

    if sys.platform == "linux" and tvm_tirx.runtime.enabled("llvm"):
        llvm = target.with_host("llvm")
        with pytest.raises(ValueError, match="only host functions"):
            build_cuda_host_object(bound, target=llvm, cubin=image)
        with pytest.raises(ValueError, match="only host functions"):
            build_cuda_host_object(mixed, target=llvm, cubin=image)

        @T.function
        def cooperative(x: T.Tensor((1,), T.int32)):
            T.cuda.launch("fill", (x.data,), T.cuda.LaunchConfig(grid=1, block=1, cooperative=True))

        with pytest.raises(
            ValueError, match=r"Unsupported direct CUDA launch attributes.*cooperative"
        ):
            build_cuda_host_object(cooperative, target=llvm, cubin=image)


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="requires LLVM codegen")
def test_same_lowered_host_ir_for_c_and_llvm():
    from tvm_tirx.backend.cuda.transform import LowerCUDAHostLaunch
    from tvm_tirx.tirx._compile import _split_host_device_modules

    mod = tvm_tirx.IRModule.from_expr(_wrapper())
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"}, host="c")
    pipeline, finalize_host, _ = tvm_tirx.tirx.get_default_tir_pipeline(
        target, mod, target_to_bind=target
    )
    host, devices = _split_host_device_modules(pipeline(mod))
    assert not devices and "call_ffi_kernel" in host.script()
    shared = finalize_host()(LowerCUDAHostLaunch()(host))
    assert "call_ffi_kernel" not in shared.script()

    # Feed the very same IRModule to both ordinary code generators.
    c = tvm_tirx.codegen.build_module(shared, tvm_tirx.target.Target("c")).inspect_source()
    llvm = tvm_tirx.codegen.build_module(shared, tvm_tirx.target.Target("llvm")).inspect_source(
        "ll"
    )
    assert c.count("#include <cuda.h>") == 1
    assert "cuLaunchKernelEx(" in c and "cuGetErrorString(" in c
    assert re.search(r"call .*@cuLaunchKernelEx\(", llvm)
    assert "TVMFFIEnvModLookupFromImports(" not in c


@pytest.fixture(scope="module")
def cubin():
    from tvm_tirx.backend.cuda.compiler import compile_source

    pytest.importorskip("cuda.bindings.nvrtc")
    return compile_source(
        'extern "C" __global__ void add_one(const float* x, float* y, int n) {'
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;"
        "  if (i < n) y[i] = x[i] + 1;"
        "}"
        'extern "C" __global__ void write_bits(unsigned short* out, unsigned short x, '
        "                                   unsigned short y) { out[0] = x; out[1] = y; }",
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
@pytest.mark.parametrize("host_kind", ["c", "llvm", "cuda_host"])
def test_standalone_raw_cubin(host_kind, cubin, tmp_path):
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

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}, host=host_kind)
    cuda = Path(find_cuda_path())
    ffi_lib = Path(libinfo.find_libtvm_ffi()).parent
    cxx = shlex.split(os.environ.get("CXX") or "c++")
    library = tmp_path / "wrapper.so"
    if host_kind in ("c", "llvm"):
        orc = pytest.importorskip("tvm_ffi_orcjit")
        obj = build_cuda_host_object(wrapper, target=target, cubin=cubin)
        path = tmp_path / "wrapper.o"
        path.write_bytes(obj)
        # Find the relocation in the compiled wrapper (the compute block may be outlined).
        disasm = subprocess.check_output(["objdump", "-dr", path], text=True)
        blocks = re.split(r"(?m)^[0-9a-f]+ <([^>]+)>:$", disasm)
        assert any(
            "wrapper" in name and re.search(r"\bR_\S+\s+cuLaunchKernelEx\b", body)
            for name, body in zip(blocks[1::2], blocks[2::2], strict=True)
        )
        relocs = subprocess.check_output(["readelf", "-Wr", path], text=True)
        assert "TVMBackend" not in relocs and "TVMFFIEnvModLookupFromImports" not in relocs
        subprocess.run(
            [
                *cxx,
                "-shared",
                "-Wl,-z,defs",
                path,
                f"-L{ffi_lib}",
                "-ltvm_ffi",
                f"-L{cuda / 'lib64'}",
                "-lcuda",
                "-lcudart",
                "-o",
                library,
            ],
            check=True,
        )
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(cuda / "lib64/libcudart.so"), mode=ctypes.RTLD_GLOBAL)
        run = orc.default_session().load_module(obj)["wrapper"]
        dev = tvm_tirx.cuda(0)
        stream = dev.create_raw_stream()
        try:
            values = np.arange(513, dtype="float32")
            x = tvm_tirx.runtime.tensor(values, dev)
            y = tvm_tirx.runtime.empty(values.shape, "float32", dev)
            with tvm_ffi.use_raw_stream(dev, stream):
                with pytest.raises(RuntimeError, match=r"cuLaunchKernelEx: .*invalid"):
                    run(x, y, 2048)
                run(x, y, 64)
                dev.sync(stream)
            np.testing.assert_array_equal(y.numpy(), values + 1)
        finally:
            dev.free_raw_stream(stream)
    else:
        host = tvm_tirx.compile(wrapper, target=target).mod
        assert not host.imports
        shutil.copyfile(build_cuda_host(host, cubin=cubin), library)

    runner = tmp_path / "runner"
    subprocess.run(
        [
            *cxx,
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


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != "linux" or not env.has_cuda(), reason="requires Linux and CUDA")
@pytest.mark.parametrize("host", ["c", "llvm"])
def test_standalone_16bit_scalar_arguments(host, cubin):
    from tvm_tirx.support.nvcc import find_cuda_path

    orc = pytest.importorskip("tvm_ffi_orcjit")

    @T.function
    def scalar_bits(out: T.Tensor((2,), T.uint16), value: T.float32):
        T.cuda.launch(
            "write_bits",
            (out.data, T.Cast("float16", value), T.Cast("bfloat16", value)),
            T.cuda.LaunchConfig(grid=1, block=1),
        )

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": env.cuda_arch()}, host=host)
    obj = build_cuda_host_object(scalar_bits, target=target, cubin=cubin)
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(Path(find_cuda_path()) / "lib64/libcudart.so"), mode=ctypes.RTLD_GLOBAL)
    run = orc.default_session().load_module(obj)["scalar_bits"]
    out = tvm_tirx.runtime.empty((2,), "uint16", tvm_tirx.cuda(0))
    run(out, 1.5)
    np.testing.assert_array_equal(out.numpy(), [0x3E00, 0x3FC0])
