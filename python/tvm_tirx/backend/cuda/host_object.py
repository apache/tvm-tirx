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
"""Lower explicit host launches into a relocatable Linux host object."""

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from tvm_tirx import tirx


def build_cuda_host_object(mod, *, cubin: bytes, target) -> bytes:
    """Compile host TIRx for the native 64-bit Linux CPU and embed one raw cubin.

    Use a CUDA target with a C or LLVM host. The relocatable ELF exports TVM-FFI
    entry points. Link it with TVM-FFI, the CUDA driver, cudart, and the C++
    runtime, or load it with TVM-FFI ORC after making the CUDA libraries visible.
    Each launch can specify grid, block, stream and dynamic shared memory.
    Kernel operand types at each site must match the cubin entry's ABI.
    """
    import tvm_ffi.cpp

    from tvm_tirx import IRModule, codegen, libinfo
    from tvm_tirx.backend.config import prepare_target
    from tvm_tirx.backend.cuda.transform import LowerCUDAHostLaunch
    from tvm_tirx.support.nvcc import find_cuda_path
    from tvm_tirx.target import Target
    from tvm_tirx.tirx._compile import _split_host_device_modules

    if not sys.platform.startswith("linux") or sys.maxsize <= 2**32:
        raise ValueError("Direct CUDA host objects require 64-bit Linux")
    if not (
        isinstance(cubin, bytes)
        and len(cubin) >= 64
        and cubin.startswith(b"\x7fELF\x02\x01\x01")
        and int.from_bytes(cubin[18:20], "little") == 190
    ):
        raise ValueError("Expected a 64-bit little-endian CUDA ELF cubin")
    if isinstance(mod, tirx.Function):
        mod = IRModule.from_expr(mod)
    target = Target(prepare_target(target, mod))
    if target.kind.name != "cuda":
        raise ValueError("Expected a CUDA device target")
    host = target.host or Target("llvm")
    if host.kind.name not in ("c", "llvm"):
        raise ValueError("Direct CUDA host objects require a C or LLVM host backend")
    pipeline, finalize_host, _ = tirx.get_default_tir_pipeline(
        target, mod, target_to_bind=target.with_host(host)
    )
    host_ir, devices = _split_host_device_modules(pipeline(mod))
    if devices or host_ir.get_attr("external_mods"):
        raise ValueError("Expected only host functions with launches into the supplied cubin")
    lowered = LowerCUDAHostLaunch()(host_ir)
    # Kernel argument slots expose their storage type only after launch lowering.
    lowered = tirx.transform.BF16ComputeLegalize()(lowered)
    lowered = tirx.transform.BF16StorageLegalize()(lowered)
    host_module = codegen.build_module(finalize_host()(lowered), host)
    cuda = Path(find_cuda_path())
    support = tvm_ffi.cpp.build(
        name="tirx_cuda_host_support",
        backend="cpu",
        output="support.o",
        sources=[str(Path(__file__).with_name("host_object_support.cc"))],
        extra_include_paths=[str(cuda / "include")],
    )
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        if host.kind.name == "c":
            host_module.write_to_file(str(work / "host.c"))
            includes = [
                *tvm_ffi.libinfo.include_paths(),
                libinfo.find_include_path(),
                cuda / "include",
                Path(__file__).parent,
            ]
            cc = shlex.split(os.environ.get("CC") or "cc")
            # C host stack storage is reused through typed tensor views.
            subprocess.run(
                [
                    *cc,
                    "-std=gnu11",
                    "-O2",
                    "-fPIC",
                    "-fno-strict-aliasing",
                    "-Werror=implicit-function-declaration",
                    "-Werror=incompatible-pointer-types",
                    *[f"-I{path}" for path in includes],
                    "-c",
                    "host.c",
                    "-o",
                    "host.o",
                ],
                cwd=work,
                check=True,
            )
        else:
            host_module.write_to_file(str(work / "host.o"))
        (work / "device.cubin").write_bytes(cubin)
        # An aligned read-only ELF symbol; never feed the image through a C++ frontend.
        assembly = (
            ".section .rodata\n.balign 64\n.hidden tirx_cuda_image\n"
            '.globl tirx_cuda_image\ntirx_cuda_image:\n.incbin "device.cubin"\n'
            '.section .note.GNU-stack,"",%progbits\n'
        )
        cxx = shlex.split(os.environ.get("CXX") or "c++")
        subprocess.run(
            [*cxx, "-x", "assembler", "-c", "-", "-o", "image.o"],
            cwd=work,
            input=assembly,
            text=True,
            check=True,
        )
        # ORC cannot resolve hidden compiler-rt/libgcc helpers from the process.
        compiler_rt = subprocess.check_output([*cxx, "-print-libgcc-file-name"], text=True).strip()
        subprocess.run(
            [*cxx, "-nostdlib", "-r", "host.o", support, "image.o", compiler_rt, "-o", "wrapper.o"],
            cwd=work,
            check=True,
        )
        return (work / "wrapper.o").read_bytes()
