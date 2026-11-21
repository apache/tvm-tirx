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
"""Bind precompiled CUDA device code to explicit TIRx kernel calls."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import tvm_ffi

from tvm_tirx import ir, tirx


def _check_kernel_symbol(name):
    if not isinstance(name, str) or not name.isascii() or not name.isidentifier():
        raise ValueError(f"Expected a CUDA kernel symbol spelled as a C identifier, got {name!r}")


def _check_cubin_image(image):
    # Both the ELF identification and e_machine must describe a CUDA image.
    if not (
        isinstance(image, bytes)
        and len(image) >= 64
        and image.startswith(b"\x7fELF\x02\x01\x01")
        and int.from_bytes(image[18:20], "little") == 190  # EM_CUDA
    ):
        raise ValueError("Expected a 64-bit little-endian CUDA ELF cubin")


def _imported_cuda_symbols(imports):
    names = set()
    for module in imports:
        if module.kind == "cuda":
            _, _, symbols = module["__tvm_cuda_binary"]()
            names.update(symbols)
    return names


def _check_import_conflicts(device_groups, imports):
    # Outlining thread regions can introduce new names after bind_cubin.
    imported = _imported_cuda_symbols(imports)
    for group in device_groups:
        for func in group.functions.values():
            symbol = func.attrs.get("global_symbol")
            if symbol in imported:
                raise ValueError(f"CUDA kernel symbol {symbol!r} is also defined by an import")


def _parameter_type(ty):
    if isinstance(ty, ir.PtrType) and not isinstance(ty.element_type, tirx.TensorMapType):
        return "handle"
    if isinstance(ty, ir.PrimType):
        return str(ty.dtype)
    raise ValueError(f"Unsupported cubin argument type {ty}")


def _kernel_metadata(mod, signatures):
    """Derive the CUDA runtime's packed launch layout from all uses of each symbol."""
    metadata = {}

    def collect(call):
        if not isinstance(call, ir.Call) or getattr(call.op, "name", "") != "tirx.call_ffi_kernel":
            return
        if not call.args or not isinstance(call.args[0], ir.StringImm):
            return
        name = call.args[0].value
        if name not in signatures:
            return
        attrs = call.attrs
        if attrs is None or attrs.num_kernel_args < 0:
            raise ValueError(f"Binding {name!r} requires an explicit CUDA LaunchConfig")
        actual = tuple(_parameter_type(arg.ty) for arg in call.args[1 : 1 + attrs.num_kernel_args])
        if actual != signatures[name]:
            raise ValueError(f"CUDA kernel {name!r} expects {signatures[name]}, got {actual}")
        info = {
            "name": name,
            "arg_types": actual,
            "cuda_launch_fields": list(attrs.launch_fields),
            "cuda_kernel_attrs": dict(attrs.kernel_attrs),
        }
        if metadata.setdefault(name, info) != info:
            raise ValueError(
                f"Calls to CUDA kernel {name!r} must use the same launch fields and attributes"
            )

    for func in mod.functions.values():
        if isinstance(func, tirx.Function):
            tvm_ffi.structural_walk(func.body, collect)
    unused = signatures.keys() - metadata.keys()
    if unused:
        raise ValueError(f"No explicit CUDA launch for declared kernels {sorted(unused)}")

    return metadata


def bind_cubin(
    mod: ir.IRModule | tirx.Function, image: bytes, *, kernels: Mapping[str, Sequence[str]]
) -> ir.IRModule:
    """Attach one cubin to a host function or module before compilation.

    ``kernels`` maps each referenced entry symbol in the image to its ordered
    parameter types. Use ``"handle"`` for device pointers. The supported scalar
    types are int32, uint32, int64, float32 and float64. Each call to a symbol
    must use the same launch fields, but their values can vary at runtime.

    Returns a new IRModule with a serializable CUDA device import. Call again
    to attach a different image. The caller supplies a trusted cubin for the
    target GPU and declares its ABI. TIRx checks calls against this declaration.
    The CUDA driver resolves the actual entry on first use.
    """
    if isinstance(mod, tirx.Function):
        mod = ir.IRModule.from_expr(mod)
    if not isinstance(mod, ir.IRModule):
        raise TypeError("bind_cubin expects an IRModule or TIRx function")
    _check_cubin_image(image)
    if not isinstance(kernels, Mapping):
        raise TypeError("kernels must map CUDA symbols to parameter types")
    if not kernels:
        raise ValueError("Declare at least one kernel from the cubin")

    imports = list(mod.get_attr("external_mods") or [])
    occupied = {gv.name_hint for gv in mod.functions}
    for func in mod.functions.values():
        if func.attrs and "global_symbol" in func.attrs:
            occupied.add(func.attrs["global_symbol"])
    occupied.update(_imported_cuda_symbols(imports))

    signatures = {}
    for name, params in kernels.items():
        _check_kernel_symbol(name)
        if name == "__tvm_cuda_binary" or name in occupied:
            raise ValueError(f"CUDA kernel symbol {name!r} is already defined")
        if isinstance(params, str) or not isinstance(params, Sequence):
            raise TypeError(f"Expected a sequence of kernel parameter types for {name!r}")
        signatures[name] = tuple(params)
        for dtype in params:
            if dtype not in ("handle", "int32", "uint32", "int64", "float32", "float64"):
                raise ValueError(f"Unsupported cubin parameter type {dtype!r} for {name!r}")

    metadata = _kernel_metadata(mod, signatures)
    load_metadata = tvm_ffi.get_global_func("runtime.LoadMetaDataFromJSON")
    create = tvm_ffi.get_global_func("ffi.Module.create.cuda", allow_missing=True)
    if create is None:
        create = tvm_ffi.get_global_func("ffi.Module.create.cuda_fallback")
    device = create(image, "cubin", load_metadata(json.dumps({"func_info": metadata})), {})
    return mod.with_attr("external_mods", [*imports, device])
