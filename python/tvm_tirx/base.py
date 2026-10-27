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
# coding: utf-8
# pylint: disable=invalid-name, import-outside-toplevel
"""Base library for TVM."""

import importlib
import importlib.util
import os
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from tvm_ffi import libinfo as tvm_ffi_libinfo
from tvm_ffi import register_global_func
from tvm_ffi.libinfo import load_lib_ctypes

from . import libinfo

# These receivers must exist before native static initialization. Keep them stable:
# native callers may cache them before the IR modules have finished importing.
# Op dispatcher installed after the IR wrappers finish loading.
_EXPOSE_OP_CALLBACK: Callable[[str], None] | None = None
# Type dispatcher installed at the same IR initialization boundary.
_EXPOSE_TYPE_CALLBACK: Callable[[str], None] | None = None
# Early native selections, removed only after successful dispatch.
_PENDING_EXPOSURES: deque[tuple[Literal["op", "type"], str]] = deque()


@register_global_func("ir.expose_op")
def _expose_op(name: str) -> None:
    """Request API exposure after the operator's registration is complete."""
    if _EXPOSE_OP_CALLBACK is None:
        _PENDING_EXPOSURES.append(("op", name))
    else:
        _EXPOSE_OP_CALLBACK(name)


@register_global_func("ir.expose_type")
def _expose_type(type_key: str) -> None:
    """Request API exposure after the type's reflection and constructor are ready."""
    if _EXPOSE_TYPE_CALLBACK is None:
        _PENDING_EXPOSURES.append(("type", type_key))
    else:
        _EXPOSE_TYPE_CALLBACK(type_key)


# ----------------------------
# library loading
# ----------------------------

# Whether only the runtime library is loaded (runtime-only wheel, or
# ``TVM_TIRX_USE_RUNTIME_LIB=1``). Set during library loading below.
_RUNTIME_ONLY = os.environ.get("TVM_TIRX_USE_RUNTIME_LIB") == "1"

# Handles of the core libraries actually loaded, keyed by basename
# (e.g. ``{"tvm_runtime": <CDLL>, "tvm_tirx_compiler": <CDLL>}``). Downstream /
# autoloaded extensions can inspect this to skip duplicate libraries
# (``"tvm_runtime" in _LOADED_LIBS``) and obtain the loaded handle.
_LOADED_LIBS = {}


# runtime is loaded RTLD_GLOBAL to expose its symbols to subsequent loads;
# compiler is loaded RTLD_LOCAL.
_LOADED_LIBS["tvm_runtime"] = load_lib_ctypes(
    "tvm_tirx", "tvm_runtime", "RTLD_GLOBAL", extra_lib_paths=libinfo.package_lib_paths()
)

if not _RUNTIME_ONLY:
    _lib_paths = [Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent]
    try:
        tvm_ffi_libinfo._find_library_by_basename(
            "tvm_tirx", "tvm_tirx_compiler", extra_lib_paths=_lib_paths
        )
    except RuntimeError:
        # A runtime-only installation need not contain the compiler library.
        _RUNTIME_ONLY = True
    else:
        _LOADED_LIBS["tvm_tirx_compiler"] = load_lib_ctypes(
            "tvm_tirx", "tvm_tirx_compiler", "RTLD_LOCAL", extra_lib_paths=_lib_paths
        )


def _load_llvm_backend(lib_paths):
    """Load one generic or native backend from the compiler installation."""
    package = "tvm_tirx"
    target_name = "tvm_tirx_llvm_backend"
    try:
        tvm_ffi_libinfo._find_library_by_basename(
            package,
            target_name,
            lib_paths,
            use_fallback_build_path=False,
            use_env_path=False,
        )
    except RuntimeError:
        # Only a native LLVM installation opts into the optional targets package.
        # LLVM-OFF installations must not load it even when it is installed.
        target_name = "tvm_tirx_llvm_backend_native"
        try:
            tvm_ffi_libinfo._find_library_by_basename(
                package,
                target_name,
                lib_paths,
                use_fallback_build_path=False,
                use_env_path=False,
            )
        except RuntimeError:
            return None
        if importlib.util.find_spec("tvm_tirx_llvm_targets") is not None:
            # Initialize wheel repair hooks before loading their dependent DLLs.
            importlib.import_module("tvm_tirx_llvm_targets")
            package = "apache-tvm-tirx-llvm-targets"
            target_name = "tvm_tirx_llvm_backend"
            tvm_ffi_libinfo._find_library_by_basename(
                package,
                target_name,
                use_fallback_build_path=False,
                use_env_path=False,
            )
    return tvm_ffi_libinfo.load_lib_ctypes(
        package, target_name, "RTLD_LOCAL", extra_lib_paths=lib_paths
    )


if not _RUNTIME_ONLY:
    _llvm_backend = _load_llvm_backend(
        [Path(_LOADED_LIBS["tvm_tirx_compiler"]._name).resolve().parent]
    )
    if _llvm_backend is not None:
        _LOADED_LIBS["tvm_tirx_llvm_backend"] = _llvm_backend

if _RUNTIME_ONLY:
    from tvm_ffi import registry as _tvm_ffi_registry

    _tvm_ffi_registry._SKIP_UNKNOWN_OBJECTS = True
