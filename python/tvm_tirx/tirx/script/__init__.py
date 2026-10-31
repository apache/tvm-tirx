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
"""Public canonical TVMScript dialect namespace."""

import importlib as _importlib
import sys as _sys
from typing import Any as _Any

_initialized = False
_initializing = False
_ENTRY_EXPORTS = ("jit", "Optional")


def _initialize() -> None:
    global _initialized, _initializing
    if _initialized or _initializing:
        return
    _initializing = True
    try:
        from tvm_tirx.ir._expose_api import _init_exposed_type_api
        from tvm_tirx.script.parser import entry, protocol_registry, register_namespace
        from tvm_tirx.tirx.layout import Axis

        from . import ir_builder as builder
        from .jit import make_jit

        globals().update(
            (name, value) for name, value in vars(builder).items() if not name.startswith("_")
        )
        globals().update(
            ir_module=entry.ir_module,
            pyfunc=entry.pyfunc,
            function=entry.make_decorator(builder, namespace_path="tirx.function"),
            inline=entry.make_macro_decorator(
                builder, namespace_path="tirx.inline", preserve_return=True, late_binding=True
            ),
            macro=entry.make_macro_decorator(
                builder, namespace_path="tirx.macro", preserve_return=False
            ),
            jit=make_jit(builder, namespace_path="tirx.jit"),
        )
        globals()["function"].builder = entry.make_decorator(
            builder,
            namespace_path="tirx.function.builder",
            definition_kind=protocol_registry.DefinitionKind.BUILDER,
        )
        globals()["function"].builder.__doc__ = """Build a TIRx function with Python body semantics.

        Use at the definition site with inspectable source. Assignments, loops,
        conditionals and returns execute as Python; explicit ``T.serial``-family
        frames, ``T.if_``/``T.then_``/``T.else_``, ``T.assign``, ``T.bind_`` and
        ``T.return_`` construct IR. Signature annotations are re-evaluated from
        source, including postponed annotations; write expressions without quotes.

        Shares ``T.function`` options, expression locations and module declarations.
        Statement locations are not injected: explicit frames, void emissions and
        IR emitted inside ordinary Python helpers may have unknown locations.
        Use ``T.call_global_var_(callee, args)`` for explicit IR function calls.
        The resulting IR prints as ordinary ``@T.function`` TVMScript.

        Examples
        --------
        Python loops execute while building; ``T.serial`` creates an IR loop:

        .. code-block:: python

            from tvm_tirx.script import tirx as T

            @T.function.builder
            def add_rows(A: T.Tensor((2, 4), T.float32), B: T.Tensor((2, 4), T.float32)):
                for row in range(2):
                    with T.serial(4) as col:
                        T.assign(B[row, col], A[row, col] + 1)

        Mixed modules share declarations, so explicit calls can refer to a later
        function:

        .. code-block:: python

            @T.ir_module
            class Module:
                @T.function.builder
                def main(A: T.Tensor((4,), T.int32)):
                    with T.serial(4) as i:
                        T.assign(A[i], T.call_global_var_(Module.add_one, [A[i]]))

                @T.function
                def add_one(value: T.int32) -> T.int32:
                    return value + 1
        """
        for name, syntax in {
            "serial": "for",
            "parallel": "for",
            "unroll": "for",
            "vectorized": "for",
            "grid": "for",
            "for_": "for",
            "if_": "if",
            "while_": "while",
        }.items():
            protocol_registry.WITH_FRAME_SUGAR[f"tirx.{name}"] = syntax
        namespace = _sys.modules[__name__]
        register_namespace("tirx", namespace)
        register_namespace("Axis", Axis)
        globals()["__all__"] = sorted(name for name in globals() if not name.startswith("_"))
        from tvm_tirx.ir._expose_api import _init_exposed_op_api
        from tvm_tirx.script.ir_builder.base import auto_emit_void_call

        _init_exposed_type_api("ir", __name__)
        _init_exposed_type_api("tirx", __name__)
        _init_exposed_op_api("ir", __name__, wrapper=auto_emit_void_call)
        _init_exposed_op_api("prim", __name__, wrapper=auto_emit_void_call)
        _init_exposed_op_api("tirx", __name__, wrapper=auto_emit_void_call)
        _initialized = True
    finally:
        _initializing = False


def __getattr__(name: str) -> _Any:
    if name == "ir_builder":
        return _importlib.import_module(__name__ + ".ir_builder")
    if name.startswith("_") and name != "__all__":
        raise AttributeError(name)
    _initialize()
    if name in globals():
        return globals()[name]
    return getattr(_importlib.import_module("tvm_tirx.tirx.script.ir_builder"), name)


# Resolve JIT annotations after the lazy dialect namespace is installed.
from .jit import OptionalAnnotation as Optional  # noqa: E402

globals().pop("jit", None)
