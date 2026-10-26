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

import pytest
import tvm_ffi

import tvm_tirx as tvm
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize(
    "target, message",
    [
        ({"kind": "c"}, "Bad bit-width for float: float128"),
        ({"kind": "llvm"}, "do not support"),
        ({"kind": "cuda", "arch": "sm_80"}, "only support even lane"),
        ({"kind": "opencl"}, "Cannot convert type float32x5 to OpenCL type"),
        (
            {"kind": "metal", "max_function_args": 31},
            "Cannot convert type float32x5 to Metal type",
        ),
        ({"kind": "webgpu"}, "only allows vector with lanes"),
    ],
    ids=lambda value: value["kind"] if isinstance(value, dict) else None,
)
def test_codegen_error_location(target, message):
    kind = target["kind"]
    if tvm_ffi.get_global_func(f"target.build.{kind}", allow_missing=True) is None:
        pytest.skip(f"{kind} codegen is not enabled")

    if kind in ("c", "llvm"):
        # Exercise the public compilation path, including nested statement and
        # expression dispatch, rather than just the lower-level build entry.
        @T.function
        def main():
            T.evaluate(T.call_extern("consume", T.FloatImm("float128", 1.0), ty="int32"))

        def build():
            return tirx.compile(main, target=tvm.target.Target(target))

        offending = 'T.FloatImm("float128", 1.0)'
    else:
        # Failure occurs while generating source, before any device compiler or
        # runtime is needed. These lanes are unsupported by each source backend.
        value = prim.Broadcast(prim.const(1.0, "float32"), 5)
        var = ir.Var("bad_value", ir.PrimType("float32x5"))
        func = (
            tirx.Function([], ir.Bind(var, value))
            .with_attr("global_symbol", "main")
            .with_attr("calling_conv", 2)
            .with_attr("tirx.kernel_launch_params", [])
        )
        mod = tvm.IRModule({"main": func})

        def build():
            return tvm.codegen.build_module(mod, tvm.target.Target(target))

        offending = "T.Broadcast(T.float32(1.0), 5)"

    with pytest.raises(RuntimeError, match=message) as caught:
        build()
    text = str(caught.value)
    assert f"Error in codegen: {kind}" in text
    assert text.count("Location (TVMScript):") == 1
    lines = text.splitlines()
    source_index = next(i for i, line in enumerate(lines) if offending in line)
    source, underline = lines[source_index : source_index + 2]
    # Underline the throw site, not an unrelated leaf or the whole function.
    start = source.index(offending)
    assert underline[start : start + len(offending)] == "^" * len(offending)
    assert not underline[:start].strip()


def test_unavailable_backend_preserves_value_error():
    from tvm_tirx.tirx._compile import _build_for_target

    if tvm_ffi.get_global_func("target.build.ext_dev", allow_missing=True) is not None:
        pytest.skip("external device codegen is registered")
    with pytest.raises(ValueError, match=r"Cannot find global function target\.build\.ext_dev"):
        _build_for_target(tvm.IRModule(), tvm.target.Target("ext_dev"))
