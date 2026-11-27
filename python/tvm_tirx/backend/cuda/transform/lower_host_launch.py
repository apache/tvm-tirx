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
"""Lower CUDA launches into ordinary host IR and external C calls."""

import tvm_ffi

from tvm_tirx import ir, tirx
from tvm_tirx.ir.transform import module_pass
from tvm_tirx.tirx import op


def _alloc(stmts, name, dtype, count=1):
    ty = tirx.TensorType((count,), dtype, align=8)
    var = ir.Var(name, ty)
    # stack_alloca("shape") allocates aligned int64 words in the host entry block,
    # even inside a CUDA device_context. Each supported scalar fits in one word.
    storage = op.stack_alloca("shape", count)
    stmts.append(ir.Bind(var, ir.Call("tirx.decl_tensor", [storage], ty_args=[ty])))
    return var


def _string_address(text):
    return op.reinterpret(op.reinterpret(ir.StringImm(text), ty="handle"), ty="uint64")


def _lower_launch(call):
    """Materialize the device ABI values and the driver's void** argument vector."""
    attrs = call.attrs
    if attrs is None or attrs.num_kernel_args < 0 or not isinstance(call.args[0], ir.StringImm):
        raise ValueError("Direct host compilation requires a named kernel with a CUDA LaunchConfig")
    split = 1 + attrs.num_kernel_args
    fields = dict(zip(attrs.launch_fields, call.args[split:], strict=True))
    dims = [f"{field}.{axis}" for field in ("grid", "block") for axis in "xyz"]
    unsupported = fields.keys() - {*dims, "dynamic_smem_bytes", "stream"}
    if attrs.kernel_attrs:
        raise ValueError("Direct CUDA host objects do not yet support kernel attributes")
    if unsupported:
        raise ValueError(f"Unsupported direct CUDA launch attributes: {sorted(unsupported)}")
    stmts = []
    null = op.reinterpret(ir.IntImm("uint64", 0), ty="handle")
    params = null
    if split > 1:
        pointers = _alloc(stmts, "parameters", "uint64", split - 1)
        for i, arg in enumerate(call.args[1:split]):
            if isinstance(arg.ty, ir.PtrType) and not isinstance(
                arg.ty.element_type, tirx.TensorMapType
            ):
                value = op.reinterpret(arg, ty="uint64")
            elif isinstance(arg.ty, ir.PrimType) and str(arg.ty.dtype) in {
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
                "float16",
                "bfloat16",
                "float32",
                "float64",
            }:
                value = arg
            else:
                raise ValueError(f"Unsupported direct CUDA kernel argument type: {arg.ty}")
            slot = _alloc(stmts, "arg", value.ty.dtype)
            stmts.append(ir.TensorStore(slot, [0], value))
            stmts.append(ir.TensorStore(pointers, [i], op.reinterpret(slot.data, ty="uint64")))
        params = op.reinterpret(pointers.data, ty="handle")

    config = _alloc(stmts, "config", "uint64", 8)
    function = _alloc(stmts, "function", "uint64")
    status = ir.Var("prepare_status", "int32")
    prepare = op.call_extern(
        "tirx_cuda_prepare_launch",
        call.args[0],
        op.reinterpret(config.data, ty="handle"),
        op.reinterpret(function.data, ty="handle"),
        *[tirx.Cast("int64", fields[key]) for key in dims],
        tirx.Cast("int64", fields.get("dynamic_smem_bytes", ir.IntImm("int32", 0))),
        fields.get("stream", null),
        int("stream" in fields),
        attrs={"c_header": "host_object_support.h"},
        ty="int32",
    )
    stmts.extend(
        [ir.Bind(status, prepare), ir.If(status != 0, ir.Evaluate(op.throw_last_error()), None)]
    )
    launch = op.call_extern(
        "cuLaunchKernelEx",
        op.reinterpret(config.data, ty="handle"),
        op.reinterpret(function[0], ty="handle"),
        params,
        null,
        attrs={"c_header": "cuda.h"},
        ty="int32",
    )
    result = ir.Var("launch_status", "int32")
    stmts.append(ir.Bind(result, launch))
    failure = []
    parts = _alloc(failure, "error_parts", "uint64", 2)
    failure.append(ir.TensorStore(parts, [0], _string_address("cuLaunchKernelEx: ")))
    lookup = op.call_extern(
        "cuGetErrorString",
        result,
        op.reinterpret(op.address_of(parts[1]), ty="handle"),
        attrs={"c_header": "cuda.h"},
        ty="int32",
    )
    failure.append(
        ir.If(lookup != 0, ir.TensorStore(parts, [1], _string_address("unknown CUDA error")), None)
    )
    failure.append(
        ir.Evaluate(
            op.call_extern(
                "TVMFFIErrorSetRaisedFromCStrParts",
                "RuntimeError",
                op.reinterpret(parts.data, ty="handle"),
                2,
                attrs={"c_header": "tvm/ffi/c_api.h"},
                ty="void",
            )
        )
    )
    failure.append(ir.Evaluate(op.throw_last_error()))
    stmts.append(ir.If(result != 0, failure, None))
    return ir.SeqStmt(stmts, loc=call.loc)


@module_pass(opt_level=0, name="LowerCUDAHostLaunch")
class LowerCUDAHostLaunch:
    """Lower explicit CUDA host launches to C ABI calls before host code generation.

    Apply to the host module after packed entry generation and before LowerTVMBuiltin.
    Each named kernel is resolved from the cubin linked with the host object.
    The resulting IR can be sent to the standard C or LLVM backend.
    """

    def transform_module(self, mod, _ctx):
        def lower_evaluate(stmt, mutator):
            call = stmt.value
            if isinstance(call, ir.Call) and getattr(call.op, "name", "") == "tirx.call_ffi_kernel":
                return _lower_launch(call)
            return mutator.default_mutate(stmt)

        def lower_call(call, mutator):
            name = getattr(call.op, "name", "")
            if (
                name == "tirx.call_extern"
                and call.args[0].value == "TVMBackendSetDevice"
                and isinstance(call.args[1], ir.IntImm)
                and int(call.args[1]) == tvm_ffi.DLDeviceType.kDLCUDA
            ):
                return op.call_extern(
                    "tirx_cuda_set_device",
                    call.args[2],
                    attrs={"c_header": "host_object_support.h"},
                    ty="int32",
                )
            if name == "tirx.call_ffi_kernel":
                raise ValueError("A direct CUDA launch must be used as a statement")
            return mutator.default_mutate(call)

        return tvm_ffi.structural_mutate(
            mod, [(ir.Evaluate, lower_evaluate), (ir.Call, lower_call)]
        )
