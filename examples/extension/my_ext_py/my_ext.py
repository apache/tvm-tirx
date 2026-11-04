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
"""A Python prototype of my_ext: the complex type, its operators, and the pass that lowers them.

It defines the same names as the Rust package in ``python/my_ext``, so ``run_example.py`` runs
with either; a process loads one of them.
"""

import tvm_ffi
from tvm_ffi.dataclasses import py_class

from tvm_tirx import ir, tirx
from tvm_tirx.ir._expose_api import _init_exposed_op_api, _init_exposed_type_api
from tvm_tirx.script.ir_builder.base import auto_emit_void_call

# begin load
_init_exposed_op_api("my_ext", __name__, wrapper=auto_emit_void_call)
_init_exposed_type_api("my_ext", __name__)
# end load


# begin type
@py_class("my_ext.ComplexType")
class ComplexType(ir.Type):
    """A complex number whose real and imaginary parts share a scalar type."""

    dtype: ir.PrimType


tvm_ffi.get_global_func("ir.expose_type")("my_ext.ComplexType")
# end type


def _require(condition, message):
    if not condition:
        raise TypeError(message)


def _check_call(call, count):
    _require(len(call.args) == count, f"expected {count} value arguments")
    _require(len(call.ty_args) == 0, "type arguments are not supported")
    _require(call.attrs is None, "call attributes are not supported")


def _float_part(ty):
    """The type of a part of a complex number: a scalar float, such as float32 or bfloat16."""
    _require(isinstance(ty, ir.PrimType), "expected a scalar float")
    dtype = ty.dtype
    _require(dtype in (f"float{dtype.bits}", f"bfloat{dtype.bits}"), "expected a scalar float")
    return ty


def _infer_complex(call):
    _check_call(call, 2)
    elem = _float_part(call.args[0].ty)
    im = _float_part(call.args[1].ty)
    _require(elem.dtype == im.dtype, "my_ext.complex expects parts of the same type")
    return ComplexType(elem)


def _complex_arg(call, index):
    ty = call.args[index].ty
    _require(isinstance(ty, ComplexType), f"{call.op.name} expects a complex value")
    _float_part(ty.dtype)
    return ty


def _infer_cmul(call):
    _check_call(call, 2)
    a, b = _complex_arg(call, 0), _complex_arg(call, 1)
    _require(a.dtype.dtype == b.dtype.dtype, "my_ext.cmul expects complex values of the same type")
    return a


def _infer_part(call):
    _check_call(call, 1)
    return _complex_arg(call, 0).dtype


# begin constructor
def _define(name, doc, args, infer_type):
    op = ir.Op.register(f"my_ext.{name}", doc)
    op.set_signature(args=args)
    op.set_attr("FInferType", infer_type)
    op.set_attr("TCallEffectKind", ir.CallEffectKind.Pure.value)
    op.expose()
    return op


COMPLEX = _define(
    "complex",
    "Construct the complex number re + im * i.",
    [("re", "The real part."), ("im", "The imaginary part.")],
    _infer_complex,
)
# end constructor
# begin operators
CMUL = _define(
    "cmul",
    "Multiply two complex numbers.",
    [("a", "The first complex value."), ("b", "The second complex value.")],
    _infer_cmul,
)
REAL = _define(
    "real", "Extract the real part of a complex number.", [("z", "The complex value.")], _infer_part
)
IMAG = _define(
    "imag",
    "Extract the imaginary part of a complex number.",
    [("z", "The complex value.")],
    _infer_part,
)
_OPS = (COMPLEX, CMUL, REAL, IMAG)
# end operators
_INFER = {"my_ext.complex": _infer_complex, "my_ext.cmul": _infer_cmul}


def _validate_call(call):
    """Infer the type of a call to one of the operators again, and compare it with its type."""
    expected = _INFER.get(call.op.name, _infer_part)(call)
    _require(
        tvm_ffi.structural_equal(expected, call.ty), "incorrect result type for complex operator"
    )


class _Lowering:
    def __init__(self):
        # Each complex local variable is replaced by two scalar local variables.
        self.bindings = {}

    def parts(self, expr):
        if isinstance(expr, ir.Call) and expr.op.same_as(COMPLEX):
            return expr.args[0], expr.args[1]
        if expr in self.bindings:
            return self.bindings[expr]
        raise ValueError("LowerComplex cannot resolve complex operand")

    def map_call(self, call):
        if any(call.op.same_as(op) for op in _OPS):
            _validate_call(call)
        if call.op.same_as(CMUL):
            a, b = self.parts(call.args[0])
            c, d = self.parts(call.args[1])
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            return ir.Call(COMPLEX, [a * c - b * d, a * d + b * c])
        if call.op.same_as(REAL) or call.op.same_as(IMAG):
            re, im = self.parts(call.args[0])
            return re if call.op.same_as(REAL) else im
        return call

    def map_bind(self, bind):
        ty = bind.var.ty
        if isinstance(ty, ComplexType):
            re, im = self.parts(bind.value)
            re_var = ir.Var(f"{bind.var.name}_re", ty.dtype)
            im_var = ir.Var(f"{bind.var.name}_im", ty.dtype)
            self.bindings[bind.var] = (re_var, im_var)
            return ir.SeqStmt([ir.Bind(re_var, re), ir.Bind(im_var, im)])
        return bind

    def map_sequence(self, seq):
        return ir.SeqStmt(list(seq.seq))


def _check_type(ty):
    """Reject a complex type, wherever it appears: on a variable, a parameter or an expression."""
    if isinstance(ty, ComplexType):
        raise ValueError("LowerComplex left an unsupported complex type")
    return ty


def _check_expr(expr):
    """Reject an expression whose type is complex."""
    _check_type(expr.ty)
    return expr


def _check_call_lowered(call):
    """Reject a call to one of the four complex operators that the rewrite did not consume."""
    if any(call.op.same_as(op) for op in _OPS):
        raise ValueError("LowerComplex left an unsupported complex call")
    return call


def _lower_function(func, _module, _ctx):
    lowering = _Lowering()
    rewritten = tvm_ffi.structural_map(
        func,
        [
            (ir.Call, lowering.map_call),
            (ir.Bind, lowering.map_bind),
            (ir.SeqStmt, lowering.map_sequence),
        ],
    )
    return tvm_ffi.structural_map(
        rewritten, [(ir.Type, _check_type), (ir.Call, _check_call_lowered), (ir.Expr, _check_expr)]
    )


def lower_complex() -> ir.transform.Pass:
    """The pass that lowers complex values to scalar arithmetic."""
    return tirx.transform.function_pass(_lower_function, opt_level=0, name="my_ext.LowerComplex")


# begin pipeline
def _pipeline(*, target=None):
    lower, finalize_host, finalize_device = tirx.get_tir_pipeline("default", target=target)
    return ir.transform.Sequential([lower_complex(), lower]), finalize_host, finalize_device


tirx.register_tir_pipeline("my_ext", _pipeline)
# end pipeline
