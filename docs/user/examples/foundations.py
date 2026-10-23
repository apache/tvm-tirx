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
"""Construct and analyze existing TIRx objects without compiling a program."""

# begin imports
import numpy as np
import tvm_ffi

from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.script import tirx as T

# end imports

# begin expression and type relationship
# begin scalar expressions
n = ir.Var("n", "int64")
two = ir.const(2, "int64")
extent = two * n
# end scalar expressions
# begin tensor expressions
tensor_type = tirx.TensorType((extent, 128), "float32")
x = ir.Var("X", tensor_type)
element = x[0, 0]
# end tensor expressions
# end expression and type relationship

assert isinstance(n, ir.Var) and n.ty == ir.PrimType("int64")
assert isinstance(extent, ir.prim.Mul) and extent.ty == n.ty
assert isinstance(x, ir.Var) and x.ty.same_as(tensor_type)
assert isinstance(x.ty, tirx.TensorType)
assert len(x.ty.shape) == 2
assert x.ty.shape[0].same_as(extent)
assert isinstance(element, ir.TensorLoad) and element.ty == ir.PrimType("float32")

# begin printing
print(extent.script())
# end printing

# begin typed values
data_pointer = x.data
assert isinstance(data_pointer, ir.Call) and isinstance(data_pointer.ty, ir.PtrType)
assert data_pointer.ty.element_type == ir.PrimType("float32")
# end typed values

# begin expression categories
assert ir.is_prim_expr(n)
assert ir.is_prim_expr(element)
assert not ir.is_prim_expr(x)
# end expression categories

# begin tensor interface
ir.assert_structural_equal(x.shape, x.ty.shape)
assert isinstance(x.dtype, tvm_ffi.dtype)
assert isinstance(x.ty.dtype, ir.PrimType)
assert x.dtype == x.ty.dtype.dtype
assert x.scope() == runtime.MemoryScope("global")
# end tensor interface

# begin predicates
positive = n > 0
equals_four = n.equal(ir.const(4, "int64"))
assert positive.ty == ir.PrimType("bool")
assert equals_four.ty == ir.PrimType("bool")
# end predicates

# begin primitive types
scalar_type = ir.PrimType("float32")
vector_type = ir.PrimType("float32x4")
assert scalar_type.is_scalar()
assert not vector_type.is_scalar()
assert vector_type.dtype.bits == 32 and vector_type.dtype.lanes == 4
assert vector_type.matches_code(runtime.DataTypeCode.FLOAT)
assert not vector_type.matches_code()
assert vector_type.matches_element_type(runtime.DataTypeCode.FLOAT, 32)
assert scalar_type().ty == scalar_type
assert scalar_type(ir.const(1, "int32")).ty == scalar_type
annotation = scalar_type[4]
assert annotation.shape == (4,)
# end primitive types

# begin composite type
integer_type = ir.PrimType("int64")
signature = ir.FuncType([integer_type], integer_type)
# end composite type
assert signature.arg_types[0].same_as(integer_type)
assert signature.ret_type.same_as(integer_type)

# begin constant tensor type
tile_type = tirx.TensorType((8, 128), "float32")
# end constant tensor type
assert tile_type.dtype == ir.PrimType("float32")
assert tuple(int(dim) for dim in tile_type.shape) == (8, 128)

# begin type states
unresolved = ir.Var("unresolved", ir.Type.missing())
assert isinstance(unresolved.ty, ir.MissingType)
assert isinstance(ir.Type.missing(), ir.MissingType)
assert not isinstance(ir.NoneType(), ir.MissingType)
assert not isinstance(ir.AnyType(), ir.MissingType)
assert ir.PrimType("void") != ir.NoneType()
# end type states

# begin variable contract
named = ir.Var(name_hint="count", ty="int64")
assert named.name == "count"
assert named.ty == ir.PrimType("int64")
assert isinstance(named.loc, ir.UnknownLoc)
pointer = ir.Var("opaque_pointer", "handle")
assert isinstance(pointer.ty, ir.PtrType)
assert pointer.ty.element_type == ir.PrimType("void")
# end variable contract

# begin constants
assert ir.const(True).ty == ir.PrimType("bool")
assert ir.const(1).ty == ir.PrimType("int32")
assert ir.const(1.0).ty == ir.PrimType("float32")
assert ir.const(np.int64(1)).ty == ir.PrimType("int32")
wide = ir.const(1 << 31, dtype="int64")
assert wide.ty == ir.PrimType("int64")
assert int(wide) == 1 << 31
assert wide.value == 1 << 31
# end constants

# begin tensor fields
assert x.ty.same_as(tensor_type)
assert tensor_type.dtype == ir.PrimType("float32")
assert x.dtype == "float32"
assert x.scope() == runtime.MemoryScope("global")
assert tensor_type.has_default_layout()

# begin storage properties
storage_type = tirx.TensorType((extent, 128), "float32", align=16, layout="default")
assert storage_type.data_alignment == 16
assert storage_type.layout is not None
# end storage properties
assert storage_type.has_default_layout()
assert storage_type.effective_layout().same_as(storage_type.layout)

implicit = tirx.TensorType((extent, 128), "float32")
inferred_layout = implicit.effective_layout()
assert isinstance(inferred_layout, tirx.Layout)
assert implicit.layout is None
assert implicit.has_default_layout()
dtype_object = tirx.TensorType((8,), tvm_ffi.dtype("float32"))
assert dtype_object.dtype == ir.PrimType("float32")
default_alignment = tirx.TensorType((8,), "float32", align=-1)
assert default_alignment.data_alignment == implicit.data_alignment
# end tensor fields

# begin tensor pointer
storage = ir.Var("storage", ir.PtrType(ir.PrimType("float32"), scope="shared"))
shared_type = tirx.TensorType(8, "float32", data=storage)
assert shared_type.scope == runtime.MemoryScope("shared")
assert len(shared_type.shape) == 1
# end tensor pointer

# begin analyzer
analyzer = sym.Analyzer()
ir.assert_structural_equal(analyzer.simplify((n + 1) - n), ir.const(1, "int64"))
with analyzer.constraint_scope(n >= 1):
    assert analyzer.can_prove(extent > 0)
assert not analyzer.can_prove(n >= 1)
# end analyzer


# begin layouts
index_map = tirx.IndexMap.from_func(lambda i, j: [i * 128 + j])
coordinates = [ir.const(1, "int64"), ir.const(2, "int64")]
mapped = index_map.map_indices(coordinates, analyzer=analyzer)
assert int(analyzer.simplify(mapped[0])) == 130

layout = T.TileLayout(T.S[2 * n, 128])
offset = layout.apply(1, 2)["m"]
assert int(analyzer.simplify(offset)) == 130
# end layouts


# begin expressions in types
def specialize_n(var):
    return ir.const(4, "int64") if var.same_as(n) else var


specialized_type = tvm_ffi.structural_map(tensor_type, (ir.Var, specialize_n))
assert int(sym.Analyzer().simplify(specialized_type.shape[0])) == 8
ir.assert_structural_equal(tensor_type.shape[0], extent)
# end expressions in types

# begin result type contract
source = ir.Var("source", tirx.TensorType((8,), "float32"))
original_pointer = source.data
other = ir.Var("other", tirx.TensorType((8,), "float16", scope="local"))
rewritten_call = ir.Call(original_pointer.op, [other], ty=original_pointer.ty)
rewritten_call.validate()
inferred_type = ir.reinfer_type(rewritten_call)
assert rewritten_call.ty != inferred_type
assert inferred_type.element_type == ir.PrimType("float16")
assert inferred_type.scope == runtime.MemoryScope("local")
assert rewritten_call.ty.element_type == ir.PrimType("float32")
repaired_call = ir.Call(rewritten_call.op, rewritten_call.args, ty=inferred_type)
repaired_call.validate()
ir.assert_structural_equal(repaired_call.ty, other.data.ty)
# end result type contract

# begin identity
another_n = ir.Var("n", "int64")
assert not n.same_as(another_n)
ir.assert_structural_equal(n, another_n, map_free_vars=True)
# end identity


# begin walk
def collect_variables(expr):
    variables = []

    def record(var):
        if not any(var.same_as(previous) for previous in variables):
            variables.append(var)

    tvm_ffi.structural_walk(expr, (ir.Var, record), order="post")
    return variables


variables = collect_variables(extent)
assert len(variables) == 1 and variables[0].same_as(n)
# end walk

# begin map
redundant = ir.prim.Add(n, ir.const(0, "int64"))


def remove_zero(expr):
    if expr.ty == ir.PrimType("int64") and isinstance(expr.b, ir.IntImm) and expr.b.value == 0:
        return expr.a
    return expr


mapped = tvm_ffi.structural_map(redundant, (ir.prim.Add, remove_zero), order="post")
assert mapped.same_as(n)
assert isinstance(redundant, ir.prim.Add)
# end map

# begin operator lookup
op = ir.Op.get("prim.exp")
assert op.name == "prim.exp"
assert [entry.name for entry in op.args_info] == ["x"]
assert op.get_attr("TCallEffectKind") == ir.CallEffectKind.Pure
# end operator lookup

# begin call contract
value = ir.const(1.0, "float32")
call = ir.Call(op, [value])
assert call.op.same_as(op)
assert call.args[0].same_as(value)
assert call.ty == ir.PrimType("float32")
# end call contract

assert isinstance(op, ir.Expr) and isinstance(call, ir.Expr)
assert ir.Call("prim.exp", [value]).op.same_as(op)
assert call.validate() is None
assert ir.reinfer_type(call) == ir.PrimType("float32")

# begin function call type
signature = ir.FuncType([ir.PrimType("float32")], ir.PrimType("float32"))
callee = ir.Var("f", signature)
function_call = ir.Call(callee, [value])
assert function_call.ty == signature.ret_type
# end function call type
assert ir.reinfer_type(function_call) == signature.ret_type

# begin call type states
# begin provisional call
provisional = ir.Call(op, [value], ty=ir.Type.missing())
assert isinstance(provisional.ty, ir.MissingType)
assert ir.reinfer_type(provisional) == call.ty
assert isinstance(provisional.ty, ir.MissingType)
# end provisional call
unknown_call = ir.Call(ir.GlobalVar("without_type"), [])
assert isinstance(unknown_call.ty, ir.MissingType)
assert isinstance(ir.reinfer_type(unknown_call), ir.MissingType)
unchecked_function_call = ir.Call(callee, [])
assert unchecked_function_call.validate() is None
# end call type states

# begin invalid call
# begin validation failure
invalid = ir.Call(op, [], ty="float32")
try:
    invalid.validate()
except TypeError as error:
    assert "Call.args expected 1 argument, got 0" in str(error)
# end validation failure
else:
    raise AssertionError("Expected the operator's validator to reject the call")
# end invalid call

restored = ir.load_json(ir.save_json(x.ty))
ir.assert_structural_equal(x.ty, restored, map_free_vars=True)


# begin error examples
def expect_error(error_type, operation, message):
    try:
        operation()
    except error_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {message}")


expect_error(TypeError, lambda: ir.Var(), "name must be a str")
expect_error(TypeError, lambda: ir.Var(1), "name must be a str")
expect_error(TypeError, lambda: ir.Var("x", 1), "ty must be a Type")
expect_error(ValueError, lambda: ir.Var("x", "invalid_dtype"), "invalid_dtype")
expect_error(
    TypeError,
    lambda: ir.Var("x", name_hint="another_x"),
    "Specify either name or name_hint, not both",
)
expect_error(
    TypeError,
    lambda: ir.const(1, ty="int32", dtype="int64"),
    "const accepts only one of ty and dtype",
)
expect_error(ValueError, lambda: ir.const(1 << 31), "exceeds range of int32")
expect_error(TypeError, lambda: ir.PrimType("handle"), "PtrType")
expect_error(ValueError, lambda: ir.PrimType("invalid_dtype"), "invalid_dtype")
expect_error(TypeError, lambda: ir.const("not a number"), "Cannot infer a constant type")
expect_error(
    TypeError,
    lambda: tirx.TensorType((8,), ir.AnyType()),
    "Tensor dtype must be a primitive type",
)
expect_error(
    ValueError,
    lambda: tirx.TensorType((8,), "float32", align=16, data_alignment=32),
    "align and data_alignment must agree",
)
expect_error(
    TypeError,
    lambda: tirx.TensorType((8,), "float32", layout="row-major"),
    "layout must be a Layout, None, or 'default'",
)
expect_error(
    TypeError,
    lambda: tirx.TensorType((8,), "float32", data=n),
    "Tensor data must be an Expr with PtrType",
)
expect_error(AttributeError, lambda: setattr(x, "shape", (8,)), "read-only")
expect_error(ValueError, lambda: bool(n > 0), "Cannot use and / or / not operator")
expect_error(ValueError, lambda: ir.assert_structural_equal(n, another_n), "StructuralEqual")
expect_error(
    ValueError,
    lambda: shared_type.effective_layout(),
    "An explicit layout is required",
)
# end error examples

print("Foundation API checks passed.")
