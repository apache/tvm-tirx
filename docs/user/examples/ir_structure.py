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

import numpy as np

from tvm_tirx import ir, runtime, sym, tirx
from tvm_tirx.script import tirx as T

# begin composite expressions
integer = ir.const(7, "int32")
pair = ir.Tuple([integer, ir.StringImm("tile")])
pair_value = ir.Var("pair", pair.ty)
first = ir.TupleGetItem(pair_value, 0)
assert pair.ty == ir.TupleType([ir.PrimType("int32"), ir.StringType()])
assert first.ty == ir.PrimType("int32")

ramp = ir.prim.Ramp(ir.const(0, "int32"), ir.const(1, "int32"), ir.const(4, "int32"))
broadcast = ir.prim.Broadcast(integer, ir.const(4, "int32"))
assert ramp.ty == ir.PrimType("int32x4")
assert broadcast.ty == ramp.ty
# end composite expressions

# begin staging expression
increment = ir.LambdaExpr(["int32"], lambda value: value + ir.const(1, "int32"))
applied = increment.apply([ir.const(3, "int32")])
assert increment.ty == ir.FuncType([ir.PrimType("int32")], ir.PrimType("int32"))
assert int(sym.Analyzer().simplify(applied)) == 4
# end staging expression

# begin statement sequence
temporary = ir.Var("temporary", "int32")
definition = ir.Bind(temporary, ir.const(7, "int32"))
use = ir.Evaluate(temporary)
sequence = ir.SeqStmt([definition, ir.SeqStmt([use])])
assert isinstance(sequence, ir.Stmt)
assert len(sequence) == 2
assert sequence[0].same_as(definition)
assert sequence[1].same_as(use)
# end statement sequence

assert sequence[0].value.ty == temporary.ty
assert ir.stmt_seq(definition, temporary)[1].value.same_as(temporary)
assert len(ir.SeqStmt([])) == 0
assert len(ir.SeqStmt(definition)) == 1

# begin invalid binding type
try:
    ir.Bind(temporary, ir.const(1, "int64"))
except RuntimeError:
    pass
else:
    raise AssertionError("Bind must reject a value whose type differs from the variable's type")
# end invalid binding type

# begin statement values
n = ir.Var("n", "int64")
A = ir.Var("A", tirx.TensorType((2 * n, 128), "int32"))
C = ir.Var("C", A.ty)
i = ir.Var("i", "int64")
j = ir.Var("j", "int64")
value = ir.Var("value", "int32")
zero = ir.const(0, "int64")
one = ir.const(1, "int64")
# end statement values

# begin statements
binding = ir.Bind(value, A[i, j] + ir.const(1, "int32"))
branch = ir.If(
    i < n,
    ir.TensorStore(C, [i, j], value),
    ir.TensorStore(C, [i, j], value + ir.const(2, "int32")),
)
inner = ir.For(j, zero, ir.const(128, "int64"), one, ir.ForKind.DEFAULT, [binding, branch])
body = ir.For(i, zero, 2 * n, one, ir.ForKind.DEFAULT, inner)
# end statements

# begin statement inspection
assert isinstance(body.body, ir.SeqStmt)
assert body.body[0].same_as(inner)
assert inner.body[0].same_as(binding)
assert branch.condition.ty == ir.PrimType("bool")
assert binding.var.same_as(value)
assert branch.then_case[0].value.same_as(value)
# end statement inspection

# begin statement region
region_local = ir.Var("region_local", "int32")
region = ir.RegionStmt(
    "tirx.compute_scope",
    [ir.StringImm("prepare")],
    [],
    None,
    ir.Bind(region_local, ir.const(7, "int32")),
)
assert region.op.name == "tirx.compute_scope"
assert not region.body_params and not region.result_vars
assert region.body[0].var.same_as(region_local)
# end statement region

# begin default return type
empty_function = tirx.Function([], [])
assert empty_function.ret_type == ir.TupleType([])
assert empty_function.ty == ir.FuncType([], ir.TupleType([]))
# end default return type

# begin function
# begin function construction
function = tirx.Function([n, A, C], body, ret_type=ir.PrimType("void"))
# end function construction
assert isinstance(function.ty, ir.FuncType)
assert function.ty.arg_types[0] == ir.PrimType("int64")
assert function.ty.arg_types[1] == A.ty
assert function.ty.ret_type == ir.PrimType("void")
assert isinstance(function.body, ir.SeqStmt)
# end function

# begin declaration
declaration = tirx.Function([n, A, C], None, ret_type=ir.PrimType("void"))
assert declaration.body is None
assert declaration.ty == function.ty
# end declaration

# begin module
# begin module construction
function = function.with_attr("global_symbol", "main")
entry = ir.GlobalVar("main")
module = ir.IRModule({entry: function}, attrs={"example": "ir-structure"})
assert module[entry].same_as(function)
assert module["main"].same_as(function)
# end module construction
assert module.attrs["example"] == "ir-structure"
assert tirx.analysis.check_well_formed(module)
# end module

# begin module names
named_module = ir.IRModule({"entry": function})
assert named_module.get_global_var("entry").name_hint == "entry"
assert named_module["entry"].attrs["global_symbol"] == "main"
# end module names

# begin global reference identity
stored = module.get_global_var("main")
fresh = ir.GlobalVar("main")
assert stored.same_as(entry)
assert not fresh.same_as(stored)
assert "main" in module and fresh not in module
# end global reference identity

assert isinstance(stored.ty, ir.MissingType)
try:
    module[fresh]
except RuntimeError:
    pass
else:
    raise AssertionError("A fresh GlobalVar cannot resolve an existing same-name definition")

# begin function attributes
annotated = function.with_attr("example.note", "constructed with IR objects")
assert "example.note" not in function.attrs
assert annotated.attrs["example.note"] == "constructed with IR objects"
# end function attributes

# begin function body update
precondition = ir.AssertStmt(
    n > ir.const(0, "int64"),
    ir.StringImm("ValueError"),
    [ir.StringImm("n must be positive")],
)
checked = function.with_body([precondition, function.body], loc=function.loc)
assert checked.body[0].same_as(precondition)
assert checked.params[0].same_as(n)
assert checked.ty == function.ty
assert len(checked.body) == 2 and len(function.body) == 1
# end function body update

assert tirx.analysis.check_well_formed(checked)

# begin module clone
variant = module.clone()
assert not variant.same_as(module)
assert variant["main"].same_as(function)

alias = variant
variant["main"] = checked
assert alias["main"].same_as(checked)
assert variant.get_global_var("main").same_as(entry)
assert module["main"].same_as(function)
# end module clone

# begin module update
variant.update({variant.get_global_var("main"): annotated})
assert variant["main"].same_as(annotated)
assert module["main"].same_as(function)
# end module update

# Exercise a conflicting update only on a disposable module.
collision = ir.IRModule({"main": function})
try:
    collision.update({"main": annotated})
except RuntimeError:
    pass
else:
    raise AssertionError("An update cannot introduce a different GlobalVar with the same name")
del variant["main"]
assert "main" not in variant and "main" in module
assert tirx.Function([], [], ret_type=ir.Type.missing()).ret_type == ir.TupleType([])


# begin script program
@T.function
def scripted(
    n: T.int64,
    A: T.Tensor((2 * n, 128), "int32"),
    C: T.Tensor((2 * n, 128), "int32"),
) -> T.void:
    for i, j in T.grid(2 * n, T.int64(128)):
        value = A[i, j] + 1
        if i < n:
            C[i, j] = value
        else:
            C[i, j] = value + 2


# end script program

# begin construction comparison
scripted_main = scripted.with_attr("global_symbol", "main")
ir.assert_structural_equal(function, scripted_main)
# end construction comparison

# begin metadata
location = ir.SourceLoc(ir.SourceName("ir_structure.py"), 1, 0, 1, 1)
located_function = function.with_body(function.body, loc=location)
assert located_function.loc.same_as(location)
# end metadata

assert isinstance(located_function.with_body(function.body).loc, ir.UnknownLoc)

# begin serialization
# begin module serialization
restored = ir.load_json(ir.save_json(module))
ir.assert_structural_equal(module, restored)
# end module serialization
assert isinstance(restored["main"], tirx.Function)
assert not restored.same_as(module)
# end serialization

# begin invalid binding
duplicate_definition = tirx.Function(
    [n], ir.Bind(n, ir.const(1, "int64")), ret_type=ir.PrimType("void")
)
assert not tirx.analysis.check_well_formed(duplicate_definition, assert_mode=False)
try:
    tirx.analysis.check_well_formed(duplicate_definition)
except RuntimeError:
    pass
else:
    raise AssertionError("A parameter cannot be defined again by Bind")
# end invalid binding

# begin escaped branch binding
predicate = ir.Var("predicate", "bool")
branch_value = ir.Var("branch_value", "int32")
conditional = ir.If(predicate, ir.Bind(branch_value, ir.const(1, "int32")), None)
branch_only = tirx.Function([predicate], conditional, ret_type=ir.PrimType("void"))
assert tirx.analysis.check_well_formed(branch_only)

escaped_branch = tirx.Function(
    [predicate], [conditional, ir.Evaluate(branch_value)], ret_type=ir.PrimType("void")
)
assert not tirx.analysis.check_well_formed(escaped_branch, assert_mode=False)
# end escaped branch binding

escaped_loop = tirx.Function([n, A, C], [body, ir.Evaluate(i)], ret_type=ir.PrimType("void"))
assert not tirx.analysis.check_well_formed(escaped_loop, assert_mode=False)

# begin invalid loop control
invalid_break = tirx.Function([], ir.Break(), ret_type=ir.PrimType("void"))
assert not tirx.analysis.check_well_formed(invalid_break, assert_mode=False)
# end invalid loop control

invalid_continue = tirx.Function([], ir.Continue(), ret_type=ir.PrimType("void"))
assert not tirx.analysis.check_well_formed(invalid_continue, assert_mode=False)
assert tirx.analysis.check_well_formed(tirx.Function([], region, ret_type=ir.PrimType("void")))
escaped_region = tirx.Function(
    [], [region, ir.Evaluate(region_local)], ret_type=ir.PrimType("void")
)
assert not tirx.analysis.check_well_formed(escaped_region, assert_mode=False)

try:
    ir.If(ir.const(True), sequence)
except TypeError as error:
    assert "else_case" in str(error)
else:
    raise AssertionError("The Python If constructor requires an explicit else_case")

try:
    ir.RegionStmt("prim.exp", [ir.const(1.0)], [], None, [])
except ValueError as error:
    assert "FRegionGetBodyParams" in str(error)
else:
    raise AssertionError("A region operator must declare its body-parameter contract")

print(function.script())

# begin execution
if not runtime.enabled("llvm"):
    raise RuntimeError("This example requires a TIRx build with LLVM enabled")

executable = tirx.compile(module, target="llvm")
for n_value in (1, 3):
    shape = (2 * n_value, 128)
    source = np.arange(np.prod(shape), dtype="int32").reshape(shape)
    output = runtime.empty(shape, "int32")
    executable(n_value, runtime.tensor(source), output)
    expected = source + 1
    expected[n_value:] += 2
    np.testing.assert_array_equal(output.numpy(), expected)
# end execution

print("IR structure: type, scope, serialization, and numerical checks passed.")
