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
"""Inspect and reconstruct the CPU/GPU functions from Quick Start.

Run from the repository root with an LLVM-enabled TIRx build:
    uv run --no-sync python docs/ir/examples/add_one.py

The loader reads the original Quick Start definitions without compiling or
running their examples. The checks below execute the CPU function and its IR
API reconstruction at several lengths. GPU checks inspect and round-trip IR
without requiring a GPU. Scalar and vector additions extend the same computation
through the IR API. A sigmoid call illustrates another registered operator.
Further call examples use an expression argument, a scalar helper function,
and a tensor allocation with type arguments and annotations.
"""

import ast
from pathlib import Path
from tempfile import TemporaryDirectory

from tvm_tirx import ir, tirx
from tvm_tirx.script import from_source
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _load_quick_start_function(filename):
    """Load the actual decorated function, stopping before compilation."""
    path = Path(__file__).resolve().parents[2] / "get_started" / "examples" / filename
    source = ast.parse(path.read_text(), filename=str(path))
    definitions = []
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "compiled" for target in node.targets
        ):
            break
        definitions.append(node)
    namespace = {"__name__": "__walkthrough__", "__file__": str(path)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["add_one"]


add_one = _load_quick_start_function("add_one_cpu.py")
gpu_add_one = _load_quick_start_function("add_one_cuda.py")

# begin parameter inspection
A, B = add_one.params
input_type = A.ty
output_type = B.ty
# end parameter inspection
n = input_type.shape[0]

# begin loop inspection
loop = add_one.body[0]
i = loop.var
# end loop inspection

# begin store inspection
store = loop.body[0]
addition = store.value
# end store inspection

# begin source location
parameter = add_one.params[0]
location = parameter.loc
source_name = location.source_name.name
line = location.start_line
# end source location

# begin gpu inspection
device_region = gpu_add_one.body[0]
index_binding = device_region.body[0]
branch = device_region.body[1]
index_expression = index_binding.value
block_call = index_expression.a.a
thread_call = index_expression.b
# end gpu inspection

# begin scalar addition
scalar_value = ir.Var("scalar_value", "float32")
scalar_plus_one = scalar_value + 1
# end scalar addition

# begin vector extension
vector_value = ir.Var("vector_value", "float32x4")
vector_plus_one = vector_value + 1
# end vector extension

# begin sigmoid call
value = ir.Var("value", "float32")
sigmoid_op = ir.Op.get("prim.sigmoid")
activated_value = ir.Call(sigmoid_op, [value])
# end sigmoid call

# begin expression argument
activated_plus_one = ir.prim.sigmoid(scalar_plus_one)
# end expression argument


# begin scalar helper
@T.function(private=True)
def add_one_scalar(value: T.float32) -> T.float32:
    return value + 1
    # end scalar helper


# begin function call
call_module = ir.IRModule({"add_one_scalar": add_one_scalar})
callee = call_module.get_global_var("add_one_scalar")
function_call = ir.Call(callee, [scalar_value], ty=add_one_scalar.ret_type)
# end function call

# begin allocation call
temporary_type = tirx.TensorType((4,), "float32", scope="local")
allocation_call = ir.Call(
    "tirx.alloc_tensor", [], ty_args=[temporary_type], attrs={"alignment": 16}
)
# end allocation call

# begin allocation binding
temporary = ir.Var("temporary", temporary_type)
allocation_binding = ir.Bind(temporary, allocation_call)
allocation_body = ir.SeqStmt([allocation_binding])
# end allocation binding


def make_add_one():
    """Construct the CPU computation directly through the IR API."""
    # begin variables
    n = ir.Var("n", "int32")
    tensor_type = tirx.TensorType((n,), "float32")
    A = ir.Var("A", tensor_type)
    B = ir.Var("B", tensor_type)
    i = ir.Var("i", "int32")
    # end variables

    # begin cpu body
    store = ir.TensorStore(B, [i], A[i] + 1)
    body = ir.SeqStmt([store])
    # end cpu body

    # begin cpu loop
    loop = ir.For(i, 0, n, 1, ir.ForKind.DEFAULT, body)
    # end cpu loop

    # begin function construction
    ir_function = tirx.Function([A, B], loop, ret_type=ir.TupleType([]))
    ir_function = ir_function.with_attr("global_symbol", "add_one")
    # end function construction
    return ir_function


ir_function = make_add_one()

# begin operator lookup
op = ir.Op.get("tirx.cuda.thread_idx")
argument_names = [argument.name for argument in op.args_info]
effect = op.get_attr("TCallEffectKind")
# end operator lookup

# begin call construction
x_call = ir.Call(op, [ir.StringImm("x")])
# end call construction

# begin shared operator
y_call = ir.Call(op, [ir.StringImm("y")])
# end shared operator

# begin inference choices
inferred_call = ir.Call(op, x_call.args)
typed_call = ir.Call(op, x_call.args, ty="int32")
provisional = ir.Call(op, x_call.args, ty=ir.Type.missing())
# end inference choices

# begin validation failure
invalid = ir.Call(op, [], ty="int32")
# end validation failure
try:
    invalid.validate()
except TypeError:
    pass
else:
    raise AssertionError("The thread-index operator requires an axis argument")

# begin provisional sigmoid
provisional_sigmoid = ir.Call(sigmoid_op, [vector_value], ty=ir.Type.missing())
# end provisional sigmoid

# begin provisional call
inferred = ir.reinfer_type(provisional_sigmoid)
rebuilt_call = ir.Call(sigmoid_op, provisional_sigmoid.args, ty=inferred)
rebuilt_call.validate()
# end provisional call

# begin module construction
module = ir.IRModule({"add_one": add_one})
# end module construction

# begin module lookup
entry = module.get_global_var("add_one")
function = module[entry]
# end module lookup

# begin module update
variant = module.clone()
variant[entry] = ir_function
# end module update

# begin printing and parsing
source = add_one.script()
restored_function = from_source(source, extra_vars={"T": T})
ir.assert_structural_equal(add_one, restored_function)
# end printing and parsing


def save_program(program, output_path):
    source = program.script(extra_config={"ir.comment_imports": False})
    output_path.write_text(source)
    restored = from_source(output_path.read_text())
    ir.assert_structural_equal(program, restored)
    return restored


def run():
    """Check source correspondence, IR construction, and CPU execution."""
    import numpy as np

    from tvm_tirx import runtime

    assert n.ty == ir.PrimType("int32")
    assert i.ty == ir.PrimType("int32")
    assert addition.ty == ir.PrimType("float32")
    assert isinstance(addition, ir.prim.Add)
    assert isinstance(addition.a, ir.TensorLoad)
    assert isinstance(addition.b, ir.FloatImm)
    assert isinstance(device_region, ir.RegionStmt)
    assert isinstance(index_binding, ir.Bind)
    assert isinstance(branch, ir.If)
    assert branch.condition.ty == ir.PrimType("bool")
    assert block_call.op.name == "tirx.cuda.block_idx"
    assert thread_call.op.name == "tirx.cuda.thread_idx"
    assert argument_names == ["axis"]
    assert effect == ir.CallEffectKind.Pure
    assert x_call.op.same_as(op)
    assert y_call.op.same_as(op)
    assert x_call.args[0].value == "x"
    assert y_call.args[0].value == "y"
    assert y_call.ty == ir.PrimType("int32")
    y_call.validate()
    assert scalar_plus_one.ty == ir.PrimType("float32")
    assert vector_plus_one.ty == ir.PrimType("float32x4")
    assert activated_value.ty == ir.PrimType("float32")
    activated_value.validate()
    ir.assert_structural_equal(activated_value, ir.prim.sigmoid(value))
    assert activated_plus_one.ty == ir.PrimType("float32")
    activated_plus_one.validate()
    ir.assert_structural_equal(activated_plus_one.args[0], scalar_plus_one)
    assert function_call.op.same_as(callee)
    assert function_call.args[0].same_as(scalar_value)
    assert function_call.ty == ir.PrimType("float32")
    assert allocation_call.ty == temporary_type
    assert allocation_call.ty_args[0].same_as(temporary_type)
    assert isinstance(allocation_call.attrs, ir.DictAttrs)
    assert allocation_call.attrs["alignment"] == 16
    allocation_call.validate()
    assert allocation_binding.value.same_as(allocation_call)
    assert allocation_binding.var.same_as(temporary)
    assert allocation_body[0].same_as(allocation_binding)
    assert inferred == ir.PrimType("float32x4")
    assert isinstance(provisional_sigmoid.ty, ir.MissingType)
    assert rebuilt_call.ty == ir.PrimType("float32x4")
    ir.assert_structural_equal(rebuilt_call, ir.prim.sigmoid(vector_value))
    assert isinstance(provisional.ty, ir.MissingType)
    assert Path(source_name).name == "add_one_cpu.py"
    assert "def add_one(A:" in Path(source_name).read_text().splitlines()[line - 1]
    ir.assert_structural_equal(x_call, thread_call)
    ir.assert_structural_equal(add_one, ir_function)
    ir.assert_structural_equal(module[entry], add_one)
    ir.assert_structural_equal(variant[entry], ir_function)
    for item in (add_one, gpu_add_one, ir_function, module, variant, call_module):
        assert tirx.analysis.check_well_formed(item)
    function_result = ir.Var("result", tirx.TensorType((1,), "float32"))
    function_call_caller = tirx.Function(
        [scalar_value, function_result],
        ir.TensorStore(function_result, [0], function_call),
        ret_type=ir.TupleType([]),
    ).with_attr("global_symbol", "call_add_one_scalar")
    function_call_module = call_module.clone()
    function_call_module["call_add_one_scalar"] = function_call_caller
    assert tirx.analysis.check_well_formed(function_call_module)
    restored_calls = from_source(function_call_module.script(), extra_vars={"T": T, "I": I})
    ir.assert_structural_equal(function_call_module, restored_calls)
    restored_gpu = from_source(gpu_add_one.script(), extra_vars={"T": T})
    ir.assert_structural_equal(gpu_add_one, restored_gpu)

    with TemporaryDirectory() as directory:
        cached = save_program(variant, Path(directory) / "add_one.py")
        original = tirx.compile(module, target="llvm")
        rebuilt = tirx.compile(cached, target="llvm")
        for length in (1, 10, 257):
            values = np.arange(length, dtype="float32")
            inputs = runtime.tensor(values)
            for compiled in (original, rebuilt):
                output = runtime.empty(inputs.shape, inputs.dtype)
                compiled["add_one"](inputs, output)
                np.testing.assert_allclose(output.numpy(), values + 1)
            print(f"add_one: length {length}, original and reconstructed functions match NumPy")

        compiled_calls = tirx.compile(function_call_module, target="c")
        library_path = str(Path(directory) / "function_calls.so")
        compiled_calls.export_library(library_path)
        loaded_calls = runtime.load_module(library_path)
        output = runtime.empty((1,), "float32")
        for input_value in (-8.0, -2.0, 0.0, 2.5, 8.0):
            loaded_calls["call_add_one_scalar"](input_value, output)
            np.testing.assert_allclose(output.numpy(), [np.float32(input_value) + 1])
        print("function call: the GlobalVar call compiles with C and matches NumPy")

    activation_output = ir.Var("output", tirx.TensorType((1,), "float32"))
    sigmoid_function = tirx.Function(
        [value, activation_output],
        ir.TensorStore(activation_output, [0], activated_value),
        ret_type=ir.TupleType([]),
    ).with_attr("global_symbol", "sigmoid")
    compiled_sigmoid = tirx.compile(sigmoid_function, target="llvm")
    output = runtime.empty((1,), "float32")
    for input_value in (-8.0, -2.0, -1.0, 0.0, 1.0, 2.0, 8.0):
        compiled_sigmoid(input_value, output)
        expected = 1 / (1 + np.exp(-np.float32(input_value)))
        np.testing.assert_allclose(output.numpy()[0], expected, rtol=1e-6, atol=1e-7)
    print("sigmoid: the documented call compiles and matches NumPy")

    composed_function = tirx.Function(
        [scalar_value, activation_output],
        ir.TensorStore(activation_output, [0], activated_plus_one),
        ret_type=ir.TupleType([]),
    ).with_attr("global_symbol", "sigmoid_after_add")
    allocation_function = tirx.Function(
        [scalar_value, activation_output],
        ir.SeqStmt(
            [
                allocation_binding,
                ir.TensorStore(temporary, [0], scalar_plus_one),
                ir.TensorStore(activation_output, [0], temporary[0]),
            ]
        ),
        ret_type=ir.TupleType([]),
    ).with_attr("global_symbol", "temporary_add_one")
    for name, function in (
        ("sigmoid_after_add", composed_function),
        ("temporary_add_one", allocation_function),
    ):
        assert tirx.analysis.check_well_formed(function)
        restored = from_source(function.script(), extra_vars={"T": T})
        ir.assert_structural_equal(function, restored)
        compiled = tirx.compile(function, target="llvm")
        for input_value in (-8.0, -2.0, 0.0, 2.5, 8.0):
            compiled(input_value, output)
            expected = np.float32(input_value) + 1
            if name == "sigmoid_after_add":
                expected = 1 / (1 + np.exp(-expected))
            np.testing.assert_allclose(output.numpy()[0], expected, rtol=1e-6, atol=1e-7)
        print(f"{name}: the documented call compiles and matches NumPy")

    print("CPU/GPU IR inspection, supplemental expressions, and round trips passed.")


if __name__ == "__main__":
    run()
