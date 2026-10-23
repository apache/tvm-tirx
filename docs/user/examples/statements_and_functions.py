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
"""Build, serialize, and execute a function with a branch and nested loops."""

import numpy as np

from tvm_tirx import ir, runtime, tirx

# begin statement values
n = ir.Var("n", "int64")
A = ir.Var("A", tirx.TensorType((2 * n, 128), "int32"))
C = ir.Var("C", A.ty)
i = ir.Var("i", "int64")
j = ir.Var("j", "int64")
value = ir.Var("value", "int32")
# end statement values

# begin statements
binding = ir.Bind(value, A[i, j] + 1)
branch = ir.If(
    i < n,
    ir.TensorStore(C, [i, j], value),
    ir.TensorStore(C, [i, j], value + 2),
)
inner = ir.For(j, 0, 128, 1, ir.ForKind.DEFAULT, [binding, branch])
body = ir.For(i, 0, 2 * n, 1, ir.ForKind.DEFAULT, inner)
# end statements

# begin function construction
function = tirx.Function([n, A, C], body, ret_type=ir.PrimType("void"))
# end function construction

# begin module construction
function = function.with_attr("global_symbol", "main")
entry = ir.GlobalVar("main")
module = ir.IRModule({entry: function})
# end module construction

# begin module serialization
restored = ir.load_json(ir.save_json(module))
ir.assert_structural_equal(module, restored)
# end module serialization

assert module[entry].same_as(module["main"])
assert tirx.analysis.check_well_formed(module)
variant = module.clone()
variant["main"] = function.with_attr("example.note", "clone")
assert "example.note" not in module["main"].attrs

# A loop variable is unavailable after its loop.
escaped = function.with_body([body, ir.Evaluate(i)])
assert not tirx.analysis.check_well_formed(escaped, assert_mode=False)

executable = tirx.compile(module, target="llvm")
for n_value in (1, 3):
    source = np.arange(2 * n_value * 128, dtype="int32").reshape(2 * n_value, 128)
    output = runtime.empty(source.shape, "int32")
    executable(n_value, runtime.tensor(source), output)
    expected = source + 1
    expected[n_value:] += 2
    np.testing.assert_array_equal(output.numpy(), expected)

print("Statement, function, and numerical examples passed.")
