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
"""Describe an integer addition and a tensor using the shared type system."""

from tvm_tirx import ir, tirx

# begin composite type
integer_type = ir.PrimType("int32")
signature = ir.FuncType([integer_type], integer_type)
# end composite type

# begin constant tensor type
tile_type = tirx.TensorType((8, 128), "float32")
# end constant tensor type

# begin scalar expressions
x = ir.Var("x", integer_type)
added = x + ir.const(1, integer_type)
# end scalar expressions

# begin tensor expressions
n = ir.Var("n", "int64")
tensor_type = tirx.TensorType((2 * n, 128), "float32")
tensor = ir.Var("A", tensor_type)
element = tensor[0, 0]
# end tensor expressions

assert signature.arg_types[0] == signature.ret_type == added.ty == x.ty
assert tile_type.dtype == element.ty == ir.PrimType("float32")
assert tensor.ty.shape[0].b.same_as(n)
print("Type and expression examples passed.")
