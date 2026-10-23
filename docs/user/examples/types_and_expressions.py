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
"""Construct types, scalar expressions, and a symbolic tensor variable."""

from tvm_tirx import ir, tirx

# begin composite type
integer_type = ir.PrimType("int64")
signature = ir.FuncType([integer_type], integer_type)
# end composite type

# begin constant tensor type
tile_type = tirx.TensorType((8, 128), "float32")
# end constant tensor type

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

assert signature.arg_types[0] == signature.ret_type == integer_type
assert tile_type.dtype == ir.PrimType("float32")
assert extent.ty == n.ty == integer_type
assert x.ty.shape[0].same_as(extent)
assert element.ty == ir.PrimType("float32")
print("Type and expression examples passed.")
