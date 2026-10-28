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
# under the License."""Inspect and transform the add_one program."""

import tvm_ffi

import tvm_tirx
from tvm_tirx.script import tirx as T

n = T.dynamic("n")


@T.function
def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
    for i in range(n):
        B[i] = A[i] + 1


# fmt: off
# [inspect.begin]
A, B = add_one.params       # parameters are Vars
n = A.ty.shape[0]           # the symbolic length n, also a Var
loop = add_one.body[0]      # for i in range(n)
print(loop.body[0].value)   # A[i] + T.float32(1.0)
# [inspect.end]
# fmt: on

# [transform.begin]
add_one_specialized = tvm_ffi.structural_map(
    add_one,
    (tvm_tirx.ir.Var, lambda v: T.int32(128) if v.same_as(n) else v),
)
add_one_specialized.show()
# [transform.end]
