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
"""Express a symbolic tensor program in TVMScript and check a source round trip."""

# TVMScript resolves dependent parameter annotations when constructing IR.

# begin tvmscript example
from __future__ import annotations

from tvm_tirx import ir, tirx
from tvm_tirx.script import from_source
from tvm_tirx.script import tirx as T


@T.function
def f(n: T.int64, A: T.Tensor((2 * n, 128), "float32")) -> T.void:
    for i, j in T.grid(2 * n, T.int64(128)):
        A[i, j] = A[i, j] + T.float32(1)


# end tvmscript example

# begin printing and parsing
source = f.script()
print(source)
restored = from_source(source, extra_vars={"T": T})
ir.assert_structural_equal(f, restored)
# end printing and parsing

assert isinstance(f, tirx.Function)
n, tensor = f.params
assert tensor.ty.shape[0].b.same_as(n)
assert f.body[0].end.b.same_as(n)
assert tirx.analysis.check_well_formed(f)
print("Printing and parsing example passed.")
