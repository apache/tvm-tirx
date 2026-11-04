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
"""Result types of the CUDA and NVSHMEM intrinsics.

A builder emits a Call as a statement only when its type is void, so a statement
intrinsic registered without a result type is silently dropped when Python code
calls it under an IRBuilder without the parser.
"""

import pytest
from tvm_ffi import structural_equal, structural_walk

from tvm_tirx import ir
from tvm_tirx.ir.op import Op
from tvm_tirx.ir.type import PrimType
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import AlreadyEmitted, IRBuilder
from tvm_tirx.tirx.script import ir_builder as I

STATEMENT_INTRINSICS = [
    "tirx.cuda.wait_until",
    "tirx.cuda.timer_init",
    "tirx.cuda.timer_start",
    "tirx.cuda.timer_end",
    "tirx.cuda.timer_finalize",
    "tirx.nvshmem.getmem_nbi_block",
]

# These constructors require the caller to pass the result type.
CALLER_TYPED = {"tirx.cuda.mma_fill", "tirx.cuda.mma_store"}


@pytest.mark.parametrize("name", STATEMENT_INTRINSICS)
def test_statement_intrinsic_registers_void(name):
    fixed = Op.get_attr_map("TFixedReturnType")
    op = Op.get(name)
    assert op in fixed and structural_equal(fixed[op], PrimType("void"))


def test_every_intrinsic_has_a_result_type():
    fixed = Op.get_attr_map("TFixedReturnType")
    infer = Op.get_attr_map("FInferType")
    untyped = {
        name
        for name in Op.list_op_names()
        if name.startswith(("tirx.cuda.", "tirx.nvshmem."))
        and Op.get(name) not in fixed
        and Op.get(name) not in infer
    }
    assert untyped == CALLER_TYPED


def test_statement_intrinsics_emit_under_a_builder():
    with IRBuilder() as builder:
        with I.function_():
            flag = I.arg_("flag", T.Tensor((1,), "int32"))
            spin = T.bind_(T.alloc_local((1,), "int32"))[0]
            ptr = flag.ptr_to([0])
            receipts = [
                T.cuda.wait_until(spin, ptr, spin >= 1, scope="gpu", ptx_type="b32"),
                T.cuda.timer_init(ptr, 0, 0, 1, 0),
                T.nvshmem.getmem_nbi.block(ptr, ptr, 1, 0),
            ]
    assert all(isinstance(receipt, AlreadyEmitted) for receipt in receipts)
    names = []
    structural_walk(builder.get(), (ir.Call, lambda call: names.append(call.op.name)))
    for name in ("tirx.cuda.wait_until", "tirx.cuda.timer_init", "tirx.nvshmem.getmem_nbi_block"):
        assert names.count(name) == 1
