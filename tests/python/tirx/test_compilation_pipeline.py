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

"""Compilation through the native TIRx pipeline."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize("pipeline", [None, "default", "tirx"])
def test_default_pipeline_allocations(pipeline):
    @T.function
    def add_one(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        temp = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(16):
            temp[i] = A[i] + T.float32(1)
        for i in range(16):
            B[i] = temp[i]

    module = tvm_tirx.tirx.compile(add_one, target="llvm", pipeline=pipeline)
    source = np.arange(16, dtype="float32")
    a = tvm_tirx.runtime.tensor(source)
    b = tvm_tirx.runtime.tensor(np.zeros(16, dtype="float32"))
    module(a, b)
    np.testing.assert_equal(b.numpy(), source + 1)


@pytest.mark.parametrize("print_after_dispatch", [None, "", "1"])
def test_explicit_tile_pass_order_and_diagnostic(monkeypatch, capfd, print_after_dispatch):
    from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

    if print_after_dispatch is None:
        monkeypatch.delenv("TVM_TIRX_PRINT_AFTER_DISPATCH_OPS", raising=False)
    else:
        monkeypatch.setenv("TVM_TIRX_PRINT_AFTER_DISPATCH_OPS", print_after_dispatch)

    @T.function
    def kernel(A: T.Tensor((2, 4), "float32", layout="default")):
        for i, j in T.grid(2, 4):
            A[i, j] = T.float32(1)

    visited = []
    layouts = {}

    @tvm_tirx.ir.transform.pass_instrument
    class RecordPasses:
        def run_after_pass(self, mod, info):
            visited.append(info.name)
            if info.name in ("tirx.TileDispatch", "tirx.LowerTensorLayouts"):
                layouts[info.name] = mod.script()

    pipeline, _, _ = tirx_pipeline(prepare_only=True, target=tvm_tirx.target.Target("llvm"))
    with tvm_tirx.ir.transform.PassContext(instruments=[RecordPasses()]):
        pipeline(tvm_tirx.IRModule({"main": kernel}))

    start = visited.index("tirx.TileDispatch")
    expected = ["tirx.TileDispatch"]
    if print_after_dispatch is not None:
        expected.append("PrintIR")
    expected.extend(["tirx.LowerTensorLayouts", "tirx.StmtSimplify", "tirx.FlattenTensor"])
    assert visited[start : start + len(expected)] == expected
    assert "TileLayout" in layouts["tirx.TileDispatch"]
    assert "TileLayout" not in layouts["tirx.LowerTensorLayouts"]
    output = capfd.readouterr().err
    assert ("PrintIR(" in output) == (print_after_dispatch is not None)
    if print_after_dispatch is not None:
        assert "TileLayout" in output


def _prepare_for_cuda(func, arch="sm_80"):
    from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    pipeline, _, _ = tirx_pipeline(prepare_only=True, target=target)
    return pipeline(tvm_tirx.IRModule({"main": func}))


def test_verify_memory_rejects_host_store_to_gpu_tensor():
    @T.function
    def kernel(A: T.Tensor((1,), "float32")):
        A[0] = T.float32(1)

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(kernel)


def test_verify_memory_rejects_host_store_to_gpu_tensor_view():
    @T.function
    def kernel(A: T.Tensor((1, 1), "float32")):
        A_view = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), "float32")])
        A_view[0] = T.float32(1)

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(kernel)


def test_verify_memory_accepts_host_address_of_gpu_tensor():
    @T.function
    def kernel(A: T.Tensor((16,), "float32"), index: T.int32):
        T.call_extern("consume_pointer", T.address_of(A[index]), ty="void")
        T.call_extern("consume_pointer", T.address_of(index), ty="void")

    _prepare_for_cuda(kernel)


def test_verify_memory_accepts_host_address_of_gpu_tensor_views():
    @T.function
    def kernel(A: T.Tensor((4, 4), "float32"), index: T.int32):
        A_decl = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32")])
        A_view = A_decl.view(16)
        T.call_extern("consume_pointer", T.address_of(A_view[index]), ty="void")

    _prepare_for_cuda(kernel)


@pytest.mark.parametrize("use_view", [False, True])
def test_verify_memory_rejects_host_load_from_gpu_tensor(use_view):
    @T.function
    def direct(A: T.Tensor((16,), "float32")):
        T.call_extern("consume_value", A[0], ty="void")

    @T.function
    def view(A: T.Tensor((4, 4), "float32")):
        A_decl = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32")])
        A_view = A_decl.view(16)
        T.call_extern("consume_value", A_view[0], ty="void")

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(view if use_view else direct)


def test_verify_memory_rejects_host_store_through_reshaped_gpu_tensor():
    @T.function
    def kernel(A: T.Tensor((4, 4), "float32")):
        A_view = A.view(16)
        A_view[0] = T.float32(1)

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(kernel)


def test_verify_memory_rejects_host_load_after_address_of():
    @T.function
    def kernel(A: T.Tensor((16,), "float32")):
        T.call_extern("consume_pointer", T.address_of(A[0]), ty="void")
        T.call_extern("consume_value", A[1], ty="void")

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(kernel)


@pytest.mark.parametrize("masked", [False, True])
def test_verify_memory_rejects_gpu_load_in_address_index(masked):
    @T.function
    def direct(A: T.Tensor((16,), "float32"), B: T.Tensor((1,), "int32")):
        B_decl = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), "int32")])
        B_view = B_decl.view(1)
        T.call_extern("consume_pointer", T.address_of(A[B_view[0]]), ty="void")

    @T.function
    def masked_index(A: T.Tensor((16,), "float32"), B: T.Tensor((1,), "int32")):
        B_decl = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), "int32")])
        B_view = B_decl.view(1)
        T.call_extern(
            "consume_pointer", T.address_of(A[T.masked_load(B_view, 0, T.bool(True))]), ty="void"
        )

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope") as error:
        _prepare_for_cuda(masked_index if masked else direct)
    diagnostic = str(error.value).split("Did you forget to bind?")[0]
    # Lowering can rename B's views; only the index load should be rejected.
    assert diagnostic.count("refers to GPU memory") == 1
    assert "Variable `A`" not in diagnostic


@pytest.mark.parametrize("store", [False, True])
def test_verify_memory_rejects_host_masked_gpu_access(store):
    @T.function
    def load(A: T.Tensor((4, 4), "float32")):
        A_decl = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32")])
        A_view = A_decl.view(16)
        T.call_extern("consume_value", T.masked_load(A_view, 0, T.bool(True)), ty="void")

    @T.function
    def write(A: T.Tensor((4, 4), "float32")):
        A_decl = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32")])
        A_view = A_decl.view(16)
        T.masked_store(A_view, T.float32(1), 0, T.bool(True))

    with pytest.raises(RuntimeError, match="accessed outside a GPU kernel scope"):
        _prepare_for_cuda(write if store else load)


def test_verify_memory_checks_other_tensormap_arguments():
    @T.function
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((1,), "int64")):
        descriptor = T.stack_alloca("tensormap", 1)
        T.cuda.tensormap_encode_tiled(
            descriptor, T.address_of(A[0]), B[0], 32, 1, descriptor_dtype="float32", rank=1
        )

    with pytest.raises(RuntimeError, match=r"Variable `B`.*GPU memory") as error:
        _prepare_for_cuda(kernel)
    diagnostic = str(error.value).split("Did you forget to bind?")[0]
    assert "Variable `A`" not in diagnostic


def test_verify_memory_accepts_automatic_tma_host_descriptors():
    @T.function
    def kernel(A: T.Tensor((32,), "float32", align=16), B: T.Tensor((32,), "float32", align=16)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((32,), "float32", layout="default", align=128)
        mbar = T.alloc_shared((1,), "uint64", align=8)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(128))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx[
                    "cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
                ],
                shared,
                A,
                box=(32,),
                coords=(0,),
                mbar=mbar.ptr_to([0]),
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        shared[lane] = shared[lane] + 1
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx["cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group"],
                B,
                shared,
                box=(32,),
                coords=(0,),
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
        T.cuda.cta_sync()

    prepared = _prepare_for_cuda(kernel, arch="sm_100a")
    source = prepared.script()
    assert source.count("T.cuda.tensormap_encode_tiled(") == 2
    assert source.rindex("T.cuda.tensormap_encode_tiled(") < source.index("with T.device_scope(")


if __name__ == "__main__":
    tvm_tirx.testing.main()
