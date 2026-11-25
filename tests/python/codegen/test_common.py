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
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T


def test_tensor_store_predicate_not_supported():
    target = "c"

    @T.function
    def func(B: T.Tensor((8,), "float32")):
        T.evaluate(
            T.call_intrin(
                "tirx.masked_store",
                B,
                T.Broadcast(1.0, 4),
                T.Ramp(0, 2, 4),
                T.Broadcast(T.bool(True), 4),
                ty="void",
            )
        )

    err_msg = "Predicated buffer store is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        tvm_tirx.compile(func, target=tvm_tirx.target.Target(target))


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
        pytest.param("rocm", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "from_device": 0}, marks=pytest.mark.gpu),
    ],
)
def test_tensor_store_predicate_not_supported_gpu(target):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.function
    def func(A: T.Tensor((2, 3), "float32"), B: T.Tensor((6,), "float32")):
        T.func_attr({"global_symbol": "main"})
        with T.launch_thread("threadIdx.x", 3) as i_0:
            T.evaluate(
                T.call_intrin(
                    "tirx.masked_store",
                    B,
                    T.Broadcast(1.0, 4),
                    T.Ramp(i_0, 1, 4),
                    T.Broadcast(T.bool(True), 4),
                    ty="void",
                )
            )

    err_msg = "Predicated buffer store is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        tvm_tirx.compile(func, target=tvm_tirx.target.Target(target))


def test_buffer_load_predicate_not_supported():
    target = "c"

    @T.function
    def func(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        for i_0 in range(4):
            B[T.Ramp(0, 2, 4)] = T.call_intrin(
                "tirx.masked_load",
                A,
                T.Ramp(i_0, 1, 4),
                T.Broadcast(T.bool(True), 4),
                ty="float32x4",
            )

    err_msg = "Predicated buffer load is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        tvm_tirx.compile(func, target=tvm_tirx.target.Target(target))


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param("opencl", marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
        pytest.param("rocm", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "from_device": 0}, marks=pytest.mark.gpu),
    ],
)
def test_buffer_load_predicate_not_supported_gpu(target):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.function
    def func(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.launch_thread("threadIdx.x", 3) as i_0:
            B[T.Ramp(0, 2, 4)] = T.call_intrin(
                "tirx.masked_load",
                A,
                T.Ramp(i_0, 1, 4),
                T.Broadcast(T.bool(True), 4),
                ty="float32x4",
            )

    err_msg = "Predicated buffer load is not supported."
    with pytest.raises(RuntimeError, match=err_msg):
        tvm_tirx.compile(func, target=tvm_tirx.target.Target(target))


@pytest.mark.parametrize(
    ("target", "qualifier"),
    [("opencl", "__global "), ("metal", "device ")],
)
def test_decl_buffer_offset_preserves_storage_scope(target, qualifier):
    @T.function
    def kernel(A_ptr: T.handle("float32", "global")):
        T.func_attr(
            {
                "calling_conv": 2,
                "global_symbol": "kernel",
                "tirx.kernel_launch_params": [],
                "tirx.noalias": True,
            }
        )
        A = T.decl_tensor(A_ptr, ty_args=[T.Tensor((8,), "float32", data=A_ptr)])
        B = T.decl_tensor(
            T.address_of(A[4]), ty_args=[T.Tensor((4,), "float32", data=T.address_of(A[4]))]
        )
        B[0] = T.float32(1)

    mod = tvm_tirx.IRModule({"kernel": kernel})
    build = tvm_ffi.get_global_func(f"target.build.{target}")
    source = build(mod, tvm_tirx.target.Target(target)).inspect_source()
    assert f"{qualifier}float* B" in source
    assert f"(({qualifier}float*)" in source


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_chained_any_results_from_global_functions(target):
    if target != "c" and not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.function
    def chain() -> T.Any:
        x: T.let[T.Any] = T.call_packed("test.ffi_chain_source", ty=T.Any)
        y: T.let[T.Any] = T.call_packed("test.ffi_chain_next", x, ty=T.Any)
        return y

    @T.function
    def discard():
        T.call_packed("test.ffi_chain_source")

    source_value = tvm_ffi.convert([41])

    @tvm_ffi.register_global_func("test.ffi_chain_source")
    def source():
        return source_value

    @tvm_ffi.register_global_func("test.ffi_chain_next")
    def next_value(x):
        return tvm_ffi.convert([x[0] + 1])

    try:
        result = tvm_tirx.compile(chain, target=target)()
        assert list(result) == [42]
        count = tvm_tirx.testing.object_use_count(source_value)
        tvm_tirx.compile(discard, target=target)()
        assert tvm_tirx.testing.object_use_count(source_value) == count
    finally:
        tvm_ffi.registry.remove_global_func("test.ffi_chain_next")
        tvm_ffi.registry.remove_global_func("test.ffi_chain_source")


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_packed_entry_cleans_up_compute_scope_failures(target):
    if target != "c" and not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.function
    def fail_after_write(self: T.handle, args: T.handle, n: T.int32, result: T.handle) -> T.int32:
        T.func_attr({"global_symbol": "__tvm_ffi_fail_after_write", "calling_conv": 1})
        T.retain_any_slot(result, 0, T.abi_field_get(args, 0, 18, ty=T.Any()))
        T.call_extern("TVMFFIErrorSetRaisedFromCStr", "ValueError", "partial output", ty="void")
        T.throw_last_error()
        return 0

    @T.function
    def entry(self: T.handle, args: T.handle, n: T.int32, result: T.handle) -> T.int32:
        T.func_attr({"global_symbol": "__tvm_ffi_entry", "calling_conv": 1})
        value: T.let[T.Any()] = T.abi_field_get(args, 0, 18, ty=T.Any())
        mode: T.let[T.int64] = T.abi_field_get(args, 1, 15, ty="int64")
        if mode == 1:
            T.call_cpacked("fail_after_write", value, ty="void")
        with T.compute_scope("managed_compute"):
            alias: T.let[T.Any()] = value
            if mode == 2:
                status: T.let[T.int32] = T.call_cpacked("fail_after_write", alias, ty="int32")
                return T.store_return(result, status)
            return T.store_return(result, alias)
        return 0

    mod = tvm_tirx.tirx.transform.LowerFFIABI()(tvm_tirx.IRModule({"entry": entry}))
    mod["fail_after_write"] = fail_after_write
    lib = tvm_tirx.codegen.build_module(mod, target)
    exe = tvm_tirx.codegen.Executable(lib).jit()
    obj = tvm_ffi.convert([7])
    original_count = tvm_tirx.testing.object_use_count(obj)
    result = exe["entry"](obj, 0)
    assert result.same_as(obj)
    assert tvm_tirx.testing.object_use_count(obj) == original_count + 1
    del result
    assert tvm_tirx.testing.object_use_count(obj) == original_count
    for mode in (1, 2):
        with pytest.raises(ValueError, match="partial output"):
            exe["entry"](obj, mode)
        assert tvm_tirx.testing.object_use_count(obj) == original_count


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_codegen_loop_step(target):
    if target != "c" and not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @T.function
    def test_loop_step(
        A: T.Tensor((1024,), "float32"),
        B: T.Tensor((1024,), "float32"),
        C: T.Tensor((1024,), "float32"),
    ):
        for i in T.serial(3, 1024, step=96):
            C[i] = A[i] + B[i]

    lib = tvm_tirx.compile(test_loop_step, target=target)

    src = lib.mod.inspect_source()
    if target == "c":
        assert src.find("for (int32_t i = 3; i < 1024; i += 96)") >= 0

    dev = tvm_tirx.cpu()
    a_np = np.random.rand(1024).astype("float32")
    b_np = np.random.rand(1024).astype("float32")
    c_np = np.zeros(1024, dtype="float32")
    a_tvm = tvm_tirx.runtime.tensor(a_np, dev)
    b_tvm = tvm_tirx.runtime.tensor(b_np, dev)
    c_tvm = tvm_tirx.runtime.tensor(c_np, dev)

    lib(a_tvm, b_tvm, c_tvm)

    c_result = c_tvm.numpy()

    # Check that the loop executes at positions 3, 99, 195, 291, 387, 483, 579, 675, 771, 867, 963
    for i in range(3, 1024, 96):
        tvm_tirx.testing.assert_allclose(c_result[i], a_np[i] + b_np[i], rtol=1e-5)

    # Assert non-touched positions remain zero
    for i in range(0, 3):
        assert c_result[i] == 0.0
    for i in range(4, 1024):
        if (i - 3) % 96 != 0:
            assert c_result[i] == 0.0


if __name__ == "__main__":
    tvm_tirx.testing.main()
