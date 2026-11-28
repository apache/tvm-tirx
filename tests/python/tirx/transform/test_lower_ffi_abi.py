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
"""Structural examples of managed ownership lowering."""

import pytest
import tvm_ffi

import tvm_tirx as tvm
from tvm_tirx.script import tirx as T


def _check_lowering(before, expected):
    lowered = tvm.tirx.transform.LowerFFIABI()(tvm.IRModule({"main": before}))
    tvm.ir.assert_structural_equal(lowered["main"], expected)


def test_packed_result_acquires_its_slot_before_last_argument_drop():
    @T.function(private=True)
    def Before(result: T.handle) -> T.int32:
        a: T.let[T.Any] = T.call_packed("first", ty=T.Any)
        alias: T.let[T.Any] = a
        b: T.let[T.Any] = T.call_packed("second", alias, ty=T.Any)
        return T.store_return(result, b)

    @T.function(private=True)
    def Expected(result: T.handle) -> T.int32:
        anys: T.let[T.handle] = T.alloc_any_frame(T.int64(2))
        args: T.let[T.handle] = T.stack_alloca("tvm_ffi_any", 1)
        T.call_packed_lowered("first", args, 0, 0, anys, 0, ty="void")
        T.retain_any_slot(anys, T.int64(1), T.abi_field_get(anys, 0, 18, ty=T.Any))
        T.drop_any_slot(anys, T.int64(0))
        T.abi_field_set(args, 0, 18, T.abi_field_get(anys, 1, 18, ty=T.Any))
        T.call_packed_lowered("second", args, 0, 1, anys, 0, ty="void")
        T.drop_any_slot(anys, T.int64(1))
        T.abi_field_set(result, 0, 18, T.abi_field_get(anys, 0, 18, ty=T.Any))
        T.abi_field_set(anys, 0, 13, 0)
        T.abi_field_set(anys, 0, 14, 0)
        T.abi_field_set(anys, 0, 15, T.int64(0))
        return 0

    _check_lowering(Before, Expected)


def test_void_wrapper_result_preserves_effect_before_returning_none():
    effect = tvm.tirx.call_extern("flush", ty="void")

    @T.function(private=True)
    def Before(result: T.handle) -> T.int32:
        return T.store_return(result, effect)

    @T.function(private=True)
    def Expected(result: T.handle) -> T.int32:
        T.call_extern("flush", ty="void")
        T.abi_field_set(result, 0, 13, 0)
        T.abi_field_set(result, 0, 14, 0)
        T.abi_field_set(result, 0, 15, T.int64(0))
        return 0

    _check_lowering(Before, Expected)


def test_literal_string_result_is_owned():
    literal = tvm.ir.StringImm("left\0right")

    @T.function(private=True)
    def Before(result: T.handle) -> T.int32:
        return T.store_return(result, literal)

    @T.function(private=True)
    def Expected(result: T.handle) -> T.int32:
        T.retain_any_slot(result, T.int64(0), literal)
        return 0

    _check_lowering(Before, Expected)


def test_exits_check_enclosing_body_owners():
    @T.function(check_well_formed=False)
    def return_with_live_owner(arg: T.Any, stop: T.bool) -> T.int32:
        live: T.let[T.Any] = arg
        with T.parallel_launch():
            if stop:
                return 0
        T.call_packed("use", live, ty="void")
        return 0

    @T.function(check_well_formed=False)
    def return_without_owners(stop: T.bool) -> T.int32:
        for i in T.parallel(4):
            if stop:
                return i
        return 0

    @T.function(check_well_formed=False)
    def break_across_region(stop: T.bool):
        for i in range(4):
            with T.parallel_launch():
                if stop:
                    break

    @T.function(check_well_formed=False)
    def continue_targeting_parallel(stop: T.bool):
        for i in T.parallel(4):
            if stop:
                continue

    @T.function(check_well_formed=False)
    def inner_serial_exits(arg: T.Any, stop: T.bool) -> T.int32:
        live: T.let[T.Any] = arg
        with T.parallel_launch():
            for i in range(4):
                if stop:
                    break
                continue
        T.call_packed("use", live, ty="void")
        return 0

    lower = tvm.tirx.transform.LowerFFIABI()
    for func, diagnostic in (
        (return_with_live_owner, "Unsupported managed host region"),
        (return_without_owners, "Managed owners require sequential loops"),
        (break_across_region, "Unsupported managed host region"),
        (continue_targeting_parallel, "Managed owners require sequential loops"),
    ):
        with pytest.raises(ValueError, match=diagnostic):
            lower(tvm.IRModule({"main": func}))
    lower(tvm.IRModule({"main": inner_serial_exits}))


def test_parallel_scratch_belongs_only_to_the_body():
    @T.function
    def parallel() -> T.int32:
        for i in T.parallel(T.call_packed("extent", ty="int32")):
            T.call_packed("worker", i, ty="void")
        result: T.let[T.int32] = T.call_packed("finish", ty="int32")
        return result

    func = tvm.tirx.transform.LowerFFIABI()(tvm.IRModule({"main": parallel}))["main"]
    calls = []
    tvm_ffi.structural_walk(func.body, (tvm.ir.Call, calls.append))
    extent, worker, finish = [
        call.args[1] for call in calls if call.op.name == "tirx.call_packed_lowered"
    ]
    loop = next(stmt for stmt in func.body.seq if isinstance(stmt, tvm.ir.For))
    assert extent.same_as(func.body.seq[0].var)
    assert finish.same_as(extent)
    assert worker.same_as(loop.body.seq[0].var)
    assert not worker.same_as(extent)
