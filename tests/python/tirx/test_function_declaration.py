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
"""Required function bodies and explicit declaration completion."""

import json
import re

import pytest
import tvm_ffi

import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.script import from_source
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as TS
from tvm_tirx.script.ir_builder import IRBuilder, decl_function, ir_module
from tvm_tirx.testing import env
from tvm_tirx.tirx.script import ir_builder as T


def test_body_is_required():
    with pytest.raises(TypeError):
        tirx.Function([], None)
    function = tirx.Function([], [])
    with pytest.raises(TypeError):
        function.with_body(None)
    assert isinstance(function.body, ir.SeqStmt)
    assert len(function.body) == 0
    ir.assert_structural_equal(function, from_source(function.script(), extra_vars={"T": TS}))


@pytest.mark.parametrize("declaration", [False, True])
def test_json_preserves_required_body_and_marker(declaration):
    function = tirx.Function([], [])
    if declaration:
        function = function.with_attr("declaration", True)
    restored = ir.load_json(ir.save_json(function))
    ir.assert_structural_equal(function, restored)
    assert isinstance(restored.body, ir.SeqStmt)
    assert ("declaration" in restored.attrs) == declaration
    if declaration:
        assert '"declaration": True' in function.script()


@pytest.mark.parametrize("missing", [False, True])
def test_json_rejects_absent_body(missing):
    graph = json.loads(ir.save_json(tirx.Function([], [])))
    fields = graph["nodes"][graph["root_index"]]["data"]
    if missing:
        del fields["body"]
    else:
        fields["body"] = len(graph["nodes"])
        graph["nodes"].append({"type": "None"})
    with pytest.raises(TypeError):
        ir.load_json(json.dumps(graph))


@pytest.mark.parametrize("inplace", [False, True])
def test_structural_mutation_requires_body_but_normalizes_statement(inplace):
    function = tirx.Function([], [])
    with pytest.raises(TypeError):
        tvm_ffi.structural_map(
            function._move() if inplace else function,
            (ir.SeqStmt, lambda _: None),
            order="pre",
        )
    function = tirx.Function([], [])
    replacement = ir.Return(ir.IntImm("int32", 1))
    changed = tvm_ffi.structural_map(
        function._move() if inplace else function,
        (ir.SeqStmt, lambda _: replacement),
        order="pre",
    )
    assert isinstance(changed.body, ir.SeqStmt)
    assert changed.body[0].same_as(replacement)


def test_builder_rejects_unfinished_declaration_and_recovers():
    with IRBuilder() as builder:
        with pytest.raises(ValueError, match="pending is declared but not defined"):
            with ir_module():
                with T.function_(decl=True):
                    T.func_name_("pending")
        with ir_module():
            with T.function_():
                T.func_name_("empty")
    assert {var.name_hint for var in builder.get().get_global_vars()} == {"empty"}


def test_builder_completes_declaration_without_marker_leakage():
    with IRBuilder() as builder:
        with ir_module() as module:
            with T.function_(decl=True, private=True, persistent=True) as frame:
                T.func_name_("identity")
                parameter = T.arg_("value", ir.PtrType("float32"))
                T.func_ret_type_(parameter.ty)
            signature = module.functions[frame.global_var]
            reference = frame.global_var
            assert len(signature.body) == 0
            assert signature.attrs["declaration"]
            call = T.call_global_var_(reference, [parameter])
            ir.assert_structural_equal(call.ty, parameter.ty)
            with frame:
                T.return_(parameter)
            assert frame.global_var.same_as(reference)
    function = builder.get()["identity"]
    assert "declaration" not in function.attrs
    assert "global_symbol" not in function.attrs
    assert function.attrs["tirx.persistent_kernel"]
    assert function.params[0].same_as(parameter)
    assert function.body[0].value.same_as(parameter)
    assert tirx.analysis.check_well_formed(function)


def test_builder_rejects_duplicate_declarations_and_definitions():
    with IRBuilder() as builder:
        with ir_module():
            with T.function_(decl=True) as frame:
                T.func_name_("once")
            with pytest.raises(ValueError, match="already exists"):
                decl_function("once", frame.function)
            with frame:
                pass
            define = tvm_ffi.get_global_func("script.ir_builder.ir.DefFunction")
            with pytest.raises(ValueError, match="already been defined"):
                define("once", frame.function)
    assert len(builder.get()["once"].body) == 0


def test_concrete_module_member_remains_a_definition():
    empty = tirx.Function([], [])

    @I.ir_module
    class Module:
        existing = empty

        @TS.function
        def caller():
            Module.existing()

    assert "declaration" not in Module["existing"].attrs
    assert Module["caller"].body[0].value.op.same_as(Module.get_global_var("existing"))


@pytest.mark.parametrize("as_module", [False, True])
def test_well_formed_requires_definitions(as_module):
    empty = tirx.Function([], [])
    declaration = empty.with_attr("declaration", True)
    valid = ir.IRModule({"empty": empty}) if as_module else empty
    invalid = ir.IRModule({"pending": declaration}) if as_module else declaration
    assert tirx.analysis.check_well_formed(valid)
    assert not tirx.analysis.check_well_formed(invalid, assert_mode=False)
    with pytest.raises(RuntimeError, match="is a declaration, but a definition is required"):
        tirx.analysis.check_well_formed(invalid)


def test_c_codegen_distinguishes_empty_definition_from_declaration():
    empty = tirx.Function([], []).with_attr("global_symbol", "empty")
    declaration = empty.with_attr("global_symbol", "pending").with_attr("declaration", True)
    module = tvm_ffi.get_global_func("target.build.c")(
        ir.IRModule({"empty": empty, "pending": declaration}), tvm_tirx.target.Target("c")
    )
    source = module.inspect_source()
    assert re.search(r"void pending\([^)]*\);", source)
    assert not re.search(r"pending\([^)]*\)\s*\{", source)
    assert re.search(r"void empty\([^)]*\)\s*\{", source)
    assert list(module.get_function("get_func_names")()) == ["empty"]


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("system_lib", [False, True])
def test_llvm_codegen_keeps_declarations_out_of_exports(system_lib):
    empty = tirx.Function([], []).with_attr("global_symbol", "empty")
    declaration = empty.with_attr("global_symbol", "pending").with_attr("declaration", True)
    caller = tirx.Function([], ir.Evaluate(tirx.call_extern("pending", ty="void"))).with_attr(
        "global_symbol", "caller"
    )
    ir_module = ir.IRModule({"empty": empty, "pending": declaration, "caller": caller})
    if system_lib:
        ir_module = ir_module.with_attr("system_lib_prefix", "")
    module = tvm_ffi.get_global_func("target.build.llvm")(ir_module, tvm_tirx.target.Target("llvm"))
    source = module.inspect_source("ll")
    assert re.search(r"declare[^\n]*void @pending\(", source)
    assert not re.search(r"define[^\n]*@pending\(", source)
    assert re.search(r"define[^\n]*void @empty\(", source)
    assert sorted(module.get_function("get_func_names")()) == ["caller", "empty"]


@pytest.mark.parametrize("target", ["c", "llvm", "cuda"])
def test_codegen_rejects_declaration_with_body(target):
    if target == "llvm" and not env.has_llvm():
        pytest.skip("need llvm")
    malformed = tirx.Function([], ir.Return(ir.IntImm("int32", 1)), ir.PrimType("int32")).with_attr(
        "declaration", True
    )
    with pytest.raises(ValueError, match="must have an empty body"):
        tvm_ffi.get_global_func(f"target.build.{target}")(
            ir.IRModule({"malformed": malformed}), tvm_tirx.target.Target(target)
        )


def test_cuda_codegen_declarations_preserve_prototypes_without_exports(monkeypatch):
    monkeypatch.setenv("TVM_TIRX_COMPILE_FORCE_FALLBACK", "1")
    empty = tirx.Function([], []).with_attr("global_symbol", "empty")
    declaration = empty.with_attr("global_symbol", "pending").with_attr("declaration", True)
    module = tvm_ffi.get_global_func("target.build.cuda")(
        ir.IRModule({"empty": empty, "pending": declaration}),
        tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"}),
    )
    source = module.inspect_source()
    assert re.search(r"void pending\([^)]*\);", source)
    assert not re.search(r"pending\([^)]*\)\s*\{", source)
    assert re.search(r"void empty\([^)]*\)\s*\{", source)
    assert list(module.get_function("__tvm_cuda_binary")()[2]) == ["empty"]
