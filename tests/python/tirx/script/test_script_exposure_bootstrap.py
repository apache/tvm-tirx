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
"""Script exposure spans native loading and the completed Python builder namespace."""

import os
import subprocess
import sys
import textwrap


def run_python(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**os.environ, "TVM_TIRX_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_receivers_precede_native_loading_and_preserve_explicit_bindings():
    run_python(
        """
        import sys
        import tvm_ffi
        from tvm_ffi import libinfo

        load = libinfo.load_lib_ctypes
        observations = []
        receivers = []

        def observed_load(package, name, *args, **kwargs):
            base = sys.modules["tvm_tirx.base"]
            assert base._EXPOSE_SCRIPT_OP_CALLBACK is None
            assert base._EXPOSE_SCRIPT_TYPE_CALLBACK is None
            assert "tvm_tirx.ir" not in sys.modules
            assert "tvm_tirx.tirx.script.ir_builder" not in sys.modules
            receivers[:] = [tvm_ffi.get_global_func("script.expose_script_op"),
                            tvm_ffi.get_global_func("script.expose_script_type")]
            handle = load(package, name, *args, **kwargs)
            observations.append((name, list(base._PENDING_SCRIPT_EXPOSURES)))
            return handle

        libinfo.load_lib_ctypes = observed_load
        import tvm_tirx
        libinfo.load_lib_ctypes = load
        from tvm_tirx import base, ir
        from tvm_tirx.script import tirx as T
        from tvm_tirx.tirx.script import ir_builder as builder
        from tvm_tirx.tirx.script.ir_builder import ir as explicit

        assert observations[0][0] == "tvm_runtime"
        pending = dict(observations)["tvm_tirx_compiler"]
        assert ("op", "prim.pow") in pending
        assert ("type", "ir.TupleType") in pending
        assert ("type", "tirx.TensorType") in pending
        assert not base._PENDING_SCRIPT_EXPOSURES
        assert base._EXPOSE_SCRIPT_OP_CALLBACK is builder.expose_script_op
        assert base._EXPOSE_SCRIPT_TYPE_CALLBACK is builder.expose_script_type
        assert T.alloc_tensor is explicit.alloc_tensor
        assert T.Ptr is explicit.Ptr
        assert T.TensorMap is explicit.TensorMap
        assert T.Tuple is explicit.Tuple
        assert tvm_ffi.structural_equal(T.Tuple(T.int32).fields[0], T.int32)

        # Saved receiver handles continue to use current implementations after import.
        calls = []
        base._EXPOSE_SCRIPT_OP_CALLBACK = lambda name: calls.append(("op", str(name)))
        base._EXPOSE_SCRIPT_TYPE_CALLBACK = lambda name: calls.append(("type", str(name)))
        receivers[0]("later.operator")
        receivers[1]("later.Type")
        assert calls == [("op", "later.operator"), ("type", "later.Type")]
        """
    )


def test_mixed_pending_order_reentrant_dispatch_and_failed_replay_retry():
    run_python(
        """
        import importlib
        import tvm_ffi
        import tvm_tirx
        from tvm_tirx import base
        from tvm_tirx.tirx.script import ir_builder as builder

        implementations = importlib.import_module("tvm_tirx.tirx.script.ir_builder.op")
        op_receiver = tvm_ffi.get_global_func("script.expose_script_op")
        type_receiver = tvm_ffi.get_global_func("script.expose_script_type")
        assert not base._PENDING_SCRIPT_EXPOSURES
        base._EXPOSE_SCRIPT_OP_CALLBACK = None
        base._EXPOSE_SCRIPT_TYPE_CALLBACK = None
        op_receiver("first")
        type_receiver("second")
        op_receiver("third")
        assert list(base._PENDING_SCRIPT_EXPOSURES) == [
            ("op", "first"), ("type", "second"), ("op", "third")]
        calls = []
        fail = True

        def expose_op(name):
            calls.append(("op", str(name)))
            if name == "first":
                # The other callback is installed already and runs immediately.
                type_receiver("reentrant")

        def expose_type(name):
            calls.append(("type", str(name)))
            if name == "second" and fail:
                raise ValueError("retry this type")

        implementations.expose_script_op = expose_op
        implementations.expose_script_type = expose_type
        try:
            importlib.reload(builder)
        except ValueError as error:
            assert "retry this type" in str(error)
        else:
            raise AssertionError("exposure failure was swallowed")
        assert calls == [("op", "first"), ("type", "reentrant"), ("type", "second")]
        assert list(base._PENDING_SCRIPT_EXPOSURES) == [("type", "second"), ("op", "third")]
        fail = False
        importlib.reload(builder)
        assert calls[-2:] == [("type", "second"), ("op", "third")]
        assert not base._PENDING_SCRIPT_EXPOSURES
        op_receiver("immediate")
        assert calls[-1] == ("op", "immediate")
        assert not base._PENDING_SCRIPT_EXPOSURES
        """
    )


def test_failed_first_import_retains_successful_root_and_nested_bindings():
    run_python(
        """
        import importlib
        import sys
        import tvm_ffi
        import tvm_tirx
        from tvm_tirx import base, ir

        builder_name = "tvm_tirx.tirx.script.ir_builder"
        assert builder_name not in sys.modules
        receiver = tvm_ffi.get_global_func("script.expose_script_op")

        def define(name):
            tvm_ffi.get_global_func("ir.RegisterOp")(name, "Return the input value.")
            op = ir.Op.get(name)
            op.set_signature(["value"])
            ir.register_op_attr(name, "TFixedReturnType", ir.PrimType("int32"))
            return op

        first = define("tirx.test_bootstrap_first")
        nested = define("tirx.test_bootstrap_nested.identity")
        receiver(first.name)
        receiver(nested.name)
        # Deliberately request an incomplete definition to inject an exposure failure.
        receiver("tirx.test_bootstrap_retry")
        try:
            importlib.import_module(builder_name)
        except AttributeError as error:
            assert "test_bootstrap_retry" in str(error)
        else:
            raise AssertionError("the incomplete definition did not fail exposure")
        assert builder_name not in sys.modules
        assert list(base._PENDING_SCRIPT_EXPOSURES) == [("op", "tirx.test_bootstrap_retry")]

        retry = define("tirx.test_bootstrap_retry")
        builder = importlib.import_module(builder_name)
        assert not base._PENDING_SCRIPT_EXPOSURES
        assert tvm_ffi.structural_equal(builder.test_bootstrap_first(1), ir.Call(first, [1]))
        assert tvm_ffi.structural_equal(builder.test_bootstrap_nested.identity(2),
                                        ir.Call(nested, [2]))
        assert tvm_ffi.structural_equal(builder.test_bootstrap_retry(3), ir.Call(retry, [3]))
        from tvm_tirx.script import tirx as T
        assert T.test_bootstrap_first is builder.test_bootstrap_first
        assert T.test_bootstrap_nested is builder.test_bootstrap_nested
        """
    )
