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
"""Operators defined in Rust are registered when their library loads, as C++ ones are.

``rust/examples/define_ir.rs`` is built by cargo into a library of its own, which a fresh
interpreter loads, as ``import tvm_tirx`` loads libtvm_tirx_rust, and then uses its operators as it
uses C++ ones: registering them also gives them their TVMScript builders. Its type is registered
with them. A copy of the library is a second library with the same blocks, as libraries that link
the same crate have.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

RUST_DIR = Path(__file__).resolve().parents[3] / "rust"


def find_cargo():
    """A cargo as new as the crate's rust-version, rustup's first, as the CMake build looks."""
    manifest = (RUST_DIR / "Cargo.toml").read_text()
    needed = re.search(r'^rust-version *= *"([0-9.]+)"', manifest, re.M).group(1)
    needed = tuple(int(part) for part in needed.split("."))
    homes = [Path(os.environ["CARGO_HOME"])] if "CARGO_HOME" in os.environ else []
    candidates = [str(home / "bin" / "cargo") for home in [*homes, Path.home() / ".cargo"]]
    for cargo in [*candidates, shutil.which("cargo")]:
        if not cargo or not os.access(cargo, os.X_OK):
            continue
        version = subprocess.run([cargo, "--version"], capture_output=True, text=True).stdout
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
        if match and tuple(int(part) for part in match.groups()) >= needed:
            return cargo
    return None


@pytest.fixture(scope="module")
def define_ir_library():
    """The example library of Rust operators and types, built by cargo."""
    cargo = find_cargo()
    if cargo is None or not shutil.which("tvm-ffi-config") or not shutil.which("tvm-tirx-config"):
        pytest.skip(
            "building rust/ needs a cargo as new as its rust-version, and tvm-ffi-config and "
            "tvm-tirx-config on PATH"
        )
    manifest = RUST_DIR / "Cargo.toml"
    subprocess.run(
        [cargo, "build", "--manifest-path", str(manifest), "--example", "define_ir"], check=True
    )
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", RUST_DIR / "target"))
    name = {"darwin": "libdefine_ir.dylib", "win32": "define_ir.dll"}.get(
        sys.platform, "libdefine_ir.so"
    )
    return target_dir / "debug" / "examples" / name


def run_python(*code, libraries):
    """Run `code` in a fresh interpreter, with the paths of `libraries` as its arguments, and
    return its standard error.

    The code runs from a file: TVMScript reads the source of the functions it parses, which Python
    keeps for code passed with `-c` only from 3.13 on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text("\n".join(textwrap.dedent(part) for part in code))
        result = subprocess.run(
            [sys.executable, str(script), *map(str, libraries)],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stderr
    return result.stderr


# With the TVMScript builders that registering the example operators gives them, TVMScript writes,
# prints and parses their calls as it does those of C++ operators: a call with a value, a call
# emitted as a statement, `Call.attrs`, and a region, whose body parameter the Rust hook creates.
ROUND_TRIP = """
    import tvm_ffi
    import tvm_tirx
    from tvm_tirx import ir
    from tvm_tirx.script import tirx as T

    # Each builder is generated from the signature of its operator under the name the printer
    # gives its calls: `T.<name>` for `tirx.<name>`, and outside `tirx.`, in the parser namespace
    # its root names.
    assert T.example_clamp.__doc__ == ir.Op.get("tirx.example_clamp").doc
    assert T.example_loop.__doc__ == ir.Op.get("tirx.example_loop").doc
    from tvm_tirx.script.parser import my_ext

    assert my_ext.hint.__doc__ == ir.Op.get("my_ext.hint").doc

    @T.function
    def func(i: T.int32, n: T.int32):
        T.evaluate(T.example_clamp(i + 1, n))
        T.example_mark()
        T.evaluate(my_ext.hint(i, attrs={"note": "hot"}))
        with T.example_loop(n) as j:
            T.evaluate(T.example_clamp(j, n))

    text = func.script()
    assert "T.example_clamp(i + 1, n)" in text, text
    assert "T.example_mark()" in text, text
    assert 'my_ext.hint(i, attrs={"note": "hot"})' in text, text
    assert "with T.example_loop(n) as j:" in text, text
    parsed = tvm_tirx.script.parse(text, extra_vars={"T": T})
    assert tvm_ffi.structural_equal(parsed, func), parsed.script()
    assert parsed.script() == text
"""


def test_library_is_loaded_on_import():
    """``import tvm_tirx`` loads libtvm_tirx_rust, when the build has it, as a backend."""
    from tvm_ffi.libinfo import _find_library_by_basename  # pylint: disable=import-outside-toplevel

    from tvm_tirx.base import _LOADED_LIBS  # pylint: disable=import-outside-toplevel

    runtime_dir = Path(_LOADED_LIBS["tvm_runtime"]._name).resolve().parent
    try:
        _find_library_by_basename("tvm_tirx", "tvm_tirx_rust", extra_lib_paths=[runtime_dir])
    except RuntimeError:
        pytest.skip("the build has no libtvm_tirx_rust (TVM_TIRX_USE_RUST)")
    assert "tvm_tirx_rust" in _LOADED_LIBS


def test_rust_operators_are_registered_by_their_library(define_ir_library):
    run_python(
        """
        import sys

        import tvm_ffi
        from tvm_tirx import ir

        # Registered when the library loads, as libtvm_tirx_rust is when `import tvm_tirx` loads
        # it, with the doc comment and signature the Rust definition gives.
        tvm_ffi.load_module(sys.argv[1])
        clamp = ir.Op.get("tirx.example_clamp")
        assert clamp.doc == "Clamp an index to the range `[0, bound)`."
        assert [arg.name for arg in clamp.args_info] == ["index", "bound"]
        mark = ir.Op.get("tirx.example_mark")
        assert mark.doc.startswith("Mark a point in the program.\\n\\nThe call has no value")

        # A Call built in Python runs the Rust type inference and validator.
        i, n = ir.Var("i", "int32"), ir.Var("n", "int32")
        call = ir.Call(clamp, [i, n])
        assert tvm_ffi.structural_equal(call.ty, i.ty)
        call.validate()
        try:
            ir.Call(clamp, [i, ir.Var("f", "float32")], ty="int32").validate()
        except TypeError as error:
            assert "`Call.args[1]` (`bound`) expected `ir.IntExpr`" in str(error), error
        else:
            raise AssertionError("a float bound passed validation")
        """,
        ROUND_TRIP,
        libraries=[define_ir_library],
    )


def test_rust_types_are_registered_by_their_library(define_ir_library):
    run_python(
        """
        import sys

        import tvm_ffi
        import tvm_tirx
        from tvm_tirx import ir
        from tvm_tirx.script import tirx as T

        # Registered with the operators when the library loads, the type gets the class its
        # TVMScript constructor names, which constructs it from its fields, as a C++ type is
        # constructed.
        tvm_ffi.load_module(sys.argv[1])
        pair = T.ExamplePair(T.int32, second=T.float32)
        assert pair.script() == "T.ExamplePair(T.int32, T.float32)"
        assert tvm_ffi.structural_equal(pair.first, T.int32)

        # TVMScript prints and parses it, and a JSON graph holds it.
        @T.function
        def func(p: T.ExamplePair(T.int32, T.float32)):
            T.evaluate(p)

        text = func.script()
        assert "p: T.ExamplePair(T.int32, T.float32)" in text, text
        parsed = tvm_tirx.script.parse(text, extra_vars={"T": T})
        assert tvm_ffi.structural_equal(parsed, func), parsed.script()
        assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(func)), func)

        # A class written for it in Python takes the name.
        @tvm_ffi.register_object("tirx.ExamplePairType")
        class ExamplePair(ir.Type):
            pass

        assert T.expose_script_type("tirx.ExamplePairType") is ExamplePair
        assert T.ExamplePair is ExamplePair
        assert type(T.ExamplePair(T.int32, T.float32)) is ExamplePair
        """,
        libraries=[define_ir_library],
    )


def test_a_library_registers_its_operators_when_python_loads_it(define_ir_library, tmp_path):
    copy = tmp_path / define_ir_library.name
    shutil.copy(define_ir_library, copy)
    stderr = run_python(
        """
        import ctypes
        import gc
        import sys

        import tvm_ffi
        from tvm_tirx import ir

        library, copy = sys.argv[1:]

        # The library of another crate: importing tvm_tirx does not register its operators.
        try:
            ir.Op.get("tirx.example_clamp")
        except Exception:
            pass
        else:
            raise AssertionError("registered before the library loaded")

        # Loading it registers them, as loading a C++ library does. The library stays loaded when
        # Python drops its module, as the operators keep its hooks.
        module = tvm_ffi.load_module(library)
        del module
        gc.collect()
        clamp = ir.Op.get("tirx.example_clamp")
        i, n = ir.Var("i", "int32"), ir.Var("n", "int32")
        assert tvm_ffi.structural_equal(ir.Call(clamp, [i, n]).ty, i.ty)

        # The copy has the blocks of the library, as two libraries that link the same crate do:
        # loading it registers nothing again.
        ctypes.CDLL(copy)
        assert ir.Op.get("tirx.example_clamp").same_as(clamp)
        """,
        ROUND_TRIP,
        libraries=[define_ir_library, copy],
    )
    assert "registering operators and types failed" not in stderr, stderr


def test_a_block_that_fails_when_its_library_loads_is_reported(define_ir_library):
    stderr = run_python(
        """
        import sys

        import tvm_ffi
        from tvm_tirx import ir

        # Another library registered the name of an example operator: the block that defines it
        # fails when the library loads, and the other block registers its operators.
        tvm_ffi.get_global_func("ir.RegisterOp")("tirx.example_mark", "")
        module = tvm_ffi.load_module(sys.argv[1])
        ir.Op.get("tirx.example_clamp")

        # `tirx_register_ir` raises the error again.
        try:
            module.tirx_register_ir()
        except Exception as error:
            assert "tirx.example_mark is registered before" in str(error), error
        else:
            raise AssertionError("the block that failed was not reported")
        """,
        libraries=[define_ir_library],
    )
    assert "registering operators and types failed" in stderr, stderr
    assert "Operator tirx.example_mark is registered before" in stderr, stderr
