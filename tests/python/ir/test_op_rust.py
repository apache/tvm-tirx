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

import tvm_tirx

RUST_DIR = Path(__file__).resolve().parents[3] / "rust"
EXTENSION_DIR = RUST_DIR.parent / "examples" / "extension_guides"


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


def cargo_build(manifest, *args):
    """Build with cargo in rust/'s target directory, shared with the crate, and return the
    directory of the debug build."""
    cargo = find_cargo()
    if cargo is None or not shutil.which("tvm-ffi-config") or not shutil.which("tvm-tirx-config"):
        pytest.skip(
            "building rust/ needs a cargo as new as its rust-version, and tvm-ffi-config and "
            "tvm-tirx-config on PATH"
        )
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", RUST_DIR / "target"))
    env = {**os.environ, "CARGO_TARGET_DIR": str(target_dir)}
    subprocess.run([cargo, "build", "--manifest-path", str(manifest), *args], check=True, env=env)
    return target_dir / "debug"


def library_name(name):
    """The file name of the shared library that cargo builds for `name`."""
    return {"darwin": f"lib{name}.dylib", "win32": f"{name}.dll"}.get(sys.platform, f"lib{name}.so")


def build_rust_example(example):
    """Build an example library of Rust operators and types with cargo."""
    debug_dir = cargo_build(RUST_DIR / "Cargo.toml", "--example", example)
    return debug_dir / "examples" / library_name(example)


@pytest.fixture(scope="module")
def define_ir_library():
    """The example library of Rust operators and types, built by cargo."""
    return build_rust_example("define_ir")


@pytest.fixture(scope="module")
def extension_package(tmp_path_factory):
    """The my_ext package of examples/extension_guides/rust, built from its declared source
    dependency and laid out as its wheel installs it."""
    debug_dir = cargo_build(EXTENSION_DIR / "rust" / "Cargo.toml")
    root = tmp_path_factory.mktemp("extension")
    package = shutil.copytree(EXTENSION_DIR / "rust" / "my_ext", root / "my_ext")
    shutil.copy(debug_dir / library_name("my_ext"), package)
    return root


def run_python(*code, libraries=(), python_path=()):
    """Run `code` in a fresh interpreter, with the paths of `libraries` as its arguments and
    `python_path` before PYTHONPATH, and return its standard error.

    The code runs from a file: TVMScript reads the source of the functions it parses, which Python
    keeps for code passed with `-c` only from 3.13 on.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.py"
        script.write_text("\n".join(textwrap.dedent(part) for part in code))
        paths = [*map(str, python_path), *filter(None, [os.environ.get("PYTHONPATH")])]
        result = subprocess.run(
            [sys.executable, str(script), *map(str, libraries)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(paths)},
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

    # Built-in destinations expose tirx operators as T.<name>. Extensions declare
    # their own API module and pass it explicitly when parsing printed source.
    assert T.example_clamp.__doc__ == ir.Op.get("tirx.example_clamp").doc
    assert T.example_loop.__doc__ == ir.Op.get("tirx.example_loop").doc
    import importlib
    import sys
    from types import ModuleType
    from tvm_tirx.ir._expose_api import _init_exposed_op_api
    from tvm_tirx.script.ir_builder.base import auto_emit_void_call

    sys.modules["my_ext"] = ModuleType("my_ext")
    _init_exposed_op_api("my_ext", "my_ext", wrapper=auto_emit_void_call)
    my_ext = importlib.import_module("my_ext")

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
    parsed = tvm_tirx.script.parse(text, extra_vars={"T": T, "my_ext": my_ext})
    assert tvm_ffi.structural_equal(parsed, func), parsed.script()
    assert parsed.script() == text
"""


# Checks of the extension beyond those of its example script: the type, printing and parsing,
# invalid calls, chained products, float64 parts, and what the lowering rejects.
EXTENSION_CHECKS = """
    # Run after examples/extension_guides/run_example.py, whose names it uses.
    import tvm_ffi
    from tvm_tirx import ir, tirx

    # Inspect the registered type and round-trip the IR through text and JSON.
    a_var, b_var = ir.Var("a", "float32"), ir.Var("b", "float32")
    z = my_ext.complex(a_var, b_var)
    assert isinstance(z.ty, my_ext.Complex)
    assert tvm_ffi.structural_equal(z.ty.dtype, ir.PrimType("float32"))
    assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(z.ty)), z.ty)
    text = complex_product.script()
    parsed = tvm_tirx.script.parse(text, extra_vars={"T": T, "my_ext": my_ext})
    assert tvm_ffi.structural_equal(parsed, complex_product)
    assert tvm_ffi.structural_equal(ir.load_json(ir.save_json(complex_product)), complex_product)
    for name in ("complex", "cmul", "real", "imag"):
        op = ir.Op.get(f"my_ext.{name}")
        assert op.get_attr("TCallEffectKind") == ir.CallEffectKind.Pure
        assert getattr(my_ext, name).__doc__ == op.doc

    lower = my_ext.lower_complex()
    lowered = lower(ir.IRModule({"main": complex_product}))
    assert "my_ext." not in lowered["main"].script()


    for original, tensor in zip((a, b, c, d), inputs):
        np.testing.assert_array_equal(tensor.numpy(), original)


    # Further checks stay out of the tutorial's code slices.
    def expect_error(error_type, message, function, *args):
        try:
            function(*args)
        except error_type as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError(f"expected {error_type.__name__}: {message}")


    # The parts are scalar floats of one type; nothing is converted or promoted.
    for dtype in ("int32", "float32x4"):
        wrong = ir.Var("wrong", dtype)
        expect_error(TypeError, "expected a scalar float", my_ext.complex, wrong, b_var)
        expect_error(TypeError, "expected a scalar float", my_ext.complex, a_var, wrong)
    wide = ir.Var("wide", "float64")
    expect_error(TypeError, "parts of the same type", my_ext.complex, a_var, wide)
    z64 = my_ext.complex(wide, wide)
    assert tvm_ffi.structural_equal(z64.ty, my_ext.Complex(T.float64))
    assert tvm_ffi.structural_equal(my_ext.real(z64).ty, T.float64)
    expect_error(TypeError, "complex values of the same type", my_ext.cmul, z, z64)
    brain = ir.Var("brain", "bfloat16")
    assert tvm_ffi.structural_equal(my_ext.complex(brain, brain).ty, my_ext.Complex(T.bfloat16))
    expect_error(TypeError, "cmul expects a complex value", my_ext.cmul, a_var, z)
    expect_error(TypeError, "cmul expects a complex value", my_ext.cmul, z, a_var)
    expect_error(TypeError, "real expects a complex value", my_ext.real, a_var)
    expect_error(TypeError, "imag expects a complex value", my_ext.imag, a_var)

    bad_type = my_ext.Complex(T.int32)
    expect_error(TypeError, "expected a scalar float", my_ext.real, ir.Var("bad", bad_type))

    # Complex function parameters and unconsumed complex values are outside this lowering.
    unsupported = tirx.Function([ir.Var("z", z.ty)], ir.SeqStmt([ir.Evaluate(0)]))
    expect_error(ValueError, "unsupported complex type", lower, ir.IRModule({"main": unsupported}))
    unsupported = tirx.Function([a_var, b_var], ir.SeqStmt([ir.Evaluate(z)]))
    expect_error(ValueError, "unsupported complex", lower, ir.IRModule({"main": unsupported}))


    # Aliases, chained products and inline constructors take the same lowering path.
    @T.function
    def cube(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        R: T.Tensor((N,), "float32"),
        Im: T.Tensor((N,), "float32"),
    ):
        for i in range(N):
            z = my_ext.complex(A[i], B[i])
            alias = z
            square = my_ext.cmul(alias, z)
            R[i] = my_ext.real(my_ext.cmul(square, my_ext.complex(A[i], B[i])))
            Im[i] = my_ext.imag(my_ext.cmul(square, z))


    cube_lowered = lower(ir.IRModule({"main": cube}))
    assert "my_ext." not in cube_lowered["main"].script()
    cube_compiled = tvm_tirx.compile(cube, target="llvm", pipeline="my_ext")
    cube_compiled(*inputs[:2], real_out, imag_out)
    expected_cube = (a + 1j * b) ** 3
    np.testing.assert_allclose(real_out.numpy(), expected_cube.real, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(imag_out.numpy(), expected_cube.imag, rtol=1e-5, atol=1e-6)


    # The same lowering handles float64 parts.
    @T.function
    def product64(
        A: T.Tensor((N,), "float64"),
        B: T.Tensor((N,), "float64"),
        R: T.Tensor((N,), "float64"),
        Im: T.Tensor((N,), "float64"),
    ):
        for i in range(N):
            z = my_ext.complex(A[i], B[i])
            product = my_ext.cmul(z, z)
            R[i] = my_ext.real(product)
            Im[i] = my_ext.imag(product)


    a64, b64 = rng.standard_normal((2, N))
    real64, imag64 = runtime.empty((N,), "float64"), runtime.empty((N,), "float64")
    product64_compiled = tvm_tirx.compile(product64, target="llvm", pipeline="my_ext")
    product64_compiled(runtime.tensor(a64), runtime.tensor(b64), real64, imag64)
    expected64 = (a64 + 1j * b64) ** 2
    np.testing.assert_allclose(real64.numpy(), expected64.real, rtol=1e-12)
    np.testing.assert_allclose(imag64.numpy(), expected64.imag, rtol=1e-12)
"""


# The validators of the Rust operators check a call whose result type is given explicitly.
VALIDATOR_CHECKS = """
    for name, args in (("complex", [a_var, b_var]), ("cmul", [z, z]), ("real", [z]), ("imag", [z])):
        invalid = ir.Call(f"my_ext.{name}", args, ty="int32")
        expect_error(TypeError, "incorrect result type", invalid.validate)
    invalid = ir.Call("my_ext.cmul", [a_var, z], ty=z.ty)
    expect_error(TypeError, "expects a complex value", invalid.validate)
"""


# The operators of the Python prototype validate the number of arguments, as their signatures
# declare; its pass checks the types of the calls it lowers.
PROTOTYPE_CHECKS = """
    invalid_calls = [
        (ir.Call(f"my_ext.{name}", args, ty="int32"), "incorrect result type")
        for name, args in (("complex", [a_var, b_var]), ("cmul", [z, z]), ("real", [z]))
    ]
    invalid_calls.append((ir.Call("my_ext.cmul", [a_var, z], ty=z.ty), "expects a complex value"))
    for invalid, message in invalid_calls:
        invalid.validate()
        invalid_func = tirx.Function([a_var, b_var], ir.SeqStmt([ir.Evaluate(invalid)]))
        expect_error(TypeError, message, lower, ir.IRModule({"main": invalid_func}))
    expect_error(TypeError, "argument", ir.Call("my_ext.real", [z, z], ty="float32").validate)
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


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
def test_extension_example(extension_package):
    script = EXTENSION_DIR / "run_example.py"
    run_python(
        script.read_text(), EXTENSION_CHECKS, VALIDATOR_CHECKS, python_path=[extension_package]
    )


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM is not enabled")
def test_extension_prototype():
    """The Python prototype of the extension runs the same example and checks."""
    script = EXTENSION_DIR / "run_example.py"
    prototype = EXTENSION_DIR / "python"
    run_python(script.read_text(), EXTENSION_CHECKS, PROTOTYPE_CHECKS, python_path=[prototype])


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

        # Reexposure preserves the first published constructor.
        original = T.ExamplePair
        @tvm_ffi.register_object("tirx.ExamplePairType")
        class ExamplePair(ir.Type):
            pass

        assert T.expose_script_type("tirx.ExamplePairType") is original
        assert T.ExamplePair is original
        assert type(T.ExamplePair(T.int32, T.float32)) is original
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
    assert "the static init block at" not in stderr, stderr


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
    assert "the static init block at" in stderr, stderr
    assert "Operator tirx.example_mark is registered before" in stderr, stderr
