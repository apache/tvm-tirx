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
"""Exercise archive hiding through CMake without depending on an installed TIRx."""

import ctypes
import json
import pathlib
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason="ELF/Mach-O archive policy")
def test_hide_static_archive_symbols(tmp_path: pathlib.Path) -> None:
    if not shutil.which("cmake") or not shutil.which("nm"):
        pytest.skip("CMake and nm are required")
    source = pathlib.Path(__file__).parent / "cmake" / "hide_static_archives"
    build = tmp_path / "build with spaces"
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "--config", "Release"], check=True)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    archive_exports = {
        "direct_value",
        "imported_value",
        "transitive_value",
        "leaf_value",
        "unknown_value",
    }
    for name, hidden in (("protected", True), ("baseline", False)):
        library = build / ("lib" + name + suffix)
        loaded = ctypes.CDLL(str(library))
        assert loaded.public_api() == 31
        flags = ["-gU"] if sys.platform == "darwin" else ["-D", "--defined-only"]
        output = subprocess.check_output(["nm", *flags, str(library)], text=True)
        symbols = {line.split()[-1] for line in output.splitlines() if line.split()}
        if sys.platform == "darwin":
            symbols = {symbol.removeprefix("_") for symbol in symbols}
        assert "public_api" in symbols
        assert "unused_archive_member" not in symbols
        assert "dynamic_value" not in symbols
        assert not (symbols & archive_exports) if hidden else archive_exports <= symbols


def test_darwin_archive_launcher(tmp_path: pathlib.Path) -> None:
    """Keep launcher arguments intact and preserve large, quoted response files."""
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake is required")
    helper = pathlib.Path(__file__).resolve().parents[2] / "cmake/utils/HideDarwinArchives.cmake"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import json, os, pathlib, shlex, sys\n"
        "args = sys.argv[2:]\n"
        "assert len(args) == 1 and args[0].startswith('@')\n"
        "args = shlex.split(pathlib.Path(args[0][1:]).read_text())\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps([os.environ['CONFIG'], args]))\n"
    )
    response = tmp_path / "input with spaces.rsp"
    response.write_text(
        '"/archive with spaces/lib.part.a" "libdynamic.dylib"\n' + "x" * 1000 + "\n"
    )
    # Exceed typical command-line limits while remaining a valid response file.
    with response.open("a") as stream:
        stream.write(("x" * 1000 + "\n") * 2300)
    output = tmp_path / "arguments.json"
    subprocess.run(
        [
            cmake,
            "-P",
            str(helper),
            "--",
            "4",
            cmake,
            "-E",
            "env",
            "CONFIG=launcher.a",
            sys.executable,
            str(driver),
            str(output),
            "@" + str(response),
        ],
        cwd=tmp_path,
        check=True,
    )
    config, arguments = json.loads(output.read_text())
    assert config == "launcher.a"
    assert arguments[:5] == [
        "-Xlinker",
        "-load_hidden",
        "-Xlinker",
        "/archive with spaces/lib.part.a",
        "libdynamic.dylib",
    ]
    assert len(arguments) == 2306
    assert not list((tmp_path / "CMakeFiles").glob("tvm_tirx_hidden_*.rsp"))
