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

"""Install the local core wheel or check the repaired optional wheel in CI."""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-core", type=Path)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if args.install_core:
        wheels = list(args.install_core.glob("apache_tvm_tirx-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one local core wheel, found {wheels}")
        subprocess.run([sys.executable, "-m", "pip", "install", str(wheels[0])], check=True)
        return
    if args.wheel is None:
        parser.error("Pass --install-core or --wheel")

    # The optional wheel must not bundle FFI or a second native LLVM backend.
    with zipfile.ZipFile(args.wheel) as wheel:
        forbidden = ("tvm_ffi", "llvm_backend_native")
        for name in wheel.namelist():
            if any(part in Path(name).name for part in forbidden):
                raise RuntimeError(f"Optional wheel bundles an FFI/native-backend library: {name}")

    # A normal core import must initialize the optional wheel's DLL loader hooks.
    from importlib import import_module
    from importlib.metadata import version

    import tvm_ffi

    from tvm_tirx import base

    get_lib_path = import_module("tvm_tirx_llvm_targets").get_lib_path

    assert version("apache-tvm-tirx") == version("apache-tvm-tirx-llvm-targets")
    loaded = Path(base._LOADED_LIBS["tvm_tirx_llvm_backend"]._name).resolve()
    assert loaded == get_lib_path().resolve(), (loaded, get_lib_path())
    targets = tvm_ffi.get_global_func("codegen.llvm_get_targets")()
    assert {"x86_64", "aarch64"}.issubset(targets), targets
    print(f"Selected optional LLVM backend {loaded}; {len(targets)} targets available")


if __name__ == "__main__":
    main()
