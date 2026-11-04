<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# TIRx Extension Example

A Python package, `my-ext`, that extends TIRx from Rust: a complex-number type,
operators that build and multiply complex values, and a pass that lowers them to
scalar arithmetic. Importing `my_ext` loads the Rust library, provides the
`my_ext.*` builders for TVMScript, and registers the `my_ext` compilation
pipeline. The tutorial in `docs/advanced/` walks through the code.

- `src/lib.rs`: the type, operators, and pass, built by cargo as a shared library
- `CMakeLists.txt`, `pyproject.toml`: build the library and install it into the package
- `python/my_ext/`: loads the library and registers the pipeline
- `run_example.py`: compiles a function that uses the extension and checks it against NumPy
- `my_ext_py/my_ext.py`: the same extension written in Python, a prototype that needs no build

## Install and Run

After building and installing TIRx from this checkout, with Rust 1.95 or newer,
run from the repository root in the same Python environment:

```bash
uv pip install scikit-build-core
uv pip install --no-build-isolation --no-deps ./examples/extension
python examples/extension/run_example.py
```

The build uses this checkout's `tvm-tirx` crate and pinned TVM FFI sources.
Disabling build isolation keeps the native libraries from the same checkout
available to cargo, instead of installing a released package in a temporary
build environment.

To run the example with the Python prototype instead, without building anything:

```bash
PYTHONPATH=examples/extension/my_ext_py python examples/extension/run_example.py
```
