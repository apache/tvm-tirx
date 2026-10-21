..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

Build from Source
=================

Build TIRx from source to develop the compiler or select its native backends.
The distribution is ``apache-tvm-tirx``; Python code imports ``tvm_tirx``.
Run the commands below from the repository root unless noted otherwise.

Set Up the Environment
----------------------

You need Git, Python 3.10 or newer, a C++20 compiler, CMake 3.26 or newer,
and Ninja. The default LLVM backend requires LLVM development libraries
(version 15 or newer), including ``llvm-config``. Install a device SDK only
if you plan to enable its runtime.

On Linux or macOS, install `Miniforge <https://github.com/conda-forge/miniforge>`__
and initialize mamba for your shell. Create an environment for Python and the
native build tools:

.. code-block:: bash

   mamba create -n tirx-dev -c conda-forge python=3.12 \
     "cmake>=3.26" ninja git "llvmdev=22" uv
   mamba activate tirx-dev

Use a system C++20 compiler, or install one with ``mamba install -c conda-forge
cxx-compiler``. On macOS, also install the Xcode command-line tools
(``xcode-select --install``). On Windows, use a Visual Studio developer shell
with the C++ workload; adapt the shell syntax and use the environment's
``Library/bin/llvm-config.exe``.

Check out TIRx and its pinned dependencies:

.. code-block:: bash

   git clone --recursive https://github.com/apache/tvm-tirx.git
   cd tvm-tirx

For an existing checkout, run ``git submodule update --init --recursive`` after
switching revisions. In particular, the native build uses the TVM-FFI submodule.

Keep native tools in the mamba environment and Python project packages in a
uv virtual environment. Select the active environment's interpreter explicitly
so uv does not choose an unrelated Python installation:

.. code-block:: bash

   uv venv --python "$(command -v python)"
   source .venv/bin/activate
   export TIRX_LLVM_CONFIG="$(command -v llvm-config)"
   "$TIRX_LLVM_CONFIG" --version
   export CMAKE_ARGS="-DTVM_TIRX_USE_LLVM=$TIRX_LLVM_CONFIG -DTVM_TIRX_USE_CUDA=OFF"
   export CMAKE_BUILD_PARALLEL_LEVEL=8

Adjust the parallelism to available memory and CPU cores. Activating ``.venv``
ensures ``uv pip install`` targets the project environment; the mamba-provided
native tools remain on ``PATH``. In a new shell, activate ``tirx-dev`` first,
then ``.venv`` from the repository root. ``uv run --no-sync`` uses those packages
without replacing the editable build.

Build the Python Package
------------------------

Install the editable package and development dependencies. scikit-build-core
runs CMake to build the native libraries:

.. code-block:: bash

   uv pip install --verbose --group dev -e .

Python edits take effect immediately. After changing C++ code or CMake options,
re-run the install with ``--reinstall-package apache-tvm-tirx``:

.. code-block:: bash

   uv pip install --verbose --reinstall-package apache-tvm-tirx -e .

Verify the package and the libraries actually loaded by this interpreter:

.. code-block:: bash

   uv run --no-sync python - <<'PYTHON'
   import tvm_tirx
   from tvm_tirx import base

   print("Version:", tvm_tirx.__version__)
   print("Python package:", tvm_tirx.__file__)
   for name, library in base._LOADED_LIBS.items():
       print(name, library._name)
   assert "tvm_tirx_compiler" in base._LOADED_LIBS
   PYTHON
   uv run --no-sync tvm-tirx-config --libfiles

``base._LOADED_LIBS`` is an internal diagnostic: it reports loaded handles,
not just candidate library paths. If paths point to another checkout, remove
stale ``PYTHONPATH`` or ``TVM_TIRX_LIBRARY_PATH`` overrides and rebuild in this
environment.

Select Backends
---------------

Pass options through ``CMAKE_ARGS`` before installing or rebuilding. For
example, enable CUDA while keeping the selected LLVM toolchain:

.. code-block:: bash

   export CMAKE_ARGS="-DTVM_TIRX_USE_LLVM=$TIRX_LLVM_CONFIG -DTVM_TIRX_USE_CUDA=ON"
   uv pip install --verbose --reinstall-package apache-tvm-tirx -e .

The options are defined in `CMakeLists.txt
<https://github.com/apache/tvm-tirx/blob/main/CMakeLists.txt>`__:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Behavior
   * - ``TVM_TIRX_USE_LLVM``
     - Defaults to ``ON``, which searches for an LLVM CMake package. A full
       ``llvm-config`` path selects that installation explicitly; ``OFF``
       disables LLVM code generation.
   * - ``TVM_TIRX_USE_CUDA``
     - Defaults to ``AUTO``. Use ``ON`` or an SDK path to require CUDA,
       or ``OFF`` for a CPU-only build. Execution needs a compatible driver
       and GPU as well as the toolkit.
   * - ``TVM_TIRX_USE_METAL``
     - Defaults to ``OFF``; enable on macOS to build the Metal runtime.
   * - ``TVM_TIRX_USE_ROCM``, ``TVM_TIRX_USE_VULKAN``, ``TVM_TIRX_USE_OPENCL``
     - Default to ``OFF``. Enable with the corresponding SDK and runtime
       available; device tests also need supported hardware.
   * - ``TVM_TIRX_USE_Z3``
     - Defaults to ``AUTO``. The Python build provides ``mlc-z3-static``;
       use ``ON`` to require the solver or ``OFF`` to disable it.
   * - ``TVM_TIRX_BUILD_TESTS``
     - Defaults to ``OFF``. Enable for the C++ test target; see :doc:`ci_cd`.

Build the C++ Libraries
-----------------------

For native development without an editable Python install, configure a separate
build directory. The TVM-FFI submodule is built as part of the same CMake graph:

.. code-block:: bash

   cmake -S . -B build_cpp -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DTVM_TIRX_USE_LLVM="$TIRX_LLVM_CONFIG" -DTVM_TIRX_USE_CUDA=OFF
   cmake --build build_cpp --parallel 8 --target tvm_tirx_compiler runtime

Libraries are written under ``build_cpp/lib``. The ``runtime`` target includes
the core runtime and enabled device runtimes. This does not install the Python
package; use the editable workflow above for Python tests and documentation.
For a multi-configuration generator such as Visual Studio, add
``--config RelWithDebInfo`` to the build command.

Continue with :doc:`ci_cd` for tests or :doc:`doc_build` for documentation.
