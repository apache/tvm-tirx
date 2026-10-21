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

Reproduce CI/CD
===============

Start with :doc:`source_build` for the editable package and ``dev`` dependencies.
Run local checks from the repository root in that environment. GitHub Actions
runs lint, macOS, and Windows checks; Jenkins runs Linux CPU, GPU, and ARM jobs.
The checked-in workflows and ``tests/scripts/`` define the exact CI commands.

Linters
-------

Pre-commit runs the hooks in `.pre-commit-config.yaml
<https://github.com/apache/tvm-tirx/blob/main/.pre-commit-config.yaml>`__, including
ASF headers, file types, whitespace, YAML/TOML, Ruff, and clang-format:

.. code-block:: bash

   uv run --no-sync pre-commit install
   uv run --no-sync pre-commit run --all-files

For a focused check, pass ``--files path/to/changed_file``. To run a single hook,
use, for example, ``pre-commit run ruff-check --all-files`` through the same
``uv run --no-sync`` prefix. Review any formatting changes before committing.
The GitHub lint workflow uses these same hooks.

C++ Tests
---------

Configure a separate Debug build with tests enabled, then build and run
``tirx_cpptest``:

.. code-block:: bash

   cmake -S . -B build_test -G Ninja -DCMAKE_BUILD_TYPE=Debug \
     -DTVM_TIRX_BUILD_TESTS=ON -DTVM_TIRX_USE_CUDA=OFF \
     -DTVM_TIRX_USE_LLVM="$TIRX_LLVM_CONFIG"
   cmake --build build_test --parallel 8 --target tirx_cpptest
   ctest --test-dir build_test -C Debug --output-on-failure

CMake uses an installed GoogleTest package or fetches it if needed. To select
a test, run ``build_test/tirx_cpptest --gtest_list_tests`` and then use
``--gtest_filter=Suite.Test`` with a name from that list. For Visual Studio,
omit ``-G Ninja`` and pass ``--config Debug`` when building.

Python Tests
------------

Start with the component you changed, then expand to the CPU suite:

.. code-block:: bash

   uv run --no-sync python -m pytest -v tests/python/ir/test_type.py
   uv run --no-sync python -m pytest -v tests/python -m "not gpu"

The ``test`` group (included by ``dev``) supplies pytest plugins, Torch, Z3,
and process-pool dependencies. Device tests require the corresponding native
backend, SDK, driver, and hardware:

.. code-block:: bash

   uv run --no-sync python -m pytest -v tests/python -m gpu

Architecture-specific tests may need a particular GPU generation. Inspect skip
reasons with ``-rs``; a skipped test is missing coverage for that environment.
The CI wrapper ``tests/scripts/task_python_unittest.sh`` defaults to ``not gpu``
and adds xdist parallelism and a faulthandler timeout. It accepts test paths;
``TVM_TEST_MARKER=gpu`` selects device tests. The GPU-only wrapper sets that
marker and runs the full Python test directory.

Run in a Jenkins Image
~~~~~~~~~~~~~~~~~~~~~~

To reproduce a Linux job's toolchain and installed dependencies, use its image
from ``ci/jenkins/docker-images.ini``. Docker must be available, and the host
architecture must match the image (use ``ci_arm`` on ARM). Start from a separate
recursive checkout so CI's ``build/``, ``dist/``, and ``.local/python`` do not
replace local development artifacts.

For the CPU job on an x86-64 host:

.. code-block:: bash

   TIRX_CI_IMAGE=$(python - <<'PYTHON'
   import configparser
   config = configparser.ConfigParser()
   config.read("ci/jenkins/docker-images.ini")
   print(config["jenkins"]["ci_cpu"])
   PYTHON
   )
   docker/bash.sh --no-gpu "$TIRX_CI_IMAGE" bash -euo pipefail -c '
     source tests/scripts/task_install_build_dependencies.sh
     tests/scripts/task_build_cpu.sh
     source tests/scripts/test_task_install_dist_wheels.sh
     tests/scripts/task_python_unittest.sh
   '

The first script builds the pinned TVM-FFI wheel and prepares build dependencies;
the CPU script builds the TIRx wheel and runs C++ tests. Sourcing the installation
scripts preserves their environment for subsequent commands. Wheels go to
``dist/``; Python tests import the wheel installation in ``.local/python``.

For ARM, select ``ci_arm`` and ``task_build_arm.sh``. For CUDA, select ``ci_gpu``
and ``task_build_gpu.sh``, omit ``--no-gpu``, and run
``task_python_unittest_gpuonly.sh`` after installing the wheels. CUDA execution
requires NVIDIA Container Toolkit and a supported GPU. The CPU job uses LLVM 17,
the ARM job LLVM 17, and the GPU job LLVM 15, as selected in their build scripts.
The GPU pipeline separately compiles ROCm, Vulkan, and OpenCL through
``task_build_gpu_other.sh``; that compile-only check does not execute those
backends. See :doc:`doc_build` for its documentation build.

Build Python Wheels
-------------------

Local Wheel
~~~~~~~~~~~

Use the same native environment and ``CMAKE_ARGS`` as the source build. This
builds a wheel for the current machine without publishing it:

.. code-block:: bash

   uv build --wheel --python .venv/bin/python --out-dir dist .

Use ``.venv/Scripts/python.exe`` on Windows. The wheel is written as
``dist/apache_tvm_tirx-<version>-py3-none-<platform>.whl``. Check its contents and
install it into a separate environment before testing, so an editable install
cannot hide missing packaged files. A local wheel linked against shared LLVM
needs that library available to the loader. On Linux, set its search path first:

.. code-block:: bash

   export LD_LIBRARY_PATH="$("$TIRX_LLVM_CONFIG" --libdir)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

On macOS, use ``DYLD_LIBRARY_PATH`` instead; on Windows, keep LLVM's DLL
directory on ``PATH``. Then install and test the wheel:

.. code-block:: bash

   uv venv --python .venv/bin/python .venv-wheel
   uv pip install --python .venv-wheel/bin/python dist/apache_tvm_tirx-*.whl pytest
   .venv-wheel/bin/python -I -m pytest -v \
     tests/python/runtime/test_tensor.py tests/python/ir/test_type.py

Keep only the intended TIRx wheel in ``dist/`` for that installation command.
A locally built wheel has not undergone the platform repair used by the publish
workflow; test it on the intended deployment platform before distributing it.

Python Compatibility
~~~~~~~~~~~~~~~~~~~~

TIRx accesses its C++ libraries through TVM-FFI rather than a TIRx extension
compiled against each CPython minor version. Accordingly, ``pyproject.toml``
sets ``wheel.py-api = "py3"``: a TIRx wheel has Python tag ``py3`` and ABI tag
``none``, and can be reused across supported Python 3 minor versions.
TVM-FFI supplies the interpreter-facing bindings as a separate dependency.

The package requires **Python >=3.10**, with classifiers currently listing
3.10 through 3.13 and no upper bound in ``Requires-Python``. The tag alone does
not establish testing on every newer interpreter. Installation also requires
compatible distributions of ``apache-tvm-ffi``, ``apache-tvm-ffi-orcjit``, NumPy,
and the other declared dependencies. These are native platform wheels, not
``py3-none-any`` wheels: OS, CPU architecture, system library baseline, and
optional GPU runtime requirements still apply.

Publication Builds
~~~~~~~~~~~~~~~~~~

The manual `publish_wheel.yml workflow
<https://github.com/apache/tvm-tirx/actions/workflows/publish_wheel.yml>`__ takes
a ``tag`` input (tag, branch, or SHA) and ``publish_repository`` (``none`` or
``pypi``). Selecting ``none`` builds and tests artifacts without uploading to
PyPI. The workflow currently builds Linux x86-64 and aarch64, macOS arm64, and
Windows AMD64 wheels. Its ``cp310-*`` selectors choose the build interpreter;
the resulting TIRx wheel still uses ``py3-none`` tags.

The `build-wheel-for-publish action
<https://github.com/apache/tvm-tirx/blob/main/.github/actions/build-wheel-for-publish/action.yml>`__
uses cibuildwheel with LLVM 22.1.0 and static LLVM linkage. Linux targets
manylinux_2_28. Linux and Windows wheels bundle a separately built CUDA runtime
library; macOS enables Metal. Platform repair and isolated smoke tests are
configured in ``pyproject.toml``. The wheel tests explicitly check LLVM and
CUDA-library expectations through ``TVM_WHEEL_EXPECT_LLVM`` and
``TVM_WHEEL_EXPECT_CUDA_RUNTIME``; they do not replace testing on actual GPUs.

Reproducing a publication build locally requires the action's LLVM prefix,
static dependencies, platform repair tools, and (where used) CUDA sidecar,
as well as its ``CIBW_*`` environment. A bare local ``uv build`` is only a
native wheel build. The action writes repaired wheels to ``wheelhouse/``;
the workflow uploads them as ``tirx-wheel-*`` artifacts. Source releases,
approval, and publication are covered in :doc:`release_process`.
