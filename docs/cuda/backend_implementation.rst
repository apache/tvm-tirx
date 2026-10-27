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

CUDA Backend Implementation
===========================

In this chapter, we will trace a CUDA program from TVMScript to device
code and locate the parts of the backend that implement each step.
This is useful when a new instruction needs support, a layout fails to
lower, or generated code differs from the intended computation. The
following chapters explain the authoring interfaces in more detail.

TIRx keeps the core IR and compiler machinery under ``tvm_tirx.ir`` and ``tvm_tirx.tirx``.  A target
backend owns its script namespaces, intrinsic builders, dispatch variants,
pipeline additions, and code-generation support under ``tvm_tirx.backend``.

Loading and registration
------------------------

``tvm_tirx.backend.load`` imports a backend and calls its registration hook.  A
backend can register the following target-owned behavior:

#. TVMScript namespaces such as ``Tx.cuda``, ``Tx.ptx``, and ``Tx.nki``;
#. tile-op dispatch implementations for that target kind;
#. target tags and, when the target owns one, compilation-pipeline entry
   points; and
#. code-generation helpers that translate backend calls to target source.

Registration imports some modules only for their side effects.  Those modules
are implementation surfaces, not additional kernel-authoring APIs.


The Source Tree
---------------

CUDA support spans three directories. The public C++ declarations,
native implementations, and Python authoring interfaces have distinct
roles:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Directory
     - Responsibility
   * - ``include/tvm/backend/cuda/``
     - Public operator declarations, typed attributes, and backend contracts.
   * - ``src/backend/cuda/``
     - Native operators, lowering passes, layout axes, code generation, and runtime support.
   * - ``python/tvm_tirx/backend/cuda/``
     - Script namespaces, PTX tables and generators, tile dispatch, compiler configuration, and profiling orchestration.

Within the Python directory, ``script.py`` exposes ``T.cuda`` and
``T.ptx``. The backend's ``register_backend()`` installs these namespaces
and loads the CUDA axis, tile, and code-generation registrations.
The namespaces construct ordinary IR objects, including calls and device
regions, which the shared compiler pipeline can process.

From a Function to a Kernel
---------------------------

The ``scale`` function in :doc:`language/first_kernel` contains tensor
parameters, a device region, an index calculation, and a tensor
store. Its compilation follows this path:

.. code-block:: text

   TVMScript function
       -> IR function with a device region
       -> target binding and tile dispatch
       -> layout lowering and tensor flattening
       -> host/device splitting and ABI lowering
       -> host module + CUDA device source
       -> device compilation and runtime launch

``python/tvm_tirx/tirx/compilation_pipeline.py`` defines the shared
pipeline. ``TileDispatch`` replaces tile calls with their selected
implementations. ``LowerTensorLayouts`` maps logical accesses to their
thread and memory coordinates, and ``FlattenTensor`` converts tensor
accesses to linear storage accesses. A function without tile calls still
passes through these stages.

``SplitHostDevice`` separates the device region into a CUDA kernel and
leaves its launch in the host function. The finalization stages also
lower profiling annotations and prepare the calling convention.
The CUDA source generator in
``src/backend/cuda/codegen/codegen_cuda.cc`` emits the device code.

With ``scale`` defined, we can inspect the generated device source:

.. code-block:: python

   from tvm_tirx import tirx
   from tvm_tirx.target import Target

   target = Target({"kind": "cuda", "arch": "sm_75"})
   executable = tirx.compile(scale, target=target)
   device_module = executable.mod.imports[0]
   print(device_module.inspect_source("cuda"))

For this single-device program, the first imported module contains the
CUDA kernel. Its source uses ``blockIdx.x`` and ``threadIdx.x`` for the
index calculation to select the element processed by each thread. The explicit target
selects the compilation architecture; choose one supported by the GPU
and toolchain that will run the program.

Operators and Code Generation
-----------------------------

CUDA operator implementations live under ``src/backend/cuda/op/``.
They define argument requirements, result types, and effects. Python
helpers expose these operators through the IR and script interfaces.

The CUDA code generator handles ordinary expressions and statements
directly. For backend intrinsics, it can look up a Python code-generation
callback through ``tirx.intrinsics.cuda.get_codegen``. The registry in
``python/tvm_tirx/backend/cuda/codegen/registry.py`` connects an operator
name to its callback. A callback can supply a device helper and the call
that invokes it.

PTX instructions use this same connection. The instruction table and
engine under ``python/tvm_tirx/backend/cuda/ptx/`` register the instruction
families and generate CUDA helpers containing inline PTX. The
:doc:`ptx_tablegen` chapter follows one instruction through this path.
Tile patterns under ``python/tvm_tirx/backend/cuda/tile/patterns/`` lower
higher-level tensor operands to these instruction calls.

Compilation and Runtime
-----------------------

``python/tvm_tirx/backend/cuda/compiler.py`` compiles CUDA source through
NVCC or NVRTC. ``backend_config.py`` resolves the architecture, compiler,
output format, and tool arguments. These settings are retained with the
device module so a deferred compilation uses the same configuration.

The implementations under ``src/backend/cuda/runtime/`` manage device
memory, module loading, and kernel launches. The launch configuration
connects the program's grid and block dimensions to the CUDA launch.
A build without CUDA runtime support can retain device source for later
compilation; that source module alone cannot execute a GPU kernel.

Where to Make a Change
----------------------

The failing stage usually narrows the implementation to inspect:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Starting point
   * - Add a PTX form
     - ``ptx/table.py``, its validation rules, and the PTX code-generation tests.
   * - Add a tensor instruction pattern
     - ``tile/patterns/`` and tests under ``tests/python/tirx/tile/cuda/``.
   * - Change CUDA axis semantics
     - ``src/backend/cuda/layout/axis.cc`` and the Python axis namespace.
   * - Change source compilation
     - ``compiler.py``, ``backend_config.py``, and ``tvm_tirx.support.nvcc``.
   * - Change device loading or launching
     - ``src/backend/cuda/runtime/`` and ``src/backend/cuda/launch/``.

The :doc:`CUDA backend API </reference/python/backend.cuda>` links the
public namespaces. For kernel authoring, continue with
:doc:`language_basics`.
