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

In this chapter, we will follow the GPU add-one program from
:doc:`Quick Start </get_started/quick_start>` through the CUDA backend.
The compilation stages connect the program's IR to device code and its
runtime launch. Their source locations help identify where to add an
instruction or investigate a lowering problem.

The source tree
---------------

CUDA support has three main parts:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Directory
     - Responsibility
   * - ``include/tvm/backend/cuda/``
     - Public C++ operator declarations, attributes, and backend contracts.
   * - ``src/backend/cuda/``
     - Native operators, lowering passes, code generation, and runtime support.
   * - ``python/tvm_tirx/backend/cuda/``
     - Script namespaces, PTX instruction tables, tile patterns, compiler configuration, and profiling orchestration.

The Python backend's ``register_backend()`` installs the ``T.cuda`` and
``T.ptx`` namespaces and loads its axis, tile, and code-generation
registrations. ``script.py`` supplies the CUDA authoring interfaces;
their calls and device regions use the shared IR object system.

From TVMScript to a CUDA kernel
-------------------------------

The GPU add-one function contains a device region, an index calculation,
and a guarded tensor store. Its compilation follows this path:

.. code-block:: text

   TVMScript -> IR function with a device region
       -> target binding and tile dispatch
       -> layout lowering and tensor flattening
       -> host/device splitting and calling convention
       -> host module + CUDA device source
       -> device compilation and runtime launch

``python/tvm_tirx/tirx/compilation_pipeline.py`` defines the shared
pipeline. ``TileDispatch`` expands tile operations into their selected
implementations. ``LowerTensorLayouts`` maps logical tensor accesses to
threads and storage, and ``FlattenTensor`` converts them to linear
accesses. In add-one, there are no tile calls to expand; the accesses to
``A[i]`` and ``B[i]`` still go through layout lowering and flattening.

``SplitHostDevice`` extracts the device region as a kernel and leaves
its launch in the host function. ``LowerIket`` then handles profiling
annotations when present, and ``MakePackedAPI`` prepares the host calling
convention. The CUDA source generator in
``src/backend/cuda/codegen/codegen_cuda.cc`` emits the device code,
including ``blockIdx.x`` and ``threadIdx.x`` for add-one's index.
The :doc:`compiling and inspecting guide <language/compiling>` shows how
to retrieve that source from a compiled program.

Instruction calls and execution
-------------------------------

Native CUDA operators under ``src/backend/cuda/op/`` define their
argument requirements, result types, and effects. For an intrinsic call,
the CUDA source generator can retrieve a Python callback through
``tirx.intrinsics.cuda.get_codegen``. The registry in
``python/tvm_tirx/backend/cuda/codegen/registry.py`` connects operator
names to callbacks that emit a CUDA expression or a device helper call.

PTX instructions use this mechanism to emit helpers containing inline
assembly. The :doc:`PTX Tablegen <ptx_tablegen>` chapter follows one
instruction from its table entry to that helper. Tile patterns in
``python/tvm_tirx/backend/cuda/tile/patterns/`` prepare instruction calls
from tensor regions before source generation.

``compiler.py`` compiles the CUDA source through NVCC or NVRTC, using
the architecture and options resolved by ``backend_config.py``. The
implementations under ``src/backend/cuda/runtime/`` load the resulting
device module, manage memory, and launch kernels with the requested
grid and block dimensions.

Where to make a change
----------------------

These source locations provide starting points for an extension or a
failing compilation stage:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Starting point
   * - Add a PTX instruction form
     - ``ptx/table.py`` and ``tests/python/tirx/codegen/test_ptx_dialect.py``.
   * - Add a tensor instruction pattern
     - ``tile/patterns/`` and ``tests/python/tirx/tile/cuda/``.
   * - Change source compilation
     - ``compiler.py``, ``backend_config.py``, and ``tvm_tirx.support.nvcc``.
   * - Change CUDA axis semantics
     - ``src/backend/cuda/layout/axis.cc`` and the Python ``axis`` namespace.
   * - Change device loading or launching
     - ``src/backend/cuda/runtime/`` and ``src/backend/cuda/launch/``.

The :doc:`CUDA backend API </reference/python/backend.cuda>` documents
the public interfaces in these modules.
