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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/native_basics.rst

CUDA Language Basics
====================

In this chapter, we will introduce native-level CUDA kernel authoring:
thread hierarchy, memory scopes, device intrinsics, and the compile, run,
and inspect loop. The examples follow Apache TVM's CUDA programming guide,
with interfaces adapted to the standalone ``tvm_tirx`` package.

What "native level" means
-------------------------

A native-level TIRx kernel reads like a structured device kernel: you place
threads yourself, allocate shared/per-thread local tensors, write loops and barriers, and
call device intrinsics directly. You explicitly choose the orchestration and
layouts; standard lowering still dispatches primitives, applies layouts, and
vectorizes or unrolls marked loops. This is the foundation the tile ops
(:doc:`/cuda/tile_operations`) are built on; everything here is what those ops
ultimately lower to, so it is also where you go when a hardware feature does not
have a primitive yet.

The authoring model
-------------------

- ``@Tx.function`` (or ``@Tx.jit`` for compile-time-specialized) kernels, written
  with ``from tvm_tirx.script import tirx as Tx``;
- ``Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(32,)))`` plus *scope-id* intrinsics for thread binding;
- ``Tx.Tensor`` parameter annotations and ``Tx.alloc_*`` scratch tensors;
- ordinary loops, branches, and scalar math;
- ``tvm_tirx.tirx.compile(mod, target=..., pipeline="tirx")`` to build, then call the
  result directly.

The examples use these imports. The ``__future__`` import lets ``@Tx.jit``
kernels reference compile-time parameters inside type annotations (see
:doc:`/cuda/language/functions`); it is harmless for ordinary kernels::

    from __future__ import annotations
    import tvm_tirx
    from tvm_tirx.script import tirx as Tx

Start here
----------

The first example scales a vector by two. It covers the complete
write, compile, run, and inspect loop before the language chapters expand
each part. The examples require CUDA support; see
:doc:`the source build guide </dev/source_build>` for installation.

.. toctree::
   :maxdepth: 1

   language/first_kernel

Language guide
--------------

.. toctree::
   :maxdepth: 1

   language/functions
   language/data_types
   language/tensors
   language/control_flow

CUDA execution and compilation
------------------------------

.. toctree::
   :maxdepth: 1

   language/threads_sync
   language/compiling

Advanced topics
---------------

.. toctree::
   :maxdepth: 1

   language/parser_utils


:doc:`The IKET profiling guide </cuda/iket_profiling>` covers the current
backend's event-tracing and capture interfaces.
