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

IKET Profiling
==============

In this chapter, we will introduce in-kernel event tracing with NVIDIA
IKET. A launch timer measures the whole kernel; event ranges and marks
help locate time spent within it. We will bracket a small affine
computation, run it under the official IKET tool, and locate the resulting
trace files.

The Profiling Interfaces
------------------------

The CUDA backend provides device annotations and host orchestration:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Interface
     - Role
   * - ``IketProfiler``
     - Construct event annotations inside TVMScript and compile an instrumented module.
   * - ``iket.run(workload, ...)``
     - Replay a saved Python entry point under the official tracker and capture processes.
   * - ``iket.profile(command, ...)``
     - Profile a separate replayable command.
   * - ``IketProfileResult``
     - Locate JSON, Perfetto, and HTML artifacts from a successful capture.

The current lowering requires an SM90 or newer GPU. The environment also
needs CUDA support, NVRTC through ``cuda-bindings``, and the official
``run-iket`` entry point supplied by ``nvidia-cutlass-dsl-libs-base``.
The host API validates that installation before starting a capture.
This chapter uses ``tvm_tirx.backend.cuda.iket``; the separate
``tvm_tirx.testing.bench.CudaProfiler`` uses a different recording API.

Annotations in a Kernel
-----------------------

Save the following kernel and the host code below in
``profile_affine.py``. The kernel computes ``B[i] = 2 * A[i] + 1`` for
32 values:

.. code-block:: python

   import torch
   from tvm_tirx.backend.cuda import iket
   from tvm_tirx.backend.cuda.iket import IketProfiler
   from tvm_tirx.script import tirx as T

   @T.function
   def profiled_affine(
       A: T.Tensor((32,), T.float32), B: T.Tensor((32,), T.float32)
   ):
       with T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,))):
           profiler = IketProfiler()
           tid = T.cuda.thread_idx("x")
           profiler.range_push("affine")
           value = A[tid]
           profiler.mark("loaded")
           B[tid] = T.float32(2) * value + T.float32(1)
           profiler.range_pop()

``range_push`` begins the named range and ``range_pop`` ends it.
The ``loaded`` mark identifies a point between the tensor read and the
remaining computation. These annotations construct IR calls; the
instrumented device code records their events during the capture.

The profiler supports two range forms:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Form
     - Use
   * - ``mark(name)``
     - Record a named checkpoint.
   * - ``range_push(name)`` / ``range_pop()``
     - Bracket nested regions in stack order.
   * - ``range_start(name)`` / ``range_end(token)``
     - Connect a start and end through the returned uint32 token.

The markers identify regions to measure. The kernel still needs the
synchronization required by its actual data dependencies.

A Capture Run
-------------

The host workload compiles the instrumented function, launches it, and
checks the output. GPU work belongs inside this callable because IKET
replays it during tracking and capture:

.. code-block:: python

   def workload():
       compiled = IketProfiler().compile(profiled_affine, target="cuda")
       a = torch.arange(32, dtype=torch.float32, device="cuda")
       b = torch.empty_like(a)
       compiled(a, b)
       torch.cuda.synchronize()
       torch.testing.assert_close(b, 2 * a + 1)

   if __name__ == "__main__":
       result = iket.run(
           workload, output_dir="iket_affine_trace", postprocess="all"
       )
       print(result.trace_json)
       print(result.perfetto)
       print(result.html)

Run the saved script with ``python profile_affine.py``.
``IketProfiler.compile`` enables the module's IKET lowering. The
``iket.run`` call starts the official tool's tracker and capture passes
with the saved script. After those passes finish, it returns the artifact
paths in the parent process. Each invocation of the example should use
a fresh output directory, or explicitly pass ``clobber=True`` to replace
an earlier capture.

A command-based workflow can instead call ``iket.profile`` from a
separate driver script with an argument list such as
``[sys.executable, "profile_workload.py"]``. That workload must compile
and run its instrumented kernel without starting another profile.

The Trace and the Lowering
--------------------------

For this single-kernel example, ``result.trace_json``, ``result.perfetto``,
and ``result.html`` identify the JSON trace, Perfetto timeline, and HTML
report. A workload with several artifacts can use the corresponding
``json_traces``, ``perfetto_traces``, and ``html_reports`` collections.
The timeline connects the ``affine`` range and ``loaded`` checkpoint to
the captured kernel execution. Instrumentation adds work, so compare
regions and measure the uninstrumented kernel separately for overall
performance.

The native ``LowerIket`` pass runs after host/device splitting. It checks
event declarations and token use, assigns event IDs, and emits the
metadata and device event helpers that the official IKET tools consume.
The implementation is in ``src/backend/cuda/transform/lower_iket.cc``;
Python orchestration lives in ``python/tvm_tirx/backend/cuda/iket.py``.

The :doc:`IKET API </reference/python/backend.cuda.iket>` describes capture
options and result fields.
