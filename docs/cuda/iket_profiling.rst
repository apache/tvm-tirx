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

In this chapter, we will add NVIDIA IKET events to a kernel and capture
a timeline of its execution. Named ranges and checkpoints help relate
execution time to individual phases of the computation. We will use a
small affine kernel to demonstrate the annotation and capture workflow.

The example requires a CUDA-enabled TIRx build, an SM90-or-newer GPU,
NVRTC through ``cuda-bindings``, and the ``run-iket`` executable supplied
by ``nvidia-cutlass-dsl-libs-base``.

Annotate a computation
----------------------

Save the following kernel and the host code below in
``profile_affine.py``. Each thread computes ``B[i] = 2 * A[i] + 1``:

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

``IketProfiler`` inserts the event annotations. ``range_push("affine")``
begins a named range, and ``range_pop()`` ends it. The ``loaded`` mark
places a checkpoint after the tensor read. These events will appear in
the captured trace. They do not replace synchronization needed by the
kernel's data dependencies.

Capture the kernel
------------------

The host workload compiles the instrumented kernel, runs it, and checks
its result. GPU initialization, compilation, and execution belong inside
``workload`` because IKET replays this callable during its tracking and
capture passes:

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
``IketProfiler.compile`` enables the profiling lowering, while
``iket.run`` replays the script under NVIDIA's tracker and capture tools.
After the capture completes, the parent process receives the result and
prints the artifact paths. Use a fresh output directory for each capture,
or pass ``clobber=True`` to replace an earlier capture.

Read the trace
--------------

``result.perfetto`` identifies the timeline file, ``result.html`` the
HTML report, and ``result.trace_json`` the JSON trace. The timeline shows
the ``affine`` range and the ``loaded`` checkpoint within the kernel's
execution. Instrumentation adds work, so measure the uninstrumented
kernel separately when assessing its overall performance.

The :doc:`IKET API </reference/python/backend.cuda.iket>` covers token-based
ranges, command-based capture with ``iket.profile``, and capture options.
The :doc:`backend implementation <backend_implementation>` explains where
the profiling annotations are lowered in the compilation pipeline.
