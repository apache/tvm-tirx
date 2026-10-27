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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/functions.rst

Defining a function
===================

The GPU example in :doc:`Quick Start </get_started/quick_start>` defines its
kernel with ``@Tx.function``. This chapter covers parameter types, symbolic
shapes, compile-time specialization with ``@Tx.jit``, and CUDA launch
configuration.

Declaring tensor parameters
---------------------------

Declare tensor parameters with ``Tx.Tensor`` annotations. The annotation accepts
shape, dtype, layout, scope, and alignment metadata:

.. code-block:: python

    from tvm_tirx.script import tirx as Tx

    @Tx.function
    def copy(A: Tx.Tensor((256,), "float32", align=16), B: Tx.Tensor((256,), "float32")):
        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=1, block=256))
        tx = Tx.cuda.thread_idx("x")
        B[tx] = A[tx]

The parameters are tensors that you index with ``A[i]`` or ``A[i, j]``.
Annotations also support :ref:`symbolic shapes <cuda-symbolic-shapes>`.

What the parameter list accepts
-------------------------------

A ``Function`` parameter is one of the following. The third column is what you
pass on the Python side when you call the compiled ``Executable``:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Annotation
     - Is
     - Pass at call time
   * - ``Tx.Tensor((d0, d1), dtype)``
     - a tensor with a declared shape and dtype
     - a tensor on the right device
   * - ``Tx.handle``
     - an opaque handle
     - a handle value
   * - ``Tx.int32`` / ``Tx.float32`` / …
     - a runtime scalar
     - a Python ``int`` / ``float``
   * - ``Tx.constexpr`` (``@Tx.jit`` only)
     - a compile-time constant
     - supplied to ``.specialize(...)``, **not** at the call

Tensors may be CUDA ``torch`` tensors (zero-copy through TVM FFI tensor
interop) or
``tvm_tirx.runtime.tensor(...)``. Arguments are positional and match the parameter
order. For example, a kernel with a scalar parameter::

    @Tx.function
    def scal(A: Tx.Tensor((256,), 'float32'), B: Tx.Tensor((256,), 'float32'), s: Tx.float32):


        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        B[tx] = A[tx] * s

    exe(a, b, 3.0)        # pass the scalar as a Python float

.. _cuda-symbolic-shapes:

Symbolic shapes
---------------

For a length that varies at runtime, declare a symbol with ``Tx.dynamic("n")``
and use it in the tensor shapes. The same compiled function can then accept
different lengths:

.. code-block:: python

    import torch
    import tvm_tirx

    n = Tx.dynamic("n")

    @Tx.function
    def scale_dyn(A: Tx.Tensor((n,), "float32"), B: Tx.Tensor((n,), "float32")):
        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(1,)))
        for i in range(n):
            B[i] = A[i] * Tx.float32(2.0)


    exe = tvm_tirx.tirx.compile(
        tvm_tirx.IRModule({"main": scale_dyn}), target=tvm_tirx.target.Target("cuda"), pipeline="tirx"
    )
    exe(torch.rand(100, device="cuda"), torch.empty(100, device="cuda"))  # n = 100
    exe(torch.rand(200, device="cuda"), torch.empty(200, device="cuda"))  # n = 200, same kernel

At each call, the host launcher reads ``n`` from the input shape, checks that
the output has the same length, and passes the data pointers and ``n`` to the
device kernel. Both calls above use the same compiled function; the caller
supplies only the two tensors.

``@Tx.function`` vs ``@Tx.jit``
-------------------------------

- ``@Tx.function`` parses the function immediately into a ``Function``. Sizes are
  whatever you wrote — concrete ints, or runtime-symbolic vars (above).
- ``@Tx.jit`` returns a ``JITFunction`` and **defers** parsing until you call ``.specialize(**constexpr)``:
  parameters annotated ``Tx.constexpr`` are baked in as compile-time constants and
  the result is an ordinary ``Function``. Use it when you want sizes/flags fixed at
  compile time (so the compiler can unroll, statically size shared memory, etc.).
  Referencing a constexpr inside an annotation (e.g. ``Tx.Tensor((N,), ...)``)
  requires ``from __future__ import annotations`` at the top of the file.

.. code-block:: python

    from __future__ import annotations
    from tvm_tirx.script import tirx as Tx

    @Tx.jit
    def add(
        A: Tx.Tensor((N,), "float32"),
        B: Tx.Tensor((N,), "float32"),
        C: Tx.Tensor((N,), "float32"),
        *,
        N: Tx.constexpr,
    ):
        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(N,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        C[tx] = A[tx] + B[tx]


    kernel = add.specialize(N=256)  # -> a Function with N = 256 baked in

So: a **symbolic shape** is one kernel whose size is resolved at run time; a
**constexpr + jit** produces a specialized kernel per value, resolved at compile
time.

The :doc:`JIT API </reference/python/tirx.script.jit>` covers optional
parameters and the full specialization rules.

Launch parameters
-----------------

Device regions
~~~~~~~~~~~~~~

``Tx.device_entry`` begins a device region. A standalone call covers the
remaining statements in the enclosing body; a ``with`` block gives the region
an explicit boundary. Its ``launch`` argument specifies the execution geometry.

CUDA indices and launch configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch geometry is explicit and independent of the indices read by the kernel.
``grid`` counts CTAs, ``block`` counts threads per CTA, and ``cluster`` counts
CTAs per cluster. Each accepts an integer or one to three dimensions; omitted
trailing dimensions are one.

.. code-block:: python

    Tx.device_entry(
        launch=Tx.cuda.LaunchConfig(grid=(GM, GN), block=128),
        kernel_attrs=Tx.cuda.KernelAttributes(min_blocks_per_sm=2),
    )
    bx, by = Tx.cuda.block_idx("x"), Tx.cuda.block_idx("y")
    warp = Tx.cuda.warp_id()
    lane = Tx.cuda.lane_id()
    tid = Tx.cuda.linear_thread_id()

``block_idx`` and ``thread_idx`` accept ``"x"``, ``"y"``, or ``"z"``.
``warp_id`` identifies the warp within a CTA, and ``lane_id`` identifies the
thread within its warp. ``linear_thread_id`` follows CUDA's x-major order
for multidimensional blocks. Warp helpers require a static block size
divisible by 32. The :doc:`CUDA script API </reference/python/backend.cuda.script>`
lists the remaining index helpers.

For a cluster launch:

.. code-block:: python

    Tx.device_entry(launch=Tx.cuda.LaunchConfig(
        grid=(NUM_CLUSTERS * 2,), block=128, cluster=(2,),
    ))
    cid = Tx.cuda.cluster_id("x")
    cx = Tx.cuda.cluster_cta_id("x")

This launches two CTAs per cluster. ``cluster_id`` identifies the cluster,
while ``cluster_cta_id`` gives the CTA's coordinate within it. Cluster launches
require a supporting GPU.

``LaunchConfig`` supplies runtime execution settings. ``KernelAttributes``
supplies compile-time properties such as launch bounds, including
``min_blocks_per_sm`` in the earlier example. The
:doc:`launch API </reference/python/backend.cuda.launch>` describes additional
settings and their requirements.
