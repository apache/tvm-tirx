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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/threads_sync.rst

CUDA C++/PTX intrinsics
=======================

In this chapter, we will call CUDA device helpers and PTX instructions
directly, then include a device function written in CUDA C++. TVMScript
exposes these interfaces through ``Tx.cuda``, ``Tx.ptx``, and
``Tx.cuda.func_call``.

Calling backend intrinsics
--------------------------

The backend provides helpers for thread synchronization, data movement,
reductions, and matrix operations:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Interface
     - Purpose
   * - ``Tx.cuda.cta_sync()``
     - Synchronize the threads in a block.
   * - ``Tx.cuda.warp_sync()``
     - Synchronize a full warp.
   * - ``Tx.cuda.warpgroup_sync(8)``
     - Synchronize a 128-thread warpgroup using named barrier 8.
   * - ``Tx.cuda.__shfl_xor_sync(mask, value, lane_mask, width)``
     - Read a value from another lane selected by XOR.
   * - ``Tx.cuda.cta_sum(value, num_warps, scratch)``
     - Reduce a value across a block using a shared scratch pointer.

For example, a warp can sum its 32 input values with
``Tx.cuda.__shfl_xor_sync``:

.. code-block:: python

    from tvm_tirx.script import tirx as Tx

    @Tx.function
    def warp_reduce(A: Tx.Tensor((32,), "float32", align=16)):
        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=1, block=32))
        lane_id = Tx.cuda.lane_id()
        v = Tx.alloc_local((1,), "float32")
        i = Tx.alloc_local((1,), "int32")
        v[0] = A[lane_id]
        i[0] = 16
        while i[0] >= 1:
            v[0] += Tx.cuda.__shfl_xor_sync(0xFFFFFFFF, v[0], i[0], 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

All lanes participate, as selected by ``0xFFFFFFFF``. The XOR distances
16, 8, 4, 2, and 1 combine all 32 values; every lane receives the sum.
The device helper calls CUDA's ``__shfl_xor_sync``:

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

The :doc:`TMA guide </cuda/tile_ops/tma>` shows a complete barrier
initialization, arrival, and phase wait around an asynchronous transfer.

Other families under ``Tx.ptx.*`` / ``Tx.cuda.*``: ``cp.async`` (LDGSTS),
``cp.async.bulk.tensor`` (TMA), ``ldmatrix`` / ``stmatrix``, ``tcgen05.*``
(Blackwell MMA), ``atomic_add``, ``fence`` … See :doc:`/reference/python/backend.cuda` for CUDA
helpers and :doc:`/cuda/ptx_tablegen` for the registered PTX forms.

Inlining raw CUDA
-----------------

For something with no intrinsic at all, inject a ``__device__`` function from a
source string with ``Tx.cuda.func_call(name, *args, ..., ty=...)``:

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """


    @Tx.function
    def k(A: Tx.Tensor((256,), "float32"), B: Tx.Tensor((256,), "float32")):

        Tx.device_entry(launch=Tx.cuda.LaunchConfig(grid=(1,), block=(256,)))
        bx = Tx.cuda.block_idx("x")
        tx = Tx.cuda.thread_idx("x")
        B[tx] = Tx.cuda.func_call("my_relu", A[tx], SRC, ty="float32")

The source is emitted verbatim and the call is wired in:

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
