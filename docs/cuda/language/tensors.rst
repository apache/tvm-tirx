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

.. Adapted from Apache TVM:
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/data_types.rst
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/tensors.rst

Types, tensors and memory
=========================

In this chapter, we will look at how CUDA kernels represent values and manage
their storage. A tensor's dtype describes its elements, and its scope selects
the memory space. We will extend the add-one computation with vector accesses,
shared storage, and thread-private values, then create views of existing data.

Element types and vector accesses
---------------------------------

Tensor elements and intermediate values can be integers, floating-point
values, or booleans. Common TIRx dtypes have the following representations
in generated CUDA source:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - TIRx dtype
     - Generated CUDA representation
   * - ``float32``
     - ``float``
   * - ``float16``
     - ``half``
   * - ``bfloat16``
     - ``nv_bfloat16``
   * - ``int32``
     - ``int``
   * - ``uint8``
     - ``uchar`` (an alias for ``unsigned char``)
   * - ``bool``
     - ``bool``
   * - ``float32x4``
     - ``float4``

The ``x4`` suffix describes four float32 lanes within one vector value.
The ``add_one_four`` function in :doc:`functions` processes four elements
per thread with a loop. For a contiguous tensor of 256 elements, we can
also read and compute those four elements as a vector:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def add_one_vector(
        A: T.Tensor((256,), "float32", align=16),
        B: T.Tensor((256,), "float32", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        vector_value = A[T.Ramp(tx * 4, 1, 4)]
        B[T.Ramp(tx * 4, 1, 4)] = vector_value + 1

``T.Ramp(base, stride, lanes)`` selects evenly spaced indices. The index
above selects ``4*tx``, ``4*tx + 1``, ``4*tx + 2``, and ``4*tx + 3``.
The tensors have float32 elements, while the load produces one
``float32x4`` intermediate. The addition adds one to each lane, and the
store writes those four results to the corresponding positions in ``B``.

``align=16`` declares 16-byte-aligned data pointers. Each four-element
group also starts at a multiple of 16 bytes, allowing aligned vector loads
and stores. The 64 threads cover all ``64 * 4 = 256`` elements.

Declaring tensors
-----------------

A parameter such as ``A: T.Tensor((128,), "float32")`` describes storage
supplied by the caller. Inside a kernel, ``T.alloc_shared`` and
``T.alloc_local`` allocate scratch tensors. Both are conveniences for
``T.alloc_tensor``, whose tensor type specifies the shape, dtype, and scope.
The ``scope`` selects the memory space:

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Scope
     - Typical interface
     - Storage
   * - ``"global"``
     - Tensor parameters (default)
     - Device global memory
   * - ``"shared"``
     - ``T.alloc_shared``
     - Shared by threads in a block; size fixed at compile time
   * - ``"shared.dyn"``
     - ``SMEMPool``
     - Shared by threads in a block; reserved at launch
   * - ``"local"``
     - ``T.alloc_local``
     - Private to each thread
   * - ``"tmem"``
     - ``T.cuda.decl_tmem``
     - Blackwell tensor memory, reserved separately

When storage already exists, ``T.decl_tensor`` creates a tensor view over
its pointer. For example, these statements inside a device region allocate
one shared tensor and declare another view of the same elements:

.. code-block:: python

    shared = T.alloc_shared((128,), "float32")
    shared_view = T.decl_tensor(
        shared.data, ty_args=[T.Tensor((128,), "float32", scope="shared")]
    )

``shared.data`` supplies the base pointer. Both names refer to the same
storage, so a write through ``shared_view`` is also visible through
``shared``. The declaration does not allocate or copy data.

The tensor's ``layout`` determines how logical indices map to storage;
``align`` specifies data-pointer alignment in bytes. The
:doc:`Tensor Layout </cuda/layouts>` guide covers explicit layouts, and the
:class:`TensorType API <tvm_tirx.tirx.TensorType>` lists the full parameters.

Shared memory
-------------

Static shared memory
~~~~~~~~~~~~~~~~~~~~

``T.alloc_shared`` creates a shared tensor with a size known at compile
time. The following kernel lets each thread read the value staged by its
neighbor, wrapping the last thread back to the first:

.. code-block:: python

    @T.function
    def shared_shift(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        shared = T.alloc_shared((128,), "float32")
        shared[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = shared[(tx + 1) % 128]

Each thread first writes one input element to ``shared``. The
``T.cuda.cta_sync()`` barrier waits until all threads in the block have
reached it, making those writes visible before the neighboring reads.
For example, thread 0 writes ``A[0]`` but reads the value that thread 1
stored from ``A[1]``.

Dynamic shared memory
~~~~~~~~~~~~~~~~~~~~~

Dynamic shared memory is reserved when the kernel launches. TIRx uses one
backing allocation, with tensor views occupying different regions within
it. :class:`SMEMPool <tvm_tirx.backend.cuda.lang.SMEMPool>` manages the offsets
and total size. Inside a device region, we can reserve two scratch tensors:

.. code-block:: python

    from tvm_tirx.backend.cuda.lang import SMEMPool

    pool = SMEMPool()
    left = pool.alloc((64,), "float32")
    right = pool.alloc((64,), "float32")
    pool.commit()

``left`` and ``right`` occupy separate regions of the same allocation.
``pool.commit()`` records the required size after the allocations, so the
launcher can reserve enough shared memory. The ``SMEMPool`` API describes
alignment, storage reuse, and specialized allocation helpers. Additional
launch settings are documented in the
:doc:`CUDA launch API </reference/python/backend.cuda.launch>`.

Per-thread local storage
------------------------

``T.alloc_local`` creates storage private to each thread. For example,
each of 64 threads can keep four input elements in a local array before
adding one and writing them back:

.. code-block:: python

    @T.function
    def add_one_local(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        values = T.alloc_local((4,), "float32")
        for k in T.unroll(4):
            values[k] = A[tx * 4 + k]
        for k in T.unroll(4):
            B[tx * 4 + k] = values[k] + 1

Every thread has its own ``values`` array. The CUDA compiler may place its
elements in registers when the accesses permit; local scope alone does not
guarantee register storage.

Mutable scalars and ``let``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An accumulator needs to change as a loop runs. A type-annotated assignment
such as ``total: T.float32 = 0.0`` creates a mutable scalar backed by a
one-element local tensor. A computed index that stays fixed can instead
use an immutable ``T.let`` binding. This version sums each thread's four
input elements into one output:

.. code-block:: python

    @T.function
    def sum_four(A: T.Tensor((256,), "float32"), B: T.Tensor((64,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        base: T.let[T.int32] = tx * 4
        total: T.float32 = 0.0
        for k in range(4):
            total += A[base + k]
        B[tx] = total

``base`` names the first element assigned to the thread and cannot be
reassigned. ``total`` accumulates the four values through repeated updates.
The :doc:`TIRx script API </reference/python/tirx.script>` also provides
``local_scalar``, ``shared_scalar``, and ``alloc_scalar`` for explicit
scalar allocation.

Tensor views and pointers
-------------------------

Tensor variables provide methods for accessing addresses and creating
views of existing storage. These operations are useful when an intrinsic
needs a pointer or a computation expects a different arrangement of the
same elements:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Expression
     - Meaning
   * - ``A.data``
     - The base data pointer
   * - ``A.ptr_to([i, j])``
     - A pointer to element ``A[i, j]``, with its element type and storage scope
   * - ``A.sub[i, :]``
     - A tensor region selecting row ``i``, for use as an instruction operand
   * - ``A.view(*shape)``
     - A view with a different shape
   * - ``A.permute(*dims)``
     - A view with the axes reordered
   * - ``A.rearrange(pattern, **sizes)``
     - A view that splits, reorders, or merges axes using a named pattern
   * - ``A.view("uint32")``
     - A view of the same bytes as 32-bit unsigned words

For a contiguous one-dimensional tensor ``A`` with 256 elements, we can
view the data as a matrix and transpose that view:

.. code-block:: python

    matrix = A.view(64, 4)
    transposed = matrix.permute(1, 0)

``matrix`` has shape ``(64, 4)`` and ``transposed`` has shape ``(4, 64)``.
``matrix[i, j]`` and ``transposed[j, i]`` both refer to ``A[i * 4 + j]``.

A tile instruction may expect separate axes for warps and lanes. We can
split the 64 rows into two groups of 32 with ``rearrange``:

.. code-block:: python

    grouped = matrix.rearrange("(warp lane) value -> warp lane value", warp=2, lane=32)

The left side describes the current shape: ``(warp lane)`` is one axis
of length ``2 * 32``, followed by the four-value axis. The right side
exposes all three axes, giving ``grouped`` shape ``(2, 32, 4)``.
``grouped[w, l, j]`` refers to ``matrix[32*w + l, j]``. The pattern names
the coordinates; the tensor's layout determines their physical placement.

``sub`` selects a region to pass to an operation. For example,
``matrix.sub[i, :]`` selects its four-element row ``i``. A tile load can
then read that region into registers, as shown in :doc:`/cuda/tile_ops/data_movement`.

A dtype view changes the storage units used to access the bytes. Inside
a device region, eight float16 elements can be viewed as four uint32 words:

.. code-block:: python

    half_values = T.alloc_local((8,), "float16")
    packed_words = half_values.view("uint32")

``packed_words`` has shape ``(4,)``. Each word contains the bits of two
adjacent float16 elements. This is the representation used by instructions
that accept packed half values; a numeric conversion uses ``cvt`` instead,
as shown in :doc:`/cuda/tile_ops/arithmetic`.

All these views and regions reuse the original storage. The
:class:`TensorType API <tvm_tirx.tirx.TensorType>` covers the available
methods and options, while :doc:`/cuda/layouts` explains explicit storage
mappings, including layouts distributed across threads.

Tensor memory
-------------

Blackwell tensor memory uses ``scope="tmem"`` and must be reserved and
released explicitly. ``T.cuda.decl_tmem`` declares a tensor at an allocated
tensor-memory address; access uses ``tcgen05`` instructions. The
:doc:`Blackwell Tensor Core guide </cuda/tile_ops/blackwell_tensor_cores>` shows the
allocation, synchronization, transfer, and release steps together. The
:func:`decl_tmem API <tvm_tirx.backend.cuda.script.decl_tmem>` describes its
arguments.
