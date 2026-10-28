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
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/functions.rst
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/control_flow.rst
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/parser_utils.rst

Defining and launching kernels
==============================

In this chapter, we will define CUDA functions, configure their launches,
and assign work to threads. We will use the GPU add-one program from
:doc:`Quick Start </get_started/quick_start>`, then extend it with a runtime
parameter, a loop, and compile-time specialization.

A CUDA function
---------------

The ``@T.function`` decorator constructs a ``tirx.Function`` from a
TVMScript definition. The add-one function takes two tensors with the same
symbolic length ``n``:

.. code-block:: python

    from tvm_tirx import tirx
    from tvm_tirx.script import tirx as T

    n = T.dynamic("n")

    @T.function
    def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,)))
        i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
        if i < n:
            B[i] = A[i] + 1

``A`` supplies the input and ``B`` receives the result. The body describes
the launch, the index assigned to each thread, and the guarded tensor store.

Parameters
----------

The parameter annotations specify what the caller supplies:

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Annotation
     - Meaning
     - Argument
   * - ``T.Tensor(shape, dtype)``
     - A tensor with the declared shape and dtype
     - A tensor on the expected device
   * - ``T.int32``, ``T.float32``, etc.
     - A runtime scalar
     - A Python ``int`` or ``float``
   * - ``T.handle``
     - An opaque handle
     - A handle value
   * - ``T.constexpr`` (``@T.jit``)
     - A compile-time parameter
     - Supplied to ``.specialize(...)``

Runtime arguments are positional and follow the parameter order. Tensors
can be CUDA PyTorch tensors or TIRx runtime tensors. For example, a scaling
function can accept its multiplier as a runtime float32 parameter:

.. code-block:: python

    @T.function
    def scale(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
        factor: T.float32,
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,)))
        i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
        if i < n:
            B[i] = A[i] * factor

For an input of 100 elements, we can supply ``factor`` after the two
tensor arguments and check the scaled values:

.. code-block:: python

    import torch

    a = torch.arange(100, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    scale_compiled = tirx.compile(scale, target="cuda")
    scale_compiled(a, b, 3.0)
    torch.testing.assert_close(b, 3 * a)

.. _cuda-symbolic-shapes:

Symbolic shapes
---------------

The symbol ``n`` appears in both parameter shapes and in the computation.
Its value comes from the input shape when the compiled function is called;
the output must have the same length. One compiled function can serve
multiple lengths:

.. code-block:: python

    add_one_compiled = tirx.compile(add_one, target="cuda")
    for length in (100, 200):
        a = torch.arange(length, dtype=torch.float32, device="cuda")
        b = torch.empty_like(a)
        add_one_compiled(a, b)
        torch.testing.assert_close(b, a + 1)

The host launcher reads ``n`` from the tensor shape and passes it to the
device kernel. The caller supplies only ``a`` and ``b``; both calls above
reuse ``add_one_compiled``.

Launch configuration
--------------------

``T.device_entry`` begins the device computation. As a standalone
statement, it covers the remaining statements in the enclosing body. An
explicit ``with T.device_entry(...):`` block sets the boundary through
indentation; :ref:`tvmscript-region-syntax` explains both forms.

In ``add_one``, ``block=(256,)`` launches 256 threads per block, and
``grid=((n + 255) // 256,)`` launches enough blocks to cover ``n`` elements.
``block_idx("x")`` gives the block index and ``thread_idx("x")`` the thread
index within that block. Their combination gives each thread its element
index ``i``. The ``if i < n`` condition skips any threads beyond the tensor.

``grid`` and ``block`` accept an integer or up to three dimensions.
``LaunchConfig`` also supports cluster and stream settings, while
``KernelAttributes`` supplies compile-time kernel properties such as launch
bounds. The :doc:`CUDA launch API </reference/python/backend.cuda.launch>`
covers these options; the
:doc:`CUDA script API </reference/python/backend.cuda.script>` lists the
coordinate helpers for blocks, threads, warps, and lanes.

.. _cuda-kernel-control-flow:

Conditions and loops
--------------------

A thread can process several elements with a loop. This version assigns
four consecutive elements to each thread, using 64 threads per block:

.. code-block:: python

    @T.function
    def add_one_four(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(64,)))
        base = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x") * 4
        for offset in T.unroll(4):
            i = base + offset
            if i < n:
                B[i] = A[i] + 1

Each block still covers up to ``64 * 4 = 256`` elements. ``T.unroll(4)``
requests expansion of the four iterations during lowering. The bounds
check applies to each element, so the last thread can handle a partial
group of four.

A regular ``range`` loop executes sequentially within each thread.
``T.vectorized`` marks a vectorized loop, and ``T.grid`` expresses nested
loops. Kernels also support ``if`` / ``else``, ``while``, ``break``, and
``continue``. Mutable counters and accumulators are described in
:doc:`tensors`; the :doc:`TIRx script API </reference/python/tirx.script>`
provides the loop helpers and their parameters.

Compile-time specialization
---------------------------

A symbolic dimension varies at runtime. When a size should be fixed before
compilation, ``@T.jit`` lets us declare it as ``T.constexpr`` and supply it
through ``.specialize(...)``. Parsing happens at specialization time, and
the result is an ordinary ``tirx.Function``.

The following definition specializes the add-one function for a fixed
length. Put ``from __future__ import annotations`` at the beginning of the
Python file so the annotation can refer to ``N`` before specialization:

.. code-block:: python

    from __future__ import annotations
    from tvm_tirx.script import tirx as T

    @T.jit
    def add_one_fixed(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        *,
        N: T.constexpr,
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((N + 255) // 256,), block=(256,)))
        i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
        if i < N:
            B[i] = A[i] + 1

    kernel = add_one_fixed.specialize(N=256)

``kernel`` has length 256 fixed in its parameter types and body. It can be
compiled with ``tirx.compile`` and called with its two tensor arguments;
``N`` is already supplied. In contrast, ``@T.function`` parses immediately,
and the symbolic ``n`` in ``add_one`` remains a runtime dimension. The
:doc:`JIT API </reference/python/tirx.script.jit>` covers optional
parameters and specialization rules.

.. _cuda-inline-functions:

Reusable code fragments
-----------------------

Several kernels may need the same bounds check and store. ``@T.inline``
lets us write that fragment once:

.. code-block:: python

    @T.inline
    def add_one_if_valid(A, B, i, length):
        if i < length:
            B[i] = A[i] + 1

The loop body in ``add_one_four`` can then use the helper:

.. code-block:: python

    for offset in T.unroll(4):
        add_one_if_valid(A, B, base + offset, n)

TVMScript expands the helper body at each call site during parsing, using
the supplied tensors, index, and length. The resulting kernel contains the
bounds checks and stores, with no separate runtime call to the helper.
The :func:`inline API <tvm_tirx.tirx.script.inline>` describes its options.
