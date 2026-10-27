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

PTX Tablegen
============

In this chapter, we will introduce the table-driven PTX interface and
follow an instruction from its TVMScript spelling to generated inline
assembly. The same table supplies operand contracts, modifier choices,
and generated helpers, so a backend developer can add an instruction
form without duplicating this information across several interfaces.

A Direct PTX Instruction
------------------------

PTX instructions name their destination registers explicitly. The
following kernel uses ``add.rn.f32`` to add two float32 values in each
thread. The ``rn`` modifier selects round-to-nearest-even:

.. code-block:: python

   from tvm_tirx.script import tirx as T

   @T.function
   def add_ptx(
       A: T.Tensor((32,), T.float32),
       B: T.Tensor((32,), T.float32),
       C: T.Tensor((32,), T.float32),
   ):
       with T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,))):
           tid = T.cuda.thread_idx("x")
           result = T.local_scalar("float32")
           T.ptx.add.rn.f32(result, A[tid], B[tid])
           C[tid] = result

``result`` declares the destination. The PTX call writes it, and the
following tensor store writes that value to ``C``. The instruction call
itself has no return value, so it appears as a statement.

The indexed spelling ``T.ptx["add.rn.f32"]`` selects the same instruction.
It is convenient for names containing tokens such as ``global``, which
need a Python keyword escape in attribute chains: ``T.ptx.ld.global_.b32``.
The selected instruction must include the modifiers required by its form.

Instruction forms
-----------------
Attribute chains fill PTX modifier slots.  Python keywords have a trailing
underscore, and PTX's ``::`` separator is written as a double underscore::

   T.ptx.ld.acquire.gpu.global_.b32(value, pointer)
   T.ptx.mbarrier.arrive.shared__cluster.b64(barrier, count)

The indexed form accepts the exact PTX spelling of a registered form and uses
the same table lookup::

   T.ptx["st.weak.shared::cta.b32"](pointer, value)

Destination registers are leading operands, so PTX calls are statements.
Predication is keyword-only::

   T.ptx.ld.global_.b32(value, pointer, pred=predicate, preserve_dst=True)

``T.ptx.pred(value)`` marks a register operand whose PTX type is ``.pred``;
``T.ptx.addr(base, byte_offset)`` forms an immediate-offset address for
instructions that accept one; and ``T.ptx.SINK`` represents PTX's sink operand
``_`` in table-marked positions.  The stub generator described below provides editor completion for
registered instruction families and modifier chains.

PTX versus CUDA helpers
-----------------------
Use ``T.ptx`` for one table-described PTX instruction.  Use ``T.cuda`` for
backend helpers that require multiple statements, C/C++ expressions, descriptor
packing, or other behavior that cannot be represented as one PTX instruction.
``T.ptx_legacy`` preserves operation construction for downstream emitters.
New kernels should use the direct ``T.ptx`` interface.


The Instruction Table
---------------------

``python/tvm_tirx/backend/cuda/ptx/table.py`` describes each instruction
shape with an ``InstructionEntry``. We can inspect the existing add entry:

.. code-block:: python

   from tvm_tirx.backend.cuda.ptx.table import TABLE

   entry = TABLE["add"]
   print(entry.op_name)
   print([operand.name for operand in entry.operands])
   print([slot.name for slot in entry.slots])

The output is:

.. code-block:: text

   tirx.ptx.add
   ['d', 'a', 'b']
   ['rnd', 'ftz', 'sat', 'type', 'srctype']

The operands describe the destination ``d`` and sources ``a`` and ``b``.
Modifier slots describe the available rounding, flush-to-zero, saturation,
and type choices; some slots are optional. The engine resolves
``add.rn.f32`` to this entry with concrete modifier values.

The entry's main components are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Component
     - Role
   * - ``operands``
     - Operand order, register or address kind, read/write direction, dtype, and register-group size.
   * - ``slots``
     - Named modifier positions, legal tokens, and whether a position is optional.
   * - ``check``
     - Cross-modifier rules that reject combinations of otherwise valid tokens.
   * - ``imm_check``
     - Validation for concrete immediate operands.
   * - ``asm_volatile``, ``orders_memory``
     - Inline-assembly retention and ordering requirements.

An entry represents one operand shape. Forms with different register
groups can have separate entries while sharing a user-facing mnemonic.

From the Table to CUDA Source
-----------------------------

The tablegen system is implemented in Python. Its main pieces are:

.. code-block:: text

   table.py       instruction shapes, operands, and modifier rules
       -> engine.py      namespace selection, IR calls, and codegen registration
       -> render.py      CUDA helpers containing inline PTX

The engine registers an operator for each entry and constructs a void
``ir.Call`` for a selected instruction. The CUDA code generator invokes
the registered callback, which renders the helper for the concrete
modifiers and operand types. For the kernel above, the helper contains:

.. code-block:: cuda

   asm volatile("add.rn.f32 %0, %1, %2;"
                : "=f"(__d) : "f"(__a), "f"(__b));

The output constraint writes ``__d``; the two input constraints read the
float operands. NVCC or NVRTC processes this CUDA source, and the CUDA
assembler validates the instruction for the selected architecture.

Generated Helpers and Coverage
------------------------------

The generators make the table inspectable without writing a kernel.
From the repository root, these commands produce helper source, Python
signatures, and a coverage summary:

.. code-block:: bash

   python -m tvm_tirx.backend.cuda.ptx.gen_helpers add -o /tmp/ptx_add.cuh
   python -m tvm_tirx.backend.cuda.ptx.gen_stubs -o /tmp/ptx_stubs.pyi
   python -m tvm_tirx.backend.cuda.ptx.gen_coverage > /tmp/ptx_coverage.txt

``gen_helpers`` enumerates the selected family's valid forms and renders
their helpers. ``gen_stubs`` exposes signatures for editor tooling, while
``gen_coverage`` summarizes the table's families and variants. Runtime
code generation uses the table directly; these files are inspection
artifacts.

A New Instruction Form
----------------------

A new form starts with its PTX operand contract: operand order, register
widths and groups, address spaces, and modifiers. The entry then describes
those operands and slots, with ``check`` or ``imm_check`` for constraints
that cannot be expressed as individual token choices. Existing entries
provide examples for arithmetic, loads, stores, and collective operations.

The new form also needs a code-generation test that constructs a kernel
and checks its emitted instruction. The tests in
``tests/python/tirx/codegen/test_ptx_dialect.py`` cover registration,
argument handling, helper generation, and assembly. A source check can
confirm the intended spelling; an assembler test must also exercise the
form on a supporting target. The table's ``cert_arch`` selects an
architecture for certification tests and does not replace that check.

The :doc:`PTX API </reference/python/backend.cuda.ptx>` exposes instruction
lookup and inspection. The next chapters connect instruction operands
to :doc:`CUDA layouts <layouts>` and :doc:`tile operations <tile_operations>`.
