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

In this chapter, we will follow ``add.rn.f32`` from a TVMScript call to
generated inline PTX. An instruction table describes operands and valid
modifiers. The backend uses that definition to check calls and generate
CUDA helpers, so a new instruction form can share one definition across
these interfaces.

Call a PTX instruction
----------------------

PTX instructions name their destination registers explicitly. The kernel
below uses ``add.rn.f32`` to add two float32 values per thread. The
``rn`` modifier selects round-to-nearest-even:

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

``result`` supplies the destination register. The PTX call writes it,
and the next statement stores that value in ``C``. The instruction call
itself has no return value.

The indexed spelling ``T.ptx["add.rn.f32"]`` selects the same form.
Attribute chains escape Python keywords with a trailing underscore,
as in ``T.ptx.ld.global_.b32``; PTX's ``::`` becomes ``__``, as in
``T.ptx.mbarrier.init.shared__cta.b64``. Indexed names use the exact PTX
spelling. ``T.cuda`` also provides helpers for operations that need
multiple statements or descriptor construction.

Inspect its table entry
-----------------------

``python/tvm_tirx/backend/cuda/ptx/table.py`` describes each instruction
operand shape with an ``InstructionEntry``. We can inspect the add entry:

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

The operands are destination ``d`` and sources ``a`` and ``b``. Modifier
slots describe rounding, flush-to-zero, saturation, and type choices;
some are optional. The engine resolves ``add.rn.f32`` to this entry with
concrete modifier values and checks that the combination is supported.

From the entry to inline assembly
---------------------------------

Three Python modules connect the definition to generated code:

.. code-block:: text

   table.py   instruction operands and modifier rules
       -> engine.py   instruction selection, IR calls, and codegen registration
       -> render.py   CUDA helpers containing inline PTX

The engine registers an operator for each entry and constructs a void
``ir.Call`` for the selected instruction. During CUDA source generation,
the registered callback renders a helper for its modifiers and operand
types. For the kernel above, that helper contains:

.. code-block:: cuda

   asm volatile("add.rn.f32 %0, %1, %2;"
                : "=f"(__d) : "f"(__a), "f"(__b));

The output constraint writes ``__d``, and the input constraints read the
two float operands. NVCC or NVRTC compiles the CUDA helper, and the
assembler checks support for the selected GPU architecture. To inspect
the add family's helpers without writing a kernel, run:

.. code-block:: bash

   python -m tvm_tirx.backend.cuda.ptx.gen_helpers add -o /tmp/ptx_add.cuh

Add an instruction form
-----------------------

A new form starts with its PTX operand order, register widths and groups,
address spaces, and modifiers. Add an entry describing that contract,
with ``check`` or ``imm_check`` for constraints that cannot be expressed
as individual modifier choices. A form with different register groups
may need a separate entry under the same user-facing mnemonic.

Add a code-generation test in
``tests/python/tirx/codegen/test_ptx_dialect.py`` to check the emitted
instruction and assemble it for a supporting target. The entry's
``cert_arch`` selects the target for certification tests. The
:doc:`PTX reference </reference/python/backend.cuda.ptx>` describes entry
fields, special operands, and generators for editor stubs and coverage
reports.
