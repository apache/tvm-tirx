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

.. _statements-and-control-flow:

Statements
==========

In this chapter, we will introduce statements, which describe the actions
a function performs and their execution order. A statement can bind an
intermediate value to a variable, write a result to a tensor, or control
execution through a branch or loop. These statements form the function's
body, which compiler passes can inspect and rewrite through the IR
interfaces.

.. _statements-and-sequences:

Statement Classes
-----------------

``ir.Stmt`` is the base class for all statements, from bindings and tensor
writes to sequences, branches, and loops:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Represents
     - Fields
   * - ``ir.Bind``
     - A variable definition
     - ``var``, ``value``
   * - ``ir.TensorStore``
     - A tensor element write; the destination type must support stores
     - ``dest``, ``indices``, ``value``
   * - ``ir.Evaluate``
     - Expression evaluation with its result discarded
     - ``value``
   * - ``ir.AssertStmt``
     - A program assertion
     - ``condition``, ``error_kind``, ``message_parts``
   * - ``ir.Return``
     - A function return
     - ``value``
   * - ``ir.Break``
     - Exit the current loop
     - None
   * - ``ir.Continue``
     - Start the next loop iteration
     - None
   * - ``ir.SeqStmt``
     - A sequence of statements in execution order
     - ``seq``
   * - ``ir.If``
     - Select a branch
     - ``condition``, ``then_case``, ``else_case``
   * - ``ir.For``
     - Repeat a body over an integer range
     - ``var``, ``begin``, ``end``, ``step``, ``kind``, ``body``, ``annotations``
   * - ``ir.While``
     - Repeat while a condition is true
     - ``condition``, ``body``
   * - ``ir.RegionStmt``
     - An operator-defined statement with a nested body
     - ``op``, ``args``, ``attrs``, ``body_params``, ``body``, ``result_vars``

Statement Sequences
-------------------

A function body often contains several statements that run one after
another. ``ir.SeqStmt`` stores these statements in execution order in
its ``seq`` field. The statements inside a loop or branch are also
stored in a ``SeqStmt``.

The CPU example writes one output element in each loop iteration:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at:     for i in range(n):
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

The indented assignment ``B[i] = A[i] + 1`` becomes an
``ir.TensorStore`` statement describing a write to a tensor element.
The loop body is a ``SeqStmt`` containing this statement. With the
tensor parameters ``A`` and ``B`` from :ref:`user-ir-var`, we can also
construct this loop body directly through the IR API, first creating
the tensor store and then placing it in a ``SeqStmt``:

.. code-block:: python

   i = ir.Var("i", "int32")
   store = ir.TensorStore(B, [i], A[i] + 1)
   body = ir.SeqStmt([store])

``ir.TensorStore`` takes the destination tensor ``B``, the index list
``[i]``, and the value expression ``A[i] + 1``. These arguments populate
its ``dest``, ``indices``, and ``value`` fields. ``ir.SeqStmt([store])``
holds this store statement as its first entry:

.. code-block:: text

   body: ir.SeqStmt
   └── seq[0]: ir.TensorStore                B[i] = A[i] + 1
       ├── dest: ir.Var                     B
       ├── indices[0]: ir.Var               i
       └── value: ir.prim.Add                A[i] + 1

When the compiled function runs, the store computes ``A[i] + 1`` and
writes the result to element ``i`` of ``B``. We can retrieve this
statement with ``body[0]``, which is shorthand for ``body.seq[0]``.
A longer sequence would hold the following statements in ``seq[1]``,
``seq[2]``, and so on, in execution order. The next section places
this ``body`` inside the CPU loop.

.. _control-flow-and-scope:

Control Flow
------------

Control-flow statements determine which parts of a program execute and
how often. TIRx provides ``ir.If`` for branching, ``ir.For`` and
``ir.While`` for loops, ``ir.Break`` and ``ir.Continue`` for loop control,
and ``ir.Return`` for returning from a function.

The statements inside a branch or loop form its body. Branch bodies are
stored in ``If.then_case`` and ``If.else_case``, while loop bodies are
stored in ``For.body`` and ``While.body``. Each present body is a
``SeqStmt`` and can contain further branches and loops. ``If.else_case``
is ``None`` when there is no else branch.

The CPU example visits every element with a loop:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at:     for i in range(n):
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

Its IR specifies a variable, begin and end bounds, a step, and an
execution kind. The same loop can be built around ``body`` with:

.. code-block:: python

   loop = ir.For(i, 0, n, 1, ir.ForKind.DEFAULT, body)

The first argument, ``i``, is the loop variable. The next three arguments
set the beginning to ``0``, the end to ``n``, and the step to ``1``.
``ForKind.DEFAULT`` selects ordered execution, and the final argument,
``body``, is the sequence repeated on each iteration. The ``For`` holds
these values in its fields:

.. code-block:: text

   loop: ir.For
   ├── var: ir.Var                           i
   ├── begin: ir.IntImm                      0
   ├── end: ir.Var                           n
   ├── step: ir.IntImm                       1
   ├── kind: ir.ForKind.DEFAULT
   └── body: ir.SeqStmt
       └── seq[0]: ir.TensorStore            B[i] = A[i] + 1

The end bound is exclusive, so ``i`` runs from 0 through ``n - 1``.
Each iteration executes the store in ``body`` with that value of ``i``.
The :class:`~tvm_tirx.ir.ForKind` API reference describes the other
execution kinds.

``i`` is available only inside ``For.body``. This is its scope, and it
matters when we place statements in the IR: the final write below lies
outside the loop and is invalid:

.. code-block:: python

   for i in range(n):
       B[i] = A[i] + 1
   B[i] = 0

The ``ir.For`` introduces the scope. Its ``SeqStmt`` body only records
statement order and does not add another scope.

The GPU example distributes this work across threads. Each thread
computes an index ``i`` and checks whether it has an element to process:

.. literalinclude:: ../get_started/examples/add_one_cuda.py
   :language: python
   :start-at:     i = T.cuda.block_idx
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

The ``if`` becomes an ``ir.If``. Its condition is an ``ir.prim.LT``
expression, and its ``then_case`` is a sequence containing the store:

.. code-block:: text

   ir.If
   ├── condition: ir.prim.LT                 i < n
   ├── then_case: ir.SeqStmt
   │   └── seq[0]: ir.TensorStore            B[i] = A[i] + 1
   └── else_case: None

The condition computes a boolean; the ``then_case`` holds the statement
that runs when it is true. With no else branch, threads whose indices
are outside the tensor perform no tensor access.

The index assignment and this branch belong to the device region introduced
by ``T.device_entry``. In the IR,
:class:`ir.RegionStmt <tvm_tirx.ir.RegionStmt>` groups these statements under
an operator that defines how the body executes. Its ``op`` identifies
``tirx.device_entry``, and its ``body`` is a ``SeqStmt`` containing an
``ir.Bind`` for the index assignment followed by the ``ir.If``. The binding
defines ``i`` within the region's scope, so the condition and tensor store
can use it. The
:ref:`Printing and Parsing <tvmscript-region-syntax>` chapter explains how
TVMScript groups these statements into the region.
