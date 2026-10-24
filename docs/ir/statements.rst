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

Statements describe actions such as binding variables, writing tensor elements,
and returning values, along with the order and conditions under which those
actions occur. This structure lets the compiler track variable scope and
preserve program behavior when moving or rewriting computations.

.. _statements-and-sequences:

Statement Nodes
---------------

``ir.Stmt`` is the base class for statement objects. Its subclasses store
operands such as values and variables, or nested statement bodies. Common
nodes include:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Represents
     - Fields
   * - ``ir.Bind``
     - A variable definition
     - ``var``, ``value``
   * - ``ir.Return``
     - A function return
     - ``value``
   * - ``ir.TensorStore``
     - A tensor element write; the destination type must support stores
     - ``dest``, ``indices``, ``value``
   * - ``ir.Evaluate``
     - Expression evaluation with its result discarded
     - ``value``
   * - ``ir.AssertStmt``
     - A program assertion
     - ``condition``, ``error_kind``, ``message_parts``

All statements also inherit a ``loc`` field for source information, defaulting to
``UnknownLoc``.

Statement Sequences
-------------------

``ir.SeqStmt`` is itself a statement. Its ``seq`` field holds statements in
execution order. Nested sequences are flattened, and a sequence introduces no
scope of its own.

For example, first create the expressions used by a small function body:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin statement values
   :end-before: # end statement values
   :prepend: from tvm_tirx import ir, tirx

A binding followed by a return forms a body that adds one to ``x``:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin statements
   :end-before: # end statements

The binding defines ``result`` for subsequent statements in the same scope.
Its variable and value must have structurally equal types. The return refers
to that same variable object.

Control Flow and Scope
----------------------

The body above always binds and returns a value. To choose a body or repeat
one, control-flow statements contain their own sequences:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Class
     - Behavior
     - Additional fields
   * - ``ir.If``
     - Select a branch
     - ``condition``, ``then_case``, ``else_case``
   * - ``ir.For``
     - Repeat a body over an integer range
     - ``var``, ``begin``, ``end``, ``step``, ``kind``, ``body``, ``annotations``
   * - ``ir.While``
     - Repeat while a condition is true
     - ``condition``, ``body``

For example, this branch returns the incremented value only for positive
inputs. Both bodies are normalized to ``SeqStmt``:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin branch
   :end-before: # end branch

``If`` and ``While`` use boolean conditions. ``If`` requires an explicit
``else_case`` argument; pass None when there is no else branch.

A loop instead repeats a body. This loop evaluates each index from zero through
three:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin loop
   :end-before: # end loop

``For`` takes an exclusive ``end`` and explicit ``step`` and ``kind`` arguments.
Its variable, bounds, and step must have matching scalar integer types after
conversion. ``ForKind.DEFAULT`` denotes sequential execution; the other kinds
permit parallel execution or request vectorization or unrolling. ``Break``
exits a loop, and ``Continue`` starts its next iteration.

These nested bodies also determine variable visibility. The loop defines ``i``
inside its body; it cannot be used after the loop. Branch-local definitions
similarly stay inside their branch. Each variable object has one definition,
and its uses refer to that object.

Registered operations can enclose bodies through ``ir.RegionStmt``. Its
``args`` and ``attrs`` refer to the enclosing scope; ``body_params`` introduces
variables inside ``body``, and ``result_vars`` introduces variables for following
statements. See :class:`~tvm_tirx.ir.RegionStmt` for the operator contract.

To check these definition and scope rules, the body needs the context of a
complete function or module. :doc:`functions_and_modules` supplies that context;
``tirx.analysis.check_well_formed`` then checks variable and tensor uses and loop
control. Invalid uses raise ``RuntimeError`` by default; ``assert_mode=False``
returns False instead.

The :download:`statements_and_functions.py <examples/statements_and_functions.py>`
example completes and runs the increment function. See
:doc:`/reference/python/ir` for statement constructors and
:doc:`/reference/python/tirx.analysis` for verification APIs.
