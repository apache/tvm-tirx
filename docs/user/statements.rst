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

Statements and Control Flow
===========================

Expressions describe values and computations. Statements organize their use:
a statement can define a variable, write to a tensor, select a branch, or repeat
a body. Together, statement sequences and nested bodies describe execution order
and variable scope.

Statement Structure
-------------------

``ir.Stmt`` is the base class for statements. All statements carry source
information in ``loc``. Concrete classes add fields for their operands and
nested statements:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Represents
     - Additional fields
   * - ``ir.SeqStmt``
     - An ordered statement sequence
     - ``seq``
   * - ``ir.Bind``
     - A variable definition
     - ``var``, ``value``
   * - ``ir.TensorStore``
     - A tensor element write
     - ``dest``, ``indices``, ``value``
   * - ``ir.Evaluate``
     - Expression evaluation with its result discarded
     - ``value``
   * - ``ir.AssertStmt``
     - A program assertion
     - ``condition``, ``error_kind``, ``message_parts``

Operands such as a binding's ``value`` or a store's ``indices`` are expressions.
``Bind`` defines a variable for subsequent statements in the same scope; its
variable and value must have structurally equal types. ``Evaluate`` can place an
effectful call in a body, while ``TensorStore`` writes through a destination
whose type supports stores.

``SeqStmt`` holds statements in execution order and flattens nested sequences.
It does not introduce a scope of its own. Function and control-flow bodies are
stored as ``SeqStmt`` objects, even when constructed from a single statement.

Control Flow and Scope
----------------------

Control-flow statements connect conditions and loop bounds to nested bodies:

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

Conditions have boolean types. ``If`` requires an explicit ``else_case``
argument; pass None when there is no else branch. ``For`` takes an exclusive
``end`` and explicit ``step`` and ``kind`` arguments. Its variable, bounds, and
step must have matching scalar integer types after conversion.
``ForKind.DEFAULT`` denotes sequential execution; the other kinds permit
parallel execution or request vectorization or unrolling. ``Break`` exits the
enclosing loop, ``Continue`` starts its next iteration, and ``Return`` exits the
function.

For example, the following values describe two tensors with shape ``(2*n, 128)``
and the variables used to access them:

.. literalinclude:: examples/ir_structure.py
   :language: python
   :start-after: # begin statement values
   :end-before: # end statement values
   :prepend: from tvm_tirx import ir, tirx

A binding computes an element value, a branch selects its store, and two loops
repeat that body over the tensor:

.. literalinclude:: examples/ir_structure.py
   :language: python
   :start-after: # begin statements
   :end-before: # end statements

``body`` is an outer ``For`` whose body contains ``inner``. The inner loop's
body contains ``binding`` followed by ``branch``. The first ``n`` rows receive
``A + 1``; the remaining rows receive ``A + 3``.

Each variable object has one definition, and its uses refer to that same
object. Here the loops define ``i`` and ``j``, and the binding defines ``value``
before both branches use it. A variable defined inside a branch or loop body
cannot be used after that body.

``ir.RegionStmt`` represents a registered operation with a lexical body. Its
``op``, ``args``, and ``attrs`` describe the operation. ``body_params`` lists
variables scoped to ``body``; ``result_vars`` lists variables defined for
following statements. The :class:`~tvm_tirx.ir.RegionStmt` reference describes
the required operator contract.

Once statements are placed in a function or module,
``tirx.analysis.check_well_formed`` checks variable definitions and scopes,
tensor uses, and loop control. Invalid uses, such as a branch-local variable
escaping its branch, raise ``RuntimeError`` by default; ``assert_mode=False``
returns False instead. :doc:`objects_and_functions` completes this program.

The :download:`ir_structure.py <examples/ir_structure.py>` example includes
scope errors and numerical checks. See :doc:`/reference/python/ir` for statement
constructors and :doc:`/reference/python/tirx.analysis` for verification APIs.
