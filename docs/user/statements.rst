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

Expressions describe values and computations. Statements place those
computations in an execution order: they define variables, write to tensors,
select branches, and repeat bodies. A function body combines these statements
into sequences and nested scopes.

Statements and Sequences
------------------------

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

``SeqStmt`` puts these statements in execution order. Its ``seq`` field holds
the sequence; nested sequences are flattened. This grouping does not introduce
a scope of its own.

The values used by statements are expressions. For example, a binding's
``value`` can be an addition, and a store's ``indices`` can contain variables
or arithmetic. ``Bind`` defines a variable from an expression's result for use
by subsequent statements in the same scope. Its variable and value must have
structurally equal types. ``TensorStore`` checks that the destination's type
supports stores.

Control Flow and Scope
----------------------

A sequence describes straight-line execution. To select or repeat statements,
control-flow nodes contain their own body sequences:

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

``If`` and ``While`` use boolean conditions. ``If`` requires an explicit
``else_case`` argument; pass None when there is no else branch. Each
present branch is stored as a ``SeqStmt``, even when supplied as one statement.

``For`` and ``While`` similarly store their bodies as sequences. ``For`` takes
an exclusive ``end`` and explicit ``step`` and ``kind`` arguments; its variable,
bounds, and step must have matching scalar integer types after conversion.
``ForKind.DEFAULT`` denotes sequential execution; the other kinds permit
parallel execution or request vectorization or unrolling. Within a loop,
``Break`` exits the loop and ``Continue`` starts its next iteration. ``Return``
exits the enclosing function.

The following body combines a binding, a branch, and two loops. First construct
the tensor variables with shape ``(2*n, 128)`` and their indices:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin statement values
   :end-before: # end statement values
   :prepend: from tvm_tirx import ir, tirx

The binding computes an element value, and the branch selects which value to
store. The loops enclose both statements to cover all rows and columns:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin statements
   :end-before: # end statements

``body.body[0]`` is ``inner``. The inner loop's body contains ``binding``
followed by ``branch``, so both branches can use ``value``. The program writes
``A + 1`` to the first ``n`` rows of ``C`` and ``A + 3`` to the remaining rows.

This nesting also determines variable scope. The loops define ``i`` and ``j``
for their bodies, and the binding defines ``value`` for the rest of the inner
body. Each variable object has one definition, and uses refer to that same
object. A variable defined inside a branch or loop body cannot be used after
that body.

The same scoping rule applies to ``ir.RegionStmt``, which lets a registered
operation enclose a lexical body. Its ``args`` and ``attrs`` are evaluated in
the enclosing scope; ``body_params`` introduces variables inside ``body``, and
``result_vars`` introduces variables for following statements. The
:class:`~tvm_tirx.ir.RegionStmt` reference describes the operator contract.

These definition and scope rules can be checked in a complete function or
module with ``tirx.analysis.check_well_formed``. It checks variable and tensor
uses and loop control. An invalid use, such as a branch-local variable escaping
its branch, raises ``RuntimeError`` by default; ``assert_mode=False`` returns
False instead. :doc:`objects_and_functions` places this body in a function and
module.

The :download:`statements_and_functions.py <examples/statements_and_functions.py>` example includes
scope errors and numerical checks. See :doc:`/reference/python/ir` for statement
constructors and :doc:`/reference/python/tirx.analysis` for verification APIs.
