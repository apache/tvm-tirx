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

Functions and Modules
=====================

In this chapter, we will introduce functions and modules as the structures
that organize IR programs. A function combines parameters, a result type,
and a statement body; a module collects named function definitions. Compiler
passes use these structures to inspect a function's interface and
implementation, locate definitions through global references, and replace
an implementation while preserving references to it.

Functions
---------

``@T.function`` defines a tensor-program function in TVMScript and
produces a :class:`tirx.Function <tvm_tirx.tirx.Function>` IR object.
The CPU example is defined as:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

The decorated name ``add_one`` refers to the resulting IR function.
Its parameters are ``A`` and ``B``, and its body contains the loop.

``ir.BaseFunc`` is the base class for all function definitions and is
also an expression. Its ``attrs`` field holds named function properties:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``attrs``
     - ``ir.DictAttrs``
     - Named function properties.

``tirx.Function`` extends ``ir.BaseFunc`` with parameters, a result type,
and a statement body:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``params``
     - Parameter variables in calling order.
   * - ``ret_type``
     - The declared result type.
   * - ``body``
     - A ``SeqStmt``, or None for a declaration without an implementation.

The parameter and result types form the function's signature, available
as an ``ir.FuncType`` through ``ty``. The ``global_symbol`` entry in
``attrs`` specifies its exported name for code generation. For public
functions, ``@T.function`` sets it to the Python function name by default.

The IR API can also build the function from the CPU example.
We start with its symbolic length, tensor parameters, and loop index:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin variables
   :end-before: # end variables
   :dedent: 4
   :prepend: from tvm_tirx import ir, tirx

``A`` and ``B`` share the tensor type with length ``n``, and ``i`` is
an int32 index variable. The store and loop from :doc:`statements`
form the computation:

.. code-block:: python

   store = ir.TensorStore(B, [i], A[i] + 1)
   body = ir.SeqStmt([store])
   loop = ir.For(i, 0, n, 1, ir.ForKind.DEFAULT, body)

The function then combines the parameters with the loop body:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin function construction
   :end-before: # end function construction
   :dedent: 4

``[A, B]`` supplies the function's parameters in calling order. The
second argument, ``loop``, is a single ``ir.For`` statement; the
constructor wraps it in a ``SeqStmt`` to form the function's ``body``.
The function writes its results through the output parameter ``B`` and
returns no value, as declared by ``ret_type=ir.TupleType([])``.

The ``with_attr`` call returns a function with ``attrs["global_symbol"]``
set to ``"add_one"``, its exported name. The symbol ``n`` appears in both
parameter types and the loop bound, and its value is derived from the
tensor shapes at runtime.

The original ``add_one`` function connects its parameters and result
type to a nested statement body:

.. code-block:: text

   add_one: tirx.Function
   ├── params[0]: ir.Var                     A
   ├── params[1]: ir.Var                     B
   ├── ret_type: ir.TupleType                no return value
   └── body: ir.SeqStmt
       └── seq[0]: ir.For                    for i in range(n)
           └── body: ir.SeqStmt
               └── seq[0]: ir.TensorStore    B[i] = A[i] + 1

The function body contains the ``For``, and the loop body contains the
``TensorStore``. We can follow these fields to inspect each statement.
Since ``add_one.body`` is a ``SeqStmt``, ``add_one.body[0]`` selects
the loop:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin loop inspection
   :end-before: # end loop inspection

``loop`` is the ``ir.For`` for ``for i in range(n)``, and ``loop.var``
is its index variable ``i``. The loop body is another ``SeqStmt``. Its
first statement writes the result to ``B[i]``:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin store inspection
   :end-before: # end store inspection

``store`` is an ``ir.TensorStore``. Its ``value`` field contains the
expression to write, so ``addition`` is the ``ir.prim.Add`` for
``A[i] + 1``.

Modules
-------

:class:`ir.IRModule <tvm_tirx.ir.IRModule>` organizes named functions into
a program. An :class:`ir.GlobalVar <tvm_tirx.ir.GlobalVar>` refers to a
function stored in the module. The module associates that reference
with an ``ir.BaseFunc`` definition, such as a ``tirx.Function``.

The program passes ``add_one`` directly to ``tirx.compile``.
A tool that manages functions by name can wrap this same function in a module:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin module construction
   :end-before: # end module construction

``IRModule`` converts the string key ``"add_one"`` into an ``ir.GlobalVar``
and associates it with the ``add_one`` definition. We can retrieve that
global reference, then use it to look up the function:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin module lookup
   :end-before: # end module lookup

``entry`` is an ``ir.GlobalVar`` used as a key in the module's
``functions`` mapping. The associated value is the function definition:

.. code-block:: text

   module: ir.IRModule
   └── functions
       └── entry (ir.GlobalVar) → add_one (tirx.Function)

Both ``module[entry]`` and ``module["add_one"]`` retrieve that definition.
The ``GlobalVar`` refers to the function in the module; the ``Var``
objects inside the function represent its parameters and loop indices.

A compiler pass can rebuild a function and replace its definition under
the same reference. For example, we can install the IR API reconstruction
``ir_function`` from above into a copy of the module:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin module update
   :end-before: # end module update

``variant[entry]`` now returns the reconstructed definition. It keeps
the same parameters, result type, and exported name. The original
``module`` retains the TVMScript definition. This is the update pattern
a pass can use while preserving the function's calling interface.
