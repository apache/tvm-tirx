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

A reusable computation needs an explicit calling interface: which values are
inputs and what kind of result it produces. ``tirx.Function`` records that
interface through parameters and a return type, alongside an optional statement
body.

``ir.IRModule`` groups function definitions under global names so they can be
referenced by calls and processed together by compiler passes. It also holds
metadata shared by the module.

Functions
---------

``ir.BaseFunc`` is the shared base class for function definitions and is itself
an expression. ``tirx.Function`` adds parameters, a return type, and an optional
statement body:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``params``
     - Parameter variables in calling order.
   * - ``ret_type``
     - The declared result type; an omitted type becomes an empty ``TupleType``.
   * - ``body``
     - A ``SeqStmt``, or None for a declaration without an implementation.
   * - ``attrs``
     - Named function properties stored in ``ir.DictAttrs``.

Using ``x`` and ``body`` from :doc:`statements_and_control_flow` completes the
increment function:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin function construction
   :end-before: # end function construction

``x`` is now a parameter definition; ``result`` remains local to the body. The
function's ``ty`` is ``ir.FuncType([x.ty], x.ty)``, matching the signature
introduced in :doc:`types_and_expressions`. Construction derives this signature
from the parameter and declared return types, without inferring a return type
from the body. Passing None for ``body`` creates a declaration without an implementation.

Modules
-------

To share this definition through a global reference, place it in an
``ir.IRModule``. The module maps ``ir.GlobalVar`` references to ``ir.BaseFunc``
definitions:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin module construction
   :end-before: # end module construction

``module[entry]`` and ``module["main"]`` retrieve the same definition. The module
key names it within IR; ``global_symbol`` specifies the exported name for code
generation. These names need not match. String keys passed to the module
constructor create global references automatically.

The stored reference can now be used as a callee, connecting this definition to
the calls introduced in :doc:`operators_and_calls`:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin global call
   :end-before: # end global call

The signature lives on ``function.ty``; insertion does not copy it to
``entry.ty``, so this call supplies its result type explicitly. As with local
variables, a fresh ``GlobalVar`` with the same name has a different identity.
Use ``module.get_global_var(name)`` to retrieve a stored reference.

When transforming a definition, ``function.with_body`` and ``with_attr`` return
a function with a replacement body or attribute while preserving the original.
``with_body`` retains the parameters, return type, and attributes; pass
``loc=function.loc`` to retain its location. Install the result with
``module[name] = replacement``, which changes the module in place. Start with
``module.clone()`` to keep the original function map. For replacements through
``module.update``, reuse the stored global reference.

A module also carries metadata: ``attrs`` stores named attributes, and
``global_infos`` stores named arrays of ``ir.GlobalInfo``. To preserve the complete
object graph for later inspection, save and reconstruct it as JSON:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin module serialization
   :end-before: # end module serialization

Structural comparison checks the representation, including corresponding bound
variables. Loading requires the graph's object types to be registered in the
receiving process.

See :class:`~tvm_tirx.tirx.Function` and :class:`~tvm_tirx.ir.IRModule` for the
complete APIs. The :download:`statements_and_functions.py <examples/statements_and_functions.py>`
example checks scopes and updates, compiles with LLVM, and verifies that an
input of three returns four.
