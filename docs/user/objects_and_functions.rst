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

A function defines parameters, a return type, and an optional statement body.
A module groups functions under global names so compiler components can look up
their definitions and process them together.

Functions
---------

``ir.BaseFunc`` is the shared base class for function definitions and is itself
an expression. ``tirx.Function`` extends it with parameters, a return type, and
an optional body. Its main fields are:

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
   * - ``attrs``
     - Named function properties stored in ``ir.DictAttrs``.

The function's ``ty`` is an ``ir.FuncType`` built from its parameter types and
``ret_type``. This signature describes how the function can be called; the body
describes its implementation. Construction does not infer the return type from
the body. Omitting ``ret_type`` produces an empty ``ir.TupleType``.

Using the parameters and loop body from :doc:`statements`, this function writes
its output to ``C`` and explicitly declares ``ir.PrimType("void")``, matching
TVMScript's ``T.void``:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin function construction
   :end-before: # end function construction

The scalar parameter ``n`` is the same variable used in the tensor shapes and
loop bounds. Passing None for ``body`` instead constructs a declaration with no
implementation. An invocation within IR is represented by an ``ir.Call``;
constructing a function does not execute its body.

``with_body`` returns a function with a replacement body and the same parameters,
return type, and attributes. Pass ``loc=function.loc`` to retain its location.
``with_attr`` returns a function with an added or replaced attribute. Both
methods preserve the original function.

Modules
-------

``ir.IRModule`` stores definitions in its ``functions`` map, whose keys are
``ir.GlobalVar`` references and whose values are ``ir.BaseFunc`` objects. Each
global name occurs once within a module. ``attrs`` stores module attributes,
and ``global_infos`` stores named arrays of ``ir.GlobalInfo`` metadata.

This module installs the function under a global reference named ``main``:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin module construction
   :end-before: # end module construction

``module[entry]`` looks up the stored reference; ``module["main"]`` resolves its
name to the same definition. The signature is stored on ``function.ty`` and is
not copied to ``entry.ty`` during insertion. Use ``module.get_global_var(name)``
to retrieve an existing reference: a fresh ``GlobalVar`` with the same name has
a different identity.

The module key names a definition within IR. The function's ``global_symbol``
attribute specifies its exported name for code generation. They need not match.
String keys, as in ``ir.IRModule({"main": function})``, create global references
automatically.

``module[name] = function`` adds or replaces a definition in place. To change
the function map independently, first call ``module.clone()``. The clone is a
new module whose existing functions are initially shared. For replacements via
``module.update``, reuse the stored ``GlobalVar`` key; a new same-name reference
conflicts with the existing one.

A module's object graph can be saved with ``ir.save_json`` and reconstructed
with ``ir.load_json``. A structural comparison checks the round trip:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin module serialization
   :end-before: # end module serialization

Structural equality compares program structure, including corresponding bound
variables; it does not prove numerical equivalence. JSON loading requires the
object types used by the graph to be registered in the receiving process.

See :class:`~tvm_tirx.tirx.Function` and :class:`~tvm_tirx.ir.IRModule` for the
complete APIs. The :download:`statements_and_functions.py <examples/statements_and_functions.py>`
example also checks updates and scopes, then compiles and runs this module.
