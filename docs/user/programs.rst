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

IR Construction and Printing
============================

The IR objects introduced in the preceding chapters can be assembled with
constructors or builder APIs, or produced by parsing TVMScript. Printing gives
those objects a source representation for inspection and editing.

TVMScript
---------

TVMScript uses Python source syntax to describe IR. The ``@T.function`` decorator
parses a function definition and returns a ``tirx.Function``. Put the following
imports at the top of a Python source file:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin construction imports
   :end-before: # end script imports

Deferred annotations let the parser resolve ``n`` in later tensor parameter
types. The source file also lets the decorator recover the function's text.
The program from :doc:`statements` and :doc:`objects_and_functions` can then be
written as:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin script function
   :end-before: # end script function definition

The annotations define the parameter types, ``T.grid`` creates nested loops,
and the assignment to ``value`` creates a binding. The branch bodies contain
tensor stores. These constructs produce the IR objects introduced in the
preceding chapters.

``@I.ir_module`` builds a module from a class. It accepts existing functions as
members:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin script module
   :end-before: # end script module definition

Functions can also be defined inside the class with ``@T.function``. Their
signatures are declared before their bodies are built, allowing references
between member functions.

Builder Contexts
----------------

The parser rewrites supported syntax into builder calls and executes those calls
during construction. Builder frames collect parameters and statements, then
assemble an IR object when their scopes finish. The resulting tensor
computations execute later, when the compiled program runs.

Python code can enter the same frames directly. For example, a loop frame
collects its body and produces an ``ir.For``:

.. code-block:: python

   from tvm_tirx.script.ir_builder import IRBuilder
   from tvm_tirx.tirx.script import ir_builder as B

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin grid pair
   :end-before: # end loop construction

``IRBuilder`` owns the construction context, and ``get()`` retrieves the result.
Here ``B.grid`` takes a ``(start, extent)`` pair, so the loop has ``begin=2`` and
exclusive ``end=5``. ``B.evaluate`` emits a statement into its body.

Similarly, ``B.function_`` collects a function's parameters and body, while
``B.ir_module`` collects functions. Nesting these frames constructs a complete
program. See
:doc:`/reference/python/tirx.script.ir_builder` for the frame APIs.

Parsing and Printing
--------------------

``script()`` returns TVMScript as a string; ``show()`` displays it with syntax
highlighting. ``from_source`` parses text into IR. For example, this round trip
checks the reconstructed function against the original:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin function source round trip
   :end-before: # end function source round trip

The printer emits namespace imports as comments by default. ``extra_vars``
supplies those bindings to the parser; source text does not inherit its caller's
local variables. Alternatively,
``main.script(extra_config={"ir.comment_imports": False})`` includes executable
imports. Printing may normalize loop syntax and symbolic declarations, so a
round trip preserves IR structure rather than the original source spelling.

See :func:`~tvm_tirx.script.parser.from_source` and
:doc:`/reference/python/script.printer` for parsing and formatting options.
The :download:`construction_and_printing.py <examples/construction_and_printing.py>` example checks
TVMScript construction, a builder loop, and a source round trip.
