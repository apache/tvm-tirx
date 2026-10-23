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

The preceding chapters assembled an increment function from individual IR
objects. TVMScript and builder frames provide other ways to assemble those same
objects. Keeping the function unchanged makes their relationship visible in
the resulting IR.

TVMScript
---------

TVMScript describes IR with Python source syntax. The ``@T.function`` decorator
parses a definition and returns a ``tirx.Function``. Put these imports in a
Python source file so the decorator can recover the function's text:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin construction imports
   :end-before: # end script imports

The increment function can then be written directly:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin script function
   :end-before: # end script function definition

The annotations provide the parameter and return types. Assigning to ``result``
creates the ``Bind`` from :doc:`statements`, and ``return`` creates its
``Return``. The result is a function containing the same two-statement body.

To give that function a module entry, ``@I.ir_module`` accepts it as a class
member:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin script module
   :end-before: # end script module definition

Functions can also be defined inside the class with ``@T.function``. Their
signatures are declared before their bodies are built, allowing references
between member functions.

Builder Contexts
----------------

The decorator constructs these objects by translating source syntax into
builder calls. Frames collect the function's parameters and statements, then
assemble the function when their scopes finish. Python code can enter these
same frames directly:

.. code-block:: python

   from tvm_tirx.script.ir_builder import IRBuilder
   from tvm_tirx.tirx.script import ir_builder as B

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin builder function
   :end-before: # end builder function

``B.arg_`` and ``B.func_ret_type_`` supply the signature; ``B.bind_`` and
``B.return_`` collect the body. ``builder.get()`` retrieves the completed
function. The example checks that ``built`` and the TVMScript ``main`` are
structurally equal. A surrounding ``B.ir_module`` frame can collect functions
into a module. See :doc:`/reference/python/tirx.script.ir_builder` for the frame
APIs.

Both routes run Python during construction. The represented addition executes
later, when the function is compiled and called.

Parsing and Printing
--------------------

Once either route has produced IR, ``script()`` turns it back into TVMScript.
It returns a string; ``show()`` displays it with syntax highlighting.
``from_source`` parses that text, closing the loop from source to IR and back:

.. literalinclude:: examples/construction_and_printing.py
   :language: python
   :start-after: # begin function source round trip
   :end-before: # end function source round trip

The printer emits namespace imports as comments by default. ``extra_vars``
supplies those bindings to the parser; source text does not inherit its caller's
local variables. Alternatively,
``main.script(extra_config={"ir.comment_imports": False})`` includes executable
imports. Printing may normalize source spelling, while the structural check
compares the reconstructed program.

See :func:`~tvm_tirx.script.parser.from_source` and
:doc:`/reference/python/script.printer` for parsing and formatting options.
The :download:`construction_and_printing.py <examples/construction_and_printing.py>`
example checks both construction routes and the source round trip.
