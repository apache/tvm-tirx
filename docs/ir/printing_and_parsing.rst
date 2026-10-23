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

.. _ir-construction-and-printing:

Printing and Parsing
====================

Readable source makes it easier to inspect generated programs, edit examples,
and compare IR before and after a transformation. TVMScript provides this source
form using Python syntax.

Printing renders IR objects as TVMScript, while parsing constructs them from
source. Function signatures, tensor shapes, symbolic dimensions, and control
flow can be written together. Python functions and script helpers also let you
reuse construction logic to generate program variants.

For example, this function increments every element of a float32 tensor with
shape ``(2*n, 128)``. Save the code in a Python file so the decorator can inspect
its source:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin tvmscript example
   :end-before: # end tvmscript example

``@T.function`` parses the definition into a ``tirx.Function``. The future
import lets the parser resolve ``n`` in the annotations. The tensor's shape and
the loop bound refer to the same IR variable, keeping ``2*n`` symbolic in both
places. The loops and indexing construct IR; the tensor computation runs when
the compiled function is called.

An ordinary Python function can generate variations of this program. To reuse
the element update across those variations, first separate its computation and
tensor write into script helpers:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin script helpers
   :end-before: # end script helpers

``@T.inline`` expands a helper and returns a value to its caller during
construction. ``@T.macro`` emits a statement fragment into the caller's body;
a return inside a macro would return from the enclosing IR function. Here,
``ir.const(1, value.ty)`` creates the increment with the element's type, so the
same helpers work with different element types.

The factory can now choose a width and dtype and build a function using those
helpers:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin program factory
   :end-before: # end program factory

``width`` and ``dtype`` are Python values read while the factory runs. ``wide``
has a fixed second dimension of 128 and float32 elements; ``narrow`` has a second
dimension of 64 and float16 elements. Their first dimension remains ``2*n``,
where ``n`` is an IR parameter supplied when the compiled function is called.
The validation and function construction happen in Python; the generated loops
describe the later tensor computation. Using Python to choose and assemble IR
this way is metaprogramming.

Printing lets you inspect what this Python code generated:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin print generated function
   :end-before: # end print generated function

``script()`` returns source text. The printed ``wide`` function contains the
selected width and dtype, with the helpers expanded into the direct update
``A[i, j] = A[i, j] + T.float32(1.0)``. The factory and helper calls have already
done their work during construction. You can use the same view after a pass to
inspect its result, or print smaller parts of the program:
``wide.params[1].ty.script()`` renders its tensor type, and
``wide.body[0].script()`` renders its outer loop.

For interactive inspection, ``show()`` displays highlighted source. Its options
can add line numbers and underline particular IR objects, such as the outer
loop:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin diagnostic display
   :end-before: # end diagnostic display

The source string can then be parsed into a new function:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin printing and parsing
   :end-before: # end printing and parsing

``from_source`` reads text using its imports and ``extra_vars``; it does not
inherit the caller's local variables. Here, ``extra_vars`` supplies ``T``
because the printer emits its import as a comment by default. For handwritten
source, these bindings can also supply Python constants, type constructors,
and script helpers. Pass ``filename`` when parsing saved text to use that name
in parser diagnostics.

The structural comparison checks the reconstructed program, including its
types and variable bindings. The printed text may use explicit symbol
declarations or nested loops in place of ``T.grid``.

This workflow also applies to modules. ``@I.ir_module`` collects the generated
functions as class members, allowing both variants to be printed and parsed
together:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin module round trip
   :end-before: # end module round trip

To obtain source with executable imports, set ``ir.comment_imports`` to False.
This is useful when saving the source to a file or parsing it in another
process. The imports provide the namespaces, so this example needs no
``extra_vars``:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin standalone source
   :end-before: # end standalone source

The :download:`complete example <examples/printing_and_parsing.py>` checks the
generated variants, shared shape symbols, and function and module round trips.
See :doc:`/reference/python/script.parser` for parsing and script-helper
interfaces, and :doc:`/reference/python/script.printer` for printer configuration.
