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

TVMScript uses Python syntax to express IR. Function signatures, tensor shapes,
symbolic dimensions, and control flow can be written together as source code.
Parsing constructs the IR objects introduced in the preceding chapters;
printing renders those objects back into TVMScript.

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

Use ``f.script()`` to obtain its TVMScript as a string, or ``f.show()`` to
display it with syntax highlighting. That text can be parsed back into IR:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin printing and parsing
   :end-before: # end printing and parsing

``extra_vars`` supplies the ``T`` namespace because the printer emits its import
as a comment by default. Printing may introduce explicit symbol declarations
or expand ``T.grid`` into nested loops. The structural comparison checks that
the reconstructed IR matches the original.

The :download:`complete example <examples/printing_and_parsing.py>` checks the
symbolic dimensions and the source round trip. See
:func:`~tvm_tirx.script.parser.from_source` and
:doc:`/reference/python/script.printer` for parsing and formatting options.
