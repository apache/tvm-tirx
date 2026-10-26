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

In this chapter, we will introduce TVMScript and the interfaces for printing
and parsing IR. TVMScript provides a Python-based notation for program
definitions, while the printer and parser convert between IR objects and
source text. These interfaces allow generated programs to be inspected,
saved, and reconstructed. Compiler extensions can reuse them with custom
builder namespaces and printer hooks.

The ``@T.function`` decorator parses a Python function definition into a
``tirx.Function``. We can revisit the CPU example:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

The symbol ``n`` is shared by the tensor shapes and loop bound.
``add_one`` is already an IR function after the decorated definition has
been parsed; the numerical computation runs when the compiled function
is called.

The ``script()`` method returns TVMScript source for an IR object:

.. code-block:: python

   source = add_one.script()
   print(source)

The printer includes the symbolic dimension declaration and the
constant's dtype in the function text:

.. code-block:: python

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + T.float32(1.0)

``[n]`` declares the symbolic dimension used by this function, so the
printed definition includes the symbol needed by its tensor shapes and
loop bound. ``T.float32(1.0)`` makes the addition's constant type explicit.

The same method works on the GPU example, whose source also includes the
launch configuration, thread-index binding, and bounds check.

Smaller objects can be inspected too. ``add_one.params[0].ty.script()``
prints the input tensor type. :meth:`add_one.show() <tvm_tirx.tirx.Function.show>`
displays highlighted source and supports options for line numbers and
object annotations.

The :doc:`TVMScript parser </reference/python/script.parser>` converts
source text back into IR through ``from_source``. We can reconstruct the
printed CPU function and check that its IR structure is preserved:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import from_source

   restored_function = from_source(source, extra_vars={"T": T})
   ir.assert_structural_equal(add_one, restored_function)

The default source comments out its imports, so ``extra_vars`` supplies
the ``T`` namespace. The dictionary key ``"T"`` matches the name in the
source text, and its value is the imported TVMScript namespace ``T``.
The reconstructed function can be inspected, transformed, or compiled
again. The GPU function can be printed and parsed through the same
interfaces.

.. _tvmscript-region-syntax:

The parser also accepts shorthand for region constructs. The GPU example
uses a standalone ``T.device_entry(...)`` statement:

.. literalinclude:: ../get_started/examples/add_one_cuda.py
   :language: python
   :start-at: @T.function
   :end-at:         B[i] = A[i] + 1

In this form, the region includes all following statements in the enclosing
block, even at the same indentation level. The enclosing block in this example
is the function body, so the region extends to the end of the function and
contains both the assignment to ``i`` and the entire ``if`` branch. This rule
applies to region constructs; ordinary calls do not capture subsequent
statements. The same function can be written with an explicit ``with`` block:

.. code-block:: python

   @T.function
   def add_one(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
       with T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,))):
           i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
           if i < n:
               B[i] = A[i] + 1

The ``with`` form includes only its indented block. Both forms produce the
same ``ir.RegionStmt`` and body described in :doc:`statements`.

A source file that can be parsed on its own needs its imports. With
``ir.comment_imports=False``, the :doc:`printer </reference/python/script.printer>`
includes them in the generated source. The following helper saves and
reloads a whole module, such as the wrapper from
:doc:`functions_and_modules`:

.. literalinclude:: examples/add_one.py
   :language: python
   :pyobject: save_program

``output_path`` is a ``pathlib.Path``. The saved file contains the
module's definition and imports, so ``from_source`` can reconstruct it
from that file. The structural check verifies that the reconstructed
module has the same IR as the original. The returned module can then
be inspected, transformed, or compiled with ``tirx.compile``.
