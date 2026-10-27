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

Symbolic Integer Analysis
=========================

In this chapter, we will introduce symbolic integer analysis for tensor
shapes, loop bounds, and element indices. A compiler pass may need to
simplify an index calculation or check that an access stays within bounds
even when the tensor length is not yet known. We will use the CPU add-one
program from :doc:`/get_started/quick_start` to see how an analyzer answers
these questions, then extend its index calculation with groups of four.

The Analyzer
------------

:class:`sym.Analyzer <tvm_tirx.sym.Analyzer>` answers queries about integer
IR expressions. It combines algebraic rules with facts supplied by a pass,
such as a loop variable's range. The same analyzer can use those facts
across several queries:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Query
     - Result
     - Purpose
   * - ``simplify``
     - An IR expression
     - Simplify a calculation under the known facts.
   * - ``int_set``
     - ``sym.IntSet``
     - Describe possible values, including symbolic bounds.
   * - ``const_int_bound``
     - ``sym.ConstIntBound``
     - Find inclusive numeric bounds.
   * - ``modular_set``
     - ``sym.ModularSet``
     - Find divisibility and remainder information.
   * - ``can_prove``, ``can_prove_equal``
     - A boolean
     - Establish a condition or equality of expression values.

Facts from the Program
----------------------

An index variable alone does not tell an analyzer its loop bounds.
The CPU example visits each tensor element with this loop:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at:     for i in range(n):
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

With ``add_one`` from the CPU example, we can retrieve the loop and its
index, then create an analyzer:

.. code-block:: python

   from tvm_tirx import ir, sym

   loop = add_one.body[0]
   i = loop.var
   n = loop.end
   analyzer = sym.Analyzer()

Inside this loop, ``0 <= i < n``. The analyzer needs this information
from the surrounding program to reason about ``i``. A ``RangeConstraint``
records the variable's domain; a ``ConstraintContext`` collects facts
for the current program point:

.. code-block:: python

   range_fact = sym.RangeConstraint(i, ir.Range(0, n))
   context = sym.ConstraintContext()
   context.append(range_fact)
   analyzer.sync_constraint_context(context)

The range starts at zero and has extent ``n``:

.. code-block:: text

   range_fact: sym.RangeConstraint
   ├── var → ir.Var                     i
   └── range: ir.Range
       ├── begin: ir.IntImm              0
       └── extent: ir.Var                n

``sync_constraint_context`` makes these facts available to subsequent
queries. The analyzer captures the context at synchronization, so later
additions need another synchronization. A pass moving to another part
of the program must supply the context valid there.

Simplifying Index Expressions
-----------------------------

A vectorized version of the loop could process four elements at a time.
For an element at index ``i``, its group starts at ``(i // 4) * 4``, and
``i % 4`` gives its lane within that group. For example, index 6 belongs
to the group starting at 4 and occupies lane 2. The analyzer can simplify
the expression that recombines the group start and lane:

.. code-block:: python

   group_start = (i // 4) * 4
   lane = i % 4
   simplified = analyzer.simplify(group_start + lane)
   print(simplified)

The output is:

.. code-block:: text

   i

``simplified`` is the IR expression a pass can use to replace the longer
index calculation.

Bounds and Divisibility
-----------------------

An analysis can also describe the values an expression may take.
For example, a neighboring-element access ``A[i + 1]`` would use this
index:

.. code-block:: python

   next_index = i + 1
   possible_values = analyzer.int_set(next_index)
   print(analyzer.simplify(possible_values.min_value))
   print(analyzer.simplify(possible_values.max_value))

The output is:

.. code-block:: text

   1
   n

For this expression, ``int_set`` returns a ``sym.IntervalSet``. Its
``min_value`` and ``max_value`` fields are integer IR expressions, so
the endpoints can depend on ``n``. The interval is inclusive: ``i + 1``
can reach ``n`` on the last iteration. This matters because the final
valid tensor index is ``n - 1``.

For numeric endpoints, ``const_int_bound`` returns a ``ConstIntBound``.
The lane expression from the grouping example has fixed bounds:

.. code-block:: python

   lane_bounds = analyzer.const_int_bound(lane)
   print(lane_bounds.min_value, lane_bounds.max_value)

The output is:

.. code-block:: text

   0 3

These endpoints are also inclusive, describing the four lane positions.
Bounds cover every possible value; for more complex expressions, an
inferred interval may also contain values that never occur.

``modular_set`` describes possible remainders. We can use it to check
the offsets at which groups begin:

.. code-block:: python

   multiples = analyzer.modular_set(group_start)
   print(multiples.coeff, multiples.base)

The output is:

.. code-block:: text

   4 0

``coeff`` and ``base`` describe values of the form ``coeff*k + base``
for integer ``k``. Every group therefore starts at a multiple of four,
a property a pass can use when checking a vector index calculation.

Proving Conditions
------------------

A proof query asks whether a condition holds for all values allowed by
the known facts. The loop's range lets us check the original access,
the neighboring-element access, and the regrouped index:

.. code-block:: python

   print(analyzer.can_prove(i < n))
   print(analyzer.can_prove(next_index < n))
   print(analyzer.can_prove_equal(group_start + lane, i))

The output is:

.. code-block:: text

   True
   False
   True

The original access satisfies its upper bound. The neighboring index
can reach ``n``, so its upper bound cannot be established for every
iteration. In general, ``False`` means that the analyzer did not prove
a condition; it does not prove the opposite condition. A pass requiring
that condition must keep the general case when the proof fails.

``can_prove_equal`` compares values rather than IR structure. Here, a
pass can establish that the grouping calculation preserves the original
index even though the two expressions have different structures.

The analysis infrastructure and its APIs are under active development.
The :doc:`tvm_tirx.sym </reference/python/sym>` reference describes the
current interfaces.
