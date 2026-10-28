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

In this chapter, we will use the add-one loop to introduce symbolic integer
analysis for tensor shapes, loop bounds, and element indices. A pass that
combines four scalar accesses into one vector access needs to preserve their
positions and establish that the whole group is within the tensor. The
length may only be known at runtime, so these decisions require reasoning about symbolic
expressions. We will start from the loop's known range, examine grouped
indices and a neighboring-element access, and then analyze the iteration
mapping and region covered by those groups.

The Analyzer
------------

:class:`sym.Analyzer <tvm_tirx.sym.Analyzer>` answers queries about integer
IR expressions. It combines algebraic rules with facts supplied by a pass,
such as a loop variable's range. The same analyzer can use those facts
across several queries:

.. list-table::
   :header-rows: 1
   :widths: 32 28 40

   * - Query
     - Arguments
     - Result
   * - ``simplify(expr)``
     - An IR expression.
     - An equivalent IR expression in a simpler form.
   * - ``int_set(expr)``
     - An integer IR expression.
     - A ``sym.IntSet`` covering its possible values, including symbolic bounds.
   * - ``const_int_bound(expr)``
     - An integer IR expression.
     - A ``sym.ConstIntBound`` with inclusive numeric bounds.
   * - ``modular_set(expr)``
     - An integer IR expression.
     - A ``sym.ModularSet`` describing divisibility and remainders.
   * - ``can_prove(expr)``
     - A boolean IR expression.
     - ``True`` if the condition is established from the known facts.
   * - ``can_prove_equal(lhs, rhs)``
     - Two integer IR expressions.
     - ``True`` if their values are established to be equal.

These calls show the required arguments. The expressions describe
calculations in the program, and the analyzer uses its known facts to
reason about their runtime values.

Facts from the Program
----------------------

The analyzer needs facts from the point in the program where a rewrite
would occur. We will use the same CPU function as in
:doc:`structural_utilities`:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

In this loop, the index satisfies ``0 <= i < n``. We can obtain the actual
loop variable and symbolic dimension from the function's fields:

.. code-block:: python

   from tvm_tirx import ir, sym, tirx

   loop = add_one.body[0]
   i = loop.var
   n = add_one.params[0].ty.shape[0]
   analyzer = sym.Analyzer()

The loop bounds belong to the loop statement, while the variable ``i``
records a name and type. A query on ``i`` therefore does not automatically
include those bounds. The pass supplies them through a ``RangeConstraint``;
a ``ConstraintContext`` stores the facts for the current program point:

.. code-block:: python

   range_fact = sym.RangeConstraint(i, ir.Range(0, n))
   context = sym.ConstraintContext()
   context.append(range_fact)
   analyzer.sync_constraint_context(context)

``ir.Range(0, n)`` describes the half-open interval ``[0, n)``: zero is
included and ``n`` is excluded. In general, ``ir.Range(begin, end)`` takes
endpoints; ``ir.Range.from_min_extent(begin, extent)`` takes a starting value
and size. Here, the range starts at zero and has extent ``n``:

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

A vectorization pass can split the loop index ``i`` into a group start
and a lane within a group of four. The address calculation may later
recombine these parts, leaving division and remainder operations that can
be removed. The group starts at ``(i // 4) * 4``, and ``i % 4`` gives its
lane. For example, index 6 belongs to the group starting at 4 and occupies
lane 2. The analyzer can simplify the recombined expression:

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

The loop range guarantees that ``A[i]`` is in bounds. A computation that
also reads ``A[i + 1]`` needs to establish a bound for that new index.
``int_set`` can bound its possible values before the pass decides whether
the access needs a guard:

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

The grouping calculation gives another bounds query: the lane must select
one of the group's four positions. Its bounds are numeric, so
``const_int_bound`` can report them directly:

.. code-block:: python

   lane_bounds = analyzer.const_int_bound(lane)
   print(lane_bounds.min_value, lane_bounds.max_value)

The output is:

.. code-block:: text

   0 3

These endpoints are also inclusive, describing the four lane positions.
Bounds cover every possible value; for more complex expressions, an
inferred interval may also contain values that never occur.

The start offset also matters for a vector access. A pass can use
``modular_set`` to check whether every group starts on a multiple of four:

.. code-block:: python

   multiples = analyzer.modular_set(group_start)
   print(multiples.coeff, multiples.base)

The output is:

.. code-block:: text

   4 0

``coeff`` and ``base`` describe values of the form ``coeff*k + base``
for integer ``k``. Every group therefore starts on a four-element boundary.
A vectorization pass can use this offset information together with the
base-address alignment and access bounds.

Proving Conditions
------------------

A compiler pass often needs a yes-or-no answer before it removes a check
or replaces an index. A proof query asks whether the required condition
holds for every value allowed by the known facts. The following calls
check the original index, the neighboring index, the regrouped index, and
the last element of its group of four:

.. code-block:: python

   group_last = group_start + 3
   print(analyzer.can_prove(i < n))
   print(analyzer.can_prove(next_index < n))
   print(analyzer.can_prove_equal(group_start + lane, i))
   print(analyzer.can_prove(group_last < n))

The output is:

.. code-block:: text

   True
   False
   True
   False

The known range proves ``i < n``, so a redundant check of this condition
inside the loop can be removed. The neighboring index can reach ``n``,
so ``i + 1 < n`` cannot be established for every iteration. That access
needs a guard or an iteration range that excludes the last element.

The final query checks the group's last index. The original index
can be valid while part of its group lies beyond the tensor: with ``n = 6``
and ``i = 5``, the group starts at 4 and ends at 7. Offsets 6 and 7 are out
of bounds. The lane range and alignment established earlier do not by
themselves justify a full vector access; a pass also needs to establish
``group_last < n`` or handle the partial group.

In general, ``False`` means that the analyzer did not prove a condition;
it does not prove the opposite condition. A pass cannot use that result
to justify removing a bounds check.

``can_prove_equal`` compares expression values. Here, its ``True`` result
confirms that a pass can replace ``group_start + lane`` with ``i`` while
preserving the element index, even though their IR structures differ.

.. _symbolic-constraint-management:

Managing Facts Across Program Points
------------------------------------

A range is one kind of premise. ``sym.ConstraintContext`` can also record
bindings and boolean conditions:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Constraint
     - Fields
     - Meaning
   * - ``sym.RangeConstraint``
     - ``var``, ``range``
     - A half-open superset of a variable's possible values, without stride.
   * - ``sym.BindingConstraint``
     - ``var``, ``value``
     - The primitive value captured at a variable's binding.
   * - ``sym.PredicateConstraint``
     - ``condition``, ``assume_always``
     - A boolean premise whose lifetime is supplied by the collector.

These are data objects; an analyzer consumes the premises after
synchronization. For example, a branch that checks ``i + 1 < n`` supplies
the additional condition needed by the neighboring-element access. A
snapshot lets us represent the true branch of that check without changing
the loop's context:

.. code-block:: python

   branch_context = context.snapshot()
   branch_context.append(sym.PredicateConstraint(i + 1 < n))
   analyzer.sync_constraint_context(branch_context)
   print(analyzer.can_prove(i + 1 < n))

   analyzer.sync_constraint_context(context)
   print(analyzer.can_prove(i + 1 < n))

The output is:

.. code-block:: text

   True
   False

Switching back removes the branch-only premise. Appending facts to a context
requires another synchronization; an unchanged context is inexpensive to
synchronize. An empty ``ConstraintContext`` removes all collected premises.
``analyzer.clone()`` makes an independent analyzer with the same accumulated
facts when an analysis needs to explore alternatives.

Appending the predicate represents a condition already known to hold at
that program point, such as entering the true branch of an existing test.
It does not prove that the test is true for every loop iteration.

TIRx's :ref:`constraint-aware traversals <ir-traversal-with-constraints>`
collect these facts from program constructs. Bindings become available
after visiting their values. Positive-step loops supply body ranges;
branches and while bodies supply guards. Primitive ``Let`` expressions
supply body-local bindings. Lazy ``if_then_else`` arguments receive separate
guards; eager ``Select`` operands do not. Ordinary assertions require a
pure condition, while ``tirx.assume_always`` supplies an explicit guarantee
of predicate truth throughout its scope.

The traversal context is live and callbacks must not append to it.
``list(constraints)`` retains a separate sequence of premise handles;
``constraints.snapshot()`` captures a context that can be extended
independently. Both still share the premise objects and their expressions,
which must remain unchanged through aliases.

A binding such as ``x = A[0]`` captures a value, while a later ``A[0]`` may
observe a write. Collection does not snapshot memory. The built-in analyzer
conservatively ignores bindings, range bounds, and predicates with impure
expressions; custom consumers must also respect premise lifetimes and
memory effects. The examples in this chapter use scalar dimensions and
indices.

.. _ir-traversal-with-constraints:

Analyzing a Guard During Traversal
----------------------------------

We have supplied facts manually so far. TIRx extends the structural
operations from :doc:`structural_utilities` with traversal that collects
constraints at each program point. We can connect that collector to an
analyzer for a redundant add-one check:

.. code-block:: python

   @T.function
   def guarded_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           if i < n:
               B[i] = A[i] + 1

   guard_analyzer = sym.Analyzer()
   proven_guards = []

   def inspect_guard(value, constraints):
       if isinstance(value, ir.If):
           guard_analyzer.sync_constraint_context(constraints)
           proven_guards.append(guard_analyzer.can_prove(value.condition))

   tirx.structural_walk_with_constraint_context(
       guarded_add_one, inspect_guard, order="pre"
   )
   print(proven_guards)

The output is ``[True]``. The callback receives the loop range valid at the
``ir.If``. Synchronization makes that range available to the analyzer, which
proves the condition. The pre-order callback runs before entering either
branch, so the proof uses the surrounding loop facts. A condition's own
premise inside its true branch would not establish that the test is redundant.

Synchronize at every callback that makes a query; this also removes facts
when traversal returns to an outer scope. These steps provide the analysis
needed by the pass example in :doc:`writing_passes`. The remaining sections
extend the grouped-index example to coordinate mappings, access regions,
and stronger proofs.

.. _symbolic-iteration-maps:

Mapping Groups and Lanes
-------------------------

The group-of-four calculation describes a coordinate transformation as
well as an arithmetic identity. To examine the whole mapping, consider a
fixed domain of sixteen indices. This example uses a fresh analyzer for that
independent domain. ``detect_iter_map`` takes the output coordinates and
the input domains:

.. code-block:: python

   k = ir.Var("k", "int32")
   map_analyzer = sym.Analyzer()
   detected = sym.detect_iter_map(
       [k // 4, k % 4],
       {k: ir.Range(0, 16)},
       check_level="bijective",
       analyzer=map_analyzer,
   )
   assert len(detected.indices) == 2, detected.errors

Each input index selects a group and lane:

.. code-block:: text

   k in [0, 16) ──→ (k // 4, k % 4)
                         │       │
                     group     lane
                     [0, 4)    [0, 4)

Bijectivity establishes that each output coordinate has exactly one input.
We can then recover that input from symbolic output coordinates:

.. code-block:: python

   group = ir.Var("group", "int32")
   lane_index = ir.Var("lane_index", "int32")
   inverse = sym.inverse_affine_iter_map(detected.indices, [group, lane_index])
   print(inverse[k])
   print(map_analyzer.can_prove_equal(inverse[k], group * 4 + lane_index))

The output is:

.. code-block:: text

   lane_index + group * 4
   True

``inverse[k]`` expresses the original coordinate in terms of the output
coordinates: group 1, lane 2 recovers index 6. The value proof confirms this
formula even though its printed operand order differs from
``group * 4 + lane_index``.

``check_level="surjective"`` checks output coverage
without requiring a unique preimage; ``"nocheck"`` skips that validation.
The enum is ``sym.iter_affine_map.IterMapLevel``. An unchecked match does
not justify inversion. Detection failure gives an empty ``indices`` array
and diagnostic ``errors``.

The detected coordinates are ``IterSumExpr`` objects, derived from
``IterMapExpr``. ``IterMark`` records a source and extent;
``IterSplitExpr`` selects and scales a component; ``IterSumExpr`` combines
components with a base. ``sym.normalize_iter_map_to_expr`` converts them
back to ordinary IR expressions.

``tirx.IndexMap`` packages a coordinate transformation for reuse:

.. code-block:: python

   split = tirx.IndexMap.from_func(lambda index: (index // 4, index % 4))
   print([int(v) for v in split.map_indices([6])])
   print([int(v) for v in split.map_shape([16])])

The output is:

.. code-block:: text

   [1, 2]
   [4, 4]

``map_indices`` applies the map and ``map_shape`` infers output extents.
``inverse`` derives an inverse over a supplied domain;
``non_surjective_inverse`` also returns a padding predicate when the image
does not fill the output space. ``is_equivalent_to`` compares coordinate
mappings with symbolic reasoning. These operations accept an analyzer to
reuse its facts. An index map describes coordinates; a pass must still
rewrite accesses and establish that the program transformation is valid.

.. _symbolic-region-analysis:

Covering an Accessed Region
---------------------------

The earlier ``int_set(i + 1)`` query covered the values of one expression.
A region query can cover several elements at each iteration. Suppose four
iterations each access a contiguous group of four elements:

.. code-block:: python

   tile = ir.Var("tile", "int32")
   region = [ir.Range.from_min_extent(tile * 4, 4)]
   covered = sym.estimate_region_strict_bound(
       region,
       {tile: ir.Range(0, 4)},
       predicate=True,
       analyzer=map_analyzer,
   )
   assert covered is not None
   print(map_analyzer.simplify(covered[0].min_value))
   print(map_analyzer.simplify(covered[0].max_value))

The output is:

.. code-block:: text

   0
   15

``predicate=True`` includes every iteration in the supplied domain. The
result covers all sixteen elements. The input region is a list of
half-open ranges; the returned integer-set endpoints are inclusive.
``ir.Range(begin, end)`` supplies endpoints, while
``ir.Range.from_min_extent(begin, extent)`` supplies a size.

Different estimators provide different coverage guarantees:

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - API
     - Interpretation
   * - ``sym.estimate_region_upper_bound``
     - Cover every accessed point, possibly adding points.
   * - ``sym.estimate_region_lower_bound``
     - Find a covered subregion, possibly discarding accessed points.
   * - ``sym.estimate_region_strict_bound``
     - Require a result without discarding or relaxing the region.

A lower bound cannot justify allocating only that subregion for all
accesses. Lower and strict estimation may return ``None``; upper bounds
may be wider than the actual region. These queries describe geometry and
do not establish memory dependencies or aliasing properties.

For example, accessing only two elements from each group would touch
``[0, 1]``, ``[4, 5]``, ``[8, 9]``, and ``[12, 13]``. An upper bound of
``[0, 13]`` covers every access and also includes the gaps. Being inside
such a bound does not imply that an element was accessed.

``sym.IntSet`` is the base integer-set interface. ``sym.IntervalSet(lo, hi)``
constructs inclusive intervals, ``IntSet.single_point`` represents one
value, and ``IntSet.vector`` covers a vector expression's lanes.
``is_nothing()`` and ``is_everything()`` inspect empty and universal sets.
``sym.int_set.pos_inf()`` and ``neg_inf()`` construct symbolic infinities;
``union_lower_bound`` computes a lower bound for a union of regions.

Stronger Proofs with Z3
-----------------------

The preceding examples use the native analyzers. More complex expressions
can benefit from stronger reasoning. ``can_prove`` accepts
``sym.ProofStrength.SYMBOLIC_BOUND`` to request symbolic-bound reasoning
and, when available, a Z3 fallback after native reasoning fails. Its default
``ProofStrength.DEFAULT`` uses native reasoning without that solver fallback.

The earlier group calculation used a constant width of four. A group can
also have a symbolic width and starting offset. In this example, ``start``,
``index``, and ``width`` are positive. Rounding ``index - start`` down to a
multiple of ``width`` and adding ``start`` gives a group start at or before
``index``:

.. code-block:: python

   start = ir.Var("start", "int32")
   index = ir.Var("index", "int32")
   width = ir.Var("width", "int32")
   proof_context = sym.ConstraintContext()
   for var in (start, index, width):
       proof_context.append(sym.PredicateConstraint(var > 0))

   prover = sym.Analyzer()
   prover.sync_constraint_context(proof_context)
   tile_start = ((index - start) // width) * width + start
   condition = tile_start <= index

   if prover.is_z3_enabled:
       prover.set_z3_timeout_ms(1000)
       assert prover.can_prove(condition, sym.ProofStrength.SYMBOLIC_BOUND)

Build with ``TVM_TIRX_USE_Z3=ON`` and check ``is_z3_enabled`` before using
solver-specific configuration or diagnostics. Installing Python's
``z3-solver`` package alone does not enable the native backend. A ``False``
proof result still means unproved, including when translation is unsupported
or the solver reaches a resource limit.

``set_z3_rlimit`` supplies a deterministic solver budget; zero disables that
limit. ``get_smtlib2(expr=None)`` exports the current problem, optionally
with a negated goal, and ``get_z3_stats()`` exposes solver statistics. These
methods and ``set_z3_timeout_ms`` require a Z3-enabled build.

Further Analysis Tools
-----------------------

The examples above cover the analysis workflow. The following index groups
additional entry points by the tasks they support; full signatures are in
the API references linked below:

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - Tool
     - Use
   * - ``Analyzer.rewrite_simplify`` and ``canonical_simplify``
     - Request a simplification stage directly. ``simplify(steps=2)`` runs
       rewriting followed by canonicalization by default.
   * - ``Analyzer.try_compare``
     - Return ``sym.CompareResult`` using directly known comparisons and
       constant offsets. ``UNKNOWN`` does not establish an ordering.
   * - ``Analyzer.int_set(expr, dom_map)``
     - Supply explicit variable domains as ``IntSet`` objects. Omitting the
       mapping uses accumulated domains.
   * - ``Analyzer.const_int_bound_is_bound``
     - Check whether the constant-bound analyzer has a variable domain.
   * - ``Analyzer.set_maximum_rewrite_steps``, ``rewrite_simplify_stats``,
       and ``reset_rewrite_simplify_stats``
     - Bound and inspect native rewriting work. Exceeding a positive step
       limit raises an error.
   * - ``sym.iter_map_simplify``
     - Simplify indices using iteration-map detection; return the original
       expressions if no match is found.
   * - ``sym.normalize_to_iter_sum``
     - Decompose an expression into an iteration sum, leaving undetected
       parts in its base.
   * - ``sym.subspace_divide``
     - Separate selected iterators' contribution from the remaining space,
       with outer and inner predicates.
   * - ``sym.deduce_bound``
     - Derive a target variable's integer set from a condition, using hint
       and relaxation domains.
   * - ``tirx.analysis.detect_linear_equation``
     - Recover coefficients and a base relative to selected variables.
   * - ``tirx.analysis.detect_clip_bound``
     - Recognize lower and upper bounds in a clip condition.

For example, linear detection recovers the strides of a flattened index:

.. code-block:: python

   row = ir.Var("row", "int32")
   column = ir.Var("column", "int32")
   coefficients = tirx.analysis.detect_linear_equation(row * 16 + column, [row, column])
   print([int(value) for value in coefficients])

The output is ``[16, 1, 0]``: the row coefficient, column coefficient, and
base. Linear and clip detection return an empty list on failure. A missing
individual clip bound can be ``None``. Iter-map detection, simplification,
normalization, subspace division, region queries, and ``IndexMap`` methods
can reuse an explicitly supplied analyzer; synchronize its facts before
calling them.

Arithmetic analysis must respect the expression's dtype and division
semantics. ``sym.allow_uint_as_index()`` explicitly opts into no-wraparound
assumptions for unsigned scalar indices during compile-time layout proofs.
The caller must justify those assumptions; they do not follow from the
unsigned dtype alone.

The native counterparts are in ``tvm/sym/analyzer.h``, ``constraint.h``,
``int_set.h``, and ``iter_affine_map.h``. :doc:`/reference/python/sym`,
:doc:`/reference/python/tirx.analysis`, and :class:`tvm_tirx.tirx.IndexMap`
provide the full signatures. Continue with :doc:`writing_passes` to apply
these tools in a transformation pipeline and inspect the resulting IR.
