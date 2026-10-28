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

Traversing and Transforming IR
==============================

In this chapter, we will use TVM FFI's structural operations to inspect and
rewrite IR. We will apply walk, map, visitor, and mutator callbacks to the
same expression, examine the traversal order, and choose which children to
process. We will then apply these tools to the add-one function from the
:doc:`IR Walkthrough </ir/overview>`.

The APIs have two independent choices: whether to inspect or replace values,
and whether the operation or the callback controls recursion:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Task
     - Automatic recursion
     - Callback-controlled recursion
   * - Inspect values
     - ``tvm_ffi.structural_walk``
     - ``tvm_ffi.structural_visit``
   * - Replace values
     - ``tvm_ffi.structural_map``
     - ``tvm_ffi.structural_mutate``

For collecting nodes or applying a rule to every node, start with walk or
map. Choose visit or mutate when a callback must select children, arrange
their processing order, or manage state around recursive calls.

API Inputs and Callback Selection
---------------------------------

The four entry points have these Python call signatures:

.. code-block:: text

   tvm_ffi.structural_walk(
       root, callbacks=(), with_def_region_kind=(), order="post")
   tvm_ffi.structural_map(
       root, callbacks=(), with_def_region_kind=(), order="post")
   tvm_ffi.structural_visit(root, callbacks=())
   tvm_ffi.structural_mutate(root, callbacks=())

``root`` is the value at which traversal starts. It can be an expression,
statement, function, module, type, or FFI container. The root itself participates
in callback selection, as do values reached through structural descent.
These operations inspect IR objects; they do not execute the program those
objects describe.

``callbacks`` selects which values a handler receives. All four functions
accept the following forms; ``handler`` stands for a callable whose arguments
and return value depend on the operation:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Form
     - Meaning
   * - ``(ir.Var, handler)``
     - Handle variables.
   * - ``((ir.prim.Add, ir.prim.Mul), handler)``
     - Handle either addition or multiplication with the same callable.
   * - ``[(ir.Var, on_var), (ir.Expr, on_expr)]``
     - Try handlers in the supplied order.
   * - ``handler``
     - Handle every value as a catch-all.
   * - ``()``
     - Install no handlers; use default structural descent throughout.

An IR class selector matches that class and its subclasses. For example,
``ir.Expr`` includes ``ir.Var`` and arithmetic expressions. Only the first
matching entry runs: selection does not automatically favor the most specific
type, and returning from a handler does not try another entry. Put specific
types before broader ones. An unmatched value still follows default descent;
selecting only ``ir.Var`` does not prevent traversal through an ``Add``.

Supported selectors include registered FFI object classes and built-in value
types such as ``int``. ``object`` and ``typing.Any`` match both inline values
and FFI objects; ``tvm_ffi.Object`` matches only object-backed FFI values.
``ir.IntImm`` selects an integer IR constant, while ``int`` selects an inline
integer value; they describe different kinds of values.

``order`` belongs to walk and map. ``"post"`` is the default; ``"pre"``
selects parent-before-children processing. The corresponding enum values are
``tvm_ffi.WalkOrder.POSTORDER`` and ``PREORDER``. Visit and mutate have no
``order`` argument because their callbacks place recursive calls themselves.

``with_def_region_kind`` belongs to walk and map. It is an alternative
callback table whose handlers also receive a ``tvm_ffi.DefRegionKind`` telling
them whether a value occurs in a definition region. Normal ``callbacks``
are tried first; this table is consulted only if none matches. Its entries
use the same selection forms above. We will use it in
:ref:`ir-definition-regions`.

The following sections distinguish two results: what a callback returns for
one value, and what the entry point returns for the whole traversal.

An Expression to Traverse
-------------------------

We will start with two integer variables and a small expression:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)
   print(expr)

The output is:

.. code-block:: text

   (x + 1) * (y + 2)

The expression has this structure. Binary arithmetic nodes store their
operands in ``a`` and ``b``:

.. code-block:: text

   Mul
   ├── a: Add
   │   ├── a: Var       x
   │   └── b: IntImm    1
   └── b: Add
       ├── a: Var       y
       └── b: IntImm    2

Each node type registers its structural fields with TVM FFI. The structural
operations follow these fields and registered child traversal, so a callback
can focus on the node types it handles. The examples below run in order and
share the variables defined above.

Structural Walk
---------------

A walk inspects values without replacing them. Its normal callback is called
as ``handler(value)``; a ``with_def_region_kind`` handler is called as
``handler(value, kind)``. The walker handles recursion. The callback's return
value controls that recursion:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Callback returns
     - Effect
   * - ``None`` or ``tvm_ffi.WalkResult.ADVANCE``
     - Continue normally.
   * - ``tvm_ffi.WalkResult.SKIP``
     - In pre-order, omit this value's children and continue elsewhere.
   * - ``tvm_ffi.VisitInterrupt(payload)``
     - Stop the whole walk and return the interrupt to the caller.

The entry point returns ``None`` on completion or the ``VisitInterrupt`` on
early exit. It does not return collected nodes or a rewritten root. Store
observations in callback state, such as a list, and read an early-exit payload
through the returned interrupt's ``value`` field.

Selecting Nodes
~~~~~~~~~~~~~~~

For example, collect the variables in the expression:

.. code-block:: python

   variables = []
   tvm_ffi.structural_walk(expr, (ir.Var, variables.append))
   print([var.name for var in variables])

The output is:

.. code-block:: text

   ['x', 'y']

``(ir.Var, variables.append)`` selects the callback by node type. The walker
still descends through the unmatched ``Mul`` and ``Add`` nodes to reach
the variables. ``append`` returns ``None``, which tells the walker to continue.
Walking leaves the input structure unchanged; keep observations in callback
state such as this list.

To see first-match selection, install both a variable handler and an expression
handler on a single variable:

.. code-block:: python

   selected = []
   tvm_ffi.structural_walk(
       x,
       [
           (ir.Var, lambda var: selected.append("variable")),
           (ir.Expr, lambda node: selected.append("expression")),
       ],
   )
   print(selected)

The output is ``['variable']``. Both selectors match ``x``, but only the first
handler runs. Its ``None`` return continues traversal; it does not request
the next handler.

Traversal Order
~~~~~~~~~~~~~~~

To see when callbacks run, we can record every expression node. ``ir.Expr``
matches its concrete subclasses, including variables and arithmetic nodes:

.. code-block:: python

   def describe(node):
       if isinstance(node, ir.Var):
           return node.name
       if isinstance(node, ir.IntImm):
           return str(int(node))
       return type(node).__name__

   for order in ("pre", "post"):
       trace = []
       tvm_ffi.structural_walk(
           expr,
           (ir.Expr, lambda node: trace.append(describe(node))),
           order=order,
       )
       print(order, trace)

The output is:

.. code-block:: text

   pre ['Mul', 'Add', 'x', '1', 'Add', 'y', '2']
   post ['x', '1', 'Add', 'y', '2', 'Add', 'Mul']

Pre-order visits a parent before its children. Post-order, the default,
visits the children before the parent. Within each binary expression, the
registered child order visits ``a`` before ``b``.

Skipping a Subtree
~~~~~~~~~~~~~~~~~~

A pre-order callback can return ``WalkResult.SKIP`` to omit the current
node's children. We can skip the addition containing ``x`` while continuing
through the rest of the expression:

.. code-block:: python

   trace = []

   def skip_left_add(node):
       trace.append(describe(node))
       if isinstance(node, ir.prim.Add) and node.a.same_as(x):
           return tvm_ffi.WalkResult.SKIP
       return None

   tvm_ffi.structural_walk(expr, (ir.Expr, skip_left_add), order="pre")
   print(trace)

The output is:

.. code-block:: text

   ['Mul', 'Add', 'Add', 'y', '2']

The first ``Add`` is observed, but its children ``x`` and ``1`` are skipped.
The second subtree is still visited. ``same_as(x)`` checks the identity of
the IR variable; another variable with the printed name ``x`` is a distinct
symbol. Return ``None`` or ``WalkResult.ADVANCE`` to continue normally.
Use pre-order for pruning, since a post-order callback runs after the
children have already been visited.

Stopping the Walk
~~~~~~~~~~~~~~~~~

``VisitInterrupt`` stops the entire traversal and carries a result:

.. code-block:: python

   def find_y(var):
       if var.same_as(y):
           return tvm_ffi.VisitInterrupt(var)
       return None

   found = tvm_ffi.structural_walk(expr, (ir.Var, find_y))
   if found is not None:
       print(found.value.name)

The output is ``y``. A completed walk returns ``None``; an interrupted walk
returns the ``VisitInterrupt``. Returning ``SKIP`` only prunes one subtree,
whereas returning an interrupt stops all remaining traversal.

Structural Map
--------------

A map applies replacement rules with automatic recursion. Its normal callback
is called as ``handler(value)``; a ``with_def_region_kind`` handler is called
as ``handler(value, kind)``. Each handler must return the value to keep at
that position: either its input unchanged or a valid replacement. The entry
point returns the resulting root. Assign that result to use the transformed IR.

The ``order`` argument determines both the callback's input and what happens
to its result:

.. list-table::
   :header-rows: 1
   :widths: 15 42 43

   * - Order
     - Callback receives
     - After the callback
   * - ``"pre"``
     - The value before child mapping.
     - Map the returned value's children, then return the rebuilt value.
   * - ``"post"``
     - The value with its children already mapped.
     - Return its result without descending into the replacement again.

In both cases, the mapper invokes a selected callback once at that position;
it does not repeatedly apply rules until nothing changes. A callback must not
modify its input in place. The examples below keep the original root available
and return new IR where needed.

Replacing Symbols
~~~~~~~~~~~~~~~~~

We can replace ``x`` with 8 and ``y`` with 3:

.. code-block:: python

   def specialize_var(var):
       if var.same_as(x):
           return ir.const(8, var.ty)
       if var.same_as(y):
           return ir.const(3, var.ty)
       return var

   bound = tvm_ffi.structural_map(expr, (ir.Var, specialize_var), order="pre")
   print(bound)
   print(expr)

The output is:

.. code-block:: text

   T.Add(8, 1) * T.Add(3, 2)
   (x + 1) * (y + 2)

The mapped result contains integer IR constants with the variables' types.
The source expression remains available. ``T.Add(8, 1)`` is the printer's
explicit form for an addition node whose operands are constants: structural
mapping replaces fields without automatically evaluating the arithmetic.

Pre-order is useful for symbol substitution: the callback chooses the value
whose children the mapper then processes. A replacement must fit the field's
IR type. Returning a bare Python integer cannot replace a typed expression
field here; ``ir.const(value, var.ty)`` constructs the required expression.

Rewriting after Children
~~~~~~~~~~~~~~~~~~~~~~~~

Post-order is useful when a parent's rewrite depends on its rewritten
children. We can evaluate constant additions and multiplications with one
callback shared by both node types:

.. code-block:: python

   def fold_arithmetic(node):
       if not isinstance(node.a, ir.IntImm) or not isinstance(node.b, ir.IntImm):
           return node
       lhs, rhs = int(node.a), int(node.b)
       value = lhs + rhs if isinstance(node, ir.prim.Add) else lhs * rhs
       return ir.const(value, node.ty)

   folded = tvm_ffi.structural_map(
       bound,
       ((ir.prim.Add, ir.prim.Mul), fold_arithmetic),
       order="post",
   )
   print(folded)

The output is ``45``. The callback replaces the two additions with 9 and 5
before it receives the enclosing multiplication, so it can replace that
node with 45 as well.

The same callback in pre-order gives a different result:

.. code-block:: python

   folded_pre = tvm_ffi.structural_map(
       bound,
       ((ir.prim.Add, ir.prim.Mul), fold_arithmetic),
       order="pre",
   )
   print(folded_pre)

The output is ``T.Mul(9, 5)``. The callback sees the multiplication before
its additions have become constants, so it returns that node unchanged.
Its children then become 9 and 5. The parent callback does not run a second
time after that descent.

Map callbacks must return an expression for these IR nodes, including when
nothing changes. Falling off a callback returns ``None``; it does not mean
"keep this node." There is no map result corresponding to ``SKIP`` or
``VisitInterrupt``. Use a mutator when a rewrite needs to choose its children.

Structural Visitor
------------------

``structural_visit`` calls its handler as ``handler(value, visitor)``.
``visitor`` is the active ``tvm_ffi.StructuralVisitor``, supplied by the
entry point rather than constructed by the callback. A matched callback
owns recursion: it chooses whether, when, and which children to visit.
Unmatched node types still follow default structural descent.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Within the callback
     - Effect
   * - ``visitor.visit(child)``
     - Run callback selection and traversal on the chosen child.
   * - ``visitor.default_visit(value)``
     - Bypass callback selection for this value and traverse its registered
       children. Child visits still use the same callbacks.
   * - Return ``None`` without recursive calls.
     - Finish this callback without visiting children.
   * - Return ``tvm_ffi.VisitInterrupt(payload)``.
     - Stop traversal and return the interrupt to the caller.

Both visitor methods return ``None`` or ``VisitInterrupt``. Return a child
interrupt from the callback so it reaches the entry point. The entry point
has the same return contract as walk: ``None`` on completion, otherwise the
interrupt. ``WalkResult`` is a walk-only control signal; visitors select
subtrees through their recursive calls instead.

Requesting Default Descent
~~~~~~~~~~~~~~~~~~~~~~~~~~

This callback records the current node but does not recurse:

.. code-block:: python

   visited = []
   tvm_ffi.structural_visit(
       expr,
       (ir.Expr, lambda node, visitor: visited.append(describe(node))),
   )
   print(visited)

The output is ``['Mul']``. The root matches the callback, which returns
without visiting children. To continue through the registered structure,
call ``visitor.default_visit(node)``:

.. code-block:: python

   visited = []

   def visit_expr(node, visitor):
       visited.append(describe(node))
       return visitor.default_visit(node)

   tvm_ffi.structural_visit(expr, (ir.Expr, visit_expr))
   print(visited)

The output is:

.. code-block:: text

   ['Mul', 'Add', 'x', '1', 'Add', 'y', '2']

``default_visit(node)`` bypasses the current callback for that node and
visits its registered children. Child visits dispatch through the same
callback table. This gives us the same order as a pre-order walk.
Moving the observation after ``default_visit`` gives post-order behavior;
propagate any ``VisitInterrupt`` returned by the recursive call before
continuing the observation.

Selecting Children
~~~~~~~~~~~~~~~~~~

``visitor.visit(child)`` dispatches on a selected child. For example, visit
only the multiplication's left operand and collect variables reached there:

.. code-block:: python

   left_variables = []

   def visit_left(node, visitor):
       return visitor.visit(node.a)

   tvm_ffi.structural_visit(
       expr,
       [
           (ir.prim.Mul, visit_left),
           (ir.Var, lambda var, visitor: left_variables.append(var.name)),
       ],
   )
   print(left_variables)

The output is ``['x']``. The ``Mul`` callback selects ``a``; the unmatched
``Add`` descends normally; the ``Var`` callback records ``x``. Nothing visits
the right subtree containing ``y``.

Use ``visit`` for a child that should enter callback dispatch. In an engine
callback, use ``default_visit`` on the current node for its normal descent.
Calling ``visitor.visit(node)`` on that same current node would invoke its
callback again rather than advance to the children.

Structural Mutator
------------------

``structural_mutate`` calls its handler as ``handler(value, mutator)``.
``mutator`` is the active ``tvm_ffi.StructuralMutator`` supplied by the
entry point. The callback returns the value to install at this position and
owns all recursive processing of its matched value:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Within the callback
     - Effect
   * - ``mutator.mutate(child)``
     - Run callback selection and mutation on the chosen child; return its result.
   * - ``mutator.default_mutate(value)``
     - Bypass this value's callback and return its default mutation, with child
       callbacks applied through the same mutator.
   * - ``return value`` without recursive calls.
     - Keep this value with its children unprocessed.
   * - ``return replacement``
     - Install the replacement without further traversal by the engine.

Unmatched values follow default mutation. The entry point returns the resulting
root, which the caller must retain. Every matched callback must return a valid
value, even when unchanged; ``None`` does not request default mutation or mean
"keep the input." There is no ``SKIP`` or ``VisitInterrupt`` control result.

Use ``mutate`` on a selected child and ``default_mutate`` on the current
value to continue through the same dispatch. Calling ``mutate`` on the current
value dispatches to its callback again. Like visit, mutate has no ``order``
parameter: placing the local rewrite before or after recursive calls chooses
the processing order.

Rewriting Selected Children
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we want to replace variables only in the left subtree. A map would
apply its variable rule throughout the expression. A mutator can select the
left operand, rewrite it, and rebuild the multiplication with its original
right operand:

.. code-block:: python

   def mutate_left(node, mutator):
       left = mutator.mutate(node.a)
       if left.same_as(node.a):
           return node
       return ir.prim.Mul(left, node.b, loc=node.loc)

   def replace_var(var, mutator):
       return ir.const(8, var.ty)

   partial = tvm_ffi.structural_mutate(
       expr,
       [
           (ir.prim.Mul, mutate_left),
           (ir.Var, replace_var),
       ],
   )
   print(partial)
   print(partial.b.same_as(expr.b))
   print(expr)

The output is:

.. code-block:: text

   T.Add(8, 1) * (y + 2)
   True
   (x + 1) * (y + 2)

``mutator.mutate(node.a)`` applies callback dispatch to the selected child.
The unmatched addition follows default mutation, reaching ``x`` and replacing
it with 8. The root callback returns a new multiplication that shares the
original right subtree. Its source location is retained through ``loc``.

Requesting Default Mutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mutator.default_mutate(node)`` rewrites the current node's children without
invoking its engine callback again on that node. We can use it to write an
explicit bottom-up transformation:

.. code-block:: python

   def mutate_and_fold(node, mutator):
       rewritten = mutator.default_mutate(node)
       if isinstance(rewritten, (ir.prim.Add, ir.prim.Mul)):
           return fold_arithmetic(rewritten)
       return rewritten

   folded_manual = tvm_ffi.structural_mutate(bound, (ir.Expr, mutate_and_fold))
   print(folded_manual)

The output is ``45``. Child mutation happens before the local fold, giving
the same result as the post-order map. A map is shorter when this automatic
order is all a rewrite needs; a mutator lets each callback arrange its own
recursive calls. As with visiting, calling ``mutator.mutate(node)`` on the
current node re-enters that node's callback. Call ``default_mutate`` for
default descent, or ``mutate`` on a chosen child.

Returned Replacements Are Final
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine does not traverse a mutator callback's returned replacement again.
We can observe this by replacing a variable with a newly constructed addition:

.. code-block:: python

   new_literals = []

   def replace_symbol(var, mutator):
       return ir.prim.Add(ir.const(10, var.ty), ir.const(20, var.ty))

   def observe_literal(value, mutator):
       new_literals.append(int(value))
       return value

   replacement = tvm_ffi.structural_mutate(
       x,
       [
           (ir.Var, replace_symbol),
           (ir.IntImm, observe_literal),
       ],
   )
   print(replacement)
   print(new_literals)

The output is:

.. code-block:: text

   T.Add(10, 20)
   []

Neither new literal is passed to ``observe_literal``. If a replacement's
children need processing, the callback must explicitly request that descent.
This lets a callback determine exactly which parts of a replacement are
already complete.

Ownership and Shared Objects
----------------------------

Container hooks visit ``Array`` and ``List`` elements, and ``Map`` and
``Dict`` values. Map and dictionary keys remain structural anchors, so a
mapping callback does not replace them. Reflected fields marked
``structural_eq="ignore"`` are also omitted from default traversal.

The examples pass ordinary Python references. Mapping and mutation preserve
the source through copy-on-write, reuse unchanged children, and return the
original node when default descent makes no changes. A map callback must
not modify its input in place. The name "mutator" describes the replacement
operation; using ``structural_mutate`` does not itself request in-place writes.
Ownership-transfer and optional native in-place hooks are separate mechanisms
covered by TVM FFI's ownership contract. For ownership-aware implementations,
a ``structural_mutate`` callback may accept a third positional argument,
``handler(value, mutator, inplace_mode)``. This integer compares with
``tvm_ffi.InplaceMode.DISALLOW`` and ``ALLOW``; ``ALLOW`` permits storage reuse
only along a uniquely owned path. Passing ``root._move()`` to map or mutate
transfers ownership of that Python reference. The two-argument callbacks and
ordinary references used in this chapter do not need that mechanism.

Walks visit occurrences rather than deduplicating shared objects:

.. code-block:: python

   shared = x + 1
   twice = ir.prim.Mul(shared, shared)
   additions = []
   tvm_ffi.structural_walk(twice, (ir.prim.Add, additions.append))
   print(len(additions))
   print(additions[0].same_as(additions[1]))

The output is:

.. code-block:: text

   2
   True

There are two occurrences of the same addition. To visit an object once,
a pre-order walk can keep a visited set and return ``SKIP`` on repeats.
Map callbacks also run at each occurrence; a rewrite that must preserve
shared outputs keeps its own memo. The default variable-remapping behavior
helps keep variable definitions and uses consistent, but it does not turn
all callbacks into a graph-wide memoized rewrite.

Working on Functions and Types
------------------------------

The same operations work on statements, functions, modules, and type objects.
For example, use the add-one function from the walkthrough:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

.. _ir-shape-specialization:

The dimension ``n`` occurs in both tensor types and the loop bound. A map
can specialize these uses together:

.. code-block:: python

   add_one_specialized = tvm_ffi.structural_map(
       add_one,
       (ir.Var, lambda var: ir.const(128, n.ty) if var.same_as(n) else var),
       order="pre",
   )
   add_one_specialized.show()

The displayed function includes:

.. code-block:: text

   @T.function
   def add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
       for i in range(128):
           B[i] = A[i] + T.float32(1.0)

Traversal reaches shape expressions in ``TensorType`` as well as the body.
The original function still has its symbolic shape. ``show()`` displays
the result, while ``script()`` returns its TVMScript source as a string.

.. _ir-definition-regions:

Definitions and Uses
~~~~~~~~~~~~~~~~~~~~

A variable may occur at its definition or at a use. Walk and map accept
``with_def_region_kind`` callbacks with arguments ``(value, kind)`` to
distinguish those positions:

.. code-block:: python

   regions = []
   tvm_ffi.structural_walk(
       add_one.body[0],
       with_def_region_kind=(
           ir.Var,
           lambda var, kind: regions.append((var.name, kind.name)),
       ),
   )
   print([kind for name, kind in regions if name == "i"])

The output is:

.. code-block:: text

   ['DEF_SIMPLE', 'NONE', 'NONE']

The first occurrence defines the loop variable; the other two use it in the
store and load indices. ``NONE`` means an ordinary use, ``DEF_SIMPLE`` marks
a simple variable definition, and ``DEF_PATTERN`` marks a pattern definition,
which can bind symbols through a type. Map callbacks with definition-region
information must return the unchanged value or a valid replacement.
Substituting a dimension use is different from removing a parameter or
loop binder; the resulting definition sites must remain valid.

Visit and mutate expose the current kind through
``visitor.def_region_kind()`` and ``mutator.def_region_kind()`` instead of
accepting a separate callback table. Each also provides
``with_def_region_kind(kind, callback)``: it calls the zero-argument callable
under that definition kind, restores the previous kind afterward, and returns
the callable's result. Default descent already enters the definition regions
registered by IR nodes; explicit scopes are for callbacks implementing their
own definition traversal.

Further Tools
-------------

``ir.assert_structural_equal(actual, expected)`` can check an independently
constructed rewrite result and report mismatches. The structural comparison,
hash, and diagnostic APIs are described in the TVM FFI
`Structural Walk and Map guide <https://github.com/apache/tvm-ffi/blob/dbdcdaceeef5b0b03251e8dea87725a1c1e2cde6/docs/concepts/structural_eq_hash.rst#structural-walk-and-map>`_.

The guide also describes custom ``__s_visit__`` and mutation hooks. Those are
a separate layer from the engine callbacks used here: for example, calling
``default_visit`` on the current value inside its own ``__s_visit__`` hook
would re-enter that hook. Use each layer's recursion contract when extending
a node type.

C++ provides ``ffi::StructuralWalk`` and ``ffi::StructuralVisit`` in
``tvm/ffi/extra/structural_visit.h``, and ``StructuralMap`` and
``StructuralMutate`` in ``tvm/ffi/extra/structural_mutate.h``. TIRx's native
``ExprVisitor`` and ``ExprMutator`` bases are in ``tvm/ir/expr_functor.h``;
``StmtExprVisitor`` and ``StmtExprMutator`` are in ``tvm/tirx/stmt_functor.h``.
These provide per-node hooks for larger native implementations.

TIRx additionally provides structural operations that carry the symbolic
constraints active at each program point. :doc:`symbolic_integer_analysis`
explains how to use them with an analyzer. :doc:`writing_passes` shows how
to package a rewrite as a pass and inspect its results.
