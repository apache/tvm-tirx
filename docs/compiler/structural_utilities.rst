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
rewrite IR. Each API is introduced with its signature, callback rules, and
examples. Arithmetic expressions show traversal and rewriting; the add-one
function from the :doc:`IR Walkthrough </ir/overview>` shows how these
operations reach function bodies and tensor types.

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

The examples use the same arithmetic expression, and each section constructs
its own inputs so it can be run independently.

Structural descent follows the fields and child traversal registered by each
node type. It also follows ``Array`` and ``List`` elements and ``Map`` and
``Dict`` values; map keys and fields marked ``structural_eq="ignore"`` are
omitted. These operations inspect and rewrite IR objects without executing
the program they describe.

Structural Walk
---------------

``tvm_ffi.structural_walk`` inspects values with automatic recursion. Use it
to collect nodes, check properties, or find a value without replacing IR.

.. code-block:: text

   tvm_ffi.structural_walk(
       root, callbacks=(), with_def_region_kind=(), order="post")

``root`` is the expression, statement, function, module, type, or FFI container
where traversal starts. ``callbacks`` accepts ``(Type, handler)``, a grouped
``((TypeA, TypeB), handler)``, a list of such entries, or a bare callable that
matches every value. The default ``()`` installs no handlers. A normal handler
is called as ``handler(value)``.

An IR class selector also matches its subclasses. Only the first matching
entry runs, so put specific types before broader ones. Unmatched values still
follow default descent. ``ir.IntImm`` selects an integer IR constant, while
``int`` selects an inline integer value; ``object`` or ``typing.Any`` matches
both inline values and FFI objects, and ``tvm_ffi.Object`` matches only FFI
objects.

``order="post"`` calls the handler after the children; ``order="pre"`` calls
it before them. The corresponding enum values are
``tvm_ffi.WalkOrder.POSTORDER`` and ``PREORDER``. The optional
``with_def_region_kind`` table uses the same selection forms, but calls
``handler(value, kind)`` with a ``tvm_ffi.DefRegionKind``. Normal callbacks are
tried first; this table is consulted only when none matches.

The callback's return value controls traversal:

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

``structural_walk`` returns ``None`` on completion or the ``VisitInterrupt`` on
early exit. It does not return collected nodes or a rewritten root. Store
observations in callback state, such as a list, and read an early-exit payload
through the returned interrupt's ``value`` field.

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

The diagram below shows the arithmetic operands. Binary arithmetic nodes
store their operands in ``a`` and ``b``:

.. code-block:: text

   Mul
   ├── a: Add
   │   ├── a: Var       x
   │   └── b: IntImm    1
   └── b: Add
       ├── a: Var       y
       └── b: IntImm    2

Structural traversal also reaches types and other registered fields omitted
from this diagram. The traces below select ``ir.Expr`` to display only
expression nodes. Each node type registers its structural fields or custom
child traversal with TVM FFI, so callbacks can focus on the types they handle.
The examples in this section run in order and share these variables.

We can now collect the variables in the expression:

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

A walk visits occurrences rather than deduplicating shared objects:

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

.. _ir-definition-regions:

The optional ``with_def_region_kind`` table distinguishes variable definitions
from uses. For a complete loop example, use the add-one function from the
:doc:`IR Walkthrough </ir/overview>`:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

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
which can bind symbols through a type.

Structural Map
--------------

``tvm_ffi.structural_map`` applies replacement rules with automatic recursion.
Use it for substitutions and rewrites that process every structural child.

.. code-block:: text

   tvm_ffi.structural_map(
       root, callbacks=(), with_def_region_kind=(), order="post")

``root`` is the expression, statement, function, module, type, or FFI container
to transform. ``callbacks`` accepts ``(Type, handler)``, a grouped
``((TypeA, TypeB), handler)``, a list of entries, or a bare catch-all callable.
The default ``()`` installs no handlers. IR class selectors match subclasses,
and only the first matching entry runs; put specific types before broad ones.
Unmatched values follow default mapping. ``object`` or ``typing.Any`` matches
all values, while ``tvm_ffi.Object`` matches only FFI objects.

A normal callback is called as ``handler(value)`` and returns the value to
install: either the input unchanged or a valid replacement. ``None`` does
not mean "keep this value." ``structural_map`` returns the transformed root;
assign this result to use the new IR.

``with_def_region_kind`` accepts the same entry forms but calls
``handler(value, kind)``, where ``kind`` is a ``tvm_ffi.DefRegionKind``. This
table is tried only if no normal callback matches. ``order`` accepts
``"post"`` (the default), ``"pre"``, or the corresponding
``tvm_ffi.WalkOrder.POSTORDER`` and ``PREORDER`` enum values.

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

First construct an expression to transform:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

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

Starting again from ``bound``, the same callback in pre-order gives a different
result:

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

We can now apply the same variable-substitution rule to a complete function.
The root changes from an arithmetic expression to a function, whose structural
children include both its parameter types and its body. We will use the
add-one function from the walkthrough:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

.. _ir-shape-specialization:

The dimension ``n`` occurs in both tensor types and the loop bound. As in
``specialize_var``, an identity test selects the symbol to replace. A map
can specialize these occurrences together:

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

With ``with_def_region_kind``, the replacement callback receives
``(value, kind)``. ``NONE`` marks an ordinary use, ``DEF_SIMPLE`` a simple
definition, and ``DEF_PATTERN`` a pattern definition that can bind symbols
through a type. The callback still returns the unchanged value or a valid
replacement. Substituting a dimension is different from removing a parameter
or loop binder: the resulting definition sites must remain valid.

Passing an ordinary Python reference preserves the source through
copy-on-write. Default mapping reuses unchanged children and copies changed
paths. Callbacks run at each occurrence of a shared object, so a rewrite that
must preserve shared outputs keeps its own memo. The default variable remap
maintains definition/use consistency during descent; it does not memoize all
callback results. Passing ``root._move()`` transfers ownership and permits
storage reuse along uniquely owned paths; map callbacks must still return
values without modifying their input in place.

Structural Visitor
------------------

``tvm_ffi.structural_visit`` inspects values while letting each matched
callback choose which children to visit and in what order.

.. code-block:: text

   tvm_ffi.structural_visit(root, callbacks=())

``root`` is the expression, statement, function, module, type, or FFI container
where visiting starts. ``callbacks`` accepts ``(Type, handler)``, a grouped
``((TypeA, TypeB), handler)``, a list of entries, or a bare catch-all callable.
The default ``()`` installs no handlers. IR class selectors match subclasses;
only the first matching entry runs, so put specific types before broad ones.
Unmatched values follow default structural descent. ``object`` or
``typing.Any`` matches all values, while ``tvm_ffi.Object`` matches only FFI
objects.

The callback is called as ``handler(value, visitor)``. The supplied
``tvm_ffi.StructuralVisitor`` carries the same callbacks into recursive calls.
A matched callback owns descent; it can choose these operations:

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
interrupt from the callback so it reaches the caller. ``structural_visit``
returns ``None`` on completion, otherwise the interrupt.
``WalkResult`` is a walk-only control signal; visitors select
subtrees through their recursive calls instead. There is no ``order`` argument;
the callback places its recursive calls before or after its local work.

We will inspect the following expression:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

   def describe(node):
       if isinstance(node, ir.Var):
           return node.name
       if isinstance(node, ir.IntImm):
           return str(int(node))
       return type(node).__name__

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

An early exit must propagate through recursive calls. This callback finds
``y`` and returns any interrupt produced while visiting children:

.. code-block:: python

   def visit_until_y(node, visitor):
       if node.same_as(y):
           return tvm_ffi.VisitInterrupt(node)
       return visitor.default_visit(node)

   visitor_found = tvm_ffi.structural_visit(expr, (ir.Expr, visit_until_y))
   print(visitor_found.value.name)

The output is ``y``. When a callback visits several children explicitly, check
each recursive result and return an interrupt before visiting the next child.

``visitor.def_region_kind()`` reports the current ``tvm_ffi.DefRegionKind``:
``NONE`` for ordinary uses, ``DEF_SIMPLE`` for simple definitions, or
``DEF_PATTERN`` for pattern definitions. Default descent enters the regions
registered by IR nodes. For custom definition traversal,
``visitor.with_def_region_kind(kind, callback)`` runs a zero-argument callable
in a scoped region and returns its result. Outside ``DEF_PATTERN``, it applies
the requested kind and restores the previous kind afterward; an enclosing
``DEF_PATTERN`` continues to apply even if a nested call requests another kind.

The callbacks above are different from a node type's custom ``__s_visit__``
hook. In an engine callback, ``default_visit(node)`` bypasses that callback;
inside the node's own ``__s_visit__`` hook, the same call would re-enter the
hook. Use the hook's child traversal contract when defining a new node type.

Structural Mutator
------------------

``tvm_ffi.structural_mutate`` combines replacement with explicit control over
recursive processing. Use it when a rewrite must select children or arrange
recursive calls around its local transformation.

.. code-block:: text

   tvm_ffi.structural_mutate(root, callbacks=())

``root`` is the expression, statement, function, module, type, or FFI container
to transform. ``callbacks`` accepts ``(Type, handler)``, a grouped
``((TypeA, TypeB), handler)``, a list of entries, or a bare catch-all callable.
The default ``()`` installs no handlers. IR class selectors match subclasses;
only the first matching entry runs, so place specific types before broader
ones. Unmatched values follow default mutation. ``object`` or ``typing.Any``
matches all values, while ``tvm_ffi.Object`` matches only FFI objects.

The callback is called as ``handler(value, mutator)`` and returns the value
to install. The supplied ``tvm_ffi.StructuralMutator`` carries the same
callbacks into recursive calls. The matched callback owns descent:

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

Unmatched values follow default mutation. ``structural_mutate`` returns the
resulting root, which the caller must retain. Every matched callback must
return a valid value, even when unchanged; ``None`` does not request default
mutation or mean "keep the input." There is no ``SKIP`` or ``VisitInterrupt``
control result.

Use ``mutate`` on a selected child and ``default_mutate`` on the current
value to continue through the same dispatch. Calling ``mutate`` on the current
value dispatches to its callback again. There is no ``order`` parameter:
placing the local rewrite before or after recursive calls chooses the
processing order.

For example, construct an expression and replace variables only in its
left operand:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

We can select just the left operand and preserve the original right operand:

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

``mutator.default_mutate(node)`` rewrites the current node's children without
invoking its callback again on that node. To perform a bottom-up constant
fold, first construct ``(8 + 1) * (3 + 2)`` as IR nodes and define a local
arithmetic rule:

.. code-block:: python

   bound = ir.prim.Mul(
       ir.prim.Add(ir.const(8, "int32"), ir.const(1, "int32")),
       ir.prim.Add(ir.const(3, "int32"), ir.const(2, "int32")),
   )

   def fold_arithmetic(node):
       if not isinstance(node.a, ir.IntImm) or not isinstance(node.b, ir.IntImm):
           return node
       lhs, rhs = int(node.a), int(node.b)
       value = lhs + rhs if isinstance(node, ir.prim.Add) else lhs * rhs
       return ir.const(value, node.ty)

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
recursive calls.

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

The supplied ``mutator`` also exposes ``def_region_kind()`` to inspect
definition/use context and ``with_def_region_kind(kind, callback)`` to run a
zero-argument callable in a scoped definition region. The kinds are
``tvm_ffi.DefRegionKind.NONE``, ``DEF_SIMPLE``, and ``DEF_PATTERN``. The scoped
method returns the callable's result and restores the previous kind afterward;
an enclosing ``DEF_PATTERN`` takes precedence over a nested requested kind.
Default mutation already applies the definition regions registered by IR nodes.

Ordinary Python references preserve the source through copy-on-write, as in
the left-operand example. Default mutation copies changed paths and reuses
unchanged children. A callback can optionally accept a third positional
argument, ``handler(value, mutator, inplace_mode)``. This integer compares with
``tvm_ffi.InplaceMode.DISALLOW`` and ``ALLOW``; ``ALLOW`` permits storage reuse
only along a uniquely owned path. Passing ``root._move()`` transfers ownership
and permits that optimization. The two-argument callbacks and ordinary
references above do not require ownership transfer.

For custom object hooks and native implementations, see the TVM FFI
`Structural Walk and Map guide <https://github.com/apache/tvm-ffi/blob/dbdcdaceeef5b0b03251e8dea87725a1c1e2cde6/docs/concepts/structural_eq_hash.rst#structural-walk-and-map>`_.
TIRx's constraint-aware variants are covered in
:doc:`symbolic_integer_analysis`; :doc:`writing_passes` shows how to package
a rewrite as a compiler pass.
