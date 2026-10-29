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
examples. We will start with small arithmetic expressions to explain traversal
order and node replacement, then work through a deeper expression.

TVM FFI provides four APIs for traversing and transforming IR:

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

Walk and map are higher-level APIs that handle recursion for you, letting
callbacks focus on inspecting or replacing nodes. Visit and mutate give
callbacks more control over which children to process and in what order.
Their callbacks make the recursive calls explicitly and can manage state
before and after each call.

These operations work on IR objects without executing the program they describe.

The API sections start from the same arithmetic expression, making their
callback behavior easy to compare. Each section includes its own setup, so
you can run it independently.

Structural Walk
---------------

``tvm_ffi.structural_walk`` automatically traverses IR. Its callbacks can
collect nodes, check properties, or stop the traversal when a matching value
is found. The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_walk(
       root, callbacks=(), with_def_region_kind=(), order="post")

``root`` is the expression, statement, function, module, type, or FFI container
where traversal starts. The ``callbacks`` parameter can take a single
``(Type, handler)`` tuple or a list or tuple of these entries. In each entry,
``Type`` is the node class to match and ``handler`` is the function to call
for each match. The following example uses a single entry to print the names
of the variables in ``(x + 1) * (y + 2)``:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

   def on_var(var):
       print(var.name)

   tvm_ffi.structural_walk(root=expr, callbacks=(ir.Var, on_var))

The output is:

.. code-block:: text

   x
   y

The expression ``(x + 1) * (y + 2)`` has a multiplication node at its root.
Its operands are two addition nodes, each containing a variable and an
integer constant. Binary arithmetic nodes store their left and right
operands in ``a`` and ``b``:

.. code-block:: text

   Mul
   ├── a: Add
   │   ├── a: Var       x
   │   └── b: IntImm    1
   └── b: Add
       ├── a: Var       y
       └── b: IntImm    2

.. _ir-default-structural-traversal:

The walker visits this structure automatically, starting at the multiplication
and following each arithmetic node's operands. This is the default structural
traversal.

With ``callbacks=(ir.Var, on_var)``, the walker calls ``on_var`` only for
variable nodes. It calls ``on_var(x)`` and then ``on_var(y)``, and each call
prints the variable's name through ``var.name``.

The following table lists the accepted callback forms, using ``on_var`` from
the example and Python's built-in ``print`` function:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, on_var)``
     - Variables are passed to ``on_var``.
   * - ``((ir.prim.Add, ir.prim.Mul), print)``
     - Additions and multiplications are passed to ``print``, one node per call.
   * - ``[(ir.Var, on_var), (ir.IntImm, print)]``
     - A Python list containing two ``(Type, handler)`` tuples. Variables are passed
       to ``on_var``; integer IR constants are passed to ``print``.
   * - ``((ir.Var, on_var), (ir.IntImm, print))``
     - The same two entries in a Python tuple instead of a list. Matching behavior
       and entry order are unchanged.
   * - ``print``
     - Every visited value is passed to ``print``, with no type filter.
   * - ``()``
     - No functions are registered in ``callbacks``. The walker still visits
       child nodes. This is the default when ``callbacks`` is omitted.

The class in a ``callbacks`` entry also matches its subclasses. For example,
``ir.Expr`` matches ``ir.Var`` because a variable is an expression. Here,
``callbacks`` is a list of two ``(Type, handler)`` tuples that both match ``x``:

.. code-block:: python

   selected = []
   tvm_ffi.structural_walk(
       root=x,
       callbacks=[
           (ir.Var, lambda var: selected.append("variable")),
           (ir.Expr, lambda node: selected.append("expression")),
       ],
   )
   print(selected)

The output is ``['variable']``. Both entries match ``x``, but the walker
checks them in order and calls only the first matching function. If the
``ir.Expr`` entry came first, the expression handler would run instead.
Returning ``None`` continues traversal; it does not try the next handler.

When no entry matches a node, the walker still visits its children. For
example, a walk with only ``(ir.Var, on_var)`` still descends through the
multiplication and additions in ``expr`` to reach ``x`` and ``y``.

``order="post"`` calls the handler after the children; ``order="pre"`` calls
it before them. The corresponding enum values are
``tvm_ffi.WalkOrder.POSTORDER`` and ``PREORDER``.

The following trace records every expression node under both orders.
``ir.Expr`` matches variables, integer constants, and arithmetic nodes:

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

A callback can also control whether traversal continues, skips children,
or stops entirely. Its return value determines what happens next:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Callback returns
     - Effect
   * - ``None`` or ``tvm_ffi.WalkResult.ADVANCE``
     - Traversal continues normally.
   * - ``tvm_ffi.WalkResult.SKIP``
     - In pre-order, this value's children are omitted; traversal continues
       elsewhere.
   * - ``tvm_ffi.VisitInterrupt(payload)``
     - The whole walk stops and returns the interrupt to the caller.

``structural_walk`` returns ``None`` on completion or ``VisitInterrupt`` on
early exit. The interrupt's ``value`` field holds the payload from the
callback.

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
symbol. A ``None`` or ``WalkResult.ADVANCE`` return continues traversal
normally. Pruning takes effect in pre-order, since a post-order callback runs
after the children have already been visited.

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

.. _ir-ffi-containers:

We use TVM FFI's data structures here: ``tvm_ffi.Array`` for sequences and
``tvm_ffi.Map`` for key-value mappings. The following example collects
variables from an array stored inside a map:

.. code-block:: python

   container_nodes = tvm_ffi.Array([x, y])
   container_groups = tvm_ffi.Map({"inputs": container_nodes})
   container_names = []

   def collect_container_var(var):
       container_names.append(var.name)

   tvm_ffi.structural_walk(
       root=container_groups, callbacks=(ir.Var, collect_container_var)
   )
   print(container_names)

The output is:

.. code-block:: text

   ['x', 'y']

``[x, y]`` initializes the FFI array. The walk follows the map's ``"inputs"``
value to that array, then visits ``x`` and ``y``. The string key ``"inputs"``
is skipped.

The same rules apply to the mutable sequence ``tvm_ffi.List`` and the mutable
mapping ``tvm_ffi.Dict``.

Each IR node type defines its children through structural fields or custom
traversal hooks. Fields marked ``structural_eq="ignore"`` are skipped.

The type in a ``callbacks`` entry determines which kinds of values match:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Matching values
   * - ``ir.IntImm``
     - Integer constants in IR, such as the ``1`` in ``x + 1``.
   * - ``int``
     - Ordinary integers, such as the elements of ``tvm_ffi.Array([1, 2])``.
   * - ``object`` or ``typing.Any``
     - All values, including IR nodes, FFI containers, and ordinary integers.
   * - ``tvm_ffi.Object``
     - FFI objects, including IR nodes and FFI containers. Ordinary integers
       do not match this type.

A walk visits every occurrence of a node, including repeated references
to the same object:

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

A variable can appear where it is defined and where it is used.
The optional ``with_def_region_kind`` argument lets a callback distinguish
these occurrences. It accepts the same callback forms as ``callbacks``,
but its handlers receive ``(value, kind)``. The ``kind`` is a
``tvm_ffi.DefRegionKind`` describing the position currently being visited.
The same variable object can therefore be visited with different kinds.

``ir.prim.Let(var, value, body)`` binds the result of ``value`` to ``var``
within ``body``. In the following expression, ``tmp`` names the result of
``expr``, and ``tmp + tmp`` uses that result twice:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(tmp, expr, tmp + tmp)

   def show_var_context(var, kind):
       print(var.name, kind.name)

   tvm_ffi.structural_walk(
       scoped_expr,
       with_def_region_kind=(ir.Var, show_var_context),
   )

The output is:

.. code-block:: text

   tmp DEF_SIMPLE
   x NONE
   y NONE
   tmp NONE
   tmp NONE

The five calls correspond to these positions:

.. list-table::
   :header-rows: 1
   :widths: 30 38 32

   * - Field
     - Variables visited
     - Kind
   * - ``Let.var``
     - ``tmp``: the variable being defined.
     - ``DEF_SIMPLE``
   * - ``Let.value``
     - ``x`` and ``y``: inputs to the bound expression.
     - ``NONE``
   * - ``Let.body``
     - ``tmp`` twice: references to the bound variable.
     - ``NONE``

``Let``'s traversal rules mark ``var`` as a definition and visit ``value``
and ``body`` as ordinary uses. The three visits to ``tmp`` receive the same
variable object: one visit at its definition and two at its uses. The
``NONE`` kind applies to uses of the bound ``tmp`` as well as the free
inputs ``x`` and ``y``.

This distinction lets an analysis count variable uses without counting the
definition itself: its callback records only visits with
``kind == tvm_ffi.DefRegionKind.NONE``.

.. _ir-pattern-definition-regions:

Some definitions also introduce symbols through a variable's type. For
example, a tensor parameter ``A`` with shape ``(n,)`` contains the symbolic
dimension ``n`` in its type. Function parameter traversal uses
``DEF_PATTERN`` for both ``A`` and the ``n`` reached through that type.

For a variable of this type, the two definition contexts affect visits as
follows:

.. list-table::
   :header-rows: 1
   :widths: 32 32 36

   * - Definition context
     - Visit to ``A``
     - Visit to ``n`` in ``A``'s type
   * - ``DEF_SIMPLE``
     - ``DEF_SIMPLE``
     - ``NONE``
   * - ``DEF_PATTERN``
     - ``DEF_PATTERN``
     - ``DEF_PATTERN``

``DEF_PATTERN`` continues into the type, while ``DEF_SIMPLE`` applies to
the variable itself. The IR node's traversal rules choose these contexts;
the callback receives the current kind from the walker.

Entries in ``callbacks`` are tried before ``with_def_region_kind``. A value
is handled by ``with_def_region_kind`` only when no entry in ``callbacks``
matches, so a variable handler in ``callbacks`` would take precedence over
``show_var_context`` in this example.

Structural Map
--------------

``tvm_ffi.structural_map`` applies replacement rules with automatic recursion,
supporting substitutions and rewrites throughout the IR. We will replace
symbolic variables with constants, then fold the resulting arithmetic.
These are separate rules: replacing a variable does not itself evaluate
the expression containing it.

.. code-block:: text

   tvm_ffi.structural_map(
       root, callbacks=(), with_def_region_kind=(), order="post")

``root`` is the expression, statement, function, module, type, or FFI container
to transform.

The ``callbacks`` parameter specifies Python replacement functions and
optional type filters. Each function receives the current value and returns
the value to install at that position. In ``callbacks=(ir.Var, specialize_var)``,
``ir.Var`` is the class to match and ``specialize_var`` is a Python function
passed without calling it. The mapper calls ``specialize_var(var)`` with
the actual variable object. The table uses functions defined in this
section's examples:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, specialize_var)``
     - ``specialize_var(var)`` returns a replacement for a variable.
   * - ``((ir.prim.Add, ir.prim.Mul), fold_arithmetic)``
     - Either operator class selects ``fold_arithmetic(node)``. Each call
       receives one node, not a pair of operands.
   * - ``[(ir.Var, specialize_var), (ir.prim.Add, fold_arithmetic)]``
     - Variables and additions have separate replacement functions.
   * - ``keep_value``
     - A function on its own has no type filter. ``keep_value`` returns its
       input unchanged.
   * - ``()``
     - The empty tuple contains no replacement callback. Default mapping
       still runs. This is the default when ``callbacks`` is omitted.

IR class selectors match subclasses, and only the first matching entry runs.
A specific handler must appear before a broader one to take precedence over it.
Unmatched values follow default mapping. ``object`` or ``typing.Any`` matches
all values, while ``tvm_ffi.Object`` matches only FFI objects.

Each callback returns the value to install: either the input unchanged or a
valid replacement. ``None`` does not mean "keep this value."
``structural_map`` returns the transformed root.

The optional ``with_def_region_kind`` table supports rewrites that distinguish
definitions from uses, as shown near the end of this section. ``order`` accepts
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
     - The mapper processes the returned value's children and returns the
       rebuilt value.
   * - ``"post"``
     - The value with its children already mapped.
     - The mapper returns the callback's result without descending into
       that replacement again.

In both cases, the mapper invokes a selected callback once at that position;
it does not repeatedly apply rules until nothing changes. A callback must not
modify its input in place. The examples below keep the original root available
and return new IR where needed.

We will start with an expression to transform:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

The root is a multiplication with two addition children. Each arithmetic
node stores its left and right operands in ``a`` and ``b``. The variables
and integer constants are also IR expressions, so a replacement for a
variable must be a compatible expression object.

A callback can return each value unchanged. Passing that function directly
applies it without a type filter:

.. code-block:: python

   def keep_value(value):
       return value

   identity = tvm_ffi.structural_map(expr, keep_value)
   print(identity.same_as(expr))

The output is ``True``. The returned root is the original object because
neither the callback nor default mapping changes any structural field.

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
``var.ty`` is the variable's IR type; ``ir.const(8, var.ty)`` constructs a
constant of that type. ``same_as`` compares symbol identities rather than
printed names, so a different variable named ``x`` would remain unchanged.
The source expression remains available. ``T.Add(8, 1)`` is the printer's
explicit form for an addition node whose operands are constants: structural
mapping replaces fields without automatically evaluating the arithmetic.

Pre-order is useful for symbol substitution: the callback chooses the value
whose children the mapper then processes. A replacement must fit the field's
IR type. Returning a bare Python integer cannot replace a typed expression
field here; ``ir.const(value, var.ty)`` constructs the required expression.

Post-order is useful when a parent's rewrite depends on its rewritten
children. The following callback evaluates constant additions, subtractions,
multiplications, and floor divisions. It keeps the input node when an operand
is nonconstant or a divisor is zero:

.. code-block:: python

   def fold_arithmetic(node):
       if not isinstance(node.a, ir.IntImm) or not isinstance(node.b, ir.IntImm):
           return node
       lhs, rhs = int(node.a), int(node.b)
       if isinstance(node, ir.prim.Add):
           value = lhs + rhs
       elif isinstance(node, ir.prim.Sub):
           value = lhs - rhs
       elif isinstance(node, ir.prim.Mul):
           value = lhs * rhs
       elif isinstance(node, ir.prim.FloorDiv) and rhs != 0:
           value = lhs // rhs
       else:
           return node
       return ir.const(value, node.ty)

   arithmetic_types = (ir.prim.Add, ir.prim.Sub, ir.prim.Mul, ir.prim.FloorDiv)
   folded = tvm_ffi.structural_map(
       bound,
       (arithmetic_types, fold_arithmetic),
       order="post",
   )
   print(folded)

The output is ``45``. The callback replaces the two additions with 9 and 5
before it receives the enclosing multiplication, so it can replace that
node with 45 as well. ``arithmetic_types`` is a tuple of alternative node
classes sharing this rule. The type filter ensures that ``node.a`` and
``node.b`` exist; variables and constants follow default mapping instead.
``int(node.a)`` reads the Python integer stored in an ``ir.IntImm``, and
``ir.const(value, node.ty)`` wraps the computed integer back in IR.

Starting again from ``bound``, the same callback in pre-order gives a different
result:

.. code-block:: python

   folded_pre = tvm_ffi.structural_map(
       bound,
       (arithmetic_types, fold_arithmetic),
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
``VisitInterrupt``. A mutator gives the callback control over child selection.

We can use the same callbacks on a deeper expression,
``((x + 1) * (y - 2)) // 4``. Its outer division depends on a multiplication,
which in turn depends on an addition and a subtraction. ``//`` constructs an
``ir.prim.FloorDiv`` node and rounds integer division down. The arithmetic
structure is:

.. code-block:: text

   FloorDiv
   ├── a: Mul
   │   ├── a: Add(x, 1)
   │   └── b: Sub(y, 2)
   └── b: IntImm(4)

The following example first substitutes ``x = 8`` and ``y = 3``, then records
each successful fold in post-order:

.. code-block:: python

   complex_expr = ((x + 1) * (y - 2)) // 4
   complex_bound = tvm_ffi.structural_map(
       complex_expr, (ir.Var, specialize_var), order="pre"
   )
   fold_steps = []

   def fold_and_record(node):
       rewritten = fold_arithmetic(node)
       if isinstance(rewritten, ir.IntImm):
           fold_steps.append((type(node).__name__, int(rewritten)))
       return rewritten

   complex_folded = tvm_ffi.structural_map(
       complex_bound, (arithmetic_types, fold_and_record), order="post"
   )
   print(complex_bound)
   print(fold_steps)
   print(complex_folded)

The output is:

.. code-block:: text

   T.Add(8, 1) * T.Sub(3, 2) // 4
   [('Add', 9), ('Sub', 1), ('Mul', 9), ('FloorDiv', 2)]
   2

The first map replaces variables while retaining the arithmetic nodes. The
second map folds the addition to 9 and the subtraction to 1 before visiting
their parent multiplication. That result, 9, becomes the dividend of the outer
division, which folds to 2. Each parent receives its rewritten children, so
one post-order traversal folds all four arithmetic nodes.

The examples so far replace free variables with constants. A definition site
has a different requirement: the variable bound by a ``Let`` expression must
remain a variable. Callbacks in ``with_def_region_kind`` receive
``(value, kind)``, so they can distinguish definitions from uses.
This table accepts the same entry forms as
``callbacks`` and is tried only when no entry in ``callbacks`` matches.
``NONE`` marks an ordinary use, ``DEF_SIMPLE`` a simple definition, and
``DEF_PATTERN`` a pattern definition that can bind symbols through a type.

The following example renames a local variable by returning the same new
symbol at its definition and both uses:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(tmp, expr, tmp + tmp)
   renamed_tmp = ir.Var("renamed_tmp", tmp.ty)
   rename_contexts = []

   def rename_tmp(var, kind):
       if var.same_as(tmp):
           rename_contexts.append(kind.name)
           return renamed_tmp
       return var

   renamed_expr = tvm_ffi.structural_map(
       scoped_expr,
       with_def_region_kind=(ir.Var, rename_tmp),
       order="pre",
   )
   print(rename_contexts)
   print(renamed_expr.var.name)
   print(renamed_expr.body.a.same_as(renamed_expr.var))
   print(renamed_expr.body.b.same_as(renamed_expr.var))

The output is:

.. code-block:: text

   ['DEF_SIMPLE', 'NONE', 'NONE']
   renamed_tmp
   True
   True

The kinds identify the three positions, while the identity checks confirm
that both uses refer to the new definition. Creating a fresh variable on
every callback invocation would give the uses different identities. The
original ``scoped_expr`` still binds ``tmp``.

Passing an ordinary Python reference preserves the source through
copy-on-write. Default mapping reuses unchanged children and copies changed
paths. Callbacks run at each occurrence of a shared object, so a rewrite that
must preserve shared outputs keeps its own memo. The default variable remap
maintains definition/use consistency during descent; it does not memoize all
callback results. Passing ``root._move()`` transfers ownership and permits
storage reuse along uniquely owned paths; map callbacks must still return
values without modifying their input in place.

.. _structural-visitor:

Structural Visit
----------------

``tvm_ffi.structural_visit`` inspects values while letting each matched
callback choose which children to visit and in what order. Unlike a walk,
a matched visitor callback does not automatically continue to the children.
We will start by observing that behavior, then add recursive calls and
control their order.

.. code-block:: text

   tvm_ffi.structural_visit(root, callbacks=())

``root`` is the expression, statement, function, module, type, or FFI container
where visiting starts.

The ``callbacks`` parameter selects functions that receive the current value
and the active visitor. For example, ``callbacks=(ir.Expr, visit_expr)``
selects expression objects for ``visit_expr``. The traversal calls it as
``visit_expr(node, visitor)``. ``callbacks`` accepts the following forms:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Expr, visit_expr)``
     - Expression nodes are passed to ``visit_expr(node, visitor)``.
   * - ``((ir.prim.Add, ir.prim.Mul), visit_expr)``
     - Either operator class selects ``visit_expr(node, visitor)``. Each
       invocation receives one matching node and the active visitor.
   * - ``[(ir.prim.Mul, visit_left), (ir.Expr, visit_expr)]``
     - Multiplications select ``visit_left``; other expressions select
       ``visit_expr``. Entries are tried in order.
   * - ``visit_expr``
     - The function is selected without a type filter for every value reached.
       Its recursive calls determine which children are reached.
   * - ``()``
     - The empty tuple contains no callback, so default visiting runs.
       This is the default when ``callbacks`` is omitted.

The function names above refer to callbacks defined in this section's examples.
IR class selectors match subclasses, and only the first matching entry runs.
A specific handler must appear before a broader one to take precedence over it.
Unmatched values follow default structural descent. ``object`` or
``typing.Any`` matches all values, while ``tvm_ffi.Object`` matches only FFI
objects.

The callback receives two arguments, ``(value, visitor)``. ``value`` is the
current object. ``structural_visit`` creates a ``tvm_ffi.StructuralVisitor``
for the traversal and passes it as ``visitor``. Recursive calls through this
visitor use the same callbacks.
A matched callback controls descent through these operations:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Within the callback
     - Effect
   * - ``visitor.visit(child)``
     - The chosen child goes through callback selection and visiting.
   * - ``visitor.default_visit(value)``
     - This value's callback is bypassed and its registered children are
       visited. Child visits still use the same callbacks.
   * - ``return None`` without recursive calls
     - The callback finishes without visiting children.
   * - ``return tvm_ffi.VisitInterrupt(payload)``
     - The interrupt reports an early exit to the calling traversal.

Both visitor methods return ``None`` or ``VisitInterrupt``. Child interrupts
reach the caller when the callback returns them. ``structural_visit``
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

``expr`` has a multiplication at its root, with ``x + 1`` in ``expr.a``
and ``y + 2`` in ``expr.b``. Additions and multiplications both store their
operands in ``a`` and ``b``. ``describe`` gives each expression a short label
so that the traces show which callbacks ran.

This callback records the current node but does not recurse:

.. code-block:: python

   visited = []
   tvm_ffi.structural_visit(
       expr,
       (ir.Expr, lambda node, visitor: visited.append(describe(node))),
   )
   print(visited)

The output is ``['Mul']``. The root matches the callback, which returns
without visiting children. A walk with this type filter would continue
automatically. Here, continuation requires a recursive call inside the
callback:

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
For the root multiplication, the sequence is: record ``Mul``, visit its
left addition and descendants, then visit its right addition and descendants.

Moving the observation after the recursive call gives post-order behavior:

.. code-block:: python

   post_visited = []

   def visit_post(node, visitor):
       interrupted = visitor.default_visit(node)
       if interrupted is not None:
           return interrupted
       post_visited.append(describe(node))
       return None

   tvm_ffi.structural_visit(expr, (ir.Expr, visit_post))
   print(post_visited)

The output is:

.. code-block:: text

   ['x', '1', 'Add', 'y', '2', 'Add', 'Mul']

The callback checks for an interrupt before recording the parent, so an early
exit in a child prevents further observations. The position of the recursive
call determines when the parent is recorded; there is no separate ``order``
argument.

``visitor.visit(child)`` dispatches on a selected child. The next example
visits only the multiplication's left operand and collects variables there:

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

A selected child enters callback dispatch through ``visitor.visit(child)``.
Default descent from the current node proceeds through
``visitor.default_visit(node)``.
Calling ``visitor.visit(node)`` on that same current node would invoke its
callback again rather than advance to the children.

Explicit child calls can also change the order within a node. Here, every
addition and multiplication visits its right operand before its left:

.. code-block:: python

   reversed_variables = []

   def visit_right_first(node, visitor):
       for child in (node.b, node.a):
           interrupted = visitor.visit(child)
           if interrupted is not None:
               return interrupted
       return None

   def record_variable(var, visitor):
       reversed_variables.append(var.name)

   tvm_ffi.structural_visit(
       expr,
       [
           ((ir.prim.Add, ir.prim.Mul), visit_right_first),
           (ir.Var, record_variable),
       ],
   )
   print(reversed_variables)

The output is ``['y', 'x']``. The root visits ``y + 2`` before ``x + 1``;
each addition likewise visits its constant before its variable. The callback
returns a child's interrupt before starting the next child, preserving early
exit across explicitly arranged calls.

Work before and after recursion can maintain state for the current path.
The following callback records the ancestry of each variable:

.. code-block:: python

   path = []
   variable_paths = []

   def visit_with_path(node, visitor):
       path.append(describe(node))
       try:
           if isinstance(node, ir.Var):
               variable_paths.append("/".join(path))
           return visitor.default_visit(node)
       finally:
           path.pop()

   tvm_ffi.structural_visit(expr, (ir.Expr, visit_with_path))
   print(variable_paths)
   print(path)

The output is:

.. code-block:: text

   ['Mul/Add/x', 'Mul/Add/y']
   []

Each recursive call sees the ancestors pushed by its callers. The ``finally``
block removes the current node on return, including an interrupted or failed
visit. The empty final path shows that state from the left subtree does not
remain when the right subtree is visited.

An early exit must propagate through recursive calls. This callback finds
``y`` and returns any interrupt produced while visiting children:

.. code-block:: python

   def visit_until_y(node, visitor):
       if node.same_as(y):
           return tvm_ffi.VisitInterrupt(node)
       return visitor.default_visit(node)

   visitor_found = tvm_ffi.structural_visit(expr, (ir.Expr, visit_until_y))
   print(visitor_found.value.name)

The output is ``y``. A callback that visits several children explicitly must
propagate any child interrupt before visiting the next child.

``visitor.def_region_kind()`` reports the current ``tvm_ffi.DefRegionKind``:
``NONE`` for ordinary uses, ``DEF_SIMPLE`` for simple definitions, or
``DEF_PATTERN`` for pattern definitions. Default descent enters the regions
registered by IR nodes. This callback reads the context while default visiting
handles a ``Let`` expression's structure:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(tmp, expr, tmp + tmp)
   variable_contexts = []

   def record_context(var, visitor):
       variable_contexts.append((var.name, visitor.def_region_kind().name))

   tvm_ffi.structural_visit(scoped_expr, (ir.Var, record_context))
   print([kind for name, kind in variable_contexts if name == "tmp"])

The output is:

.. code-block:: text

   ['DEF_SIMPLE', 'NONE', 'NONE']

The callback receives the same variable object at three positions: one
definition and two uses. Its context comes from the active visitor rather
than an additional callback parameter.

For custom definition traversal,
``visitor.with_def_region_kind(kind, callback)`` runs a zero-argument callable
in a scoped region and returns its result. Outside ``DEF_PATTERN``, it applies
the requested kind and restores the previous kind afterward; an enclosing
``DEF_PATTERN`` continues to apply even if a nested call requests another kind.
For example,
``visitor.with_def_region_kind(tvm_ffi.DefRegionKind.DEF_SIMPLE, lambda: visitor.visit(node.var))``
visits a selected definition child in a simple-definition region and returns
any interrupt from that visit. Default visiting already handles this for
``Let``; an explicit child traversal must preserve the appropriate regions.

The callbacks above are different from a node type's custom ``__s_visit__``
hook. In an engine callback, ``default_visit(node)`` bypasses that callback;
inside the node's own ``__s_visit__`` hook, the same call would re-enter the
hook. A custom node hook follows its own child traversal contract.

.. _structural-mutator:

Structural Mutate
-----------------

``tvm_ffi.structural_mutate`` combines replacement with explicit control over
recursion. Its callbacks can select which children to rewrite and arrange
recursive calls around their local transformations. A callback returns the
complete result for its current position, including any child changes it
chooses to make. We will first compare returning an unchanged node with
requesting default mutation, then rewrite a selected child and fold a deeper
expression.

.. code-block:: text

   tvm_ffi.structural_mutate(root, callbacks=())

``root`` is the expression, statement, function, module, type, or FFI container
to transform.

The ``callbacks`` parameter selects functions that receive the current value
and the active mutator, then return the replacement value. In
``callbacks=(ir.Var, replace_var)``, ``ir.Var`` is the class to match and
``replace_var`` is a Python function. The engine calls it as
``replace_var(var, mutator)``. The accepted forms for ``callbacks`` are:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, replace_var)``
     - Variables are passed to ``replace_var(var, mutator)``.
   * - ``((ir.prim.Add, ir.prim.Mul), mutate_and_fold)``
     - Either operator class selects ``mutate_and_fold(node, mutator)``.
       Each invocation receives one matching node and the active mutator.
   * - ``[(ir.prim.Mul, mutate_left), (ir.Var, replace_var)]``
     - Multiplications select ``mutate_left`` and variables select ``replace_var``.
   * - ``mutate_and_fold``
     - The function is selected without a type filter. Its recursive calls
       determine which children receive callbacks.
   * - ``()``
     - The empty tuple contains no callback, so default mutation runs.
       This is the default when ``callbacks`` is omitted.

The functions above are defined in this section's examples. IR class selectors
match subclasses, and only the first matching entry runs. A specific handler
must appear before a broader one to take precedence over it. Unmatched values
follow default mutation. ``object`` or ``typing.Any`` matches all values, while
``tvm_ffi.Object`` matches only FFI objects.

The callback receives two arguments, ``(value, mutator)``, and returns the
value to install. ``value`` is the current object. ``structural_mutate``
creates a ``tvm_ffi.StructuralMutator`` for the traversal and passes it as
``mutator``. Recursive calls through this mutator use the same callbacks.
The matched callback controls descent through these operations:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Within the callback
     - Effect
   * - ``mutator.mutate(child)``
     - The chosen child goes through callback selection and mutation. The
       method returns the resulting child.
   * - ``mutator.default_mutate(value)``
     - This value's callback is bypassed and default mutation returns the
       node with its rewritten children. Child mutations use the same callbacks.
   * - ``return value`` without recursive calls
     - This value is retained with its children unprocessed.
   * - ``return replacement``
     - The replacement is installed without further traversal by the engine.

Unmatched values follow default mutation. ``structural_mutate`` returns the
resulting root. Every matched callback must return a valid value, even when
unchanged; ``None`` does not request default
mutation or mean "keep the input." There is no ``SKIP`` or ``VisitInterrupt``
control result.

``mutator.mutate(child)`` applies callback dispatch to a selected child.
``mutator.default_mutate(node)`` applies default descent to the current node,
with child mutations using the same callbacks. Calling ``mutate`` on the
current value dispatches to its callback again. There is no ``order`` parameter:
placing the local rewrite before or after recursive calls chooses the
processing order.

We will start with the same expression used by the other APIs:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

The multiplication's ``a`` and ``b`` fields contain ``x + 1`` and ``y + 2``.
The following variable rule returns a typed IR constant. A separate callback
matches the root multiplication and returns it without recursion:

.. code-block:: python

   def replace_var(var, mutator):
       return ir.const(8, var.ty)

   def keep_multiplication(node, mutator):
       return node

   unprocessed = tvm_ffi.structural_mutate(
       expr,
       [(ir.prim.Mul, keep_multiplication), (ir.Var, replace_var)],
   )
   print(unprocessed.same_as(expr))

The output is ``True``. The multiplication callback returns the original
root, so neither variable is reached. Returning a node unchanged does not
request recursive mutation of that node.

A callback can instead request default mutation of its current node:

.. code-block:: python

   def mutate_expr(node, mutator):
       return mutator.default_mutate(node)

   all_replaced = tvm_ffi.structural_mutate(
       expr,
       [(ir.Var, replace_var), (ir.Expr, mutate_expr)],
   )
   print(all_replaced)

The output is ``T.Add(8, 1) * T.Add(8, 2)``. ``default_mutate`` processes
both operands and returns a node containing their results. Each child goes
through callback selection again. Variables match the first entry and become
constants; additions and the multiplication match the broader ``ir.Expr``
entry and request default mutation. This is why the more specific variable
entry appears first. Arithmetic remains unevaluated until a callback folds it.

If only ``(ir.Var, replace_var)`` were registered, unmatched arithmetic nodes
would follow default mutation automatically and produce the same result.
The explicit ``mutate_expr`` callback shows how a matched node can request
that behavior.

We can now select just the left operand and preserve the original right
operand:

.. code-block:: python

   def mutate_left(node, mutator):
       left = mutator.mutate(node.a)
       if left.same_as(node.a):
           return node
       return ir.prim.Mul(left, node.b, loc=node.loc)

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
it with 8. This recursive call returns the new left operand; it does not
assign to ``node.a``. The root callback installs that result by constructing
a new multiplication. If the left operand were unchanged, the identity check
would return the original multiplication instead.

The right operand is passed directly to the constructor, so ``y`` is never
offered to ``replace_var`` in this traversal. The ``True`` identity check
confirms that the output shares that entire subtree with the input.
``loc=node.loc`` preserves the multiplication's source location. The final
line shows that the input expression still contains both symbolic variables.

``mutator.default_mutate(node)`` rewrites the current node's children without
invoking its callback again on that node. The next example constructs
``((8 + 1) * (3 - 2)) // 4`` as IR, then folds it from its leaves to its root
with a local rule:

.. code-block:: python

   constant_expr = ir.prim.FloorDiv(
       ir.prim.Mul(
           ir.prim.Add(ir.const(8, "int32"), ir.const(1, "int32")),
           ir.prim.Sub(ir.const(3, "int32"), ir.const(2, "int32")),
       ),
       ir.const(4, "int32"),
   )

   def fold_arithmetic(node):
       if not isinstance(node.a, ir.IntImm) or not isinstance(node.b, ir.IntImm):
           return node
       lhs, rhs = int(node.a), int(node.b)
       if isinstance(node, ir.prim.Add):
           value = lhs + rhs
       elif isinstance(node, ir.prim.Sub):
           value = lhs - rhs
       elif isinstance(node, ir.prim.Mul):
           value = lhs * rhs
       elif isinstance(node, ir.prim.FloorDiv) and rhs != 0:
           value = lhs // rhs
       else:
           return node
       return ir.const(value, node.ty)

   arithmetic_types = (ir.prim.Add, ir.prim.Sub, ir.prim.Mul, ir.prim.FloorDiv)

   def mutate_and_fold(node, mutator):
       rewritten = mutator.default_mutate(node)
       if isinstance(rewritten, arithmetic_types):
           return fold_arithmetic(rewritten)
       return rewritten

   folded_manual = tvm_ffi.structural_mutate(constant_expr, (ir.Expr, mutate_and_fold))
   print(folded_manual)

The output is ``2``. The explicit recursive call processes the addition and
subtraction before multiplication, and multiplication before division. This
gives the same result as the post-order map. ``rewritten`` is the current
node with its child results installed: when division is reached after descent,
its operands are the constants 9 and 4. The local rule returns the constant 2.
The ``isinstance`` check keeps the arithmetic rule from accessing ``a`` and
``b`` on variables or literals, which also match ``ir.Expr``.

The explicit constructors preserve the arithmetic as IR rather than letting
Python evaluate ``((8 + 1) * (3 - 2)) // 4`` before traversal starts. A map is
shorter when uniform post-order folding is all a rewrite needs; a mutator
allows a different sequence of recursive calls for each matched node.

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
already complete. For example, ``mutator.default_mutate(replacement)`` would
process those new children without dispatching on the addition itself.

Rewriting a bound variable requires consistent identities at its definition
and uses. ``mutator.def_region_kind()`` reports ``tvm_ffi.DefRegionKind.NONE``
for ordinary uses, ``DEF_SIMPLE`` for simple definitions, or ``DEF_PATTERN``
for pattern definitions. Default mutation enters the regions registered by
IR nodes. The following callback creates a new symbol at a simple definition
and reuses it at later occurrences:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(tmp, expr, tmp + tmp)

   def rename_local(var, mutator):
       replacement = mutator.var_remap_get(var)
       if replacement is not None:
           return replacement
       if mutator.def_region_kind() == tvm_ffi.DefRegionKind.DEF_SIMPLE:
           replacement = ir.Var(var.name + "_renamed", var.ty)
           mutator.var_remap_set(var, replacement)
           return replacement
       return var

   renamed_local = tvm_ffi.structural_mutate(scoped_expr, (ir.Var, rename_local))
   print(renamed_local.var.name)
   print(renamed_local.body.a.same_as(renamed_local.var))
   print(renamed_local.body.b.same_as(renamed_local.var))
   print(scoped_expr.var.name)

The output is:

.. code-block:: text

   tmp_renamed
   True
   True
   tmp

``var_remap_get(var)`` looks up a replacement by variable identity and returns
``None`` when none is recorded. ``var_remap_set(var, replacement)`` records the
mapping for this mutator. The callback creates ``tmp_renamed`` at the definition;
both uses then retrieve that same object. The free inputs ``x`` and ``y``
remain unchanged. Because a matched callback handles the variable itself,
it consults the remap explicitly rather than assuming that default mutation
will apply it afterward. This remap handles variable identities; it is not a
cache for every rewritten arithmetic node.

For custom definition traversal,
``mutator.with_def_region_kind(kind, callback)`` runs a zero-argument callable
in a scoped region and returns its result. For example,
``mutator.with_def_region_kind(tvm_ffi.DefRegionKind.DEF_SIMPLE, lambda: mutator.mutate(node.var))``
returns the result of mutating a selected definition child. The previous kind
is restored afterward; an enclosing ``DEF_PATTERN`` takes precedence over a
nested requested kind. Default mutation already performs the required
definition traversal for ``Let``.

Ordinary Python references preserve the source through copy-on-write, as in
the left-operand example. Default mutation copies changed paths and reuses
unchanged children. A callback can optionally accept a third positional
argument, ``handler(value, mutator, inplace_mode)``. This integer compares with
``tvm_ffi.InplaceMode.DISALLOW`` and ``ALLOW``; ``ALLOW`` permits storage reuse
only along a uniquely owned path. Passing ``root._move()`` transfers ownership
and permits that optimization. The two-argument callbacks and ordinary
references above do not require ownership transfer.

The TVM FFI
`Structural Walk and Map guide <https://github.com/apache/tvm-ffi/blob/dbdcdaceeef5b0b03251e8dea87725a1c1e2cde6/docs/concepts/structural_eq_hash.rst#structural-walk-and-map>`_
describes custom object hooks and native implementations.
TIRx's constraint-aware variants are covered in
:doc:`symbolic_integer_analysis`; :doc:`writing_passes` shows how to package
a rewrite as a compiler pass.
