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

This chapter introduces TVM FFI's structural operations for inspecting and
rewriting IR. We explain how each operation works and show how to use its API.

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

``root`` is the value to traverse, such as an IR expression, a function,
or an FFI container. The ``callbacks`` parameter accepts a ``(Type, handler)``
tuple or a Python list or tuple of these pairs. ``Type`` is the class to
match, and ``handler`` is called with each matching value. The following
example prints the variable names in ``(x + 1) * (y + 2)``:

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

Each IR node type defines its child relationships. The walker follows
them automatically; this is the default structural traversal.

With ``callbacks=(ir.Var, on_var)``, only variable nodes are passed to
``on_var``. The calls ``on_var(x)`` and ``on_var(y)`` each print the
variable's ``name``.

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
     - A Python list of two ``(Type, handler)`` pairs. Variables are passed to
       ``on_var``; integer IR constants are passed to ``print``.
   * - ``((ir.Var, on_var), (ir.IntImm, print))``
     - The same two pairs in a Python tuple, with the same matching order.
   * - ``print``
     - Every visited value is passed to ``print``, with no type filter.
   * - ``()``
     - The empty tuple selects no handlers. The walker still visits
       child nodes. This is the default when ``callbacks`` is omitted.

A class in ``callbacks`` also matches objects of its subclasses. Since
``ir.Var`` is a subclass of ``ir.Expr``, both pairs below match ``x``:

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

The output is ``['variable']``. The walker checks the pairs in order and
calls only the first matching handler. If the ``ir.Expr`` pair came first,
the expression handler would run instead.
Returning ``None`` continues traversal; it does not try the next handler.

When no handler matches a node, the walker still visits its children. For
example, a walk with only ``(ir.Var, on_var)`` still descends through the
multiplication and additions in ``expr`` to reach ``x`` and ``y``.

``order`` controls when a matching handler runs. ``order="post"`` (the default)
calls it after the node's children have been visited; ``order="pre"`` calls it
before them. The corresponding enum values are
``tvm_ffi.WalkOrder.POSTORDER`` and ``tvm_ffi.WalkOrder.PREORDER``.

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

The pre-order trace records ``Mul`` before its ``Add`` children; the
post-order trace records it after them. In both cases, each binary expression's
left operand ``a`` is visited before its right operand ``b``.

A callback can also control whether traversal continues, skips children,
or stops entirely. Its return value determines what happens next:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Callback returns
     - Effect
   * - ``None`` or ``tvm_ffi.WalkResult.ADVANCE``
     - The walk continues.
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

The first ``Add`` is recorded, but its children ``x`` and ``1`` are skipped.
The right subtree is still visited. ``same_as(x)`` checks IR object identity;
another variable with the printed name ``x`` is a distinct symbol. Skipping
children takes effect in pre-order, since a post-order callback runs after
the children have already been visited.

``VisitInterrupt`` stops the entire traversal and carries a result:

.. code-block:: python

   def find_y(var):
       if var.same_as(y):
           return tvm_ffi.VisitInterrupt(var)
       return None

   found = tvm_ffi.structural_walk(expr, (ir.Var, find_y))
   if found is not None:
       print(found.value.name)

The output is ``y``. ``found`` holds the interrupt returned by ``find_y``,
and ``found.value`` is the matching variable ``y``. The walk ends there,
before visiting the remaining constant ``2``.

.. _ir-ffi-containers:

A walk can also start from a collection of IR nodes. We use TVM FFI's data
structures here: ``tvm_ffi.Array`` for sequences and ``tvm_ffi.Map`` for
key-value mappings. The following example collects variables from an array
stored inside a map:

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

FFI containers can hold ordinary values as well as IR objects. The callback
type determines which of these values match:

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

The same IR object can be referenced from more than one place.
``structural_walk`` visits each reference:

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

Both elements of ``additions`` refer to ``shared``. A pre-order handler can
remember which objects it has already processed and return ``SKIP`` when it
sees one again. The handler still runs for every reference; ``SKIP`` prevents
another descent into the repeated node's children.

.. _ir-definition-regions:

``with_def_region_kind`` lets a callback tell whether a variable is being
defined or used at each visit. It accepts the same callback forms as
``callbacks``, but its handlers receive ``(value, kind)``. ``kind`` is the
``tvm_ffi.DefRegionKind`` at the current position; it can differ between
visits to the same variable object.

``ir.prim.Let(var, value, body)`` binds the result of ``value`` to ``var``
within ``body``. In this example, ``tmp`` is bound to the result of ``expr``
and used twice in ``tmp + tmp``:

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

``Let`` marks ``var`` as a definition. All three visits to ``tmp`` refer
to the same object, but the two visits in ``body`` are uses and receive
``NONE``. The visits to ``x`` and ``y`` in ``value`` are also uses.

An analysis that counts variable uses can record only visits where
``kind == tvm_ffi.DefRegionKind.NONE``.

.. _ir-pattern-definition-regions:

A variable's type can itself contain variables. For a tensor parameter
``A`` with shape ``(n,)``, the type of ``A`` contains the dimension variable
``n``. Function parameter definitions use ``DEF_PATTERN``, which remains
active as the walker enters the type. Both ``A`` and ``n`` therefore reach
the callback with that kind.

For the same tensor variable, visiting under each definition kind gives:

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

IR node types define these contexts in their traversal rules. The walker
passes the active kind to the callback.

``with_def_region_kind`` is consulted only when no handler in ``callbacks``
matches the value. The ``Let`` example leaves ``callbacks`` empty so that
``show_var_context`` handles every variable.

Structural Map
--------------

``tvm_ffi.structural_map`` rewrites IR by applying replacement functions and
automatically processing child nodes. It returns the rewritten root.
The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_map(
       root, callbacks=(), with_def_region_kind=(), order="post")

``root`` is the value to transform, such as an IR expression, a function,
or an FFI container. The ``callbacks`` parameter accepts a ``(Type, handler)``
tuple or a Python list or tuple of these pairs. ``Type`` is the class to match,
and ``handler(value)`` returns the replacement for each matching value. The
following example replaces ``x`` with 8 and ``y`` with 3:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

   def specialize_var(var):
       if var.same_as(x):
           return ir.const(8, var.ty)
       if var.same_as(y):
           return ir.const(3, var.ty)
       return var

   bound = tvm_ffi.structural_map(root=expr, callbacks=(ir.Var, specialize_var))
   print(bound)
   print(expr)

The output is:

.. code-block:: text

   T.Add(8, 1) * T.Add(3, 2)
   (x + 1) * (y + 2)

``ir.Var`` selects variable nodes, and ``specialize_var`` returns a typed IR
constant for each selected symbol. ``same_as`` compares variable identities;
``var.ty`` gives the type that ``ir.const`` uses for the replacement. The final
``return var`` keeps any other variable unchanged.

The mapper follows the arithmetic operands, replaces the variables, and
rebuilds their enclosing additions and multiplication. The returned ``bound``
contains those changes; ``expr`` retains the original variables. The printer
uses ``T.Add(8, 1)`` for an addition of two IR constants. Mapping preserves
that addition until a rule explicitly replaces it with its result.

Mapping also processes the elements of FFI sequences and the values of FFI
mappings, described in :ref:`the FFI container overview <ir-ffi-containers>`.

Each handler returns a value compatible with the field it rewrites. For the
expression fields above, that means an IR expression. A handler that keeps
a value returns the value it receives; ``None`` is not an "unchanged" signal.
Map handlers return values without modifying their inputs in place.

The other callback forms follow the same matching rules as walk:

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, specialize_var)``
     - Variables are passed to ``specialize_var``.
   * - ``((ir.prim.Add, ir.prim.Mul), lambda node: node)``
     - Both operator classes share a handler that returns the current node.
   * - ``[(ir.Var, specialize_var), (ir.Expr, lambda node: node)]``
     - Variables use ``specialize_var``; other expressions keep their current value.
   * - ``lambda value: value``
     - A function on its own matches every reached value and returns it unchanged.
   * - ``()``
     - No handlers are selected from ``callbacks``. Child mapping still runs.

A Python tuple can hold the same pairs as the list above. Classes also match
their subclasses, and only the first matching handler runs. In the list,
``ir.Var`` comes before ``ir.Expr`` so that variable replacement takes
precedence. Nodes with no matching handler still have their children mapped.

``order`` determines when the handler runs and which children it sees. It
accepts ``"post"`` (the default), ``"pre"``, or the corresponding
``tvm_ffi.WalkOrder.POSTORDER`` and ``tvm_ffi.WalkOrder.PREORDER`` values:

.. list-table::
   :header-rows: 1
   :widths: 15 42 43

   * - Order
     - Handler receives
     - After the handler
   * - ``"post"``
     - A node whose children have already been mapped.
     - The returned replacement becomes the result at this position.
   * - ``"pre"``
     - The node before its children are mapped.
     - The mapper processes the returned value's children and rebuilds that value.

Constant folding shows why this distinction matters. A folding rule replaces
an operation on constants with its computed result. Binary arithmetic nodes
store their operands in ``a`` and ``b``. This rule handles addition,
subtraction, multiplication, and floor division when both operands are
integer constants; a zero divisor leaves the division unchanged:

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
       bound, callbacks=(arithmetic_types, fold_arithmetic), order="post"
   )
   folded_pre = tvm_ffi.structural_map(
       bound, callbacks=(arithmetic_types, fold_arithmetic), order="pre"
   )
   print("post:", folded)
   print("pre:", folded_pre)

The output is:

.. code-block:: text

   post: 45
   pre: T.Mul(9, 5)

In post-order, the two additions become 9 and 5 before the multiplication's
handler runs. It therefore receives constant operands and returns 45. In
pre-order, the multiplication's handler sees additions and returns the node
unchanged. Its children then become 9 and 5, but its handler does not run
again after that descent.

The type tuple keeps ``fold_arithmetic`` focused on nodes with ``a`` and
``b`` operands. ``int(node.a)`` reads a Python integer from an ``ir.IntImm``;
``ir.const(value, node.ty)`` turns the computed result back into typed IR.
Each handler runs once at its position. A post-order replacement is final;
the mapper does not descend into it again or keep applying rules until the
result stops changing.

The same rules apply to the deeper expression ``((x + 1) * (y - 2)) // 4``.
Here, ``//`` creates an ``ir.prim.FloorDiv`` node for integer division rounded
down. The arithmetic structure is:

.. code-block:: text

   FloorDiv
   ├── a: Mul
   │   ├── a: Add(x, 1)
   │   └── b: Sub(y, 2)
   └── b: IntImm(4)

After substituting the variables, a post-order map can record each fold as
the results move from the inner operations to the outer division:

.. code-block:: python

   complex_expr = ((x + 1) * (y - 2)) // 4
   complex_bound = tvm_ffi.structural_map(
       complex_expr, callbacks=(ir.Var, specialize_var)
   )
   fold_steps = []

   def fold_and_record(node):
       rewritten = fold_arithmetic(node)
       if isinstance(rewritten, ir.IntImm):
           fold_steps.append((type(node).__name__, int(rewritten)))
       return rewritten

   complex_folded = tvm_ffi.structural_map(
       complex_bound, callbacks=(arithmetic_types, fold_and_record), order="post"
   )
   print(complex_bound)
   print(fold_steps)
   print(complex_folded)

The output is:

.. code-block:: text

   T.Add(8, 1) * T.Sub(3, 2) // 4
   [('Add', 9), ('Sub', 1), ('Mul', 9), ('FloorDiv', 2)]
   2

The addition and subtraction fold to 9 and 1. The multiplication receives
those constants and folds to 9; the division then receives 9 and 4 and folds
to 2. One post-order traversal completes all four folds because each parent
receives its rewritten children.

Definition sites constrain which replacements are valid. A ``Let`` must
still bind a variable, so replacing its ``var`` field with an integer
constant would be invalid. ``with_def_region_kind`` gives a handler the
definition/use context described in
:ref:`Structural Walk <ir-definition-regions>`. Its handlers receive
``(value, kind)`` and return the value to install. They run only when no
handler in ``callbacks`` matches.

This example renames a local variable by returning one new symbol at its
definition and both uses:

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

``rename_contexts`` records only visits to ``tmp``: its definition and two
uses. Both identity checks are true because the new binding and its uses
share ``renamed_tmp``. Creating a separate variable at each occurrence would
give the uses different identities. Other variables are returned unchanged.

With an ordinary Python reference, mapping preserves the source through
copy-on-write: changed paths are copied and unchanged children are reused.
Handlers still run at each reference to a shared object. A rewrite that needs
one shared replacement per original object can keep a memo of its handler
results; the default variable remap handles definition/use consistency during
descent, not arbitrary handler results.

Passing ``root._move()`` transfers ownership and permits storage reuse along
uniquely owned paths. Map handlers still return values without modifying
their inputs in place. A rewrite that needs to choose which children to
process can use ``structural_mutate``, described below.

.. _structural-visitor:

Structural Visit
----------------

``tvm_ffi.structural_visit`` inspects IR while letting each matching callback
choose which children to visit. The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_visit(root, callbacks=())

``root`` is the value where visiting starts. ``callbacks`` accepts a
``(Type, handler)`` tuple or a Python list or tuple of these pairs. ``Type``
is the class to match, and the handler receives ``(value, visitor)``.
``structural_visit`` creates the ``tvm_ffi.StructuralVisitor`` passed as
``visitor``. Its ``visit(child)`` method continues the traversal on a child
with the same callbacks.

The following example visits only the multiplication's left operand and
collects the variable names found there. Binary arithmetic nodes store their
left and right operands in ``a`` and ``b``:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)
   left_variables = []

   def visit_left(node, visitor):
       return visitor.visit(node.a)

   def record_var(var, visitor):
       left_variables.append(var.name)

   tvm_ffi.structural_visit(
       root=expr,
       callbacks=[(ir.prim.Mul, visit_left), (ir.Var, record_var)],
   )
   print(left_variables)

The output is ``['x']``. The root multiplication matches ``visit_left``,
which selects its left operand ``node.a``, the addition ``x + 1``. That
addition has no matching handler, so its children are visited automatically.
The variable ``x`` then reaches ``record_var``. The right operand ``node.b``
is never visited, so ``y`` is absent from the result.

A matched handler controls all further visits from its node. Returning
``None`` without a recursive call leaves its children unvisited. Nodes with
no matching handler follow their default child traversal.

Callback selection uses the same rules as walk: class filters match
subclasses, and only the first matching handler runs. A tuple of classes can
share a handler. A function passed on its own matches every value reached by
its recursive calls. With ``callbacks=()``, no handlers are selected and
default visiting processes the whole structure.

The visitor's methods and the handler's return value control the traversal:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Within the handler
     - Effect
   * - ``visitor.visit(child)``
     - The child goes through the same callback selection as the root.
   * - ``visitor.default_visit(node)``
     - The current node's handler is bypassed and its children are visited.
       Child visits still use the same callbacks.
   * - ``return None`` without recursive calls
     - The handler finishes without visiting children.
   * - ``return tvm_ffi.VisitInterrupt(payload)``
     - The handler reports an early exit to its caller.

``visit(child)`` applies callback selection to the child. ``default_visit(node)``
continues through the current node's children without calling that node's
handler again. Calling ``visit(node)`` on the current node would select the
same handler again.

Default visiting makes it possible to do work both before and after the
children. This callback records both points in one traversal:

.. code-block:: python

   def describe(node):
       if isinstance(node, ir.Var):
           return node.name
       if isinstance(node, ir.IntImm):
           return str(int(node))
       return type(node).__name__

   visited = []
   post_visited = []

   def visit_expr(node, visitor):
       visited.append(describe(node))
       interrupted = visitor.default_visit(node)
       if interrupted is not None:
           return interrupted
       post_visited.append(describe(node))
       return None

   tvm_ffi.structural_visit(expr, callbacks=(ir.Expr, visit_expr))
   print(visited)
   print(post_visited)

The output is:

.. code-block:: text

   ['Mul', 'Add', 'x', '1', 'Add', 'y', '2']
   ['x', '1', 'Add', 'y', '2', 'Add', 'Mul']

The first list records each node before its children; the second records it
after them. The callback places the observation on either side of
``default_visit``, so ``structural_visit`` needs no ``order`` parameter.
Both visitor methods return ``None`` or ``VisitInterrupt``. Returning a
child's interrupt preserves the early exit through the recursive calls.
The outer ``structural_visit`` call likewise returns ``None`` on completion
or the interrupt on early exit.

Explicit child calls can change the order within a node. This handler
visits the right operand before the left for additions and multiplications:

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
       callbacks=[
           ((ir.prim.Add, ir.prim.Mul), visit_right_first),
           (ir.Var, record_variable),
       ],
   )
   print(reversed_variables)

The output is ``['y', 'x']``. The multiplication visits ``y + 2`` before
``x + 1``. Each addition also visits its constant before its variable.

A variable handler can return ``VisitInterrupt`` when it finds ``y``.
``visit_right_first`` propagates that result through its recursive calls:

.. code-block:: python

   def visit_until_y(var, visitor):
       if var.same_as(y):
           return tvm_ffi.VisitInterrupt(var)
       return None

   visitor_found = tvm_ffi.structural_visit(
       expr,
       callbacks=[
           ((ir.prim.Add, ir.prim.Mul), visit_right_first),
           (ir.Var, visit_until_y),
       ],
   )
   print(visitor_found.value.name)

The output is ``y``. When the right subtree returns the interrupt,
``visit_right_first`` returns it before starting the left subtree.
``WalkResult.SKIP`` belongs to ``structural_walk``; a visitor chooses its
subtrees through recursive calls and propagates an interrupt to stop.

A callback can also maintain state around those calls. This handler keeps
the current path while visiting children and restores it on return:

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

   tvm_ffi.structural_visit(expr, callbacks=(ir.Expr, visit_with_path))
   print(variable_paths)
   print(path)

The output is:

.. code-block:: text

   ['Mul/Add/x', 'Mul/Add/y']
   []

Each child sees the ancestors added by its callers. ``finally`` removes the
current label on return, including when a child interrupts or raises an error.
The printed paths each run from ``Mul`` to a variable. The final empty list
confirms that every added path element was removed.

``visitor.def_region_kind()`` reads the definition/use context described in
:ref:`Structural Walk <ir-definition-regions>`. A handler can read this
context while default visiting follows a ``Let`` expression:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(tmp, expr, tmp + tmp)
   variable_contexts = []

   def record_context(var, visitor):
       variable_contexts.append((var.name, visitor.def_region_kind().name))
       return visitor.default_visit(var)

   tvm_ffi.structural_visit(scoped_expr, callbacks=(ir.Var, record_context))
   for name, kind in variable_contexts:
       print(name, kind)

The output is:

.. code-block:: text

   tmp DEF_SIMPLE
   x NONE
   y NONE
   tmp NONE
   tmp NONE

The visitor reports the context of each occurrence through a method. The
``tmp`` object appears once at its definition and twice at its uses. The
recursive call in ``record_context`` also follows any children in a variable's
type, under the regions established by the node's traversal rules.

For a handler that selects definition fields manually,
``visitor.with_def_region_kind(kind, fn)`` runs ``fn()`` with that context
and returns its result. ``fn`` takes no arguments and can call
``visitor.visit`` on the chosen definition child. The previous context is
restored afterward. An enclosing ``DEF_PATTERN`` remains active even when a
nested call requests another kind. Its effect on variable types is described
in :ref:`the pattern-definition example <ir-pattern-definition-regions>`.

Here, ``default_visit`` bypasses callbacks passed to ``structural_visit``.
A custom node's ``__s_visit__`` hook implements the node's own child
traversal. Calling ``default_visit`` on that same node from inside its hook
would enter the hook again. Such a hook visits its child values directly.

.. _structural-mutator:

Structural Mutate
-----------------

``tvm_ffi.structural_mutate`` rewrites IR while letting each matching callback
choose which children to process. Each handler returns the result for its
current position. The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_mutate(root, callbacks=())

``root`` is the value to transform. ``callbacks`` accepts a ``(Type, handler)``
tuple or a Python list or tuple of these pairs. ``Type`` is the class to
match; ``handler(value, mutator)`` returns the value to install.
``structural_mutate`` creates the ``tvm_ffi.StructuralMutator`` passed as
``mutator``. Its ``mutate(child)`` method returns the result of rewriting a
chosen child.

The following example replaces variables only in the multiplication's left
operand. Binary arithmetic nodes store their left and right operands in
``a`` and ``b``:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

   def replace_var(var, mutator):
       if var.same_as(x):
           return ir.const(8, var.ty)
       if var.same_as(y):
           return ir.const(3, var.ty)
       return var

   def mutate_left(node, mutator):
       left = mutator.mutate(node.a)
       if left.same_as(node.a):
           return node
       return ir.prim.Mul(left, node.b, loc=node.loc)

   partial = tvm_ffi.structural_mutate(
       root=expr,
       callbacks=[(ir.prim.Mul, mutate_left), (ir.Var, replace_var)],
   )
   print(partial)
   print(partial.b.same_as(expr.b))
   print(expr)

The output is:

.. code-block:: text

   T.Add(8, 1) * (y + 2)
   True
   (x + 1) * (y + 2)

The multiplication handler calls ``mutator.mutate(node.a)`` on ``x + 1``.
The addition has no matching handler, so default mutation processes its
children. ``replace_var`` replaces ``x`` with the typed constant 8.

The recursive call returns the new left operand. ``mutate_left`` puts it
into a new multiplication and keeps ``node.b`` as the right operand.
``loc=node.loc`` retains the multiplication's source location. If the left
operand is unchanged, the identity check returns the original node instead.

The right operand is never passed to ``mutate``, so ``y`` remains a variable
in the result. The ``True`` identity check confirms that the output shares
the right subtree with ``expr``. The last line shows that
the source expression still contains both variables.

Callback classes also match subclasses, and only the first matching handler
runs. Grouped classes can share a handler, and a function passed on its own
has no type filter. Every selected handler controls its recursive calls.
Nodes with no matching handler follow default mutation; ``callbacks=()``
uses default mutation throughout the structure.

The mutator's methods and the handler's return value determine the result:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Within the handler
     - Effect
   * - ``mutator.mutate(child)``
     - The child goes through callback selection. The method returns its rewritten value.
   * - ``mutator.default_mutate(node)``
     - The current node's handler is bypassed and a node containing its rewritten
       children is returned. Child mutations use the same callbacks.
   * - ``return node`` without recursive calls
     - The node and its children are retained without further processing.
   * - ``return replacement``
     - The replacement becomes the result at this position.

Calling ``mutate(node)`` on the current node selects its handler again.
``default_mutate(node)`` continues into its children without that repeated
call. A child result takes effect when the handler includes it in the value
it returns, as ``mutate_left`` does when rebuilding the multiplication.

For an IR expression field, every handler returns the unchanged expression
or a compatible replacement. ``None`` is not a request for default mutation
or for keeping the input. Mutators return replacement values; ``SKIP`` and
``VisitInterrupt`` are not mutation control results.

The position of recursive calls also determines the rewrite order. A handler
can first request ``default_mutate`` and then fold the resulting operation.
The following example applies this to ``((x + 1) * (y - 2)) // 4`` while
replacing its variables with 8 and 3:

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
   complex_expr = ((x + 1) * (y - 2)) // 4

   def mutate_and_fold(node, mutator):
       rewritten = mutator.default_mutate(node)
       return fold_arithmetic(rewritten)

   folded_manual = tvm_ffi.structural_mutate(
       complex_expr,
       callbacks=[
           (ir.Var, replace_var),
           (arithmetic_types, mutate_and_fold),
       ],
   )
   print(folded_manual)
   print(complex_expr)

The output is:

.. code-block:: text

   2
   (x + 1) * (y - 2) // 4

The variable handler returns constants. ``mutate_and_fold`` first calls
``default_mutate``, so ``rewritten`` contains the child results before
``fold_arithmetic`` reads its operands. The addition and subtraction become
9 and 1, the multiplication becomes 9, and the division becomes 2.
``fold_arithmetic`` computes a result only when both operands are integer
constants and keeps a division with a zero divisor unchanged.

Children are rewritten before the local folding rule runs, just as in a
post-order map. Here, the handler controls that sequence explicitly;
``structural_mutate`` has no ``order`` parameter.

A returned replacement is final. The mutator does not automatically visit
its new children, as this variable replacement shows:

.. code-block:: python

   new_literals = []

   def replace_symbol(var, mutator):
       return ir.prim.Add(ir.const(10, var.ty), ir.const(20, var.ty))

   def observe_literal(value, mutator):
       new_literals.append(int(value))
       return value

   replacement = tvm_ffi.structural_mutate(
       x,
       callbacks=[(ir.Var, replace_symbol), (ir.IntImm, observe_literal)],
   )
   print(replacement)
   print(new_literals)

The output is:

.. code-block:: text

   T.Add(10, 20)
   []

``replace_symbol`` returns the addition containing 10 and 20, so neither
new literal reaches ``observe_literal``. A handler that needs to process
the replacement's children can explicitly call
``mutator.default_mutate(replacement)`` before returning its result.

Variable definitions require consistent replacements at their uses.
``mutator.def_region_kind()`` reads the context described in
:ref:`Structural Walk <ir-definition-regions>`. The mutator also provides a
substitution table: ``var_remap_get(var)`` looks up a replacement by identity, and
``var_remap_set(var, replacement)`` records one for later recursive calls.

This callback renames a local ``Let`` variable at its simple definition and
reuses the new symbol at both uses:

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

   renamed_local = tvm_ffi.structural_mutate(scoped_expr, callbacks=(ir.Var, rename_local))
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

At the definition, the callback creates ``tmp_renamed`` and records it for
``tmp``. Both body references then retrieve the same replacement. For the
input variables ``x`` and ``y``, the lookup returns ``None`` and the context
is an ordinary use, so the handler returns the original variable. The source
still binds ``tmp``.

The lookup is explicit because a matched handler processes the variable
itself. Default variable mutation does not run after the handler returns.
The substitution table stores variable replacements; it does not cache
every rewritten arithmetic node.

Default mutation enters the definition contexts registered by each IR node.
For a handler that selects definition fields manually,
``mutator.with_def_region_kind(kind, fn)`` runs the zero-argument function
``fn()`` in that context and returns its result. ``fn`` can mutate the chosen
definition child. The previous context is restored afterward; an enclosing
``DEF_PATTERN`` remains active even when a nested call requests another kind.

Ordinary Python references preserve the source through copy-on-write.
Default mutation copies changed paths and shares unchanged children, as in
the left-operand example. A handler can optionally accept a third argument,
``handler(value, mutator, inplace_mode)``. ``inplace_mode`` is an integer
that can be compared with
``tvm_ffi.InplaceMode.DISALLOW`` and ``tvm_ffi.InplaceMode.ALLOW``.
``DISALLOW`` requires preserving the input; ``ALLOW`` permits storage reuse
along uniquely owned paths. Passing ``root._move()`` transfers ownership
and permits that optimization. The examples above use
ordinary references and two-argument handlers.

The TVM FFI
`Structural Walk and Map guide <https://github.com/apache/tvm-ffi/blob/dbdcdaceeef5b0b03251e8dea87725a1c1e2cde6/docs/concepts/structural_eq_hash.rst#structural-walk-and-map>`_
describes custom object hooks and native implementations.
TIRx's constraint-aware variants are covered in
:doc:`symbolic_integer_analysis`; :doc:`writing_passes` shows how to package
a rewrite as a compiler pass.
