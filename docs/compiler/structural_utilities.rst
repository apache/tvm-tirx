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

By default, traversal follows the structural fields and child traversal hooks
registered by each node type. For containers, it visits ``Array`` and ``List``
elements and ``Map`` and ``Dict`` values, while skipping map keys. Fields marked
``structural_eq="ignore"`` are also skipped. These operations work on IR objects
without executing the program they describe.

The API sections start from the same arithmetic expression, making their
callback behavior easy to compare. Each section includes its own setup, so
you can run it independently.

Structural Walk
---------------

``tvm_ffi.structural_walk`` automatically traverses IR. Its callbacks can
collect nodes, check properties, or stop the traversal when a matching value
is found.

.. code-block:: text

   tvm_ffi.structural_walk(
       root, callbacks=(), with_def_region_kind=(), order="post")

``root`` is the expression, statement, function, module, type, or FFI container
where traversal starts. ``callbacks`` determines which Python function runs
for each visited value. In a ``(Type, handler)`` entry, ``Type`` is a class
such as ``ir.Var``, and ``handler`` is a callable supplied by the caller.
A callable supports function-call syntax: it can be a Python function or,
for example, a list's ``append`` method.
The walker calls that function with the actual matching object as its single
argument. For example, ``(ir.Var, on_var)`` means that the walker calls
``on_var(var)`` whenever it visits a variable.

The function names in the following table are defined in the examples below:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, on_var)``
     - Variables are passed to ``on_var``.
   * - ``((ir.prim.Add, ir.prim.Mul), on_arithmetic)``
     - Either an addition or a multiplication is passed to the same
       ``on_arithmetic`` function.
   * - ``[(ir.Var, on_name), (ir.IntImm, on_int)]``
     - Variables are passed to ``on_name``; integer IR constants are passed
       to ``on_int``. Each list entry pairs a type with its function.
   * - ``on_value``
     - The function is passed on its own, with no type filter. It can receive
       any kind of value visited by the walker.
   * - ``()``
     - The empty tuple contains no callback entries. The walker still follows
       default structural traversal. This is the default when ``callbacks``
       is omitted.

The notation ``handler(value)`` describes how the walker calls a callback.
The ``value`` argument is the current object, such as an ``ir.Var`` or an
``ir.IntImm``. The caller passes the function itself in ``callbacks``; the
walker supplies its argument during traversal.

An IR class selector also matches its subclasses. Only the first matching
entry runs. A specific handler must appear before a broader one to take
precedence over it. Unmatched values still follow default descent.
``ir.IntImm`` selects an integer IR constant, while
``int`` selects an inline integer value; ``object`` or ``typing.Any`` matches
both inline values and FFI objects, and ``tvm_ffi.Object`` matches only FFI
objects.

``order="post"`` calls the handler after the children; ``order="pre"`` calls
it before them. The corresponding enum values are
``tvm_ffi.WalkOrder.POSTORDER`` and ``PREORDER``. The optional
``with_def_region_kind`` table uses the same selection forms, but calls
``handler(value, kind)`` with a ``tvm_ffi.DefRegionKind``. Entries in
``callbacks`` are tried first; this table is consulted only when none matches.

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
early exit. It does not return collected nodes or a rewritten root. Callbacks
can record observations in state such as a list. An early-exit payload is
available through the returned interrupt's ``value`` field.

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

The walker follows the child traversal registered by each node type. For this
arithmetic expression, those children are the operands shown above. The
expression traces below select ``ir.Expr``, which includes variables,
integer constants, and arithmetic nodes.
The examples in this section run in order and share these variables.

We can now collect the variables in the expression:

.. code-block:: python

   variables = []

   def on_var(var):
       variables.append(var)

   tvm_ffi.structural_walk(expr, (ir.Var, on_var))
   print([var.name for var in variables])

The output is:

.. code-block:: text

   ['x', 'y']

The walker calls ``on_var(x)`` and ``on_var(y)``. The callback receives the
variable objects themselves, which is why the final line can read their
``name`` fields. The unmatched ``Mul`` and ``Add`` nodes still receive default
descent. ``on_var`` returns ``None`` implicitly, so traversal continues.

A tuple of classes shares one callback across those classes. Here, additions
and multiplications both call ``on_arithmetic``:

.. code-block:: python

   operator_names = []

   def on_arithmetic(node):
       operator_names.append(type(node).__name__)

   tvm_ffi.structural_walk(
       expr, ((ir.prim.Add, ir.prim.Mul), on_arithmetic)
   )
   print(operator_names)

The output is ``['Add', 'Add', 'Mul']``. The type tuple lists alternatives:
``on_arithmetic`` receives one node per call, whether that node is an
addition or a multiplication.

A list of entries assigns different functions to different types:

.. code-block:: python

   leaves = []

   def on_name(var):
       leaves.append(("var", var.name))

   def on_int(value):
       leaves.append(("int", int(value)))

   tvm_ffi.structural_walk(
       expr, [(ir.Var, on_name), (ir.IntImm, on_int)]
   )
   print(leaves)

The output is:

.. code-block:: text

   [('var', 'x'), ('int', 1), ('var', 'y'), ('int', 2)]

Here, ``on_name`` receives ``ir.Var`` objects and ``on_int`` receives
``ir.IntImm`` objects. The type entries determine which function runs;
the functions read the fields or convert the values they receive.

A callable passed on its own, without a ``(Type, handler)`` entry, is also
called a bare callable. It has no type filter. Its body can check types
before accessing fields specific to a node class:

.. code-block:: python

   bare_variables = []
   visited_types = []

   def on_value(value):
       visited_types.append(type(value).__name__)
       if isinstance(value, ir.Var):
           bare_variables.append(value.name)

   tvm_ffi.structural_walk(expr, on_value)
   print(visited_types)
   print(bare_variables)

The output is:

.. code-block:: text

   ['Var', 'IntImm', 'Add', 'Var', 'IntImm', 'Add', 'Mul']
   ['x', 'y']

``on_value`` receives all seven expression nodes, including arithmetic nodes
and integer constants. Only variable names enter ``bare_variables``, because
the callback body performs that type check itself.

The next example places a variable handler before an expression handler to
illustrate first-match selection:

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
from uses. ``ir.prim.Let(var, value, body)`` binds a value to a local variable
while evaluating its body. Here, ``tmp`` is defined once and used twice in
``tmp + tmp``:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(tmp, expr, tmp + tmp)
   regions = []
   tvm_ffi.structural_walk(
       scoped_expr,
       with_def_region_kind=(
           ir.Var,
           lambda var, kind: regions.append((var.name, kind.name)),
       ),
   )
   print([kind for name, kind in regions if name == "tmp"])

The output is:

.. code-block:: text

   ['DEF_SIMPLE', 'NONE', 'NONE']

The first occurrence defines ``tmp``; the other two use it in the body.
``NONE`` means an ordinary use, ``DEF_SIMPLE`` marks a simple variable
definition, and ``DEF_PATTERN`` marks a pattern definition, which can bind
symbols through a type. The inputs ``x`` and ``y`` remain ordinary uses.

Structural Map
--------------

``tvm_ffi.structural_map`` applies replacement rules with automatic recursion,
supporting substitutions and rewrites throughout the IR.

.. code-block:: text

   tvm_ffi.structural_map(
       root, callbacks=(), with_def_region_kind=(), order="post")

``root`` is the expression, statement, function, module, type, or FFI container
to transform. ``callbacks`` specifies Python replacement functions and
optional type filters. Each function receives the current value and returns
the value to install at that position. The table uses functions defined in
this section's examples:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, specialize_var)``
     - ``specialize_var(var)`` returns a replacement for a variable.
   * - ``((ir.prim.Add, ir.prim.Mul), fold_arithmetic)``
     - The same ``fold_arithmetic(node)`` function handles either operator class.
   * - ``[(ir.Var, specialize_var), (ir.prim.Add, fold_arithmetic)]``
     - Variables and additions have separate replacement functions.
   * - ``keep_value``
     - A function on its own has no type filter. ``keep_value`` returns its
       input unchanged.
   * - ``()``
     - The empty tuple supplies no replacement callback. Default mapping
       still runs. This is the default when ``callbacks`` is omitted.

IR class selectors match subclasses, and only the first matching entry runs.
A specific handler must appear before a broader one to take precedence over it.
Unmatched values follow default mapping. ``object`` or ``typing.Any`` matches
all values, while ``tvm_ffi.Object`` matches only FFI objects.

Each callback returns the value to install: either the input unchanged or a
valid replacement. ``None`` does not mean "keep this value."
``structural_map`` returns the transformed root.

``with_def_region_kind`` accepts the same entry forms, but its callbacks receive
two arguments, ``(value, kind)``, where ``kind`` is a ``tvm_ffi.DefRegionKind``.
This table is tried only if no callback in ``callbacks`` matches. ``order`` accepts
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

We will start with an expression to transform:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

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
node with 45 as well.

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

With ``with_def_region_kind``, the replacement callback receives
``(value, kind)``. ``NONE`` marks an ordinary use, ``DEF_SIMPLE`` a simple
definition, and ``DEF_PATTERN`` a pattern definition that can bind symbols
through a type. The callback still returns the unchanged value or a valid
replacement. Definition sites must remain valid, including the variable bound
by a ``Let`` expression.

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
where visiting starts. A callback is a Python function that receives the
current value and the active visitor. ``callbacks`` determines which function
handles each value, with the following forms:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Expr, visit_expr)``
     - Expression nodes are passed to ``visit_expr(node, visitor)``.
   * - ``((ir.prim.Add, ir.prim.Mul), visit_expr)``
     - Either operator class selects the same ``visit_expr`` function.
   * - ``[(ir.prim.Mul, visit_left), (ir.Expr, visit_expr)]``
     - Multiplications select ``visit_left``; other expressions select
       ``visit_expr``. Entries are tried in order.
   * - ``visit_expr``
     - The function is selected without a type filter for every value reached.
       Its recursive calls determine which children are reached.
   * - ``()``
     - The empty tuple supplies no callback, so default visiting runs.
       This is the default when ``callbacks`` is omitted.

The function names above refer to callbacks defined in this section's examples.
IR class selectors match subclasses, and only the first matching entry runs.
A specific handler must appear before a broader one to take precedence over it.
Unmatched values follow default structural descent. ``object`` or
``typing.Any`` matches all values, while ``tvm_ffi.Object`` matches only FFI
objects.

The callback receives two arguments, ``(value, visitor)``. ``value`` is the
current object; the supplied ``tvm_ffi.StructuralVisitor`` carries the same
callbacks into recursive calls.
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

A selected child enters callback dispatch through ``visitor.visit(child)``.
Default descent from the current node proceeds through
``visitor.default_visit(node)``.
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

The output is ``y``. A callback that visits several children explicitly must
propagate any child interrupt before visiting the next child.

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
hook. A custom node hook follows its own child traversal contract.

Structural Mutator
------------------

``tvm_ffi.structural_mutate`` combines replacement with explicit control over
recursion. Its callbacks can select which children to rewrite and arrange
recursive calls around their local transformations.

.. code-block:: text

   tvm_ffi.structural_mutate(root, callbacks=())

``root`` is the expression, statement, function, module, type, or FFI container
to transform. A callback is a Python function that receives the current value
and the active mutator, then returns the replacement value. ``callbacks``
selects that function using the following forms:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, replace_var)``
     - Variables are passed to ``replace_var(var, mutator)``.
   * - ``((ir.prim.Add, ir.prim.Mul), mutate_and_fold)``
     - Either operator class selects the same ``mutate_and_fold`` function.
   * - ``[(ir.prim.Mul, mutate_left), (ir.Var, replace_var)]``
     - Multiplications select ``mutate_left`` and variables select ``replace_var``.
   * - ``mutate_and_fold``
     - The function is selected without a type filter. Its recursive calls
       determine which children receive callbacks.
   * - ``()``
     - The empty tuple supplies no callback, so default mutation runs.
       This is the default when ``callbacks`` is omitted.

The functions above are defined in this section's examples. IR class selectors
match subclasses, and only the first matching entry runs. A specific handler
must appear before a broader one to take precedence over it. Unmatched values
follow default mutation. ``object`` or ``typing.Any`` matches all values, while
``tvm_ffi.Object`` matches only FFI objects.

The callback receives two arguments, ``(value, mutator)``, and returns the
value to install. ``value`` is the current object. The supplied
``tvm_ffi.StructuralMutator`` carries the same
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
invoking its callback again on that node. To fold
``((8 + 1) * (3 - 2)) // 4`` from its leaves to its root, construct the
arithmetic nodes and define a local rule:

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
gives the same result as the post-order map. A map is shorter when this order
is all a rewrite needs; a mutator lets each callback arrange its own recursive
calls.

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

The TVM FFI
`Structural Walk and Map guide <https://github.com/apache/tvm-ffi/blob/dbdcdaceeef5b0b03251e8dea87725a1c1e2cde6/docs/concepts/structural_eq_hash.rst#structural-walk-and-map>`_
describes custom object hooks and native implementations.
TIRx's constraint-aware variants are covered in
:doc:`symbolic_integer_analysis`; :doc:`writing_passes` shows how to package
a rewrite as a compiler pass.
