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

Operators and Calls
===================

In this chapter, we will introduce operators and calls. A registered
operator provides the shared argument requirements, type rules, and effect
information for a computation. A call refers to an operator or function
and supplies argument expressions for that use. We will examine how
the IR records these definitions and calls, infers result types, and
checks arguments.

Operators
---------

Calls to the same registered operator share an
:class:`ir.Op <tvm_tirx.ir.Op>` definition. ``ir.Op`` records the
operator's name, argument descriptions, and attributes, together with rules
for type inference and argument validation. Each call supplies its own
argument expressions, and the compiler uses the shared rules to infer its
result type and validate its arguments.

The GPU example uses the registered operator ``tirx.cuda.thread_idx`` to
obtain a thread coordinate. We can retrieve its definition with
``ir.Op.get`` and inspect its argument information and attributes:

.. code-block:: python

   from tvm_tirx import ir
   op = ir.Op.get("tirx.cuda.thread_idx")
   argument_names = [argument.name for argument in op.args_info]
   effect = op.get_attr("TCallEffectKind")

``argument_names`` is ``["axis"]``: ``op.args_info`` describes an argument
named ``axis``. The actual axis value is supplied by each call. ``effect``
is ``ir.CallEffectKind.Pure``, declaring that calls to this operator have
no memory side effects. These argument requirements and attributes apply
to every call that refers to ``op``. Compiler passes use this shared
information to infer call result types, validate arguments, and inspect
effects.

Calls
-----

:class:`ir.Call <tvm_tirx.ir.Call>` represents one use of an operator or
function with specific argument expressions. Its ``op`` field references
the expression being called, known as the callee, and ``args`` holds the
arguments for this use. The main fields describe what is called, what
is passed to it, and what type of value it produces:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``op``
     - The callee: a registered operator or another function-valued expression.
   * - ``args``
     - Value arguments in calling order.
   * - ``ty_args``
     - Types supplied as arguments to the callee, empty by default.
   * - ``attrs``
     - Operator-specific configuration or annotations for this call,
       stored as an ``Attrs`` object, or ``None``.
   * - ``ty``
     - The result type of this call, inferred or explicitly supplied.

The GPU example uses two calls to compute the element index ``i``:

.. literalinclude:: ../get_started/examples/add_one_cuda.py
   :language: python
   :start-at:     i = T.cuda.block_idx
   :end-at:     i = T.cuda.block_idx
   :dedent: 4

``T.cuda.block_idx("x")`` supplies the block coordinate, and
``T.cuda.thread_idx("x")`` supplies the thread's coordinate within that
block. Each helper constructs an ``ir.Call``. The multiplication and
addition use the calls' results as operands:

.. code-block:: text

   ir.prim.Add
   ├── a: ir.prim.Mul
   │   ├── a: ir.Call                   T.cuda.block_idx("x")
   │   └── b: ir.IntImm                 256
   └── b: ir.Call                       T.cuda.thread_idx("x")

The thread-index call refers to the registered ``tirx.cuda.thread_idx``
operator. We can construct it directly through the IR API:

.. code-block:: python

   op = ir.Op.get("tirx.cuda.thread_idx")
   x_call = ir.Call(op, [ir.StringImm("x")])

``ir.Op.get`` retrieves the shared operator definition. ``ir.Call``
creates an expression whose ``op`` refers to that definition and whose
``args`` contains the axis argument. Its ``ty`` records the int32 result:

.. code-block:: text

   x_call: ir.Call
   ├── op → shared ir.Op                tirx.cuda.thread_idx
   ├── args[0]: ir.StringImm             "x"
   └── ty: ir.PrimType                   int32

``op.args_info[0].name`` is the parameter name ``"axis"``;
``x_call.args[0]`` supplies its argument value ``"x"``. The call describes
reading the current thread's x-axis coordinate when the compiled GPU
program runs.

The constructor also accepts the registered name directly:
``ir.Call("tirx.cuda.thread_idx", [ir.StringImm("x")])``.

The program uses the x axis. As an additional example, we can construct
a call for the y axis with the same operator:

.. code-block:: python

   y_call = ir.Call(op, [ir.StringImm("y")])

Both ``x_call.op`` and ``y_call.op`` reference the shared ``op``. Each
call has its own ``args`` array, containing ``"x"`` or ``"y"``, and
both calls produce an int32 coordinate:

.. code-block:: text

   x_call.op ──┐
               ├──> op: ir.Op  tirx.cuda.thread_idx
   y_call.op ──┘

Expression Arguments
~~~~~~~~~~~~~~~~~~~~

A call's arguments can be computations as well as variables and constants.
As an extension of the scalar add-one expression, we can apply sigmoid to
``scalar_value + 1``. The variable ``scalar_value`` has type
``ir.PrimType("float32")``:

.. code-block:: python

   scalar_value = ir.Var("scalar_value", "float32")
   scalar_plus_one = scalar_value + 1
   activated_plus_one = ir.prim.sigmoid(scalar_plus_one)

``ir.prim.sigmoid(...)`` constructs and returns an ``ir.Call`` object.
``activated_plus_one`` refers to this call, which is also a primitive
expression because its result type is ``ir.PrimType("float32")``.
We can construct the same call explicitly:

.. code-block:: python

   sigmoid_op = ir.Op.get("prim.sigmoid")
   activated_plus_one = ir.Call(sigmoid_op, [scalar_plus_one])

In both forms, the call's ``op`` refers to the registered ``prim.sigmoid``
operator, and ``args[0]`` refers to the addition expression ``scalar_plus_one``:

.. code-block:: text

   activated_plus_one: ir.Call
   ├── op → shared ir.Op                prim.sigmoid
   ├── args[0]: ir.prim.Add              scalar_value + 1
   │   ├── a: ir.Var                    scalar_value
   │   └── b: ir.FloatImm               1.0
   └── ty: ir.PrimType                   float32

When the compiled program runs, sigmoid receives the result of the
addition. This expression therefore computes
``1 / (1 + exp(-(scalar_value + 1)))`` and produces a float32 value.

Function Calls
~~~~~~~~~~~~~~

The callee can also be a function defined in IR. A function stored in an
``IRModule`` is referenced through an ``ir.GlobalVar``. For example, a
small helper can put the scalar add-one computation in a function:

.. code-block:: python

   @T.function(private=True)
   def add_one_scalar(value: T.float32) -> T.float32:
       return value + 1

``private=True`` makes this an internal helper. We can place it in a
module and construct a call that passes ``scalar_value`` as its argument:

.. code-block:: python

   call_module = ir.IRModule({"add_one_scalar": add_one_scalar})
   callee = call_module.get_global_var("add_one_scalar")
   function_call = ir.Call(callee, [scalar_value], ty=add_one_scalar.ret_type)

``callee`` is the module's global reference to ``add_one_scalar``.
``function_call.op`` refers to this ``GlobalVar``, and
``function_call.args[0]`` supplies the value for the helper's parameter
``value``. We supply the helper's float32 return type through ``ty``.
The call holds the reference and argument; the module holds the function's
definition. Other function-valued IR expressions can also serve as callees.

Type Arguments and Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some operators need types or configuration in addition to value arguments.
For example, a temporary tensor allocation needs the type of tensor to
allocate. This additional example describes storage for four float32
elements:

.. code-block:: python

   from tvm_tirx import tirx
   temporary_type = tirx.TensorType((4,), "float32", scope="local")
   allocation_call = ir.Call(
       "tirx.alloc_tensor", [], ty_args=[temporary_type], attrs={"alignment": 16}
   )

``ty_args[0]`` supplies ``temporary_type`` to the allocation operator.
Its type rule also uses this type as the call's result, so
``allocation_call.ty`` is a ``tirx.TensorType``. A call's result can
therefore be tensor-valued as well as scalar- or vector-valued.

``attrs`` supplies annotations for this allocation. The dictionary becomes
an ``ir.DictAttrs`` object, and ``allocation_call.attrs["alignment"]``
records the annotation value ``16``. The operator's definition determines
which attributes it accepts and how they are used. These attributes belong
to this call; the attributes on ``ir.Op`` describe the shared operator.

Calls in Function Bodies
~~~~~~~~~~~~~~~~~~~~~~~~

The allocation call above describes a tensor allocation, but has not yet
been inserted into a function body. To let later statements refer to the
allocated tensor, we bind the call's result to a variable with ``ir.Bind``.
We then place this binding in a ``SeqStmt``, which arranges statements in
execution order:

.. code-block:: python

   temporary = ir.Var("temporary", temporary_type)
   allocation_binding = ir.Bind(temporary, allocation_call)
   allocation_body = ir.SeqStmt([allocation_binding])

``allocation_binding.value`` refers to ``allocation_call``, and
``allocation_binding.var`` is ``temporary``. Later statements in the
same scope can use this variable to access the temporary tensor.

When we do not need to retain a call's result, we can use ``ir.Evaluate``
to construct a statement that evaluates the call and discards its result.
For example, ``tirx.masked_store`` writes to memory and returns ``void``.
Even with a ``void`` result type, ``ir.Call`` is an expression, while
``SeqStmt`` accepts only statements, so we wrap the call in ``ir.Evaluate``.
The operator's ``TCallEffectKind`` attribute is ``ir.CallEffectKind.UpdateState``,
so compiler passes cannot discard the call merely because its result is unused.

.. _types-and-validation:

Type Inference and Validation
-----------------------------

Each call stores its result type in ``ty``. When ``ty`` is omitted,
``ir.Call`` uses the operator's registered rule or the callee's function
signature to infer it. An explicitly supplied ``ty`` is stored as given;
``MissingType`` marks a result whose type has not yet been determined.

A tool may need to keep an unfinished call while it assembles the IR.
An explicit ``ir.Type.missing()`` lets it record the call's structure
and defer result-type inference until the inputs are ready. With the
thread-index operator and argument introduced above, we can compare
automatic inference, an explicit type, and a deferred type:

.. code-block:: python

   inferred_call = ir.Call(op, x_call.args)
   typed_call = ir.Call(op, x_call.args, ty="int32")
   provisional = ir.Call(op, x_call.args, ty=ir.Type.missing())

``inferred_call`` and ``typed_call`` both store an int32 result type.
``provisional`` retains ``MissingType`` because a type was supplied
explicitly.

The operator's definition also supplies the argument requirements checked
by :meth:`validate() <tvm_tirx.ir.Call.validate>`. Validation is separate
from construction, so tools can assemble a call before checking it.
For example, a rebuilt call might accidentally omit the axis argument:

.. code-block:: python

   invalid = ir.Call(op, [], ty="int32")
   invalid.validate()

The explicit result type bypasses inference, allowing this incomplete
call to be constructed. ``invalid.validate()`` raises ``TypeError``
because the operator requires an axis argument. Type inference can also
reject arguments while computing a result type.

A transformation may also change the type of a call's input. For example,
the scalar sigmoid call produces float32, but a version whose input is
``vector_value`` should produce float32x4. We can construct a provisional
call with this four-lane variable as its argument and leave the result
type undetermined:

.. code-block:: python

   provisional_sigmoid = ir.Call(sigmoid_op, [vector_value], ty=ir.Type.missing())

The new argument is now in place. A pass can compute the corresponding
result type with :func:`ir.reinfer_type <tvm_tirx.ir.reinfer_type>` and
record it in a replacement call:

.. code-block:: python

   inferred = ir.reinfer_type(provisional_sigmoid)
   rebuilt_call = ir.Call(sigmoid_op, provisional_sigmoid.args, ty=inferred)
   rebuilt_call.validate()

``inferred`` is ``ir.PrimType("float32x4")``, matching the new input.
The inference function returns this type without changing
``provisional_sigmoid.ty``, which remains ``MissingType``.
``rebuilt_call`` stores the inferred type and is validated before a pass
uses it in place of the scalar call. The original scalar call keeps its
float32 type.
