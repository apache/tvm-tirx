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

.. _expressions-and-types:

Types and Expressions
=====================

In this chapter, we will introduce types and expressions and explain how
they work together in the IR. Types describe properties such as dtypes,
tensor shapes, and function signatures, while expressions represent values
and computations. Compiler passes can work with these objects directly to
inspect or rewrite a program without executing it.

.. _user-ir-type:

Types
-----

``ir.Type`` is the base class for all IR types, including primitive types
for scalar and vector values, tensor types, and function types:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Describes
     - Fields
   * - ``ir.PrimType``
     - A scalar or vector dtype
     - ``dtype``
   * - ``ir.PtrType``
     - A pointer's element type and memory scope
     - ``element_type``, ``scope``
   * - ``ir.TupleType``
     - Ordered tuple field types
     - ``fields``
   * - ``ir.FuncType``
     - Function argument and return types
     - ``arg_types``, ``ret_type``
   * - ``tirx.TensorType``
     - A tensor's shape, element type, and storage properties
     - ``shape``, ``dtype``, ``scope``, ``data_alignment``, ``layout``

Next, we will look at two common types in tensor programs in more detail:
``ir.PrimType`` for scalar and vector values, and ``tirx.TensorType`` for tensors.

.. _user-ir-prim-type:

PrimType
~~~~~~~~

:class:`ir.PrimType <tvm_tirx.ir.PrimType>` describes integer,
floating-point, and boolean values, either as scalars or vectors. Its
``dtype`` records the element kind, bit width, and number of lanes.

The program declares its dynamic length as:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at: n = T.dynamic

where ``n`` is a 32-bit integer symbol with type ``ir.PrimType("int32")``.
The loop index ``i`` has the same primitive type.

We can express the addition ``A[i] + 1`` directly through the IR API,
with a scalar variable standing for the input element:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin scalar addition
   :end-before: # end scalar addition
   :prepend: from tvm_tirx import ir

``scalar_value`` has type ``ir.PrimType("float32")``, so it represents a single
32-bit floating-point value. The result ``scalar_plus_one`` has the
same type, just like ``A[i] + 1`` in the program.

.. _user-ir-vector-expression:

``PrimType`` also supports vector values. We can extend the same addition
to four float32 values at once:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin vector extension
   :end-before: # end vector extension

The type of ``vector_value`` is ``ir.PrimType("float32x4")``. The ``x4`` suffix
specifies four float32 lanes within one vector value. The addition adds
one to each lane, and ``vector_plus_one`` has the same vector type.

.. _user-tensor-type:

TensorType
~~~~~~~~~~

:class:`tirx.TensorType <tvm_tirx.tirx.TensorType>` is an IR type object
that describes a whole tensor. It records the tensor's dimensions and
the type of each element.

The program declares its input and output as:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: def add_one(
   :end-at: def add_one(

``T.Tensor((n,), "float32")`` is the TVMScript notation for a tensor type;
the parser represents it as a ``tirx.TensorType`` object. Its ``shape``
field contains one dimension, ``n``, and its ``dtype`` field is
``ir.PrimType("float32")``:

.. code-block:: text

   tirx.TensorType                  T.Tensor((n,), "float32")
   ├── shape[0]: ir.Var              n
   └── dtype: ir.PrimType            float32

The dimension ``n`` is symbolic: it is not fixed to a particular number
when the function is compiled. Each call determines ``n`` from the length
of ``A``. Since ``B`` uses the same symbol in its type, its length must
match ``A``. The loop then uses ``n`` as its bound.

Expressions
-----------

``ir.Expr`` is the base class for all expressions, which represent program
values and the operations that produce them. Each expression's ``ty`` field
records its result type:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``ty``
     - ``ir.Type``
     - The type of value represented by the expression.

Expressions include variables, constants, arithmetic, tensor reads, and calls.
These classes store their values or operands in the following fields:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Classes (examples)
     - Represents
     - Additional fields
   * - ``ir.Var``
     - A variable
     - ``name``
   * - ``ir.IntImm``, ``ir.FloatImm``, ``ir.StringImm``
     - Literal constants
     - ``value``
   * - ``ir.prim.Add``, ``ir.prim.Sub``, ``ir.prim.Mul``
     - Binary arithmetic
     - ``a``, ``b``
   * - ``ir.prim.EQ``, ``ir.prim.LT``
     - Comparisons
     - ``a``, ``b``
   * - ``ir.Op``
     - A registered operator
     - ``name``, ``doc``, ``args_info``
   * - ``ir.Call``
     - A call to an operator or function
     - ``op``, ``args``, ``attrs``, ``ty_args``
   * - ``ir.TensorLoad``
     - A tensor element read
     - ``source``, ``indices``

.. _user-ir-const:

Primitive Expressions
~~~~~~~~~~~~~~~~~~~~~

A primitive expression is an expression whose ``ty`` is an ``ir.PrimType``.
Variables, tensor reads, arithmetic, and calls can all be primitive expressions,
depending on the type of value they produce.

**Scalar expressions.** The CPU example uses the int32 expressions ``n``
and ``i`` for the loop bound and index. The tensor read ``A[i]`` and the
addition ``A[i] + 1`` both produce a single float32 value. In the GPU
example, ``i < n`` produces a boolean that decides whether a thread
writes its result. These scalar result types are stored as ``PrimType``
objects in ``ty``.

``A`` itself represents the whole tensor and has a ``TensorType``.
The expression ``A[i]`` reads element ``i`` of this tensor. TVMScript
represents the read as an ``ir.TensorLoad`` node. Its ``source`` field
refers to ``A``, and ``indices`` contains ``i``, selecting the element
to read:

.. code-block:: text

   ir.TensorLoad                    A[i]
   ├── ty: ir.PrimType               float32
   ├── source: ir.Var                A
   │   └── ty: tirx.TensorType       (n,) of float32
   └── indices[0]: ir.Var            i

**Vector expressions.** A primitive expression can also produce a vector
value. The earlier add-one result ``vector_plus_one`` has type
``ir.PrimType("float32x4")``, representing four float32 lanes within
one value.

A call whose result type is a ``PrimType`` also belongs to this category.
For example, the GPU expression ``T.cuda.thread_idx("x")`` is an
``ir.Call`` with result type ``ir.PrimType("int32")``.

.. _user-ir-prim-operations:

Primitive Operations
~~~~~~~~~~~~~~~~~~~~

The ``ir.prim`` namespace provides node classes and construction functions
for arithmetic, comparisons, boolean logic, casts, and other primitive
computations. Python operators such as ``+``, ``*``, and ``<`` combine
expressions into arithmetic and comparison nodes. Python functions such
as ``ir.prim.sigmoid`` build calls to registered operators:
``ir.prim.sigmoid(scalar_value)`` returns an ``ir.Call`` for ``prim.sigmoid``.
The :doc:`ir.prim </reference/python/ir.prim>` API reference lists the
available classes and construction helpers.

In the CPU example, ``A[i] + 1`` becomes an ``ir.prim.Add``. A binary node
stores its left operand in ``a`` and its right operand in ``b``:

.. code-block:: text

   ir.prim.Add                  A[i] + 1
   ├── a: ir.TensorLoad          A[i]
   └── b: ir.FloatImm            1.0

The arithmetic constructor derives the result type from its operands.
Since ``A[i]`` is float32, it converts the source literal ``1`` to the
float32 constant shown as ``ir.FloatImm``. The result is also float32.

These constructors also support vector operands. In ``vector_value + 1``,
the scalar constant is broadcast to all four lanes, producing the
four-lane expression ``vector_plus_one``.

The GPU example combines multiplication, addition, and comparison to
compute and check each thread's element index:

.. literalinclude:: ../get_started/examples/add_one_cuda.py
   :language: python
   :start-at:     i = T.cuda.block_idx
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

The multiplication by 256 becomes an ``ir.prim.Mul``, and the sum becomes
an ``ir.prim.Add``. The condition ``i < n`` is an ``ir.prim.LT`` whose
``a`` and ``b`` operands are the index and tensor length:

.. code-block:: text

   ir.prim.LT                       i < n
   ├── a: ir.Var                    i
   └── b: ir.Var                    n

The comparison checks whether the thread has a valid element to process.

.. _user-ir-var:

Var
~~~

``ir.Var`` represents a named value with a type. Variables can be
function parameters, loop indices, or intermediate values defined by
bindings.

The tensor parameters ``A`` and ``B`` in the program are represented by
``ir.Var`` objects. In the CPU example, we can retrieve them from
``add_one.params`` and inspect their types through ``ty``:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin parameter inspection
   :end-before: # end parameter inspection

Each parameter is a ``Var`` with its own ``name`` and ``ty`` fields.
For ``A``, these are:

.. code-block:: text

   A: ir.Var
   ├── name: "A"
   └── ty: tirx.TensorType           (n,) of float32

``input_type`` refers to this tensor type. ``output_type`` comes from
``B.ty`` in the same way.

TVMScript creates these parameter variables when it parses the function.
We can also create variables directly with ``ir.Var(name, ty)``. The
scalar add-one example in PrimType uses this constructor:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin scalar addition
   :end-before: # end scalar addition
   :prepend: from tvm_tirx import ir

``ir.Var("scalar_value", "float32")`` creates a variable whose ``name``
is ``"scalar_value"`` and whose ``ty`` is ``ir.PrimType("float32")``.
We then use this variable in ``scalar_value + 1`` to construct the
addition expression ``scalar_plus_one``.

.. _user-ir-call:

Call
~~~~

:class:`ir.Call <tvm_tirx.ir.Call>` represents a call to an operator or
function. Its ``op`` field identifies the callee, and ``args`` holds
the argument expressions. A registered operator is represented by
``ir.Op``.

In the GPU example, the index expression contains calls to
``T.cuda.block_idx("x")`` and ``T.cuda.thread_idx("x")``.
These helpers call registered CUDA operators. We can construct the
thread-index call directly:

.. code-block:: python

   from tvm_tirx import ir

   op = ir.Op.get("tirx.cuda.thread_idx")
   x_call = ir.Call(op, [ir.StringImm("x")])

``ir.Op.get`` retrieves the shared operator definition. ``ir.Call``
creates a separate expression whose ``op`` refers to that definition and
whose ``args`` contains the arguments for this use:

.. code-block:: text

   x_call: ir.Call
   ├── op → shared ir.Op            tirx.cuda.thread_idx
   ├── args[0]: ir.StringImm         "x"
   └── ty: ir.PrimType               int32

The string argument selects the x axis. Other calls can refer to the same
operator with their own arguments. The index arithmetic uses the coordinates
returned by the two calls in the program.

The same representation supports calls to registered operators and
other function-valued IR expressions. For example, we can call the
built-in ``prim.sigmoid`` operator to compute
``1 / (1 + exp(-value))``:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin sigmoid call
   :end-before: # end sigmoid call
   :prepend: from tvm_tirx import ir

``activated_value`` has the same structure, with a variable as its
argument:

.. code-block:: text

   activated_value: ir.Call
   ├── op: ir.Op                    prim.sigmoid
   ├── args[0]: ir.Var               value
   └── ty: ir.PrimType               float32

The operator's type rule gives the call the same type as its input, so
``activated_value.ty`` is ``ir.PrimType("float32")``. The helper
``ir.prim.sigmoid(value)`` is a shorter way to construct the same call.

Both examples use ``ir.Call`` to describe an operation that the compiled
program performs when it runs. The :doc:`operators_and_calls` chapter
examines how operator definitions provide type inference and argument
validation for these calls.

.. _user-ir-source-locations:

Source Locations
----------------

IR nodes can carry source locations to help diagnostics point back to
the original code. The ``loc`` field stores this information, and
:class:`ir.SourceLoc <tvm_tirx.ir.SourceLoc>` records the source file
and range.

The first parameter of ``add_one`` is ``A``. We can retrieve the
filename and starting line number of its declaration:

.. literalinclude:: examples/add_one.py
   :language: python
   :start-after: # begin source location
   :end-before: # end source location

The code follows these fields from the parameter to its source location:

.. code-block:: text

   parameter: ir.Var                 A
   └── loc: ir.SourceLoc
       ├── source_name: ir.SourceName
       │   └── name                  path to add_one_cpu.py
       └── start_line                line of A's declaration

``source_name`` and ``line`` hold the path and line number reached
through these fields, identifying ``A``'s declaration in
``add_one_cpu.py``.
