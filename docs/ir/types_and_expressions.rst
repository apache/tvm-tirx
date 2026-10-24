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

Expressions describe program values and computations, and types describe
properties of those values, such as dtypes, tensor shapes, and function
signatures. The compiler uses this information to check operation inputs and
determine result types without running the program.

.. _user-ir-type:
.. _user-ir-prim-type:

Types
-----

``ir.Type`` is the base class for type objects. It defines the following field:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``loc``
     - ``ir.Location``
     - Source information for diagnostics. Defaults to ``UnknownLoc`` and does
       not affect structural comparison.

Concrete subclasses inherit ``loc`` and add fields describing different kinds
of values. Common examples include:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Describes
     - Additional fields
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

Some of these fields refer to other type objects. For example, ``FuncType``
stores the types of a function's arguments and result:

.. literalinclude:: examples/types_and_expressions.py
   :language: python
   :start-after: # begin composite type
   :end-before: # end composite type
   :prepend: from tvm_tirx import ir, tirx

``signature`` describes a function that takes and returns int32 values.
``integer_type.dtype`` holds the int32 dtype descriptor;
``signature.arg_types[0]`` and ``signature.ret_type`` both refer to the
``integer_type`` object. Tuple fields and pointer element types similarly
refer to other type objects.

.. _user-tensor-type:

``tirx.TensorType`` combines an element type with shape and storage information:

.. literalinclude:: examples/types_and_expressions.py
   :language: python
   :start-after: # begin constant tensor type
   :end-before: # end constant tensor type

``tile_type`` describes an ``(8, 128)`` tensor. Its ``shape`` contains integer
expressions for ``8`` and ``128``, and its ``dtype`` is
``ir.PrimType("float32")``. Dimensions can also use variables and arithmetic.

Use ``==`` to compare type structure and ``same_as`` to compare object identity.

.. _user-ir-var:

Expressions
-----------

``ir.Expr`` is the base class for expression objects. It defines these fields:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``ty``
     - ``ir.Type``
     - Result type. ``ir.MissingType`` marks type information that has not yet
       been supplied or computed.
   * - ``loc``
     - ``ir.Location``
     - Source information for diagnostics. Defaults to ``UnknownLoc`` and does
       not affect structural comparison.

Subclasses inherit ``ty`` and ``loc`` and add fields for their stored values or
operands. Common expression forms include:

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
   * - ``ir.Call``
     - A call to an operator or function
     - ``op``, ``args``, ``attrs``, ``ty_args``
   * - ``ir.TensorLoad``
     - A tensor element read
     - ``source``, ``indices``

Arithmetic and comparison nodes for primitive values are available under
``ir.prim``. The binary nodes above store their left and right operand
expressions in ``a`` and ``b``.

Registered operations such as ``prim.exp`` are identified by ``ir.Op``
expressions and invoked through ``ir.Call``. For example,
``ir.prim.exp(value)`` builds a ``Call`` whose ``op`` is
``ir.Op.get("prim.exp")`` and whose ``args`` contains ``value``.

Operands and call arguments are themselves expressions, allowing these forms
to be combined into larger computations.

.. _user-ir-const:

To describe ``x + 1``, create a variable with the integer type above and a
constant of that type. Adding them constructs an ``Add`` expression:

.. literalinclude:: examples/types_and_expressions.py
   :language: python
   :start-after: # begin scalar expressions
   :end-before: # end scalar expressions

``x`` is a ``Var`` and ``added`` is an ``Add``. Both carry ``integer_type`` in
their ``ty`` field. Reuse the same variable object when a symbol appears in
several places: another ``ir.Var("x", integer_type)`` is a distinct symbol,
even with the same name.

A tensor variable follows the same pattern, with a ``TensorType`` in its
``ty`` field. Its shape can reuse the integer expressions described above:

.. literalinclude:: examples/types_and_expressions.py
   :language: python
   :start-after: # begin tensor expressions
   :end-before: # end tensor expressions

The first dimension stores the expression ``2*n``. The same ``n`` can appear
in loop bounds and indices. The element load ``tensor[0, 0]`` is another
expression, with type ``ir.PrimType("float32")``.

The symbol ``n`` has a known int64 type even though its value is not yet known.
Constructing these types and expressions does not allocate tensor storage.

.. _user-ir-register-type:

Register New Types
------------------

New types can be added from Python or Rust without modifying the C++ core.
Define a subclass of ``ir.Type`` and register it under its own key. Its
reflected fields let structural comparison and printing handle it in the same
way as built-in types.

In Python, use ``tvm_ffi.dataclasses.py_class``; in Rust, use
``tvm_ffi::rs_class`` and ``tvm_ffi::def_object_ref!``.

For example, this type describes a pair of values by the types of its two
components. In Python, the fields are annotated class attributes, and the
constructor takes them in order:

.. code-block:: python

   from tvm_ffi.dataclasses import py_class
   from tvm_tirx import ir


   @py_class("my_ext.PairType")
   class PairType(ir.Type):
       first: ir.Type
       second: ir.Type


   pair = ir.Var("p", PairType(ir.PrimType("int32"), ir.PrimType("float32")))

In Rust, the object struct starts with the ``TypeObj`` it extends, and the
generated ``new`` for ``PairTypeObj`` takes that parent object and then the
fields:

.. code-block:: rust

   use tvm_tirx::ir::{Location, Type, TypeObj};
   use tvm_tirx::tvm_ffi::{self, def_object_ref, rs_class};

   #[rs_class("my_ext.PairType", final)]
   pub struct PairTypeObj {
       base: TypeObj,
       pub first: Type,
       pub second: Type,
   }

   def_object_ref!(pub PairType, Type, PairTypeObj);

   impl PairType {
       pub fn new(first: impl Into<Type>, second: impl Into<Type>) -> Self {
           PairTypeObj::new(TypeObj::new(Location::unknown()), first, second).into()
       }
   }

Once registered, the type can be assigned to variables and used inside other
types, just like a built-in type.

See :class:`~tvm_tirx.tirx.TensorType` for the complete constructor and storage
options, and :doc:`/reference/python/ir` for expression and type APIs. The
:download:`types_and_expressions.py <examples/types_and_expressions.py>` example contains the
runnable code and additional API checks.
