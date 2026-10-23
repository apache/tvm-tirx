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

Expressions and Types
=====================

``ir.Expr`` represents a value or computation in a program, such as a variable,
a constant, or an addition. ``ir.Type`` is the base class for objects describing
the types of those values. Concrete types record a scalar's dtype, a tensor's
shape and element type, or a function's argument and return types.

.. _user-ir-var:

Expressions
-----------

All expressions share two fields:

* ``ty``: an ``ir.Type`` describing the expression's result.
* ``loc``: an ``ir.Location`` recording source information for diagnostics.
  It defaults to ``UnknownLoc`` when no location is supplied.

Concrete expression classes add fields for their stored values or operands:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Represents
     - Additional fields
   * - ``ir.Var``
     - A variable
     - ``name``
   * - ``ir.IntImm``
     - An integer constant
     - ``value``
   * - ``ir.prim.Mul``
     - A multiplication
     - ``a``, ``b``
   * - ``ir.Call``
     - A call
     - ``op``, ``args``, ``attrs``, ``ty_args``
   * - ``ir.TensorLoad``
     - A tensor element read
     - ``source``, ``indices``

Operands are themselves expressions, so computations are built by connecting
these objects. Their fields describe the computation; ``ty`` describes the
result's type. Arithmetic and comparisons, calls, and tensor reads all use this
representation.

.. _user-ir-const:

For example, ``ir.Var`` creates a variable and ``ir.const`` creates a constant
expression. Multiplying them constructs a ``Mul`` with two expression operands:

.. literalinclude:: examples/foundations.py
   :language: python
   :start-after: # begin scalar expressions
   :end-before: # end scalar expressions
   :prepend: from tvm_tirx import ir, tirx

``n`` is a ``Var`` and ``extent`` is a ``Mul``. Both carry
``ir.PrimType("int64")`` in their ``ty`` field.

Reuse the same variable object when a symbol appears in several places. Creating
another ``ir.Var("n", "int64")`` creates a distinct symbol, even with the same
name.

.. _user-ir-type:
.. _user-ir-prim-type:

Types
-----

All type objects carry a ``loc`` field for source information, defaulting to
``UnknownLoc``. Concrete types add fields describing the properties of values:

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

Types can contain other types: tuple fields, pointer element types, and
function signatures refer to further ``ir.Type`` objects. For example:

.. literalinclude:: examples/foundations.py
   :language: python
   :start-after: # begin composite type
   :end-before: # end composite type

``signature`` describes a function taking one int64 argument and returning an
int64 result. Use ``==`` to compare type structure and ``same_as`` to compare
object identity.

.. _user-tensor-type:

The tensor layer adds ``tirx.TensorType`` as a subclass of ``ir.Type``. A tensor
variable is an ordinary ``ir.Var`` whose ``ty`` refers to a ``TensorType``:

.. literalinclude:: examples/foundations.py
   :language: python
   :start-after: # begin tensor expressions
   :end-before: # end tensor expressions

The first dimension stores ``extent``, the multiplication expression using
``n``. That same variable can appear in loop bounds and indices. The element
load ``x[0, 0]`` has type ``ir.PrimType("float32")``.

The symbol ``n`` has a known int64 type even though its value is not yet known.
``ir.MissingType`` marks type information that has not been supplied or computed.
Constructing these types and expressions does not allocate tensor storage.

See :class:`~tvm_tirx.tirx.TensorType` for the complete constructor and storage
options, and :doc:`/reference/python/ir` for expression and type APIs. The
:download:`foundations.py <examples/foundations.py>` example contains the
runnable code and additional API checks.
