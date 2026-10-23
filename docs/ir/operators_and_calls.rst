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

A common call representation lets new operators reuse the existing IR structure
and traversal tools. It keeps an operation's shared rules separate from the
inputs and attributes of each invocation.

``ir.Op`` identifies a registered operation such as ``prim.exp``, while
``ir.Call`` records one invocation. Calls can also refer to function-valued
expressions, including variables and global function references. Both ``Op``
and ``Call`` are expressions.

Operators
---------

An operator's registration describes accepted arguments and can supply effect
information, type inference, and validation rules. Calls to the operator share
these properties. Retrieve the registered object with ``ir.Op.get(name)``.

For example, ``prim.exp`` declares one argument named ``x`` and is a pure
operation. Both properties can be inspected on the registered object:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin operator lookup
   :end-before: # end operator lookup
   :prepend: from tvm_tirx import ir

Calls
-----

The operator describes ``prim.exp`` in general. A ``Call`` combines it with the
inputs for one invocation. It inherits ``ty`` and ``loc`` from ``ir.Expr`` and
adds these fields:

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
     - Explicit type arguments, empty by default.
   * - ``attrs``
     - Attributes for this invocation, or None. A dictionary becomes
       ``ir.DictAttrs``.

``ty_args`` supplies types as inputs to the operation; ``ty`` describes its
result. Call attributes belong to one invocation, while the operator's
registration supplies the rules used to interpret them.

For example, this call applies the registered exponential operator to a float32
constant:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin call contract
   :end-before: # end call contract

The call's ``ty`` is ``ir.PrimType("float32")``. A variable with a function
type can also be the callee:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin function call type
   :end-before: # end function call type

.. _types-and-validation:

Type Inference and Validation
-----------------------------

The examples omit ``ty``, so construction derives the result type from the
callee. For an ``Op``, it uses the registered fixed return type if available,
otherwise the type-inference callback. A call to another callee with an ``ir.FuncType`` uses
that signature's return type. Without an applicable rule, the result is
``MissingType``. An explicit type, including ``MissingType``, bypasses inference.

The operator's argument contract is checked separately by its validator.
Construction can reject inputs during inference, but does not run that
validator. ``call.validate()`` runs it if one is registered, returning None on
success and raising an exception on failure.
``validate()`` performs no checks for a non-operator callee or an operator
without a validator.

For ``prim.exp``, an explicit result type lets the constructor record a call
with no arguments. Validation then rejects the argument count:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin validation failure
   :end-before: # end validation failure
   :append: invalid.validate()  # raises TypeError: expected one argument

To derive a result type later, use ``ir.reinfer_type``. It applies the same
inference rules to the current inputs and returns a type without modifying
``call.ty`` or running validation:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin provisional call
   :end-before: # end provisional call

The stored ``provisional.ty`` remains missing; ``inferred`` is float32.
A transformation must keep the inputs and stored result type consistent;
validation alone does not generally establish their agreement.

Register New Operators
----------------------

Registering new operators makes TIRx extensible: TIRx supports users to build
on the installed package and add operators from the Rust side, while the C++
core stays stable.

New operators are defined in Rust ``register_ops!`` blocks. Each block contains
one ``OpDef`` chain per operator, which supplies the registration described
above: the name and documentation, an argument signature that generates the
validator, a type-inference callback, and attributes such as the effect kind.
A registered operator joins the same registry as the built-in operators, so
Python, C++, and Rust code use it the same way.

For example, this block registers ``my_ext.clamp``, which clamps an integer
index to ``[0, bound)``. A call has the type of its index, and ``validate()``
rejects arguments that are not integer expressions:

.. code-block:: rust

   use tvm_tirx::ir::op_attr::CALL_EFFECT_KIND;
   use tvm_tirx::ir::sig::{self, IntExpr};
   use tvm_tirx::ir::{CallEffectKind, CallObj, OpDef, Type};
   use tvm_tirx::tvm_ffi::Result;

   fn index_type(call: &CallObj) -> Result<Type> {
       Ok(call.args.get(0)?.ty.clone())
   }

   tvm_tirx::register_ops! {
       OpDef::new("my_ext.clamp", "Clamp an index to the range `[0, bound)`.")
           .signature((
               sig::arg::<IntExpr>("index", "The index."),
               sig::arg::<IntExpr>("bound", "The exclusive upper bound."),
           ))
           .set_infer_type(index_type)
           .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
   }

Once the operators are registered in Rust, they can be used as built-in operators
on Python side.

See :doc:`/reference/python/ir` for the complete operator and call APIs. The
:download:`operators_and_calls.py <examples/operators_and_calls.py>` example includes runnable
calls, re-inference, and validation failures.
