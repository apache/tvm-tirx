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

IR Overview
===========

An intermediate representation (IR) describes a program in a form that compiler
components can inspect and transform. It makes values, computations, control
flow, and types explicit, allowing analyses and transformations to work with the
program's structure independently of the syntax used to construct it.

TIRx's C++, Python, and Rust interfaces share a common IR object system through
TVM FFI, allowing compiler components written in different languages to work
together. The IR can be extended with custom operators and types, and its
existing interfaces can be used to build custom analyses, transformation passes,
and backends.

The following diagram shows the main IR classes and how they relate to one another.

.. code-block:: text

   TVM FFI object system
   ├── ir.Expr                         Expressions
   │   ├── ir.Var                      Variables
   │   ├── ir.Call                     Calls
   │   └── ir.BaseFunc                  Function definitions
   │       └── tirx.Function            Tensor-program functions
   ├── ir.Type                         Types
   │   ├── ir.PrimType                  Scalar and vector types
   │   ├── ir.FuncType                  Function signatures
   │   └── tirx.TensorType              Tensor types
   ├── ir.Stmt                         Statements
   │   ├── ir.Bind                      Bindings
   │   ├── ir.If                        Branches
   │   ├── ir.For                       Loops
   │   └── ir.TensorStore               Tensor writes
   └── ir.IRModule                     Global function definitions

A function that returns ``x + 1`` brings these families together: a type
describes its input and result, expressions describe the addition, and
statements form its body. The following chapters construct that function and
place it in a module, starting with :doc:`foundations`.
