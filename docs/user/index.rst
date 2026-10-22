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

.. _what-tirx-provides:

Overview
========

TIRx is an Open ML Compiler Foundation. It provides reusable infrastructure for
building and extending machine learning compilers, including an intermediate
representation, symbolic analysis, transformation passes, code generation, and
runtime support.

A shared IR connects program construction, analysis, and transformation.
Expressions represent program values and computations, and types describe
properties such as tensor shapes, element types, and memory scopes. Analyses
and passes use this representation to reason about and transform programs.

You can use individual components or combine them into a compiler workflow.
When you need additional functionality, you can build on the same components
to add custom analyses, transformation passes, operators, types, or backend
implementations.
