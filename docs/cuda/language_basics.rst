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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/native_basics.rst

CUDA Language Basics
====================

In this chapter, we will introduce the CUDA interfaces for writing and running
TIRx kernels. You choose the launch configuration, assign work to threads,
allocate shared and per-thread storage, and synchronize data accesses where
needed. Loops, branches, and device intrinsics describe the computation, which
TIRx compiles into CUDA device code. The following sections explain these
interfaces and show how to inspect the resulting IR and generated CUDA.

* :doc:`language/functions` covers function parameters, symbolic shapes,
  compile-time specialization, and launch configuration.
* :doc:`language/data_types` explains scalar and vector dtypes, their CUDA
  representations, and pointers.
* :doc:`language/tensors` covers shared and per-thread storage, mutable
  scalars, and tensor views.
* :doc:`language/control_flow` introduces conditionals, loop forms, and
  thread election.
* :doc:`language/threads_sync` shows how to call device intrinsics and
  include custom CUDA functions.
* :doc:`language/compiling` explains compiler configuration and how to
  inspect the IR and generated CUDA.
* :doc:`language/parser_utils` introduces helpers for reusable code
  fragments and compile-time metadata.

.. toctree::
   :hidden:
   :maxdepth: 1

   language/functions
   language/data_types
   language/tensors
   language/control_flow
   language/threads_sync
   language/compiling
   language/parser_utils
