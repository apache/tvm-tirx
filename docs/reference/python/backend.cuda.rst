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

tvm_tirx.backend.cuda
=====================

.. automodule:: tvm_tirx.backend.cuda
   :no-members:

.. currentmodule:: tvm_tirx.backend.cuda

Namespaces
----------

* :doc:`tvm_tirx.backend.cuda.op <backend.cuda.op>`
* :doc:`tvm_tirx.backend.cuda.lang <backend.cuda.lang>`
* :doc:`tvm_tirx.backend.cuda.layout <backend.cuda.layout>`
* :doc:`tvm_tirx.backend.cuda.script <backend.cuda.script>`
* :doc:`tvm_tirx.backend.cuda.transform <backend.cuda.transform>`
* :doc:`tvm_tirx.backend.cuda.axis <backend.cuda.axis>`
* :doc:`tvm_tirx.backend.cuda.backend_config <backend.cuda.backend_config>`
* :doc:`tvm_tirx.backend.cuda.codegen <backend.cuda.codegen>`
* :doc:`tvm_tirx.backend.cuda.compiler <backend.cuda.compiler>`
* :doc:`tvm_tirx.backend.cuda.iket <backend.cuda.iket>`
* :doc:`tvm_tirx.backend.cuda.launch <backend.cuda.launch>`
* :doc:`tvm_tirx.backend.cuda.ptx <backend.cuda.ptx>`
* :doc:`tvm_tirx.backend.cuda.runtime.locality_domain <backend.cuda.runtime.locality_domain>`
* :doc:`tvm_tirx.backend.cuda.thread_scope <backend.cuda.thread_scope>`
* :doc:`tvm_tirx.backend.cuda.tile <backend.cuda.tile>`

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   backend.cuda.axis
   backend.cuda.backend_config
   backend.cuda.codegen
   backend.cuda.compiler
   backend.cuda.iket
   backend.cuda.launch
   backend.cuda.ptx
   backend.cuda.runtime.locality_domain
   backend.cuda.thread_scope
   backend.cuda.tile

Classes
-------

.. autosummary::
   :toctree: generated/

   BackendConfig

Functions
---------

.. autosummary::
   :toctree: generated/

   export_cuda_host
   jit_cuda_host
   register_backend
   script_namespaces
   script_namespace

Launch a kernel from a precompiled cubin
----------------------------------------

On Linux, a host TVMScript function can launch a named entry from a cubin built
for the target GPU. For example, suppose ``add_one.cubin`` exports the CUDA kernel
``add_one(const float* x, float* y, int n)`` with C linkage:

.. code-block:: python

   from pathlib import Path

   import tvm_tirx
   from tvm_tirx.backend.cuda import jit_cuda_host
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def wrapper(x: T.Tensor((n,), T.float32), y: T.Tensor((n,), T.float32)):
       if n > 0:
           T.cuda.launch(
               "add_one", (x.data, y.data, T.Cast("int32", n)),
               T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256),
           )

   target = tvm_tirx.target.Target("cuda", host="cuda_host")
   host = tvm_tirx.compile(wrapper, target=target).mod
   module = jit_cuda_host(host, cubins={"add_one": Path("add_one.cubin").read_bytes()})
   # Given CUDA tensors x and y, invoke module["wrapper"](x, y).

The tuple supplies kernel arguments in the cubin's ABI order and types. The
host computes launch geometry for each call. Compilation uses the system C++
compiler and the TVM-FFI build cache. The generated host library embeds the cubin
and launches it with the CUDA driver.
