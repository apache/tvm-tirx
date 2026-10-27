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

tvm_tirx.script.parser
======================

.. automodule:: tvm_tirx.script.parser
   :no-members:

.. currentmodule:: tvm_tirx.script.parser

The corresponding builder form below omits generated lexical helpers and source
locations; it illustrates the construction rather than quoting generated output.

.. code-block:: python

    from tvm_tirx.script.ir_builder import IRBuilder
    from tvm_tirx.tirx.script import ir_builder as B

    with IRBuilder() as builder:
        with B.function_():
            B.func_name_("example")
            a = B.arg_("a", B.int32)
            B.emit_(B.evaluate(a + 1))
    example = builder.get()

A namespace extending the TIRx builder can customize ``emit_`` as follows:

.. code-block:: python

    from types import SimpleNamespace
    from tvm_tirx.script.parser import make_decorator, register_namespace

    M = SimpleNamespace(**vars(B))

    def emit(value, *, loc=None):
        print(type(value).__name__)
        return B.emit_(value, loc=loc)

    M.emit_ = emit
    M.function = make_decorator(M, namespace_path="custom.function")
    register_namespace("custom", M)

Use ``@M.function`` at the definition site. The registered namespace supplies
these hooks; ``tvm_tirx.script.ir_builder.parser_protocol`` defines their shared
contracts and language-specific placeholders, not a complete language.

Extension API modules declare their own destinations with
``_init_exposed_op_api`` or ``_init_exposed_type_api`` from
``tvm_tirx.ir._expose_api``. Exposure publishes selected definitions to those
destinations; it does not add default names to the parser. An imported object
such as ``custom_type.BOX`` is available to ``@T.function`` through normal Python
scope. When parsing printed text, supply the extension through a source import,
``extra_vars``, or an explicit ``register_namespace("custom_type", custom_type)``.

``T.register_script_namespace("custom", namespace)`` binds the supplied module
or namespace instance as ``T.custom``. This explicit attribute binding does not
register a standalone ``custom`` parser default. ``T.expose_script_op`` and
``T.expose_script_type`` still select definitions for declared destinations;
they return the built-in script binding for supported built-in prefixes and
``None`` for custom roots. Read custom APIs from the extension's declared module.

Functions
---------

.. autosummary::
   :toctree: generated/

   register_namespace
   register_namespace_initializer
   make_decorator
   make_macro_decorator
   pyfunc
   ir_module
   from_source

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``parse``
     - :obj:`tvm_tirx.script.parser.from_source`

Namespaces
----------

.. toctree::
   :maxdepth: 1

   script.parser.protocol_registry
