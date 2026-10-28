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

Writing and Debugging Passes
============================

In this chapter, we will turn the shape specialization from
:doc:`structural_utilities` into a compiler pass. We will combine it with
statement simplification, check the resulting program, and inspect the
intermediate stages. The example continues the add-one program used in
the :doc:`IR Walkthrough </ir/overview>`.

Starting from a Function
------------------------

Consider the add-one loop with the extra bounds check introduced in the
:ref:`constraint-aware traversal example <ir-traversal-with-constraints>`:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir, tirx
   from tvm_tirx.script import from_source
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def guarded_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           if i < n:
               B[i] = A[i] + 1

A pass takes an ``ir.IRModule`` and returns a module, so we first place the
function in one:

.. code-block:: python

   before = ir.IRModule({"guarded_add_one": guarded_add_one})

The module gives transformations a common entry point for named functions.
Its functions remain directly accessible as IR objects, as described in
:doc:`/ir/functions_and_modules`.

Packaging the Shape Specialization
----------------------------------

A function pass adapts a transformation over one ``tirx.Function`` to the
module interface. The following factory takes a dimension and its replacement
extent, then returns a pass configured for that substitution. Its inner
callback performs the actual rewrite:

.. code-block:: python

   def specialize_dimension(dimension, extent):
       replacement = ir.const(extent, dimension.ty)

       @tirx.transform.function_pass(opt_level=0, name="SpecializeDimension")
       def transform(func, mod, ctx):
           return tvm_ffi.structural_map(
               func,
               (ir.Var, lambda v: replacement if v.same_as(dimension) else v),
               order="pre",
           )

       return transform

   specialize = specialize_dimension(n, 128)
   specialized = specialize(before)
   print(int(specialized["guarded_add_one"].params[0].ty.shape[0]))

The output is ``128``. The callback receives a function, its module, and
the active ``PassContext`` containing execution settings. It returns the
resulting function, and the pass manager applies it to each TIRx function
in the module. This factory selects a specific dimension object; unrelated
variables with the same printed name are unaffected. Its replacement
extent must be valid for the tensor types being specialized.

The two calls have separate roles: ``specialize_dimension(n, 128)`` creates
a pass object, and ``specialize(before)`` runs that pass on the module.
``opt_level=0`` makes this pass eligible in a sequential pipeline whose
optimization level is at least zero.

For a body-only change, ``func.with_body(new_body)`` preserves the function's
other fields. ``tirx.transform.Apply(callback)`` is a shorter adapter when
the callback only needs the function. ``ir.transform.module_pass`` instead
adapts a ``(module, context) -> module`` callback for transformations that add
or remove functions or coordinate changes across function boundaries.
Both pass decorators also accept classes implementing ``transform_function``
or ``transform_module``, respectively.

Combining Transformation Stages
-------------------------------

Specialization changes the shapes and loop bound. The function still
contains ``if i < 128`` inside a loop that already guarantees that condition.
The guard was also redundant in the original dynamic loop: ``range(n)``
already guarantees ``i < n``. The two passes therefore demonstrate separate
changes: specialization fixes the shapes, and
``tirx.transform.StmtSimplify`` removes the redundant branch using loop facts.
We can sequence them:

.. code-block:: python

   pipeline = ir.transform.Sequential([
       specialize_dimension(n, 128),
       tirx.transform.StmtSimplify(),
   ])

   with ir.transform.PassContext(opt_level=2):
       after = pipeline(before)

   after["guarded_add_one"].show()

The displayed function includes:

.. code-block:: python

   @T.function
   def guarded_add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
       for i in range(128):
           B[i] = A[i] + T.float32(1.0)

The stages produce these changes:

.. code-block:: text

   before       shapes (n,),   loop range(n),   guard i < n
      │ SpecializeDimension
   specialized  shapes (128,), loop range(128), guard i < 128
      │ tirx.StmtSimplify
   after        shapes (128,), loop range(128), direct tensor store

``PassContext`` supplies the optimization level, required and disabled pass
names, configuration, and instruments. ``Sequential`` uses that context to
select eligible passes and runs them in the listed order.
Setting ``opt_level=2`` selects among the passes supplied to this pipeline;
it does not construct a pipeline automatically.
``PassContext.current()`` retrieves the active context;
``PassContext.list_configs()`` lists registered configuration options.

The simplifier connects the tools from the preceding chapters. In
``src/tirx/transform/stmt_simplify.cc``, ``StmtSimplifier`` derives from
``StmtExprMutatorWithConstraintContext``. It obtains a synchronized analyzer
to simplify expressions and conditions. The loop range supplies the premise
for eliminating this guard. More involved rewrites must also account for
effects, types, and scope; the native pass handles those concerns for its
supported transformations.

Checking the Result
--------------------

We can check the result's well-formedness and inspect the expected loop:

.. code-block:: python

   assert tirx.analysis.check_well_formed(after)
   result = after["guarded_add_one"]
   print(int(result.params[0].ty.shape[0]))
   print(type(result.body[0].body[0]).__name__)
   print(before["guarded_add_one"].params[0].ty.shape[0].same_as(n))

The output is:

.. code-block:: text

   128
   TensorStore
   True

The result has the fixed shape and direct tensor store, while the source
module still has its symbolic shape. To check the whole function, construct
an independent expected program and compare their structures:

.. code-block:: python

   expected = from_source(
       """
   @T.function
   def guarded_add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
       for i in range(128):
           B[i] = A[i] + 1
   """,
       extra_vars={"T": T},
   )
   ir.assert_structural_equal(result, expected)

This comparison checks the intended IR, including the loop and tensor
accesses. Well-formedness checks variable definitions and scope, tensor
access invariants, loop control, and thread bindings. It raises on invalid
IR by default; ``assert_mode=False`` requests a boolean result.
Neither check establishes semantic equivalence for arbitrary transformations.
A pass also needs a correctness argument and execution tests for the behavior
it changes, on a build with the relevant backend.

Other checks address more local questions:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Tool
     - Use
   * - ``tirx.analysis.undefined_vars(fragment, defs=...)``
     - Find variable uses without preceding definitions, with optional
       enclosing definitions supplied by the caller.
   * - ``call.validate()``
     - Invoke the operator's registered call validator, if present.
   * - ``ir.reinfer_type(call)``
     - Derive a result type from current inputs without changing ``call.ty``
       or invoking its validator.
   * - ``tirx.analysis.verify_memory(function)``
     - Check for illegal host-side access to device memory at a stage with
       the required target and device-scope information.

Changing arguments may require rebuilding dependent types and binders.
Preserve source locations, attributes, and annotations when reconstructing
nodes so subsequent diagnostics and passes retain their context.
``ir.unique_name_supply.UniqueNameSupply`` helps reserve or generate printed
names; a new variable still needs a fresh identity and valid definition site.

Inspecting Intermediate Programs
--------------------------------

When a result differs from expectations, inspect where the difference first
appears. ``script()`` returns TVMScript source and ``show()`` displays it.
Pass instruments automate inspection over a pipeline. For example, print
the input to statement simplification and its output:

.. code-block:: python

   from tvm_tirx.ir.transform.instrument import PassPrintingInstrument

   printing = PassPrintingInstrument(
       ["tirx.StmtSimplify"],
       ["tirx.StmtSimplify"],
   )
   with ir.transform.PassContext(opt_level=2, instruments=[printing]):
       inspected = pipeline(before)

The first printed program still contains ``if i < 128``; the second has the
direct store. Pass names match exactly; a pass object's ``info.name`` gives
the name to use. ``PrintBeforeAll`` and ``PrintAfterAll`` observe all executed
passes, and ``ir.transform.PrintIR`` can be inserted
as an explicit pipeline stage. Printer annotations and source highlighting
are described in :doc:`/ir/printing_and_parsing`.

A custom instrument can check each stage. We can combine one with timing
and numbered source dumps:

.. code-block:: python

   from tvm_tirx.ir.transform.instrument import DumpIR, PassTimingInstrument

   @ir.transform.pass_instrument
   class CheckWellFormed:
       def run_after_pass(self, mod, info):
           tirx.analysis.check_well_formed(mod)

   timing = PassTimingInstrument()
   with ir.transform.PassContext(
       opt_level=2,
       instruments=[CheckWellFormed(), timing, DumpIR("validation/pass_ir")],
   ):
       observed = pipeline(before)
       profile = timing.render()

   print(profile)

The printed profile lists executed passes and their timings; the measured
durations vary between runs. ``timing.render()`` must run inside its context,
and the resulting string remains available afterward. ``DumpIR`` saves files
such as ``000_SpecializeDimension.py`` and ``001_tirx.StmtSimplify.py`` in
``validation/pass_ir``. These contain the intermediate programs we inspected
above. Instruments also observe nested passes, so the sequential pipeline
itself produces another observation. The custom checker is appropriate here
because every stage is expected to produce well-formed IR.

Reproducing a Stage
-------------------

A saved intermediate program can be parsed for a focused experiment. Include
imports when producing standalone source:

.. code-block:: python

   stage_source = after.script(extra_config={"ir.comment_imports": False})
   restored_stage = from_source(stage_source)
   ir.assert_structural_equal(after, restored_stage)

For registered object graphs, JSON provides another representation:

.. code-block:: python

   restored_graph = ir.load_json(ir.save_json(after))
   ir.assert_structural_equal(after, restored_graph)

These forms preserve the program for inspection and reconstruction. Execution
also needs the relevant build, operator registrations, and external resources.

Native Implementation Tools
---------------------------

The Python pipeline above completes the transformation and debugging example.
For implementations in C++, the following headers provide tools for pass
management, analysis, and scoped state.

Native passes use ``tvm/ir/transform.h`` for the pass manager and
``tvm/tirx/stmt_functor_with_constraint_context.h`` for constraint-aware
statement mutation. ``tvm::SideEffect`` from ``tvm/ir/expr.h`` supports effect
queries; ``ScopeStack`` and ``With`` from ``tvm/ir/scope_stack.h`` and
``with_context.h`` support scoped native state. The native analysis headers
``tvm/ir/analysis.h`` and ``tvm/tirx/analysis.h`` also expose undefined-variable
analysis, checks, ``CalculateExprComplexity``, and ``FindEntryFunc``.

See :doc:`/reference/python/ir.transform`,
:doc:`/reference/python/ir.transform.instrument`, and
:doc:`/reference/python/tirx.transform` for the full pass interfaces, and
:doc:`/dev/source_build` for building the backends used to execute a program.
