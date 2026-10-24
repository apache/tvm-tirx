# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Traverse and transform TIRx with the facts available at each use.

Callbacks receive a ``tvm_tirx.sym.ConstraintContext``: iterate it to read facts,
or use ``list(context)`` to retain an independent list. Bind/Assert add facts for
later statements; If/While/lazy calls add guards, For/thread launch add ranges,
and Let adds a body binding. Mutation uses the transformed values.
Call and RegionStmt use registered operator hooks; other calls follow ordinary
descent, and other regions isolate their body facts.

Callbacks control descent. Normal exits and interrupts restore outer facts;
errors propagate without restoration, so discard the traversal after failure.
The C++ structural visitor and mutator docs describe the full collection rules.
"""

import tvm_ffi

from . import _ffi_api


@tvm_ffi.register_object("tirx.StructuralVisitorWithConstraintContext")
class StructuralVisitorWithConstraintContext(tvm_ffi.StructuralVisitor):
    """Visits values using their structural visit behavior while maintaining
    the constraint context relevant to each callback.

    Construct an instance for default traversal, or use the visitor supplied to
    a :func:`structural_visit_with_constraint_context` callback to control recursion.

    Attributes
    ----------
    constraints : tvm_tirx.sym.ConstraintContext
        Effective symbolic facts active at the current value. Read them in
        ancestor-first order by iterating this context. Do not append during traversal.

    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.StructuralVisitorWithConstraintContext)

    def default_visit(self, value):
        """Visit a value's children with the applicable symbolic facts.

        In a traversal callback, use this method to continue through all
        children without calling the callback again on ``value`` itself.
        Child callbacks still receive this visitor and its active facts. Intermediate
        sequence and lazy-call argument arrays receive no callback.

        Parameters
        ----------
        value : Any
            Value whose children should be visited.

        Returns
        -------
        result : tvm_ffi.VisitInterrupt or None
            The first interrupt returned by a child, or ``None`` when traversal
            completes. Propagate an interrupt to stop the enclosing traversal.

        """
        return _ffi_api.StructuralVisitorWithConstraintContextDefaultVisit(self, value)


@tvm_ffi.register_object("tirx.StructuralMutatorWithConstraintContext")
class StructuralMutatorWithConstraintContext(tvm_ffi.StructuralMutator):
    """Mutates values using their structural mutation behavior while maintaining
    the constraint context relevant to each callback.

    Constraints use the resulting expressions and bindings.

    Construct an instance for default transformation, or use the mutator
    supplied to a :func:`structural_mutate_with_constraint_context` callback to
    control recursion. Both ``mutate`` and ``default_mutate`` preserve their
    input values, copying nodes whose children change.

    Attributes
    ----------
    constraints : tvm_tirx.sym.ConstraintContext
        Effective symbolic facts active at the current value. Read them in
        ancestor-first order by iterating this context. Do not append during traversal.

    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.StructuralMutatorWithConstraintContext)

    def default_mutate(self, value):
        """Transform a value's children with the applicable symbolic facts.

        In a traversal callback, use this method to transform all children
        without calling the callback again on ``value`` itself. Child callbacks
        still receive this mutator and its active facts. Intermediate sequence and
        lazy-call argument arrays receive no callback.

        Parameters
        ----------
        value : Any
            Value whose children should be transformed. It is preserved even
            when the result changes.

        Returns
        -------
        result : Any
            The original value if no child changes, otherwise a transformed
            copy with the resulting children.

        """
        return _ffi_api.StructuralMutatorWithConstraintContextDefaultMutate(self, value)


def _pre_order(order):
    if order not in ("pre", "post"):
        raise ValueError("order must be 'pre' or 'post'")
    return order == "pre"


def structural_walk_with_constraint_context(root, callback, *, order="post"):
    """Walk a value structurally with its active symbolic facts.

    The walk visits each occurrence as a tree. Sequence statements and lazy
    if_then_else arguments are visited directly, without callbacks on their
    intermediate arrays. Deduplication can use a pre-order callback that returns
    ``tvm_ffi.WalkResult.SKIP`` on repeats.

    Parameters
    ----------
    root : Any
        Root value to traverse.
    callback : callable
        Called as ``callback(value, constraints)`` for each visited value. The
        second argument is the active ``tvm_tirx.sym.ConstraintContext``;
        callbacks must not append to it.
        Return ``None`` or ``tvm_ffi.WalkResult.ADVANCE`` to continue,
        ``tvm_ffi.WalkResult.SKIP`` to prune a pre-order subtree, or
        ``tvm_ffi.VisitInterrupt(payload)`` to stop traversal.
    order : {"post", "pre"}, optional
        Visit children before their parent (the default), or the parent before
        its children. A scoped node's post-order callback sees the outer facts
        after its body scope has ended.

    Returns
    -------
    result : tvm_ffi.VisitInterrupt or None
        The interrupt returned by a callback, or ``None`` if traversal completes.

    Raises
    ------
    ValueError
        If ``order`` is neither ``"pre"`` nor ``"post"``.

    """
    return _ffi_api.StructuralWalkWithConstraintContext(root, callback, _pre_order(order))


def structural_visit_with_constraint_context(root, callback):
    """Visit a value structurally with a callback that owns child traversal.

    The callback receives ``(value, visitor)``. It may call ``visitor.visit``
    for selected children or ``visitor.default_visit`` for the value's default
    descent. Returning without either call prunes that value's subtree.
    ``visitor.constraints`` contains the active symbolic facts.

    Parameters
    ----------
    root : Any
        Root value to traverse.
    callback : callable
        Called as ``callback(value, visitor)`` with a
        :class:`StructuralVisitorWithConstraintContext`. Return ``None`` to continue
        or a ``tvm_ffi.VisitInterrupt`` to stop; propagate child interrupts.

    Returns
    -------
    result : tvm_ffi.VisitInterrupt or None
        The interrupt returned by a callback, or ``None`` if traversal completes.

    """
    return _ffi_api.StructuralVisitWithConstraintContext(root, callback)


def structural_map_with_constraint_context(root, callback, *, order="post"):
    """Structurally map a value with its active symbolic facts.

    The callback returns the unchanged input or a replacement value and must
    not mutate its input in place. Callbacks run at each occurrence with the
    same intermediate-array exception as walk. A graph rewrite that must
    preserve sharing keeps its own callback memo.

    Parameters
    ----------
    root : Any
        Root value to transform. A regular Python reference is preserved
        through copy-on-write.
    callback : callable
        Called as ``callback(value, constraints)`` with the active
        ``tvm_tirx.sym.ConstraintContext`` (do not append during traversal).
        Return the value to preserve it or a replacement; do not modify the
        callback's input in place.
    order : {"post", "pre"}, optional
        Transform children before invoking the callback (the default), or
        invoke the callback first and traverse its returned value's children.

    Returns
    -------
    result : Any
        The transformed root value.

    Raises
    ------
    ValueError
        If ``order`` is neither ``"pre"`` nor ``"post"``.

    """
    return _ffi_api.StructuralMapWithConstraintContext(root, callback, _pre_order(order))


def structural_mutate_with_constraint_context(root, callback):
    """Mutate a value with a callback that owns recursive mutation.

    The callback receives ``(value, mutator)``. It may call ``mutator.mutate``
    for selected children or ``mutator.default_mutate`` for the matched value's
    default mutation. Its returned value is final and is not traversed again.
    ``mutator.constraints`` contains the active symbolic facts.

    Parameters
    ----------
    root : Any
        Root value to transform. A regular Python reference is preserved
        through copy-on-write.
    callback : callable
        Called as ``callback(value, mutator)`` with a
        :class:`StructuralMutatorWithConstraintContext`. Return the mutated owning
        value.

    Returns
    -------
    result : Any
        The transformed root value.

    """
    return _ffi_api.StructuralMutateWithConstraintContext(root, callback)
