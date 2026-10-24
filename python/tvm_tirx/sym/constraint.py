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
"""Data-only symbolic premises; no analyzer ingestion or normalization."""

import tvm_ffi

from . import _ffi_api


@tvm_ffi.register_object("sym.Constraint")
class Constraint(tvm_ffi.Object):
    """A symbolic binding, range, or predicate, with collector-defined lifetime."""


@tvm_ffi.register_object("sym.BindingConstraint")
class BindingConstraint(Constraint):
    """The primitive value captured when ``var`` is bound.

    The RHS ``value`` is retained unchanged, including mutable reads. This is
    data about a binding, not permission to re-evaluate its RHS after writes.

    Parameters
    ----------
    var : tvm_tirx.ir.Var
        The variable whose bound value is recorded.
    value : tvm_tirx.ir.PrimExpr
        The RHS expression at the binding site, including any mutable reads.
    """

    def __init__(self, var, value):
        self.__init_handle_by_constructor__(_ffi_api.BindingConstraint, var, value)


@tvm_ffi.register_object("sym.RangeConstraint")
class RangeConstraint(Constraint):
    """A variable contained in a half-open range, without stride information.

    Parameters
    ----------
    var : tvm_tirx.ir.Var
        The variable whose possible values lie in the range.
    domain : tvm_tirx.ir.Range
        A half-open superset of the variable's possible values, exposed as
        the constraint's ``range`` field.
    """

    def __init__(self, var, domain):
        self.__init_handle_by_constructor__(_ffi_api.RangeConstraint, var, domain)


@tvm_ffi.register_object("sym.PredicateConstraint")
class PredicateConstraint(Constraint):
    """A boolean ``condition`` whose validity is supplied by its collector.

    Parameters
    ----------
    condition : tvm_tirx.ir.PrimExpr
        The boolean predicate supplied by the collector.
    assume_always : bool, optional
        Defaults to ``False``. If ``True``, the caller guarantees that the predicate
        remains true throughout the active constraint scope, including after mutations.
        This guarantees predicate truth; referenced values may still change.
    """

    def __init__(self, condition, assume_always=False):
        self.__init_handle_by_constructor__(_ffi_api.PredicateConstraint, condition, assume_always)


@tvm_ffi.register_object("sym.ConstraintContext")
class ConstraintContext(tvm_ffi.Object):
    """Tracks the constraints relevant to each point in a traversal.

    Holds variable bindings, value ranges, and predicates collected during traversal.
    Callbacks can inspect these constraints when processing the current expression or statement.

    Parameters
    ----------
    parent : ConstraintContext or None, optional
        Parent context, or ``None`` to construct an empty root.
    parent_local_prefix : int, optional
        Number of parent-local entries to capture, between zero and the parent's
        ``local_size()``, inclusive. Defaults to zero and must be zero for a root.

    """

    def __init__(self, parent=None, parent_local_prefix=0):
        self.__init_handle_by_constructor__(_ffi_api.ConstraintContext, parent, parent_local_prefix)

    def append(self, constraint):
        """Append a local premise without changing any existing prefix.

        Parameters
        ----------
        constraint : Constraint
            Premise to append. Existing premises must not be modified through
            aliases, including changes to the meaning of their referenced IR.
        """
        _ffi_api.ConstraintContextAppend(self, constraint)

    def local_size(self):
        """Return the local entry count, excluding captured ancestors."""
        return _ffi_api.ConstraintContextLocalSize(self)

    def __iter__(self):
        """Iterate ancestor-first, capturing the local bounds when iteration is requested.

        Entries are read lazily. Later appends do not enter this iteration, and
        no mutable backing-list alias is exposed. Use ``list(context)`` for an
        independent list; the constraint objects and their referenced IR remain shared.
        """
        stack = []
        context, count = self, self.local_size()
        while context is not None:
            stack.append((context, count))
            context, count = _ffi_api.ConstraintContextGetParent(context)
        return (
            _ffi_api.ConstraintContextGetLocal(context, index)
            for context, count in reversed(stack)
            for index in range(count)
        )
