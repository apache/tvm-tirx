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
"""Unified type system in the project."""

import keyword
import sys
from collections.abc import Callable
from typing import ClassVar
from weakref import ref

import numpy as np
import tvm_ffi

from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import MemoryScope

from . import _ffi_api
from ._exposure import _api_attribute, _api_slot, _bind_api, _relative_name
from .base import Node, Scriptable


@tvm_ffi.register_object("ir.Type")
class Type(Node, Scriptable):
    """The base class of all types.

    ``__expr_methods__`` names methods exposed on expressions of this type.
    Each method receives the type instance and then the expression operand:
    ``expr.method(...)`` is equivalent to ``expr.ty.method(expr, ...)``.
    ``__expr_properties__`` maps read-only expression property names to
    getters taking ``(type_instance, expression)``. Ordinary expression
    attributes take precedence, then declared properties, then methods.
    Subclasses inherit these declarations or replace them explicitly.
    """

    __expr_methods__: tuple[str, ...] = ()
    __expr_properties__: ClassVar[dict[str, Callable]] = {}

    @staticmethod
    def missing():
        """Construct a MissingType for missing type information."""
        return MissingType()

    def __eq__(self, other):
        """Compare two types for structural equivalence."""
        return bool(tvm_ffi.structural_equal(self, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Compares two TVM types by referential equality."""
        return self.is_(other)


@tvm_ffi.register_object("ir.MissingType")
class MissingType(Type):
    """Type information that has not been supplied or computed.

    Unlike AnyType or Void, this is not a concrete type and must be resolved
    before a boundary that requires fully typed IR.
    """

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.MissingType)


@tvm_ffi.register_object("ir.NoneType")
class NoneType(Type):
    """The concrete type whose only value is None."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.NoneType)


@tvm_ffi.register_object("ir.AnyType")
class AnyType(Type):
    """The top type, which admits any value."""

    def __init__(self, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.AnyType, loc)


@tvm_ffi.register_object("ir.TensorRegionType")
class TensorRegionType(Type):
    """The type of a multi-dimensional :class:`tvm_tirx.ir.TensorRegion`."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.TensorRegionType)


@tvm_ffi.register_object("ir.OpaqueType")
class OpaqueType(Type):
    """Type marker for opaque values that must be removed from finished IR."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.OpaqueType)


@tvm_ffi.register_object("ir.PrimType")
class PrimType(Type):
    """A primitive scalar or vector data type, including the ``void`` sentinel.

    Parameters
    ----------
    dtype : str or tvm_ffi.dtype
        Element kind, bit width, and optional vector lane count, such as ``float32x4``.
    """

    def __init__(self, dtype):
        self.__init_handle_by_constructor__(_ffi_api.PrimType, dtype)

    def __call__(self, expr=None):
        """Create a variable, construct a typed literal, or cast an expression."""
        from ._constant import const
        from .expr import Var
        from .prim import _ffi_api as prim_ffi_api

        if expr is None:
            return Var("", self)
        if isinstance(expr, str) and str(self.dtype) not in ("bool", "void", ""):
            expr = float(expr)
        if isinstance(expr, int | float | np.integer | np.floating | np.bool_):
            return const(expr, self, UNKNOWN_LOC)
        return prim_ffi_api._cast(expr, self, UNKNOWN_LOC)

    def __getitem__(self, shape):
        """Describe local storage with this element type and the given shape."""
        return _PrimTypeArrayAnnotation(self, shape if isinstance(shape, tuple) else (shape,))

    def __eq__(self, other):
        if isinstance(other, str):
            return self.dtype == other
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        dtype = self.dtype
        return hash((dtype.type_code, dtype.bits, dtype.lanes))

    def __str__(self):
        return str(self.dtype)

    def matches_code(self, *codes) -> bool:
        """Return whether this type has any of the given DLPack dtype codes."""
        type_code = self.dtype.type_code
        return any(type_code == int(code) for code in codes)

    def matches_element_type(self, code, bits: int) -> bool:
        """Return whether this type has the given scalar element code and bits."""
        dtype = self.dtype
        return dtype.type_code == int(code) and dtype.bits == bits

    def is_scalar(self) -> bool:
        """Return whether this type has exactly one fixed lane."""
        return self.dtype.lanes == 1


class _PrimTypeArrayAnnotation:
    """A shaped primitive annotation, interpreted by a dialect's declaration builder."""

    __slots__ = ("dtype", "shape")

    def __init__(self, dtype: PrimType, shape: tuple):
        self.dtype = dtype
        self.shape = shape


@tvm_ffi.register_object("ir.StringType")
class StringType(Type):
    """Semantic string type, independent of the function's calling convention."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.StringType)


@tvm_ffi.register_object("ir.PtrType")
class PtrType(Type):
    """A pointer to an element type in a specified memory scope.

    Parameters
    ----------
    element_type : tvm_tirx.ir.Type
        The type of pointer's element.

    scope : str or tvm_tirx.runtime.MemoryScope
        The storage scope into which the pointer addresses.
    """

    def __init__(self, element_type, scope=MemoryScope("global"), *, loc=UNKNOWN_LOC):
        self.__init_handle_by_constructor__(_ffi_api.PtrType, _normalize_type(element_type), scope)
        if loc is not UNKNOWN_LOC:
            # Keep the script constructor's source-context behavior on the class alias.
            from tvm_tirx.script.ir_builder.base import (
                at,  # pylint: disable=import-outside-toplevel
            )

            at(loc, self)


@tvm_ffi.register_object("ir.TupleType")
class TupleType(Type):
    """The type of tuple values.

    Parameters
    ----------
    fields : List[Type]
        The fields in the tuple
    """

    def __init__(self, fields, loc: Location = UNKNOWN_LOC):
        self.__init_handle_by_constructor__(
            _ffi_api.TupleType, [_normalize_type(field) for field in fields], loc
        )


@tvm_ffi.register_object("ir.FuncType")
class FuncType(Type):
    """Function type.

    Records the argument types in order and a single return type.

    Parameters
    ----------
    arg_types : List[tvm_tirx.ir.Type]
        The argument types

    ret_type : tvm_tirx.ir.Type
        The return type.
    """

    def __init__(self, arg_types, ret_type):
        self.__init_handle_by_constructor__(
            _ffi_api.FuncType,
            [_normalize_type(arg) for arg in arg_types],
            _normalize_type(ret_type),
        )


def _normalize_type(value) -> Type:
    """Normalize supported explicit type arguments without invoking arbitrary factories.

    Bare concrete Type classes denote their nullary instances. Primitive dtype
    strings become PrimType values without constructing a variable or requiring
    a builder frame.
    """
    if isinstance(value, Type):
        return value
    if isinstance(value, type) and issubclass(value, Type) and value is not Type:
        return value()
    if isinstance(value, str | tvm_ffi.dtype):
        return PrimType(value)
    raise TypeError(f"Expected an IR type or primitive dtype, got {type(value).__name__}")


_EXPOSED_TYPES = {}
_TYPE_API_DESTINATIONS = {}
_TYPE_API_BINDINGS = {}


def _exposed_type_class(type_key):
    info = tvm_ffi.core._lookup_or_register_type_info_from_type_key(type_key)
    cls = info.type_cls
    if cls is None or "__init__" not in vars(cls):
        parent = info.parent_type_info
        base = parent.type_cls
        if base is None or "__init__" not in vars(base):
            base = _exposed_type_class(parent.type_key)
        namespace, _, name = type_key.rpartition(".")
        cls = tvm_ffi.register_object(type_key)(
            type(name, (base,), {"__slots__": (), "__module__": namespace})
        )
    return cls


def _publish_type(type_key, prefix, module_name):
    path = _relative_name(type_key, prefix)
    if path is None:
        return None
    namespace, separator, name = path.rpartition(".")
    names = [name]
    if len(name) > 4 and name.endswith("Type"):
        short = name[:-4]
        names = [short, name] if keyword.iskeyword(short) else [short]
    cls = None
    result = None
    for name in names:
        path = namespace + separator + name
        slot = _api_slot(module_name, path)
        if slot is None:
            continue
        container, leaf = slot
        if cls is None:
            cls = _exposed_type_class(type_key)
        existing = _api_attribute(container, leaf)
        previous = _TYPE_API_BINDINGS.get((module_name, path))
        if previous is not None and (
            previous[0]() is not sys.modules[module_name]
            or previous[1]() is not container
            or existing is not previous[2]
        ):
            previous = None
        existing_info = getattr(existing, "__tvm_ffi_type_info__", None)
        same_type = getattr(existing_info, "type_key", None) == type_key
        prefixes = _TYPE_API_DESTINATIONS[module_name]
        priority = prefixes.index(prefix)
        previous_priority = -1
        if previous is not None:
            previous_priority = max(
                (
                    i
                    for i, source in enumerate(prefixes)
                    if _relative_name(previous[3], source) is not None
                ),
                default=-1,
            )
        if existing is None or same_type or (previous is not None and priority > previous_priority):
            _bind_api(container, leaf, cls)
            if existing is None or previous is not None:
                try:
                    owner = ref(container)
                except TypeError:
                    # SimpleNamespace does not support weak references. Keep the
                    # nested namespace identity without retaining its root module.
                    def owner(container=container):
                        return container

                _TYPE_API_BINDINGS[module_name, path] = (
                    ref(sys.modules[module_name]),
                    owner,
                    cls,
                    type_key,
                )
            else:
                _TYPE_API_BINDINGS.pop((module_name, path), None)
            result = cls
        else:
            if not callable(existing):
                raise ValueError(
                    f"Type {type_key!r} conflicts with a non-callable Python attribute"
                )
            result = existing
    return result


def _expose_type(type_key):
    type_key = str(type_key)
    _EXPOSED_TYPES[type_key] = None
    for module_name, prefixes in tuple(_TYPE_API_DESTINATIONS.items()):
        for prefix in tuple(prefixes):
            _publish_type(type_key, prefix, module_name)


def _init_exposed_type_api(prefix, target_module_name):
    """Initialize APIs for explicitly exposed types and subscribe to future exposures.

    Later prefixes declared for a destination take precedence over earlier generated
    aliases. Handwritten bindings retain precedence, and re-exposing the same type
    refreshes its class. Type suffixes follow the script printer's constructor names.
    """
    prefixes = _TYPE_API_DESTINATIONS.setdefault(target_module_name, [])
    if prefix not in prefixes:
        prefixes.append(prefix)
    for type_key in tuple(_EXPOSED_TYPES):
        _publish_type(type_key, prefix, target_module_name)
