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
"""Install and update API destinations for explicitly selected IR definitions.

Native bootstrap forwards completed-definition selections here. Destination
initializers retain module/prefix subscriptions and replay earlier selections.
Existing callable bindings win; publication only fills missing APIs.
"""

import keyword
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import tvm_ffi

if TYPE_CHECKING:
    from .op import Op
    from .type import Type

_OpAPI = Callable[..., Any]
_OpWrapper = Callable[[_OpAPI], _OpAPI]


class _OpBinding(NamedTuple):
    """One generated callable and the policy that adapted it for a destination."""

    # None constructs raw calls; a wrapper adds destination-specific behavior.
    wrapper: _OpWrapper | None
    # Reused by destinations with this exact wrapper identity.
    api: _OpAPI


class _OpDestination(NamedTuple):
    """Resolved target namespace and construction policy for an op API surface."""

    # Resolved at registration; re-registration replaces a discarded module.
    namespace: ModuleType
    # None publishes raw constructors; a wrapper adapts them for this surface.
    wrapper: _OpWrapper | None


# Explicit op selections and their generated callables, shared by wrapper identity.
_EXPOSED_OP_API_MAP: dict[str, list[_OpBinding]] = {}
# Target module/prefix pairs mapped to their resolved namespace and op wrapper.
_EXPOSED_OP_API_MODULE_MAP: dict[tuple[str, str], _OpDestination] = {}
# Ordered set of explicitly selected type keys; the None values are unused.
# Keep keys after publication to replay to destinations initialized later.
_EXPOSED_TYPES: dict[str, None] = {}
# Target module/prefix pairs mapped to their resolved type API namespace.
_EXPOSED_TYPE_API_MODULE_MAP: dict[tuple[str, str], ModuleType] = {}


def _api_attribute(container: object, name: str) -> Any:
    """Read a binding without triggering lazy module imports.

    Explicit namespace objects keep their descriptor/dynamic lookup behavior.
    Errors from those lookups propagate to the publisher.

    Parameters
    ----------
    container : object
        Module or explicit namespace object holding the binding.
    name : str
        Attribute name to look up in the namespace.

    Returns
    -------
    value : Any
        Existing binding, or None when the attribute is absent.
    """
    return (
        vars(container).get(name)
        if isinstance(container, ModuleType)
        else getattr(container, name, None)
    )


def _relative_name(name: str, prefix: str) -> str | None:
    """Return the portion of a name within a destination prefix.

    Parameters
    ----------
    name : str
        Fully qualified op name or type key.
    prefix : str
        Namespace prefix matched on a complete dotted component boundary.
        An empty prefix matches every name.

    Returns
    -------
    relative_name : str or None
        Name after the prefix and separator, the full name for an empty prefix,
        or None when the name does not begin with the prefix and a dot.
    """
    if not prefix:
        return name
    return name[len(prefix) + 1 :] if name.startswith(prefix + ".") else None


def _get_or_create_namespace(root: object, path: str) -> object:
    """Resolve a dotted path under root, creating missing namespace objects.

    Existing namespace objects retain their attribute semantics; missing
    components become SimpleNamespace bindings. This helper stores no state
    and does not import modules. Attribute lookup and binding errors propagate.

    Parameters
    ----------
    root : object
        Destination module or namespace object from which to resolve the path.
    path : str
        Dotted namespace path relative to root; an empty path selects root.

    Returns
    -------
    namespace : object
        Existing or newly created namespace at the requested path.
    """
    namespace = root
    for name in path.split(".") if path else ():
        child = _api_attribute(namespace, name)
        if child is None:
            child = SimpleNamespace()
            _set_api_and_export(namespace, name, child)
        namespace = child
    return namespace


def _set_api_and_export(container: object, name: str, value: Any) -> Any:
    """Assign an attribute and add its name to an existing __all__ list.

    Assignment unconditionally overwrites an existing attribute; callers own
    collision validation and first-binding precedence. Append the name only when
    __all__ is already a list and does not contain it. Never create __all__.

    Parameters
    ----------
    container : object
        Writable namespace with an instance dictionary to receive the binding.
    name : str
        Attribute name to assign and export.
    value : Any
        Object to publish under the attribute name.

    Returns
    -------
    value : Any
        The exact supplied object, unchanged.

    Raises
    ------
    AttributeError
        If the namespace rejects the attribute assignment.
    TypeError
        If the namespace has no instance dictionary.
    """
    setattr(container, name, value)
    exports = vars(container).get("__all__")
    if isinstance(exports, list) and name not in exports:
        exports.append(name)
    return value


def _make_op_api(op: "Op", module_name: str) -> _OpAPI:
    """Create a constructor governed by the completed Op's signature and attributes.

    The callable binds positional/named operands and reflected attrs, then builds a
    raw Call. Region ops instead enter their existing builder path on invocation.
    Argument and constructor errors propagate; constructing the callable emits no IR.

    Parameters
    ----------
    op : Op
        Completed native operator whose signature, attributes and docs define the API.
    module_name : str
        Destination name assigned to the generated callable's __module__.

    Returns
    -------
    api : _OpAPI
        Unwrapped callable that constructs a raw Call or enters a region builder.
    """
    from .attrs import make_node  # pylint: disable=import-outside-toplevel
    from .expr import Call  # pylint: disable=import-outside-toplevel
    from .location import UNKNOWN_LOC, Location  # pylint: disable=import-outside-toplevel

    def call(
        *args,
        attrs=None,
        ty_args=None,
        loc: Location = UNKNOWN_LOC,
        ty=None,
        **kwargs,
    ):
        # Bind named operands using the registered signature, without inventing
        # defaults or interpreting arbitrary Python wrapper signatures.
        is_region = op.get_attr("FRegionGetBodyParams") is not None
        body_params = kwargs.pop("body_params", None) if is_region else None
        operands = list(args)
        for index, info in enumerate(op.args_info):
            if index < len(args):
                if info.name in kwargs:
                    raise TypeError(f"{op.name}: multiple values for {info.name!r}")
            elif info.name in kwargs:
                operands.append(kwargs.pop(info.name))
            else:
                raise TypeError(f"{op.name}: missing operand {info.name!r}")
        if kwargs or (attrs is None and op.attrs_type_key):
            if not op.attrs_type_key:
                raise TypeError(f"{op.name}: unexpected keyword operands {tuple(kwargs)}")
            if attrs is not None:
                raise TypeError(f"{op.name}: cannot mix attrs with attribute keywords")
            attrs = make_node(op.attrs_type_key, **kwargs)
        if is_region:
            from tvm_tirx.script.ir_builder.stmt import region

            if (
                ty is not None
                or ty_args is not None
                or not isinstance(loc, Location)
                or not loc.same_as(UNKNOWN_LOC)
            ):
                raise TypeError(
                    f"{op.name}: region builders do not accept ty, ty_args or an explicit loc"
                )
            return region(op, operands, body_params=body_params, attrs=attrs)
        return Call(
            op,
            operands,
            attrs=attrs,
            ty_args=ty_args,
            loc=loc,
            ty=ty,
        )

    call.__name__ = op.name.rsplit(".", 1)[-1]
    call.__module__ = module_name
    call.__doc__ = op.doc or f"Construct a call to {op.name}."
    return call


def _publish_op(
    name: str, prefix: str, namespace: ModuleType, wrapper: _OpWrapper | None
) -> _OpAPI | None:
    """Publish one selected op to a matching destination.

    Existing callables win. A new API is created once per wrapper identity,
    then reused without double wrapping. Wrapper and binding errors propagate.

    Parameters
    ----------
    name : str
        Fully qualified completed op name already retained in the selection map.
    prefix : str
        Component prefix to remove from the name; empty matches the full name.
    namespace : ModuleType
        Resolved destination root beneath which the remaining name is published.
    wrapper : _OpWrapper or None
        Adapter from a new raw callable to its destination callable; None keeps
        the raw constructor. Existing bindings are not adapted again.

    Returns
    -------
    api : _OpAPI or None
        Existing or newly installed callable, or None if the prefix does not match.

    Raises
    ------
    ValueError
        If the destination name is occupied by a non-callable binding.
    KeyError
        If a missing destination binding needs an op absent from the selection map.
    """
    path = _relative_name(name, prefix)
    if path is None:
        return None
    namespace_path, _, leaf = path.rpartition(".")
    container = _get_or_create_namespace(namespace, namespace_path)
    existing = _api_attribute(container, leaf)
    if existing is not None:
        if not callable(existing):
            raise ValueError(f"Op {name!r} conflicts with a non-callable Python attribute")
        return existing
    apis = _EXPOSED_OP_API_MAP[name]
    call = next((binding.api for binding in apis if binding.wrapper is wrapper), None)
    if call is None:
        from .op import Op  # pylint: disable=import-outside-toplevel

        call = _make_op_api(Op.get(name), namespace.__name__)
        if wrapper is not None:
            call = wrapper(call)
        apis.append(_OpBinding(wrapper, call))
    return _set_api_and_export(container, leaf, call)


def _expose_op(name: str) -> None:
    """Record a completed op selection and synchronously update its destinations.

    Publication errors propagate. The retained selection lets destination
    initialization retry publication without discovering unselected registry ops.

    Parameters
    ----------
    name : str
        Fully qualified native op name whose definition is ready for publication.

    Raises
    ------
    ValueError
        If a matching destination has a non-callable binding at the op name.
    """
    name = str(name)
    _EXPOSED_OP_API_MAP.setdefault(name, [])
    for (_, prefix), destination in _EXPOSED_OP_API_MODULE_MAP.items():
        _publish_op(name, prefix, destination.namespace, destination.wrapper)


def _init_exposed_op_api(
    prefix: str, target_module_name: str, *, wrapper: _OpWrapper | None = None
) -> None:
    """Initialize APIs for explicitly exposed ops and subscribe to future exposures.

    Register the module/prefix pair, then replay previous selections. Repeated
    initialization preserves existing exports and shares same-policy callables.
    Re-registration refreshes the resolved namespace. Publication errors propagate
    and leave the destination registered for retry.

    Parameters
    ----------
    prefix : str
        Component prefix selecting op names; an empty prefix selects all names.
        The matched prefix is removed from names published in this destination.
    target_module_name : str
        Name of the already loaded destination module in sys.modules.
    wrapper : _OpWrapper or None, optional
        Adapter for newly generated callables. None publishes raw constructors;
        the same adapter identity shares callables across destinations.

    Raises
    ------
    KeyError
        If the target module is not loaded.
    ValueError
        If replay encounters a non-callable binding at an exposed op name.
    """
    namespace = sys.modules[target_module_name]
    _EXPOSED_OP_API_MODULE_MAP[target_module_name, prefix] = _OpDestination(namespace, wrapper)
    for name in _EXPOSED_OP_API_MAP:
        _publish_op(name, prefix, namespace, wrapper)


def _exposed_type_class(type_key: str) -> "type[Type]":
    """Return the registered Python class, providing a reflected constructor if needed.

    Missing parent wrappers are constructed without selecting them for exposure.
    Registry/reflection errors propagate to the requesting destination.

    Parameters
    ----------
    type_key : str
        Fully qualified native type key with completed reflection information.

    Returns
    -------
    cls : type[Type]
        Registered Python type wrapper with a constructor, creating one if needed.
    """
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


def _publish_type(type_key: str, prefix: str, namespace: ModuleType) -> Callable[..., Any] | None:
    """Publish a selected type's printer aliases to a matching destination.

    Type suffixes are shortened, retaining a valid suffixed name for keywords.
    Existing callables always win, including aliases of an older class for the
    same type key. Reflection and binding errors propagate.

    Parameters
    ----------
    type_key : str
        Fully qualified selected native type key with completed reflection.
    prefix : str
        Component prefix to remove from the key; empty matches the full key.
    namespace : ModuleType
        Resolved destination root beneath which the type aliases are published.

    Returns
    -------
    binding : Callable[..., Any] or None
        Existing or newly installed callable for the last alias, or None if the
        prefix does not match. A keyword alias returns its valid suffixed binding.

    Raises
    ------
    ValueError
        If a type alias is occupied by a non-callable binding.
    """
    path = _relative_name(type_key, prefix)
    if path is None:
        return None
    namespace_path, _, name = path.rpartition(".")
    container = _get_or_create_namespace(namespace, namespace_path)
    names = [name]
    if len(name) > 4 and name.endswith("Type"):
        short = name[:-4]
        names = [short, name] if keyword.iskeyword(short) else [short]
    result = None
    for name in names:
        existing = _api_attribute(container, name)
        if existing is not None:
            if not callable(existing):
                raise ValueError(
                    f"Type {type_key!r} conflicts with a non-callable Python attribute"
                )
            result = existing
        else:
            result = _set_api_and_export(container, name, _exposed_type_class(type_key))
    return result


def _expose_type(type_key: str) -> None:
    """Retain a completed type selection and update matching destination aliases.

    Errors propagate while the selection remains available for later replay.

    Parameters
    ----------
    type_key : str
        Fully qualified native type key whose reflection and constructor are ready.

    Raises
    ------
    ValueError
        If a matching destination has a non-callable binding at a type alias.
    """
    type_key = str(type_key)
    _EXPOSED_TYPES[type_key] = None
    for (_, prefix), namespace in _EXPOSED_TYPE_API_MODULE_MAP.items():
        _publish_type(type_key, prefix, namespace)


def _init_exposed_type_api(prefix: str, target_module_name: str) -> None:
    """Initialize APIs for explicitly exposed types and subscribe to future exposures.

    Register the module/prefix pair, then replay previous selections. Existing
    bindings remain unchanged, so the first published callable wins.
    Re-registration refreshes the resolved namespace. Errors propagate and leave
    the destination registered for a later retry.

    Parameters
    ----------
    prefix : str
        Component prefix selecting type keys; an empty prefix selects all keys.
        The matched prefix is removed before deriving destination aliases.
    target_module_name : str
        Name of the already loaded destination module in sys.modules.

    Raises
    ------
    KeyError
        If the target module is not loaded.
    ValueError
        If replay encounters a non-callable binding at a type alias.
    """
    namespace = sys.modules[target_module_name]
    _EXPOSED_TYPE_API_MODULE_MAP[target_module_name, prefix] = namespace
    for type_key in _EXPOSED_TYPES:
        _publish_type(type_key, prefix, namespace)


def _install_exposure_callbacks() -> None:
    """Connect stable native receivers and drain selections queued before IR loaded.

    Call after the IR wrappers are defined. Each queued request is removed only
    after successful publication; an error leaves it available for import retry.
    Native global receivers themselves are never replaced.
    """
    from .. import base  # pylint: disable=import-outside-toplevel

    base._EXPOSE_OP_CALLBACK = _expose_op
    base._EXPOSE_TYPE_CALLBACK = _expose_type
    while base._PENDING_EXPOSURES:
        kind, name = base._PENDING_EXPOSURES[0]
        (_expose_op if kind == "op" else _expose_type)(name)
        base._PENDING_EXPOSURES.popleft()
