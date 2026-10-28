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
"""Reflection-driven text trees for inspecting IR and extension objects."""

from collections.abc import Mapping
from typing import Any

import tvm_ffi

_SCALARS = (bool, int, float, complex, str, bytes)
_SEQUENCES = (list, tuple, tvm_ffi.Array, tvm_ffi.List)
_MAPPINGS = (Mapping, tvm_ffi.Map, tvm_ffi.Dict)
_FFI_VALUES = (tvm_ffi.Object, tvm_ffi.Array, tvm_ffi.List, tvm_ffi.Map, tvm_ffi.Dict)


def _scalar(value):
    if isinstance(value, tvm_ffi.dtype):
        return str(value)
    if isinstance(value, str):
        return repr(str(value))
    if value is None or type(value) in _SCALARS:
        return repr(value)
    return None


def _identity(value):
    if isinstance(value, _FFI_VALUES):
        return ("ffi", value.__chandle__())
    return ("python", id(value))


def _fields(value):
    # dataclasses.fields excludes legacy register_object classes. Keep this
    # adapter local until FFI exposes field enumeration for both class kinds.
    info = type(value).__tvm_ffi_type_info__
    chain = []
    while info is not None:
        chain.append(info)
        info = info.parent_type_info
    for info in reversed(chain):
        yield from info.fields


def _is_default(field, value):
    if not field.c_has_default:
        return False
    default = field.c_default
    if value is None or default is None:
        return value is default
    if type(value) is type(default) and type(value) in _SCALARS:
        return value == default
    if isinstance(value, _FFI_VALUES) and isinstance(default, _FFI_VALUES):
        return value.same_as(default)
    return False


class _TreeFormatter:
    def __init__(self, max_depth, hide_fields, show_defaults, inline_width):
        self.max_depth = max_depth
        self.hide_fields = frozenset(hide_fields)
        self.show_defaults = show_defaults
        self.inline_width = inline_width

    def _parts(self, value):
        """Return a node name, layout kind, and ordered labelled children."""
        if isinstance(value, _SEQUENCES):
            return type(value).__name__, "sequence", list(enumerate(value))
        if isinstance(value, _MAPPINGS):
            items = list(value.items())
            if all(_scalar(key) is not None for key, _ in items):
                items.sort(key=lambda item: (type(item[0]).__name__, _scalar(item[0])))
                return (
                    type(value).__name__,
                    "mapping",
                    [(_scalar(key), child) for key, child in items],
                )
            # Object keys have structure too: show each key/value pair rather
            # than invoking a possibly dialect-specific repr on the key.
            return type(value).__name__, "sequence", list(enumerate(items))
        if isinstance(value, tvm_ffi.Object):
            name = type(value).__tvm_ffi_type_info__.type_key.rsplit(".", 1)[-1]
            entries = []
            for field in _fields(value):
                if field.name in self.hide_fields:
                    continue
                child = getattr(value, field.name)
                if not self.show_defaults and _is_default(field, child):
                    continue
                entries.append((field.name, child))
            return name, "object", entries
        return type(value).__name__, "opaque", []

    def _limited(self, depth):
        return self.max_depth is not None and depth >= self.max_depth

    def _compact(self, value, depth, ancestors, budget, remaining=4):
        scalar = _scalar(value)
        if scalar is not None:
            return scalar if len(scalar) <= budget else None
        identity = _identity(value)
        name, kind, entries = self._parts(value)
        if identity in ancestors:
            return "<cycle>"
        if kind == "opaque":
            return f"<{name}>"
        if self._limited(depth) and entries:
            return f"{name}(...)"
        if remaining == 0:
            return None
        opening, closing = {"sequence": ("[", "]"), "mapping": ("{", "}")}.get(
            kind, (f"{name}(", ")")
        )
        text = opening
        ancestors = ancestors | {identity}
        for index, (label, child) in enumerate(entries):
            prefix = "" if index == 0 else ", "
            if kind == "object":
                prefix += f"{label}="
            elif kind == "mapping":
                prefix += f"{label}: "
            available = budget - len(text) - len(prefix) - len(closing)
            if available < 0:
                return None
            compact = self._compact(child, depth + 1, ancestors, available, remaining - 1)
            if compact is None or len(compact) > available:
                return None
            text += prefix + compact
        return text + closing if len(text) + len(closing) <= budget else None

    def lines(self, value, depth=0, ancestors=frozenset()):
        scalar = _scalar(value)
        if scalar is not None:
            return [scalar]
        identity = _identity(value)
        if identity in ancestors:
            return ["<cycle>"]
        name, kind, entries = self._parts(value)
        if kind == "opaque":
            return [f"<{name}>"]
        if self._limited(depth) and entries:
            return [f"{name}(...)"]
        if self.inline_width:
            compact = self._compact(value, depth, ancestors, self.inline_width)
            if compact is not None and len(compact) <= self.inline_width:
                return [compact]
        if kind == "object":
            scalars = [(label, _scalar(child)) for label, child in entries]
            header = [f"{label}={text}" for label, text in scalars if text is not None]
            name += "(" + ", ".join(header) + ")" if header else ""
            entries = [(label, child) for label, child in entries if _scalar(child) is None]
        lines = [name]
        ancestors = ancestors | {identity}
        for index, (label, child) in enumerate(entries):
            last = index == len(entries) - 1
            branch, padding = ("└── ", "    ") if last else ("├── ", "│   ")
            label = f"[{label}]" if kind == "sequence" else str(label)
            child_lines = self.lines(child, depth + 1, ancestors)
            lines.append(f"{branch}{label}: {child_lines[0]}")
            lines.extend(padding + line for line in child_lines[1:])
        return lines


def format_tree(
    value: Any,
    *,
    max_depth: int | None = None,
    hide_fields: tuple[str, ...] = ("loc",),
    show_defaults: bool = False,
    inline_width: int = 80,
) -> str:
    """Render reflected fields as a compact text tree.

    TIRx nodes and downstream FFI objects use the same field-based layout.
    Field names label edges; primitive fields appear in node headers. Small
    children are printed inline, while larger children use tree branches.
    Sequence wrappers, including single-statement SeqStmt nodes, are retained.
    This inspection format is not a serialization format.

    Parameters
    ----------
    value : Any
        IR or FFI object, container, or scalar to inspect.
    max_depth : int, optional
        Maximum number of object/container edges from the root (depth zero).
        Deeper children are represented by ``...``. None means no depth limit.
    hide_fields : tuple[str, ...]
        Reflected field names to omit on every object. Pass ``()`` to include
        source locations. Container keys are not filtered.
    show_defaults : bool
        Include fields with reflected defaults. By default only matching
        primitive defaults or identical object defaults are omitted; absent
        metadata never causes zero, False, or empty values to be hidden.
    inline_width : int
        Maximum width of a compact child representation, excluding indentation
        and its edge label. Zero disables inline compaction. This is not a
        maximum line length; scalar values are never truncated.

    Returns
    -------
    text : str
        A tree without a trailing newline. Ancestor cycles are shown as
        ``<cycle>``; shared children can appear at multiple positions. Maps
        with scalar keys are sorted by key type and representation. Other map
        entries are shown as key/value pairs in their iteration order.

    Examples
    --------
    .. code-block:: python

        from tvm_tirx import ir

        print(ir.format_tree(ir.Var("i", "int32")))
        # Var(ty=PrimType(dtype=int32), name='i')
    """
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("max_depth must be a nonnegative integer or None")
    if type(inline_width) is not int or inline_width < 0:
        raise ValueError("inline_width must be a nonnegative integer")
    if isinstance(hide_fields, str) or any(not isinstance(name, str) for name in hide_fields):
        raise TypeError("hide_fields must be a sequence of field names")
    formatter = _TreeFormatter(max_depth, hide_fields, show_defaults, inline_width)
    return "\n".join(formatter.lines(value))
