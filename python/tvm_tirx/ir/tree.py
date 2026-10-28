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

from .expr import Var

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
    def __init__(
        self, max_depth, hide_fields, show_defaults, inline_width, fold_fields, expand_fields
    ):
        self.max_depth = max_depth
        self.hide_fields = frozenset(hide_fields)
        self.show_defaults = show_defaults
        self.inline_width = inline_width
        self.layouts = dict.fromkeys(fold_fields, "fold")
        self.layouts.update(dict.fromkeys(expand_fields, "expand"))

    def _reference(self, value, depth, region):
        if isinstance(value, Var) and depth and region is None:
            return value.name if value.name.isidentifier() else repr(value.name)
        return None

    def _parts(self, value, region=None):
        """Return a node name, layout kind, and ordered labelled children."""
        if isinstance(value, _SEQUENCES):
            return (
                type(value).__name__,
                "sequence",
                [(i, child, region, "auto") for i, child in enumerate(value)],
            )
        if isinstance(value, _MAPPINGS):
            items = list(value.items())
            if all(_scalar(key) is not None for key, _ in items):
                items.sort(key=lambda item: (type(item[0]).__name__, _scalar(item[0])))
                return (
                    type(value).__name__,
                    "mapping",
                    [(_scalar(key), child, region, "auto") for key, child in items],
                )
            # Object keys have structure too: show each key/value pair rather
            # than invoking a possibly dialect-specific repr on the key.
            return (
                type(value).__name__,
                "sequence",
                [(i, pair, region, "auto") for i, pair in enumerate(items)],
            )
        if isinstance(value, tvm_ffi.Object):
            name = type(value).__tvm_ffi_type_info__.type_key.rsplit(".", 1)[-1]
            entries = []
            for field in _fields(value):
                if field.name in self.hide_fields:
                    continue
                child = getattr(value, field.name)
                if (
                    not self.show_defaults
                    and field.name not in self.layouts
                    and _is_default(field, child)
                ):
                    continue
                role = field.c_structural_eq
                if role not in ("def-simple", "def-pattern"):
                    # A definition pattern can contain nested variable objects.
                    # Stop inherited context at a variable declaration; its
                    # type can still carry definition flags on its own fields.
                    role = (
                        "def-pattern"
                        if region == "def-pattern"
                        and not isinstance(value, Var)
                        and role != "ignore"
                        else None
                    )
                entries.append((field.name, child, role, self.layouts.get(field.name, "auto")))
            return name, "object", entries
        return type(value).__name__, "opaque", []

    def _limited(self, depth):
        return self.max_depth is not None and depth >= self.max_depth

    def _folded(self, value):
        if isinstance(value, tvm_ffi.Object):
            name = type(value).__tvm_ffi_type_info__.type_key.rsplit(".", 1)[-1]
        else:
            name = type(value).__name__
        return f"{name}(...)"

    def _compact(self, value, depth, ancestors, budget, region=None, layout="auto", remaining=4):
        if layout == "expand":
            return None
        scalar = self._reference(value, depth, region) or _scalar(value)
        if scalar is not None:
            return scalar if len(scalar) <= budget else None
        identity = _identity(value)
        name, kind, entries = self._parts(value, region)
        if identity in ancestors:
            return "<cycle>"
        if layout == "fold":
            return self._folded(value)
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
        for index, (label, child, role, child_layout) in enumerate(entries):
            prefix = "" if index == 0 else ", "
            if kind == "object":
                prefix += f"{label}="
            elif kind == "mapping":
                prefix += f"{label}: "
            available = budget - len(text) - len(prefix) - len(closing)
            if available < 0:
                return None
            compact = self._compact(
                child, depth + 1, ancestors, available, role, child_layout, remaining - 1
            )
            if compact is None or len(compact) > available:
                return None
            text += prefix + compact
        return text + closing if len(text) + len(closing) <= budget else None

    def _header_text(self, value, depth, ancestors, region, layout):
        if layout == "expand":
            return None
        text = self._reference(value, depth, region) or _scalar(value)
        if text is not None:
            return text
        if layout == "fold":
            return "<cycle>" if _identity(value) in ancestors else self._folded(value)
        if not self.inline_width:
            return None
        _, kind, entries = self._parts(value, region)
        if kind == "object":
            # Scalar-bearing leaf objects (e.g. constants or declarations) can
            # share the header. Structural children still get their own edges.
            if not any(_scalar(child) is not None for _, child, _, _ in entries):
                return None
        elif kind == "sequence":
            if any(
                self._reference(child, depth + 1, role) is None and _scalar(child) is None
                for _, child, role, _ in entries
            ):
                return None
        text = self._compact(value, depth, ancestors, self.inline_width, region)
        return text if text is not None and len(text) <= self.inline_width else None

    def _edge(self, label, value, depth, ancestors, region, layout):
        identity = _identity(value)
        if (
            layout == "auto"
            and isinstance(value, _SEQUENCES)
            and len(value)
            and not self._limited(depth)
            and identity not in ancestors
        ):
            for i, child in enumerate(value):
                yield from self._edge(
                    f"{label}[{i}]", child, depth + 1, ancestors | {identity}, region, "auto"
                )
        else:
            yield label, value, depth, ancestors, region, layout

    def lines(self, value, depth=0, ancestors=frozenset(), region=None, layout="auto"):
        scalar = _scalar(value)
        if layout != "expand":
            scalar = self._reference(value, depth, region) or scalar
        if scalar is not None:
            return [scalar]
        identity = _identity(value)
        if identity in ancestors:
            return ["<cycle>"]
        if layout == "fold":
            return [self._folded(value)]
        name, kind, entries = self._parts(value, region)
        if kind == "opaque":
            return [f"<{name}>"]
        if self._limited(depth) and entries:
            return [f"{name}(...)"]
        if self.inline_width and layout != "expand":
            compact = self._compact(value, depth, ancestors, self.inline_width, region)
            if compact is not None and len(compact) <= self.inline_width:
                return [compact]
        ancestors = ancestors | {identity}
        if kind == "object":
            header, children = [], []
            for label, child, role, child_layout in entries:
                text = self._header_text(child, depth + 1, ancestors, role, child_layout)
                if text is not None:
                    header.append(f"{label}={text}")
                else:
                    children.append((label, child, role, child_layout))
            name += "(" + ", ".join(header) + ")" if header else ""
            entries = children
        lines = [name]
        edges = []
        for label, child, role, child_layout in entries:
            label = f"[{label}]" if kind == "sequence" else str(label)
            edges.extend(self._edge(label, child, depth + 1, ancestors, role, child_layout))
        for index, (label, child, child_depth, child_ancestors, role, child_layout) in enumerate(
            edges
        ):
            last = index == len(edges) - 1
            branch, padding = ("└── ", "    ") if last else ("├── ", "│   ")
            child_lines = self.lines(child, child_depth, child_ancestors, role, child_layout)
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
    fold_fields: tuple[str, ...] = (),
    expand_fields: tuple[str, ...] = (),
) -> str:
    """Render reflected fields as a compact text tree.

    TIRx nodes and downstream FFI objects use the same field-based layout.
    Field names label edges; scalar and compact leaf fields appear in node
    headers. Expanded arrays fold into indexed edges such as ``seq[0]``.
    SeqStmt nodes themselves are retained. Variables use their names at use
    sites and their reflected fields at definition sites (and at the root).
    Definition sites follow FFI's ``def-simple`` / ``def-pattern`` field flags;
    custom structural visitor hooks are not interpreted by this formatter.
    This inspection format is not a serialization format.

    Parameters
    ----------
    value : Any
        IR or FFI object, container, or scalar to inspect.
    max_depth : int, optional
        Maximum number of object/container edges from the root (depth zero).
        Deeper children are represented by ``...``. Folded arrays still count
        toward depth. None means no depth limit.
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

    fold_fields : tuple[str, ...]
        Exact reflected field names to summarize as ``Type(...)``. Primitive
        values and variable references keep their scalar/name representation.
        This applies at every occurrence, including extension fields. Selected
        fields are included even when they match a reflected default.
    expand_fields : tuple[str, ...]
        Exact reflected field names to put on separate detail edges, bypassing
        inline compaction and variable-name shorthand for that field. Selected
        arrays retain their container level. Descendants follow normal rules;
        expansion does not open a folded ancestor. Selected fields are shown
        even when they match a reflected default.
        A name cannot appear in both fold_fields and expand_fields. Hidden
        fields remain hidden, and max_depth still limits expansion. These
        options select field names, not paths, array indices, or mapping keys.

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
    options = []
    for key, names in (
        ("hide_fields", hide_fields),
        ("fold_fields", fold_fields),
        ("expand_fields", expand_fields),
    ):
        if isinstance(names, str):
            raise TypeError(f"{key} must be a sequence of field names")
        names = tuple(names)
        if any(not isinstance(name, str) for name in names):
            raise TypeError(f"{key} must be a sequence of field names")
        options.append(names)
    hide_fields, fold_fields, expand_fields = options
    if set(fold_fields) & set(expand_fields):
        raise ValueError("fold_fields and expand_fields must not overlap")
    formatter = _TreeFormatter(
        max_depth, hide_fields, show_defaults, inline_width, fold_fields, expand_fields
    )
    return "\n".join(formatter.lines(value))
