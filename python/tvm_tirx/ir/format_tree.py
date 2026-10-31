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
"""Text trees for inspecting IR and extension objects."""

import enum
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import tvm_ffi
from tvm_ffi.dataclasses import Enum as _FFIEnum

from .expr import Expr, Var
from .function import BaseFunc
from .op import Op
from .stmt import For, ForKind
from .type import PrimType

FieldSelector = str | tuple[type, str]

# Keep the public defaults in the same selector form as the keyword options.
# Normalize them privately before applying exact-key caller overrides.
DEFAULT_FOLD: tuple[FieldSelector, ...] = (
    (Var, "ty"),
    "source",
    "dest",
    "indices",
    "op",
    (For, "begin"),
    (For, "end"),
    (For, "step"),
)
"""Fields inlined with their parent by default."""

DEFAULT_EXPAND: tuple[FieldSelector, ...] = ()
"""Fields explicitly expanded by default; other objects expand automatically."""

DEFAULT_HIDE: tuple[FieldSelector, ...] = (
    "loc",
    (BaseFunc, "attrs"),
    (BaseFunc, "ret_type"),
    (Expr, "ty"),
)
"""Fields omitted by default."""

DEFAULT_FOLD_FORMATTERS = {
    PrimType: lambda node: str(node.dtype),
    Op: lambda node: node.name,
    Var: lambda node: node.name if node.name.isidentifier() else repr(node.name),
}
"""Standard folded representations, keyed by class."""

_SCALARS = (bool, int, float, complex, str, bytes)
_SEQUENCES = (list, tuple, tvm_ffi.Array, tvm_ffi.List)
_MAPPINGS = (Mapping, tvm_ffi.Map, tvm_ffi.Dict)
_FFI_VALUES = (tvm_ffi.Object, tvm_ffi.Array, tvm_ffi.List, tvm_ffi.Map, tvm_ffi.Dict)


def _scalar(value):
    if isinstance(value, enum.Enum | _FFIEnum):
        return f"{type(value).__name__}({value.name})"
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
    # adapter local until FFI exposes enumeration for both class kinds.
    info = type(value).__tvm_ffi_type_info__
    chain = []
    while info is not None:
        chain.append(info)
        info = info.parent_type_info
    for info in reversed(chain):
        yield from info.fields


def _selector(selector):
    if isinstance(selector, str) and selector:
        return tvm_ffi.Object, selector
    if (
        isinstance(selector, tuple)
        and len(selector) == 2
        and isinstance(selector[0], type)
        and issubclass(selector[0], tvm_ffi.Object)
        and isinstance(selector[1], str)
        and selector[1]
    ):
        return selector
    raise TypeError("selectors must be nonempty field names or (FFI class, field name) pairs")


class _TreeFormatter:
    def __init__(self, max_depth, show_defaults, fold, expand, hide, fold_formatters):
        self.max_depth = max_depth
        self.show_defaults = show_defaults
        self.rules = {
            _selector(selector): action
            for action, selectors in (
                ("hide", DEFAULT_HIDE),
                ("fold", DEFAULT_FOLD),
                ("expand", DEFAULT_EXPAND),
            )
            for selector in selectors
        }
        self.explicit = set()
        # Normalize first: a bare name and its Object pair are the same key.
        # Applying collections in this order gives expand > fold > hide.
        for action, selectors in (("hide", hide), ("fold", fold), ("expand", expand)):
            if selectors is None:
                continue
            if isinstance(selectors, (str, bytes)):
                raise TypeError(f"{action} must be a collection of field selectors")
            for selector in selectors:
                key = _selector(selector)
                self.rules[key] = action
                self.explicit.add(key)
        self.fold_formatters = dict(DEFAULT_FOLD_FORMATTERS)
        if fold_formatters is not None:
            if not isinstance(fold_formatters, Mapping):
                raise TypeError("fold_formatters must be a mapping of FFI classes to callables")
            for cls, callback in fold_formatters.items():
                if not isinstance(cls, type) or not issubclass(cls, tvm_ffi.Object):
                    raise TypeError("fold_formatters keys must be FFI classes")
                if not callable(callback):
                    raise TypeError("fold_formatters values must be callable")
                self.fold_formatters[cls] = callback

    def _rule(self, value, field):
        # Python MRO also includes unregistered IR family bases and downstream
        # Python classes; native type-info ancestry alone would miss those.
        for cls in type(value).__mro__:
            key = (cls, field)
            if key in self.rules:
                return self.rules[key], key in self.explicit
        return "auto", False

    def _parts(self, value, region):
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
            # Object keys have structure too; never invoke a dialect's repr.
            return (
                type(value).__name__,
                "sequence",
                [(i, pair, region, "auto") for i, pair in enumerate(items)],
            )
        if isinstance(value, tvm_ffi.Object):
            name = type(value).__tvm_ffi_type_info__.type_key.rsplit(".", 1)[-1]
            if isinstance(value, Var):
                var_name = value.name if value.name.isidentifier() else repr(value.name)
                name = f"{var_name} = {name}"
            entries = []
            for field in _fields(value):
                if isinstance(value, Var) and field.name == "name":
                    continue  # The declaration name is already to the left of '='.
                action, explicit = self._rule(value, field.name)
                if action == "hide":
                    continue
                child = getattr(value, field.name)
                if (
                    not self.show_defaults
                    and not explicit
                    and field.c_has_default
                    and tvm_ffi.structural_equal(child, field.c_default, map_free_vars=False)
                ):
                    continue
                role = field.c_structural_eq
                if role not in ("def-simple", "def-pattern"):
                    # Pattern roles propagate into types/containers, but stop
                    # at Var declarations. Their own type fields set new roles.
                    role = (
                        "def-pattern"
                        if region == "def-pattern"
                        and not isinstance(value, Var)
                        and role != "ignore"
                        else None
                    )
                if isinstance(value, For) and field.name == "kind" and child in ForKind:
                    child = ForKind(child)  # Stored as an int; show the enum name.
                entries.append((field.name, child, role, action))
            return name, "object", entries
        return type(value).__name__, "opaque", []

    def render(
        self,
        value,
        depth=0,
        ancestors=frozenset(),
        region=None,
        folded=False,
        expand_reference=False,
    ):
        scalar = _scalar(value)
        if scalar is not None:
            return [scalar]
        if (
            isinstance(value, Var)
            and depth
            and region is None
            and not expand_reference
            and not folded
        ):
            return [value.name if value.name.isidentifier() else repr(value.name)]
        identity = _identity(value)
        if identity in ancestors:
            return ["<cycle>"]
        name, kind, entries = self._parts(value, region)
        if kind == "opaque":
            return [f"<{name}>"]
        if self.max_depth is not None and depth >= self.max_depth and entries:
            return [f"{name}(...)"]
        if folded:
            for cls in type(value).__mro__:
                if cls in self.fold_formatters:
                    text = self.fold_formatters[cls](value)
                    if not isinstance(text, str) or "\n" in text or "\r" in text:
                        raise TypeError("fold_formatters callbacks must return single-line text")
                    return [text]
        ancestors = ancestors | {identity}
        if folded or not entries:
            opening, closing = {"sequence": ("[", "]"), "mapping": ("{", "}")}.get(
                kind, (f"{name}(", ")")
            )
            parts = []
            for label, child, role, _ in entries:
                # An enclosing fold is sticky: descendants may hide, but their
                # expansion rules cannot reopen this inline subtree.
                text = self.render(child, depth + 1, ancestors, role, folded=True)[0]
                if kind == "object":
                    text = f"{label}={text}"
                elif kind == "mapping":
                    text = f"{label}: {text}"
                parts.append(text)
            return [opening + ", ".join(parts) + closing]
        header, edges = [], []
        for label, child, role, action in entries:
            if kind == "object" and (
                action == "fold" or (action == "auto" and _scalar(child) is not None)
            ):
                text = self.render(child, depth + 1, ancestors, role, folded=True)[0]
                header.append(f"{label}={text}")
            else:
                label = f"[{label}]" if kind == "sequence" else str(label)
                # Only explicit expansion opens a Var use into a declaration;
                # the normal structural fallback keeps references name-only.
                edges.extend(
                    self._edges(label, child, depth + 1, ancestors, role, action == "expand")
                )
        lines = [name + ("(" + ", ".join(header) + ")" if header else "")]
        for index, (label, child_lines) in enumerate(edges):
            branch, padding = ("└── ", "    ") if index == len(edges) - 1 else ("├── ", "│   ")
            lines.append(f"{branch}{label}: {child_lines[0]}")
            lines.extend(padding + line for line in child_lines[1:])
        return lines

    def _edges(self, label, value, depth, ancestors, region, expand_reference):
        identity = _identity(value)
        if (
            isinstance(value, _SEQUENCES)
            and len(value)
            and (self.max_depth is None or depth < self.max_depth)
            and identity not in ancestors
        ):
            for index, child in enumerate(value):
                yield from self._edges(
                    f"{label}[{index}]",
                    child,
                    depth + 1,
                    ancestors | {identity},
                    region,
                    expand_reference,
                )
        else:
            yield (
                label,
                self.render(value, depth, ancestors, region, expand_reference=expand_reference),
            )


def format_tree(
    value: Any,
    *,
    fold: Iterable[FieldSelector] | None = None,
    expand: Iterable[FieldSelector] | None = None,
    hide: Iterable[FieldSelector] | None = None,
    fold_formatters: Mapping[type, Callable[[Any], str]] | None = None,
    max_depth: int | None = None,
    show_defaults: bool = False,
) -> str:
    """Display an object's fields as a text tree.

    Objects expand into labelled edges, arrays into indexed edges such as
    ``seq[0]``, and scalar fields stay in headers. SeqStmt wrappers remain.
    Expanded variable definitions display ``name = Var(...)`` at each
    occurrence, even when several definitions refer to the same variable.
    Variable uses and folded variables display names. Explicit expansion
    reveals the details of a variable use.

    Field selectors are bare names that match every object, or ``(class, name)``
    pairs that match a class and its subclasses. More-specific classes take
    precedence. Caller selections replace the default action for the exact
    same selector; if it appears in multiple caller options, expand wins over
    fold, which wins over hide. Explicit fold or expand selections also expose
    default-valued fields. Container keys are not field selectors. None and
    empty collections retain the standard defaults. Formatting never changes the supplied
    collections or the shared defaults.

    Parameters
    ----------
    value : Any
        IR object, extension object, container, or scalar to inspect.
    fold : iterable of str or (class, str), optional
        Fields to inline with their parent. Standard fold selections are
        listed in :data:`~tvm_tirx.ir.DEFAULT_FOLD`. By default, variable types,
        source/destination, indices, operation references, and loop bounds
        are folded. Folding
        includes the visible subtree; descendants stay folded or hidden even
        if an inner field requests expansion.
    expand : iterable of str or (class, str), optional
        Fields to show on separate tree edges. Standard expansion selections
        are listed in :data:`~tvm_tirx.ir.DEFAULT_EXPAND`. Objects expand automatically
        unless another rule folds or hides them. Expanding a variable use shows its declaration.
    hide : iterable of str or (class, str), optional
        Fields to omit. Standard hide selections are listed in
        :data:`~tvm_tirx.ir.DEFAULT_HIDE`: source locations, function attributes/return
        types, and expression types except variable types.
    fold_formatters : mapping of class to callable, optional
        Partial overrides of :data:`~tvm_tirx.ir.DEFAULT_FOLD_FORMATTERS`. The most specific
        matching callback takes an object and returns single-line text. These
        callbacks apply only inside folds; expansion shows the object's fields.
    max_depth : int, optional
        Maximum number of object/container edges from the root (depth zero).
        Deeper children display ``...``. Indexed arrays still count toward
        depth. None means no depth limit. Cycles display ``<cycle>``.
    show_defaults : bool
        Include fields structurally equal to their declared defaults. Different
        types and distinct free variables remain distinguishable. By default,
        matching fields are omitted; fields without declared defaults remain
        visible. Hidden fields stay hidden.
        For loops omit default sequential kind, empty annotations, and an
        ``int32`` unit step; differently typed or nonunit steps remain visible.

    Returns
    -------
    text : str
        A tree without a trailing newline. Folded text is not width-truncated;
        expand selected fields to split long headers. Shared children can
        appear repeatedly. Maps with scalar keys sort by key type and text;
        other maps show key/value pairs in iteration order. This inspection
        text is not a serialization format.

    Examples
    --------
    .. code-block:: python

        from tvm_tirx import ir

        print(ir.format_tree(ir.Var("i", "int32")))
        # i = Var(ty=int32)
        print(ir.format_tree(ir.Var("i", "int32"), expand=[(ir.Var, "ty")]))
        # i = Var
        # └── ty: PrimType(dtype=int32)
    """
    if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
        raise ValueError("max_depth must be a nonnegative integer or None")
    formatter = _TreeFormatter(max_depth, show_defaults, fold, expand, hide, fold_formatters)
    return "\n".join(formatter.render(value))
