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
# pylint: disable=invalid-name
"""Primitive operators in the TVM IR."""

from collections.abc import Sequence

import tvm_ffi

from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api
from .expr import Expr


@tvm_ffi.register_object("ir.Op")
class Op(Expr):
    """A registered primitive operator used as a :class:`Call` callee.

    Retrieve the shared operator with ``Op.get(name)``. Its registration supplies
    argument signatures, attributes, validation, and result-type inference;
    constructing a Call does not itself execute the operator.
    """

    def __init__(self):
        raise RuntimeError("Cannot create op, use get instead")

    @staticmethod
    def get(op_name):
        """Get a registered operator by name.

        Parameters
        ----------
        op_name : str
            The canonical operator name.

        Returns
        -------
        Op
            A handle to the registered operator.
        """
        return _ffi_api.GetOp(op_name)

    @staticmethod
    def list_op_names():
        """List registered operator names in unspecified order.

        Returns
        -------
        list[str]
            The registered operator names.
        """
        return _ffi_api.ListOpNames()

    @staticmethod
    def list_attr_names():
        """List registered attribute column names in unspecified order.

        Returns
        -------
        list[str]
            The attribute column names, such as ``"TCallEffectKind"``.
        """
        return _ffi_api.ListOpAttrNames()

    @staticmethod
    def get_attr_map(attr_name):
        """Get the map from operators to their values of an attribute.

        This is the Python counterpart of C++ ``Op::GetAttrMap``. The map reads the
        registry's storage directly, so later registrations, overrides and removals
        are visible through it, and it can be cached.

        Parameters
        ----------
        attr_name : str
            Attribute name, such as ``"TCallEffectKind"``.

        Returns
        -------
        OpAttrMap or None
            The attribute map, or None when no operator has registered the attribute.
        """
        column = _ffi_api.GetOpAttrColumn(attr_name)
        return None if column is None else OpAttrMap(attr_name, column)

    def set_attr(self, attr_name, value, override=False):
        """Set an operator attribute.

        Parameters
        ----------
        attr_name : str
            Attribute column name.
        value : object
            Non-None attribute value.
        override : bool, optional
            Replace an existing value if True. Duplicate registration otherwise
            raises ValueError. Cached views observe replacements; no history is kept.

        Returns
        -------
        None
        """
        self._set_attr(attr_name, value, override)

    def set_signature(self, args=(), *, ty_args=(), var_args=None, var_ty_args=None) -> None:
        """Replace this Op's argument signature and validate Call arity.

        Each entry is a name string or a ``(name, doc)`` tuple of strings.
        Fixed entries form required prefixes; a variadic entry allows zero or
        more additional arguments. Calls must have exactly the required count
        without a tail, or at least that count with one. This method does not
        check argument types or Call attrs. It replaces a generated typed
        validator with a count-only one, while an existing custom validator
        keeps precedence.

        Parameters
        ----------
        args : sequence[str or tuple[str, str]], optional
            Fixed value arguments, in order.
        ty_args : sequence[str or tuple[str, str]], optional
            Fixed type arguments, in order.
        var_args : str or tuple[str, str], optional
            Variadic value-argument tail.
        var_ty_args : str or tuple[str, str], optional
            Variadic type-argument tail.

        Returns
        -------
        None
        """

        def parse_entry(entry, label):
            if isinstance(entry, str):
                return entry, ""
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and all(isinstance(part, str) for part in entry)
            ):
                return entry
            raise TypeError(f"{label} must be a name string or a (name, doc) tuple")

        def parse_fixed(entries, label):
            if not isinstance(entries, Sequence) or isinstance(entries, str | bytes):
                raise TypeError(f"{label} must be a sequence of signature entries")
            parsed = [parse_entry(entry, label) for entry in entries]
            return [name for name, _ in parsed], [doc for _, doc in parsed]

        arg_names, arg_docs = parse_fixed(args, "args")
        ty_arg_names, ty_arg_docs = parse_fixed(ty_args, "ty_args")
        value_tail = [] if var_args is None else list(parse_entry(var_args, "var_args"))
        type_tail = [] if var_ty_args is None else list(parse_entry(var_ty_args, "var_ty_args"))
        self._set_signature(arg_names, arg_docs, ty_arg_names, ty_arg_docs, value_tail, type_tail)

    def expose_script(self) -> None:
        """Expose the completed operator definition in TVMScript.

        Call after setting the signature and attributes. Early requests wait
        until the script builder is initialized; explicit builders take precedence.
        """
        tvm_ffi.get_global_func("script.expose_script_op")(self.name)


class OpAttrMap:
    """The values of one operator attribute, keyed by operator.

    Mirrors C++ ``OpAttrMap``. Obtain one with :py:meth:`Op.get_attr_map`. Lookups
    see later registrations, overrides and removals; register values with
    :py:meth:`Op.set_attr`. Like the registry itself, the map is not synchronized
    with concurrent registration.
    """

    def __init__(self, attr_name, column):
        self._attr_name = attr_name
        self._column = column

    @property
    def attr_name(self):
        """The attribute name."""
        return self._attr_name

    def __contains__(self, op):
        if not isinstance(op, Op):
            return False
        index = op.index
        return index < len(self._column) and self._column[index] is not None

    def __getitem__(self, op):
        if op not in self:
            name = op.name if isinstance(op, Op) else op
            raise KeyError(f"Attribute {self._attr_name} has not been registered for {name}")
        return self._column[op.index]

    def get(self, expr, default=None):
        """Get the attribute of an operator with a fallback.

        Parameters
        ----------
        expr : Expr
            The operator. Any other expression yields ``default``.

        default : object
            The value to return when ``expr`` is not an Op or has no value.

        Returns
        -------
        object
            The registered value or ``default``.
        """
        return self[expr] if expr in self else default


def register_op_attr(op_name, attr_key, value=None, override=False):
    """Register an operator property of an operator by name.

    Parameters
    ----------
    op_name : str
        The name of operator

    attr_key : str
        The attribute name.

    value : object, optional
        The value to set

    override : bool, optional
        Replace an existing value if True; otherwise duplicate registration raises
        ValueError. Cached views observe replacements; no priority history is kept.

    Returns
    -------
    result : object or function
        The registered value when supplied, or a decorator that registers and
        returns its argument. The named Op must already be registered.
    """

    def _register(v):
        """internal register function"""
        Op.get(op_name).set_attr(attr_key, v, override)
        return v

    return _register(value) if value is not None else _register


def _make_op_api(op, module_name):
    """Build a callable whose operands and result are governed by its Op."""
    from .attrs import make_node  # pylint: disable=import-outside-toplevel
    from .expr import Call  # pylint: disable=import-outside-toplevel

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
