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
# pylint: disable=unrecognized-inline-option
"""Function data types."""

from collections.abc import Sequence

import tvm_ffi

import tvm_tirx
import tvm_tirx.runtime
from tvm_tirx.ir import BaseFunc, Scriptable, SeqStmt, Stmt, Var
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api


@tvm_ffi.register_object("tirx.Function")
class Function(BaseFunc, Scriptable):
    """A TIRx function definition or declaration.

    The function's ``ty`` is a FuncType derived from parameter types and
    ``ret_type``. The body is normalized to a SeqStmt.
    A declaration has an empty body and the ``declaration`` attribute set to True.

    Parameters
    ----------
    params : list[tvm_tirx.ir.Var]
        Input parameters, in calling order.
    body : Stmt or Sequence[Stmt]
        The function body. An empty sequence is a valid body.
    ret_type : tvm_tirx.ir.Type, optional
        The declared return type. Omitting it or supplying MissingType produces
        an empty TupleType, the type returned by the C++ VoidType() helper.
        Construction does not infer a return type from the body.
    attrs : tvm_tirx.ir.DictAttrs, optional
        Function attributes, empty by default.
    loc : Location, optional
        The source location of the function.
    """

    body: SeqStmt

    def __init__(
        self,
        params,
        body: Stmt | Sequence[Stmt],
        ret_type=None,
        attrs=None,
        loc: Location = UNKNOWN_LOC,
    ):
        if ret_type is None:
            ret_type = tvm_tirx.ir.Type.missing()
        param_list = []
        for x in params:
            x = tvm_ffi.convert(x) if not isinstance(x, tvm_ffi.Object) else x
            if not isinstance(x, Var):
                raise TypeError("params can only contain Var or Var")
            param_list.append(x)

        if attrs is None:
            attrs = tvm_tirx.ir.make_node("ir.DictAttrs")

        self.__init_handle_by_constructor__(
            _ffi_api.Function,
            param_list,
            body,
            ret_type,
            attrs,
            loc,
        )  # type: ignore

    def with_body(self, new_body: Stmt | Sequence[Stmt], loc: Location = UNKNOWN_LOC):
        """Create a function with the same parameters, return type, and attributes but a new body.

        Parameters
        ----------
        new_body : Stmt | Sequence[Stmt]
            The new body.

        loc : Location, optional
            The source location of the new function. Defaults to UNKNOWN_LOC;
            pass ``loc=self.loc`` to preserve the original location.

        Returns
        -------
        new_func : Function
            The created new function.
        """
        return Function(
            self.params,
            new_body,
            ret_type=self.ret_type,
            attrs=self.attrs,
            loc=loc,
        )


def renew_def(func: Function) -> Function:
    """Renew variable definitions and remap their uses in a function.

    Definition regions follow the registered structural hooks. Free variables
    and function attributes are preserved, and the function signature is rebuilt.

    Parameters
    ----------
    func : Function
        The function to renew.

    Returns
    -------
    Function
        The function with fresh variable definitions.
    """
    return _ffi_api.RenewDef(func)
