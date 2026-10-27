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
"""Concrete TIRx construction operations over the shared native IRBuilder stack."""

from tvm_tirx import base as _base
from tvm_tirx.ir.type import _init_type_api
from tvm_tirx.script.ir_builder import *  # noqa: F403 - Re-export the public namespace.
from tvm_tirx.script.ir_builder import __all__ as _shared_exports
from tvm_tirx.script.ir_builder import constexpr, dtype, dynamic

from . import ir as _native
from .ir import *  # noqa: F403 - Re-export the public namespace.
from .ir import __all__ as _ir_exports
from .ir import boolean as bool
from .op import *  # noqa: F403 - Re-export the public namespace.
from .op import __all__ as _op_exports
from .op import _get_script_namespace as __getattr__
from .op import expose_script_op, expose_script_type
from .parser_protocol import *  # noqa: F403 - Re-export the public namespace.
from .parser_protocol import __all__ as _protocol_exports
from .utils import (
    frame_scope as frame_scope,
)
from .utils import (
    tensor_indices as tensor_indices,
)

supports_mutable_declarations = True

__all__ = [
    *_shared_exports,
    *_ir_exports,
    *_op_exports,
    *_protocol_exports,
    "dynamic",
    "dtype",
    "constexpr",
    "bool",
    "tensor_indices",
    "frame_scope",
    "supports_mutable_declarations",
]


_init_type_api("tirx", __name__)

# Explicit builders and type aliases take precedence over generated bindings.
# Install both callbacks before replay so reentrant requests can run immediately.
_base._EXPOSE_SCRIPT_OP_CALLBACK = expose_script_op
_base._EXPOSE_SCRIPT_TYPE_CALLBACK = expose_script_type
while _base._PENDING_SCRIPT_EXPOSURES:
    _kind, _name = _base._PENDING_SCRIPT_EXPOSURES[0]
    if _kind == "op":
        expose_script_op(_name)
    else:
        expose_script_type(_name)
    # Retain a failed request so initialization can be retried.
    _base._PENDING_SCRIPT_EXPOSURES.popleft()
