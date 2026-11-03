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
"""Load the shared library of a TIRx extension into its Python module."""

from pathlib import Path

import tvm_ffi
from tvm_ffi.libinfo import load_lib_module


def load_extension(
    module_name: str,
    library: str,
    *,
    lib_dir: str | Path | None = None,
    package: str | None = None,
    prefix: str | None = None,
) -> tvm_ffi.Module:
    """Load an extension library and publish its types and operators into a Python module.

    An extension library, such as one built from Rust with the ``tvm-tirx`` crate, registers its
    types and operators when it loads. This function makes the Python module ``module_name``,
    usually the ``__init__`` of the extension's package, the destination of their builders, then
    loads the library, so ``<module>.<name>(...)`` writes their calls in TVMScript.

    Parameters
    ----------
    module_name : str
        Name of the already imported Python module that receives the builders, usually
        ``__name__``.
    library : str
        Name of the shared library without its platform prefix and suffix, such as ``"my_ext"``
        for ``libmy_ext.so``.
    lib_dir : str or Path or None, optional
        Directory searched first for the library, such as the package directory.
    package : str or None, optional
        Distribution whose installed files are searched for the library. Defaults to
        ``library``.
    prefix : str or None, optional
        Namespace of the extension's operator names and type keys, such as ``"my_ext"`` for
        ``my_ext.cmul``. Defaults to ``module_name``.

    Returns
    -------
    library : tvm_ffi.Module
        The loaded library, whose exported functions are its attributes.

    Raises
    ------
    KeyError
        If ``module_name`` is not an imported module.
    RuntimeError
        If the library is not found.

    Examples
    --------
    .. code-block:: python

        # my_ext/__init__.py
        from pathlib import Path

        import tvm_tirx

        _LIB = tvm_tirx.load_extension(__name__, "my_ext", lib_dir=Path(__file__).parent)
    """
    # pylint: disable=import-outside-toplevel
    from .ir._expose_api import _init_exposed_op_api, _init_exposed_type_api
    from .script.ir_builder.base import auto_emit_void_call

    if prefix is None:
        prefix = module_name
    # Destinations come first, so definitions published while the library loads reach the module.
    _init_exposed_op_api(prefix, module_name, wrapper=auto_emit_void_call)
    _init_exposed_type_api(prefix, module_name)
    extra_lib_paths = None if lib_dir is None else [Path(lib_dir)]
    return load_lib_module(package or library, library, extra_lib_paths=extra_lib_paths)
