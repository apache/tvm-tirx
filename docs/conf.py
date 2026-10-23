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

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

os.environ["TVM_FFI_BUILD_DOCS"] = "1"

build_exhale = os.environ.get("BUILD_CPP_DOCS", "0") == "1"

_DOCS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

project = "tvm-tirx"
html_title = project
author = "Apache TVM Community"
extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

if build_exhale:
    extensions.append("exhale")

breathe_default_project = "tvm-tirx"

breathe_projects = {"tvm-tirx": "./_build/doxygen/xml"}

exhaleDoxygenStdin = """
INPUT = ../include
INCLUDE_PATH = ../include ../3rdparty/tvm-ffi/include
PREDEFINED             += TVM_TIRX_DLL= TVM_RUNTIME_DLL= TVM_RUNTIME_DLL_EXPORT= \
                          TVM_FFI_DLL= TVM_FFI_DLL_EXPORT= TVM_FFI_INLINE= \
                          TVM_FFI_EXTRA_CXX_API= TVM_FFI_WEAK= TVM_FFI_DOXYGEN_MODE \
                          TVM_FFI_COLD_CODE= \
                          TVM_FFI_PREDICT_FALSE(x)=x TVM_FFI_PREDICT_TRUE(x)=x \
                          __cplusplus=202002
EXCLUDE_SYMBOLS        += *detail* *TypeTraits* std tvm::ffi tvm::ffi::* TVM_* \
                          *_type_* *use_default_type_traits_v* *object_ref_contains_v* \
                          WithAttr MatchesOneOf MakeConst FInferType TFixedReturnType \
                          FRegionGetBodyParams FRegionValidate FTileOpDispatch
EXTRACT_ALL            = YES
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = YES
WARNINGS               = YES
# Doxygen < 1.13 warns on FFI trait specializations whose primary templates are not in INPUT.
WARN_AS_ERROR          = NO
"""

exhaleAfterTitleDescription = """
This page contains the full API index for the C++ API.
"""

exhale_args = {
    "containmentFolder": "reference/cpp/generated",
    "rootFileName": "index.rst",
    "doxygenStripFromPath": "../include",
    "rootFileTitle": "Full API Index",
    "createTreeView": False,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": exhaleDoxygenStdin,
    "afterTitleDescription": exhaleAfterTitleDescription,
}
cpp_id_attributes = [
    "TVM_TIRX_DLL",
    "TVM_RUNTIME_DLL",
    "TVM_RUNTIME_DLL_EXPORT",
    "TVM_FFI_DLL",
    "TVM_FFI_DLL_EXPORT",
    "TVM_FFI_INLINE",
    "TVM_FFI_EXTRA_CXX_API",
    "TVM_FFI_WEAK",
    "TVM_FFI_COLD_CODE",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "_stubs"]
html_theme = "sphinx_book_theme"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_custom_sections = ["Usage"]

# Optional kernel dependencies are not needed to render their Python APIs.
autodoc_mock_imports = ["torch", "triton"]
# The Triton adapter checks this minimum version at import time. Its remaining
# imports are handled by autodoc's mocks; no Triton code executes during a build.
_triton = ModuleType("triton")
_triton.__version__ = "3.3.0"
_triton.__path__ = []
sys.modules.setdefault("triton", _triton)

html_theme_options = {
    "repository_url": "https://github.com/apache/tvm-tirx",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "show_toc_level": 2,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
}

# Case-insensitive filesystems need distinct class/function stub names.
autosummary_filename_map = {
    "tvm_tirx.ir.Var": "tvm_tirx.ir.Var_class",
    "tvm_tirx.runtime.Tensor": "tvm_tirx.runtime.Tensor_class",
    "tvm_tirx.runtime.tensor": "tvm_tirx.runtime.tensor_function",
    "tvm_tirx.ir.prim.Div": "tvm_tirx.ir.prim.Div_class",
    "tvm_tirx.ir.prim.div": "tvm_tirx.ir.prim.div_function",
    "tvm_tirx.ir.prim.FloorDiv": "tvm_tirx.ir.prim.FloorDiv_class",
    "tvm_tirx.ir.prim.floordiv": "tvm_tirx.ir.prim.floordiv_function",
    "tvm_tirx.ir.prim.FloorMod": "tvm_tirx.ir.prim.FloorMod_class",
    "tvm_tirx.ir.prim.floormod": "tvm_tirx.ir.prim.floormod_function",
    "tvm_tirx.ir.prim.Max": "tvm_tirx.ir.prim.Max_class",
    "tvm_tirx.ir.prim.max": "tvm_tirx.ir.prim.max_function",
    "tvm_tirx.ir.prim.Min": "tvm_tirx.ir.prim.Min_class",
    "tvm_tirx.ir.prim.min": "tvm_tirx.ir.prim.min_function",
    "tvm_tirx.support.utils.FileLock": "tvm_tirx.support.utils.FileLock_class",
    "tvm_tirx.support.utils.filelock": "tvm_tirx.support.utils.filelock_function",
    "tvm_tirx.runtime.Device": "tvm_tirx.runtime.Device_class",
}


_STUBS = {
    "_stubs/cpp_index.rst": "reference/cpp/generated/index.rst",
}


def _prepare_stub_files() -> None:
    """Move stub files into place if they do not already exist."""
    for src, dst in _STUBS.items():
        src_path = _DOCS_DIR / src
        dst_path = _DOCS_DIR / dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists():
            dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")


def _mark_exhale_root_orphan(app, docname, source):
    """Prepend :orphan: to exhale-generated root so it stays out of the sidebar."""
    if docname == "reference/cpp/generated/index":
        source[0] = ":orphan:\n\n" + source[0]


_NESTED_TYPE_OF = re.compile(r"is a nested type of :ref:`exhale_(?:class|struct|union)_(\w+)`")
_CLASS_DIRECTIVE = re.compile(r"\.\. doxygen(class|struct|union):: (.+)\n(?:   :.*(?:\n|$))*")


def _link_nested_class_to_enclosing_class(app, docname, source):
    """Link a nested class page to the declaration its enclosing class page renders.

    Breathe renders public and protected nested classes inside the enclosing
    class, and Exhale also gives each nested class its own page. Rendering the
    nested class again would declare the same C++ entity twice.
    """
    prefix = "reference/cpp/generated/"
    match = _NESTED_TYPE_OF.search(source[0]) if docname.startswith(prefix) else None
    if match is None:
        return
    parent = ET.parse(_DOCS_DIR / breathe_projects[breathe_default_project] / f"{match[1]}.xml")
    inner = parent.find(f"compounddef/innerclass[@refid='{docname[len(prefix) :]}']")
    if inner is not None and inner.get("prot") != "private":
        source[0] = _CLASS_DIRECTIVE.sub(
            r"Documented in the enclosing class as :cpp:\1:`\2`.\n", source[0], count=1
        )


def setup(app):
    _prepare_stub_files()
    app.connect("source-read", _mark_exhale_root_orphan)
    app.connect("source-read", _link_nested_class_to_enclosing_class)
