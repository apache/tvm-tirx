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
"""Binding helpers for explicitly selected IR APIs."""

import sys
from types import ModuleType, SimpleNamespace


def _api_attribute(container, name):
    # Modules may lazily initialize whole dialects. Explicit namespace objects
    # own their attribute semantics, including descriptors and instruction chains.
    return (
        vars(container).get(name)
        if isinstance(container, ModuleType)
        else getattr(container, name, None)
    )


def _relative_name(name, prefix):
    if not prefix:
        return name
    return name[len(prefix) + 1 :] if name.startswith(prefix + ".") else None


def _api_slot(module_name, path):
    """Find a destination without importing it or invoking module lazy lookup."""
    container = sys.modules.get(module_name)
    if container is None:
        return None
    *parents, leaf = path.split(".")
    for name in parents:
        child = _api_attribute(container, name)
        if child is None:
            child = SimpleNamespace()
            _bind_api(container, name, child)
        container = child
    return container, leaf


def _bind_api(container, name, value):
    setattr(container, name, value)
    exports = vars(container).get("__all__")
    if isinstance(exports, list) and name not in exports:
        exports.append(name)
    return value
