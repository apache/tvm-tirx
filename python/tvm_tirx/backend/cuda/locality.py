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
"""CUDA locality domains: per-domain device memory and topology queries.

Some GPUs (e.g. Rubin) split their SMs and device memory into locality domains;
an SM reads memory of its own domain faster than memory of another one. A
tensor is placed in domain ``k`` by allocating it with the memory scope
``global.locality_domain.<k>``::

    from tvm_tirx.backend.cuda import locality

    dev = tvm_tirx.cuda(0)
    for k in range(locality.locality_domain_count(dev)):
        scope = locality.locality_domain_scope(k)  # "global.locality_domain.<k>"
        x = tvm_tirx.runtime.empty((n,), "float32", dev, mem_scope=scope)
        assert locality.locality_domain_of(x) == k

A kernel reads the domain of the SM it runs on with
``T.cuda.mov_sreg(32, "locality_domain_id")`` (PTX ISA 9.5) and can then work on
the data of its own domain. Memory of any domain stays accessible from every
SM; only the access speed differs. A device without locality domains reports
one domain, and ``global.locality_domain.0`` then behaves like ``global``.
"""

import tvm_ffi

__all__ = [
    "MAX_LOCALITY_DOMAINS",
    "locality_domain_count",
    "locality_domain_of",
    "locality_domain_scope",
    "locality_domain_sm_count",
]

#: Number of ``global.locality_domain.<k>`` memory scopes registered (k < this).
MAX_LOCALITY_DOMAINS = 8


def locality_domain_count(device) -> int:
    """Return the number of locality domains of a CUDA device (1 if it has none).

    Parameters
    ----------
    device : tvm_tirx.runtime.Device
        A CUDA device, e.g. ``tvm_tirx.cuda(0)``.
    """
    return int(tvm_ffi.get_global_func("runtime.GetCudaLocalityDomainCount")(device.index))


def locality_domain_sm_count(device) -> int:
    """Return the number of SMs guaranteed to each locality domain of a CUDA device.

    A device without locality domains reports all of its SMs. On a device with
    domains, SMs beyond ``count * per_domain`` are supplemental SMs; a kernel
    tells them apart with ``T.cuda.mov_sreg(32, "locality_domain_supplemental_sm")``.

    Parameters
    ----------
    device : tvm_tirx.runtime.Device
        A CUDA device, e.g. ``tvm_tirx.cuda(0)``.
    """
    return int(
        tvm_ffi.get_global_func("runtime.GetCudaLocalityDomainMultiprocessorCount")(device.index)
    )


def locality_domain_scope(domain: int) -> str:
    """Return the memory scope that places an allocation in locality domain ``domain``.

    Parameters
    ----------
    domain : int
        The locality domain, in ``[0, locality_domain_count(device))``.
    """
    if not 0 <= domain < MAX_LOCALITY_DOMAINS:
        raise ValueError(
            f"locality domain {domain} is outside the registered scopes [0, {MAX_LOCALITY_DOMAINS})"
        )
    return f"global.locality_domain.{domain}"


def locality_domain_of(tensor) -> int:
    """Return the locality domain a CUDA tensor's memory belongs to, or -1 if it has none.

    Memory allocated with the plain ``global`` scope is not localized and reports -1.

    Parameters
    ----------
    tensor : tvm_tirx.runtime.Tensor or any DLPack-compatible tensor
        A tensor in CUDA device memory.
    """
    return int(tvm_ffi.get_global_func("runtime.GetCudaTensorLocalityDomain")(tensor))
