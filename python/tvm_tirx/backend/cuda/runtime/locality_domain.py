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

    from tvm_tirx.backend.cuda.runtime import locality_domain

    dev = tvm_tirx.cuda(0)
    for k in range(locality_domain.get_num_locality_domains(dev)):
        scope = locality_domain.locality_domain_scope(k)  # "global.locality_domain.<k>"
        x = tvm_tirx.runtime.empty((n,), "float32", dev, mem_scope=scope)
        assert locality_domain.get_locality_domain_id(x) == k

The query names and their semantics follow ``torch.cuda.green_contexts``
(``get_num_locality_domains``, ``is_localization_supported``) and torch's
``locality_domain_id``. Each query wraps the tvm-ffi function
``runtime.cuda.<Name>`` of the same name, e.g. ``get_num_locality_domains`` is
``runtime.cuda.GetNumLocalityDomains``.

Like ``cudaMalloc`` memory, a localized tensor is ready on every stream once it
is allocated. It is freed in the order of the legacy default stream, after the
work queued on it and on every blocking stream. Work on a nonblocking stream
must be synchronized before the last reference to the tensor goes away.

A kernel reads the domain of the SM it runs on with
``T.cuda.mov_sreg(32, "locality_domain_id")`` (PTX ISA 9.5) and can then work on
the data of its own domain. Memory of any domain stays accessible from every
SM; only the access speed differs. A device without locality domains, or with
a CUDA build or driver older than 13.4, reports one domain:
``global.locality_domain.0`` then behaves like ``global``, and
``get_locality_domain_id`` returns None for its memory.
"""

import tvm_ffi

from tvm_tirx.runtime import Device

from . import _ffi_api

__all__ = [
    "MAX_LOCALITY_DOMAINS",
    "get_locality_domain_id",
    "get_num_locality_domains",
    "get_num_sms_per_locality_domain",
    "is_localization_supported",
    "locality_domain_scope",
]

#: Number of ``global.locality_domain.<k>`` memory scopes registered (k < this).
MAX_LOCALITY_DOMAINS = 8


def _device_id(device: Device | int) -> int:
    if isinstance(device, int):
        return device
    if device.dlpack_device_type() != tvm_ffi.DLDeviceType.kDLCUDA:
        raise ValueError(f"Expected a CUDA device, got {device}")
    return device.index


def get_num_locality_domains(device: Device | int = 0) -> int:
    """Return the number of locality domains of a CUDA device.

    A device without locality domains, or a CUDA driver older than 13.4,
    reports 1.

    Parameters
    ----------
    device : tvm_tirx.runtime.Device or int
        A CUDA device, e.g. ``tvm_tirx.cuda(0)``, or its device id.
    """
    return int(_ffi_api.GetNumLocalityDomains(_device_id(device)))


def is_localization_supported(device: Device | int = 0) -> bool:
    """Return whether memory can be localized on a CUDA device.

    True when the CUDA driver supports locality domains and the device has two
    or more of them.

    Parameters
    ----------
    device : tvm_tirx.runtime.Device or int
        A CUDA device, e.g. ``tvm_tirx.cuda(0)``, or its device id.
    """
    return bool(_ffi_api.IsLocalizationSupported(_device_id(device)))


def get_num_sms_per_locality_domain(device: Device | int = 0) -> int:
    """Return the number of SMs guaranteed to each locality domain of a CUDA device.

    A device without locality domains reports all of its SMs. On a device with
    domains, SMs beyond ``num_domains * sms_per_domain`` are supplemental SMs; a
    kernel tells them apart with
    ``T.cuda.mov_sreg(32, "locality_domain_supplemental_sm")``.

    Parameters
    ----------
    device : tvm_tirx.runtime.Device or int
        A CUDA device, e.g. ``tvm_tirx.cuda(0)``, or its device id.
    """
    return int(_ffi_api.GetNumSMsPerLocalityDomain(_device_id(device)))


def get_locality_domain_id(tensor) -> int | None:
    """Return the locality domain a CUDA tensor's memory belongs to.

    Returns None for memory that is not localized, e.g. memory allocated with
    the plain ``global`` scope, or with ``global.locality_domain.0`` on a device
    that reports a single locality domain.

    Parameters
    ----------
    tensor : tvm_tirx.runtime.Tensor or any DLPack-compatible tensor
        A tensor in CUDA device memory.
    """
    domain = _ffi_api.GetLocalityDomainId(tensor)
    return None if domain is None else int(domain)


def locality_domain_scope(locality_domain_id: int) -> str:
    """Return the memory scope that places an allocation in a locality domain.

    Parameters
    ----------
    locality_domain_id : int
        The locality domain, in ``[0, get_num_locality_domains(device))``.
    """
    if not 0 <= locality_domain_id < MAX_LOCALITY_DOMAINS:
        raise ValueError(
            f"locality domain {locality_domain_id} is outside the registered scopes "
            f"[0, {MAX_LOCALITY_DOMAINS})"
        )
    return f"global.locality_domain.{locality_domain_id}"
