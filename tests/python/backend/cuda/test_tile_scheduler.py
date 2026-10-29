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
"""Persistent scheduler ordering and compile-time mode selection."""

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@pytest.mark.parametrize("serpentine", [False, True])
def test_scheduler_parses_only_selected_ordering(monkeypatch, serpentine):
    def unselected_ordering(*_args):
        pytest.fail("the parser visited the unselected scheduler ordering")

    unselected = "_update_group_major" if serpentine else "_update_serpentine"
    monkeypatch.setattr(ClusterPersistentScheduler2D, unselected, unselected_ordering)

    @T.function(private=True)
    def schedule(output: T.Tensor((2,), "int32")):
        scheduler = ClusterPersistentScheduler2D(
            "sched", 4, 4, num_clusters=3, l2_group_size=2, serpentine=serpentine
        )
        scheduler.init(0)
        output[0] = scheduler.m_idx
        output[1] = scheduler.n_idx

    assert isinstance(schedule, tvm_tirx.tirx.Function)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("num_clusters", [1, 3, 20])
def test_scheduler_default_persistent_traversal(num_clusters):
    # Scheduler arithmetic can run on the CPU without a CUDA device.
    @T.function
    def schedule(output: T.Tensor((16, 2), "int32")):
        scheduler = ClusterPersistentScheduler2D(
            "sched", 4, 4, num_clusters=num_clusters, l2_group_size=2
        )
        for cluster_id in T.serial(num_clusters):
            scheduler.init(cluster_id)
            while scheduler.valid():
                output[scheduler.work_idx, 0] = scheduler.m_idx
                output[scheduler.work_idx, 1] = scheduler.n_idx
                scheduler.next_tile()

    expected = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
        (2, 3),
        (3, 3),
    ]

    output = tvm_tirx.runtime.tensor(np.full((16, 2), -1, dtype="int32"))
    tvm_tirx.compile(schedule, target="llvm")(output)
    np.testing.assert_array_equal(output.numpy(), expected)
