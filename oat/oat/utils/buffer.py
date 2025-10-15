# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import tree

from oat.types import RewardData


class UniformBuffer(object):
    def __init__(self, max_len: int):
        self._max_len = int(max_len)
        self._storage = None
        self._n = 0
        self._idx = 0

    def extend(self, batch: RewardData):
        if self._storage is None:
            sample_batch = tree.map_structure(lambda t: t[0], batch)
            self._storage = tree.map_structure(
                lambda t: torch.empty(
                    (self._max_len,) + t.shape, dtype=t.dtype, device=t.device
                ),
                sample_batch,
            )

        num_steps = len(batch.pair_features)
        indices = torch.arange(self._idx, self._idx + num_steps) % self._max_len
        tree.map_structure(lambda a, x: assign(a, indices, x), self._storage, batch)
        self._idx = (self._idx + num_steps) % self._max_len
        self._n = min(self._n + num_steps, self._max_len)

    def sample(self, batch_size: int) -> RewardData:
        if batch_size > self._n:
            return None
        start_indices = np.random.choice(self._n, batch_size, replace=False)
        base_idx = 0 if self._n < self._max_len else self._idx
        all_indices = (start_indices + base_idx) % self._max_len
        return tree.map_structure(lambda a: a[all_indices], self._storage)

    def get_all(self) -> RewardData:
        all_indices = np.arange(self.size)
        return tree.map_structure(lambda a: a[all_indices], self._storage)

    @property
    def size(self):
        return self._n


def assign(a, i, x):
    a[i] = x
