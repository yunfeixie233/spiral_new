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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Tuple

import torch

Metric = Dict[str, Any]


class DAPAlgo(Enum):
    DPO = 0
    IPO = 1
    SLiC = 2
    SimPO = 3
    BNF = 4
    LR_DPO = 5


class RLAlgo(Enum):
    PPO = 100


class SFTAlgo(Enum):
    SFT = 200


@dataclass
class Transition:
    obs: str
    action: str
    rewards: float
    done: bool

    prompt: str
    prompt_ids: list
    response: str
    response_ids: list
    response_logprobs: list

    response_is_truncated: bool
    action_is_formatted: bool

    loss_mask: bool = True
    info: Metric = None

    def format(self):
        return {
            "obs": self.obs,
            "action": self.action,
            "reward": self.rewards,
            "done": int(self.done),
            "prompt": self.prompt,
            "response": self.response,
        }


@dataclass
class PreferenceData:
    prompt: str
    chosen_response: str
    rejected_response: str
    chosen_id: int = 0
    chosen_feature: torch.Tensor = None
    rejected_feature: torch.Tensor = None
    init_clash: bool = False
    loss_mask: bool = True
    is_model_data: bool = False
    info: Metric = None


@dataclass
class TransitionData:
    """Contains single-turn transition data."""

    prompt: str
    prompt_ids: List[int]
    response: str
    response_ids: List[int]
    response_logprobs: List[float]
    rewards: List[float]
    loss_mask: bool = True
    info: Metric = None


@dataclass
class TrajectoryData:
    """Contains multi-turn trajectory data."""

    trajectory_ids: List[int]
    num_turns: int
    response_token_ranges: List[Tuple[int]]
    turn_weights: List[float] = None  # weighted SFT / RL
    messages: List[dict] = None
    info: Metric = None


class RewardData(NamedTuple):
    pair_features: torch.Tensor  # (B, 2, d)
    loss_masks: torch.Tensor  # (B,)
