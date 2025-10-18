# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import textarena as ta
from textarena.envs.registration import register

# Kuhn Poker (two-player)
register(
    id="KuhnPoker-v1",
    entry_point="spiral.envs.KuhnPoker.env:KuhnPokerEnv",
    ante=1,
    max_rounds=5,
)

# Simple Negotiation (two-player)
register(
    id="SimpleNegotiation-v1",
    entry_point="spiral.envs.SimpleNegotiation.env:SimpleNegotiationEnv",
    max_turns=10,
)

# Liar's Dice (two-player)
register(
    id="LiarsDice-v1",
    entry_point="spiral.envs.LiarsDice.env:LiarsDiceEnv",
)

# Truth and Deception (two-player)
register(
    id="TruthAndDeception-v1",
    entry_point="spiral.envs.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=6,
)

# Pig Dice (two-player)
register(
    id="PigDice-v1",
    entry_point="spiral.envs.PigDice.env:PigDiceEnv",
    winning_score=25,
    max_turns=10,
)

# Tic Tac Toe (two-player)
register(
    id="TicTacToe-v1",
    entry_point="spiral.envs.TicTacToe.env:TicTacToeEnv",
)

# Briscola (two-player card game)
register(
    id="Briscola-v1",
    entry_point="spiral.envs.Briscola.env:BriscolaEnv",
)

# Colonel Blotto (two-player strategy)
register(
    id="ColonelBlotto-v1",
    entry_point="spiral.envs.ColonelBlotto.env:ColonelBlottoEnv",
    num_fields=3,
    num_total_units=20,
    num_rounds=10,
)

# Indian Poker (two-player betting game)
register(
    id="IndianPoker-v1",
    entry_point="spiral.envs.IndianPoker.env:IndianPokerEnv",
    max_rounds=5,
    starting_chips=100,
)

# Two Dollar Negotiation (two-player negotiation)
register(
    id="TwoDollar-v1",
    entry_point="spiral.envs.TwoDollar.env:TwoDollarEnv",
    total_amount=2.00,
    max_rounds=10,
    error_allowance=3,
)


def make_env(env_id: str, use_llm_obs_wrapper: bool):
    env = ta.make(env_id)
    
    # List of environments that append available actions on each get_observation()
    # These need the FirstLastObservationWrapper to avoid accumulation
    envs_with_action_messages = ["KuhnPoker-v1"]
    
    if use_llm_obs_wrapper:
        if env_id in envs_with_action_messages:
            logging.info(f"[ENV] Using FirstLastObservationWrapper for {env_id}")
            env = ta.wrappers.FirstLastObservationWrapper(env=env)
        else:
            logging.info(f"[ENV] Using LLMObservationWrapper for {env_id}")
            env = ta.wrappers.LLMObservationWrapper(env=env)
    else:
        logging.info(f"[ENV] Using FirstLastObservationWrapper for {env_id}")
        env = ta.wrappers.FirstLastObservationWrapper(env=env)
    return env

def make_vec_env(env_id: str, num_envs: int, **kw_args):
    return [make_env(env_id, **kw_args) for _ in range(num_envs)]
