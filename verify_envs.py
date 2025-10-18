#!/usr/bin/env python3
"""Quick verification for newly integrated environments."""

import sys
sys.path.insert(0, '/ephemeral/games-workspace/spiral_new')

from spiral.envs import make_env
from spiral.agents.utils import get_valid_action_parser

print("Testing newly integrated environments...\n")

envs_to_test = [
    "Briscola-v1",
    "ColonelBlotto-v1", 
    "IndianPoker-v1",
    "TwoDollar-v1"
]

for env_id in envs_to_test:
    print(f"Testing {env_id}...")
    env = make_env(env_id, use_llm_obs_wrapper=True)
    env.reset(num_players=2, seed=42)
    player_id, observation = env.get_observation()
    
    if env_id in ["Briscola-v1", "ColonelBlotto-v1"]:
        parser = get_valid_action_parser(env_id)
        actions = parser(observation)
        print(f"  ✓ {env_id}: {len(actions)} actions parsed")
    else:
        print(f"  ✓ {env_id}: chat-based (infinite action space)")

print("\nAll environments verified successfully!")

