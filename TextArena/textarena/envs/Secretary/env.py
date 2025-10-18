import random
import re
from typing import Dict, Any, Optional, Tuple, List

import textarena as ta


class SecretaryEnv(ta.Env):

    def __init__(self, N: int = 20):
        self.N = N
        self.value_sampler = random.random
        self.action_space = re.compile(r'\[(accept|continue)\]')

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.SinglePlayerState(num_players=num_players, max_turns=self.N, seed=seed)
        draws: List[float] = [self.value_sampler() for _ in range(self.N)]
        self.state.reset(game_state=dict(draws=draws, accepted_idx=None, current_idx=0), player_prompt_function=self._prompt)
        self._show_next_value()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You will observe {self.N} hidden values sequentially.\nAt each step type [accept] to pick the *current* value, "
            "or [continue] to skip it and see the next one.\nIf you never accept, you are forced to take the final value.\n"
            "You win (reward = 1) **only** if the value you ultimately pick is the highest of all."
        )
    
    def _show_next_value(self):
        self.state.add_observation(message=f"The current value is {self.state.game_state['draws'][self.state.game_state['current_idx']]:.4f}.", observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION)
        self.state.game_state['current_idx'] += 1

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ta.ObservationType.PLAYER_ACTION)

        # Validate action format
        m = self.action_space.fullmatch(action.strip())
        if m is None:
            self.state.set_invalid_move(reason="Action must be either [accept] or [continue].")
            return self.state.step()

        choice = m.group(1)
        if choice == "accept": self._resolve(accepted_at=self.state.game_state['current_idx']-1)
 
        # Auto-accept on the very last turn if no decision yet
        if (self.state.game_state['current_idx'] >= self.N or choice=="accept"):
            self._resolve(accepted_at=self.state.game_state['current_idx']-1)
        else:                   
            self._show_next_value()
        return self.state.step()

    def _resolve(self, accepted_at: int):
        draws = self.state.game_state["draws"]
        won = draws[accepted_at] == max(draws)
        message=f"You accepted value {draws[accepted_at]:.4f} at draw {accepted_at + 1}/{self.N}. The best overall was {max(draws):.4f}. "
        self.state.set_outcome(reward=1.0 if won else 0.0, reason=message+("Perfect choice! 🎉" if won else "Not the maximum."))
