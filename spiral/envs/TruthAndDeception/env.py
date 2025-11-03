import importlib.resources
import json
import logging
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.TruthAndDeception.renderer import create_board_str


class TruthAndDeceptionEnv(ta.Env):
    """Environment for Truth and Deception Game"""

    def __init__(self, max_turns: Optional[int] = 5, data_path: Optional[str] = None, max_retry: Optional[int] = 3):
        """
        Initialize the Truth and Deception game.

        Roles:
            - Player 0 is the deceiver (gets potentially incorrect information about which fact is correct)
            - Player 1 is the guesser

        Args:
            max_turns (int): Maximum number of conversation turns.
            data_path (str): Path to the JSON file containing the facts.
            max_retry (int): Maximum number of retries when <answer> parsing fails.
        """
        assert (
            max_turns % 2 == 0
        ), f"Please use an even number of max turns. Current max_turns: {max_turns}"

        self.max_turns = max_turns
        self.max_retry = max_retry
        self._load_facts(data_path=data_path)

        # Match both "Fact 1" and "[Fact 1]" (with or without brackets)
        self.guess_fact1_pattern = re.compile(r"(?:\[Fact\s+1\]|Fact\s+1)", re.IGNORECASE)
        self.guess_fact2_pattern = re.compile(r"(?:\[Fact\s+2\]|Fact\s+2)", re.IGNORECASE)
        self.boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)

    def _parse_answer_from_action(self, action: str) -> str:
        """
        Parse \boxed{...} from action. If no tags found, return the original action.
        This serves as a safety net in case the training pipeline doesn't parse.
        
        Args:
            action: Raw action string, potentially containing thinking and \boxed{...}
            
        Returns:
            Parsed action with only the content from \boxed{...}, or original action if no tags
        """
        if not action:
            return action
            
        # Extract content from \boxed{...}
        boxed_match = self.boxed_pattern.search(action)
        if boxed_match:
            parsed_content = boxed_match.group(1).strip()
            logging.debug(f"Environment parsed \\boxed content: '{parsed_content[:50]}...'")
            return parsed_content
        
        # No tags found, return original action
        # This handles cases where action is already parsed by training pipeline
        return action

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)
    
    def _load_facts(self, data_path: Optional[str]) -> None:
        """Load the facts from the specified JSON file.

        Args:
            data_path (str): Path to the JSON file containing the facts.
        """
        try:
            if data_path is not None:
                if not os.path.exists(data_path):
                    raise FileNotFoundError(
                        f"Facts data file not found at: {data_path}"
                    )
                with open(data_path, "r", encoding="utf-8") as file:
                    self.facts_data = json.load(file)
            else:
                files = importlib.resources.files("textarena.envs.TruthAndDeception")
                with files.joinpath("facts.json").open("r") as file:
                    self.facts_data = json.load(file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load facts data: {str(e)}")

    def _prompt(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> str:
        """Generate the initial prompt for a player"""
        # Internal role mapping for game logic (not exposed in observations)
        internal_roles = {0: "Deceiver", 1: "Guesser"}
        player_role = internal_roles[player_id]
        
        prompt = f"You are Player {player_id}, and you are the {player_role} in a game.\n"
        if player_role == "Deceiver":
            # For the Deceiver, use the potentially incorrect information
            prompt += (
                f"You know which of the following two facts is correct:\n"
                f"Fact 1: {game_state['fact1']['fact']} ({'correct' if game_state['deceiver_told_fact1_correct'] else 'wrong'})\n"
                f"Fact 2: {game_state['fact2']['fact']} ({'correct' if game_state['deceiver_told_fact2_correct'] else 'wrong'})\n"
                "Your goal is to convince the other player to choose the incorrect fact.\n"
                f"You will have a conversation for {self.state.max_turns - 1} turns.\n"
                "After that, the other player will make their guess.\n\n"
                "IMPORTANT: During the conversation, DO NOT directly state '[Fact 1]' or '[Fact 2]' in your messages. "
                "Instead, discuss the facts using reasoning, examples, or explanations to persuade the guesser. "
                "Only the guesser will state '[Fact 1]' or '[Fact 2]' in the final guessing round.\n\n"
                "On your turn, you can think about your strategy, then put your message to the guesser inside \\boxed{your message here}.\n"
            )
        elif player_role == "Guesser":
            prompt += (
                f"You need to determine which of the following two facts is correct:\n"
                f"Fact 1: {game_state['fact1']['fact']}\n"
                f"Fact 2: {game_state['fact2']['fact']}\n"
                f"You will have a conversation with the other player for {self.state.max_turns - 2} turns.\n"
                "What the other player said may be right or wrong.\n"
                "After that, you will make your guess.\n\n"
                "IMPORTANT: During the conversation, DO NOT directly state '[Fact 1]' or '[Fact 2]' in your messages. "
                "Instead, ask questions and discuss the facts to gather information. "
                "You will only state '[Fact 1]' or '[Fact 2]' when prompted for your final guess.\n\n"
                "On your turn, you can think about the facts, then put your message to the other player inside \\boxed{your message here}.\n"
            )
        else:
            raise ValueError(
                f"Unexpected role: {player_role}. Expected 'Deceiver' or 'Guesser'."
            )
        return prompt

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the game to its initial state."""
        # Initialize game state
        self.state = ta.TwoPlayerState(
            num_players=num_players,
            max_turns=self.max_turns,
            seed=seed
        )

        # Set random seed if provided for reproducible results
        if seed is not None:
            random.seed(seed)

        selected_facts = random.choice(self.facts_data)

        facts = [
            (
                selected_facts["facts"]["fact1"],
                selected_facts["correct_fact"] == "fact1",
            ),
            (
                selected_facts["facts"]["fact2"],
                selected_facts["correct_fact"] == "fact2",
            ),
        ]
        random.shuffle(facts)

        # Determine what the Deceiver will be told (50% chance of being correct)
        deceiver_info_is_correct = random.choice([True, False])
        
        if deceiver_info_is_correct:
            # Tell the Deceiver the correct information
            deceiver_told_fact1_correct = facts[0][1]
            deceiver_told_fact2_correct = facts[1][1]
        else:
            # Tell the Deceiver the opposite (incorrect information)
            deceiver_told_fact1_correct = not facts[0][1]
            deceiver_told_fact2_correct = not facts[1][1]

        game_state = {
            "fact1": {"fact": facts[0][0], "is_correct": facts[0][1]},
            "fact2": {"fact": facts[1][0], "is_correct": facts[1][1]},
            "correct_fact": facts[0][0] if facts[0][1] else facts[1][0],
            "wrong_fact": facts[0][0] if facts[1][1] else facts[1][0],
            # What the Deceiver is told about which facts are correct
            "deceiver_told_fact1_correct": deceiver_told_fact1_correct,
            "deceiver_told_fact2_correct": deceiver_told_fact2_correct,
            # Whether the information given to the Deceiver is actually correct
            "deceiver_info_is_correct": deceiver_info_is_correct,
        }
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._prompt,
            role_mapping={0: "Deceiver", 1: "Guesser"}
        )
        
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process the player's action"""
        
        # Safety net: Parse <answer> tags if present, otherwise use raw action
        parsed_action = self._parse_answer_from_action(action)

        conversation_turns = self.state.max_turns - 2

        # Normal conversation phase
        if self.state.turn < conversation_turns:
            # During conversation, reject responses that look like guesses
            if self.guess_fact1_pattern.search(parsed_action) or self.guess_fact2_pattern.search(parsed_action):
                reason = f"Player {self.state.current_player_id} cannot guess during the conversation phase. Please engage in conversation."
                self.state.set_invalid_move(reason=reason)
                return self.state.step()
            
            self.state.add_observation(
                from_id=self.state.current_player_id,
                message=parsed_action,
                observation_type=ta.ObservationType.PLAYER_ACTION
            )
            return self.state.step()

        # Transition to guessing phase after conversation turns
        elif (
            self.state.turn == conversation_turns and self.state.current_player_id == 0
        ):
            # During Deceiver's final conversation turn, also reject guess-like responses
            if self.guess_fact1_pattern.search(parsed_action) or self.guess_fact2_pattern.search(parsed_action):
                reason = f"Player {self.state.current_player_id} cannot guess during the conversation phase. Please engage in conversation."
                self.state.set_invalid_move(reason=reason)
                return self.state.step()
            
            # Add Deceiver's final conversation message
            self.state.add_observation(
                from_id=self.state.current_player_id,
                message=parsed_action,
                observation_type=ta.ObservationType.PLAYER_ACTION
            )

            # Normal step to rotate to Guesser
            done, info = self.state.step()

            # Show guessing prompt to Guesser
            if not done:
                message = "Now guess which of the two facts are correct by returning '\\boxed{[Fact 1]}' or '\\boxed{[Fact 2]}'."
                self.state.add_observation(
                    message=message,
                    observation_type=ta.ObservationType.GAME_MESSAGE
                )

            return done, info

        elif self.state.turn == conversation_turns + 1:
            if self.state.current_player_id != 1:
                self.state.set_invalid_move(
                    reason="Only the Guesser can make the final guess."
                )
                return self.state.step()

            self.state.add_observation(
                from_id=self.state.current_player_id,
                message=parsed_action,
                observation_type=ta.ObservationType.PLAYER_ACTION
            )

            # Check if the guess is in the correct format
            guess_fact1 = self.guess_fact1_pattern.search(parsed_action)
            guess_fact2 = self.guess_fact2_pattern.search(parsed_action)
            
            if guess_fact1 or guess_fact2:
                # Evaluate guess based on the REAL correct fact, not what the Deceiver was told
                if (
                    guess_fact1
                    and self.state.game_state["fact1"]["is_correct"]
                ) or (
                    guess_fact2
                    and self.state.game_state["fact2"]["is_correct"]
                ):
                    # Correct guess - Guesser wins
                    winner_id = self.state.current_player_id
                    reason = (
                        f"Player {self.state.current_player_id} guessed correct fact."
                    )
                else:
                    # Wrong guess - Deceiver wins
                    winner_id = 1 - self.state.current_player_id
                    reason = (
                        f"Player {self.state.current_player_id} guessed the wrong fact."
                    )

                # Set state winner
                self.state.set_winner(player_id=winner_id, reason=reason)
            else:
                reason = f"Player {self.state.current_player_id} did not make their guess in the correct format."
                self.state.set_invalid_move(reason=reason)

            return self.state.step()

        # Should not reach here
        else:
            self.state.add_observation(
                from_id=self.state.current_player_id,
                message=parsed_action,
                observation_type=ta.ObservationType.PLAYER_ACTION
            )
            return self.state.step()
