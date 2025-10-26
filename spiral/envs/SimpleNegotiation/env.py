import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.SimpleNegotiation.renderer import create_board_str


class SimpleNegotiationEnv(ta.Env):
    """Environment for the SimpleNegotiation Game."""

    def __init__(self, max_turns: Optional[int] = 10):
        self.max_turns = max_turns
        self.resource_names = ["Wood", "Gold"]
        self.base_values = {"Wood": 10, "Gold": 30}

        self.accept_pattern = re.compile(r"\[Accept\]", re.IGNORECASE)
        self.deny_pattern = re.compile(r"\[Deny\]", re.IGNORECASE)
        self.offer_pattern = re.compile(
            r"\[Offer:\s*(?:I\s+(?:give|offer)\s+)?([^\[\]]+?)\s*\.*\]",
            re.IGNORECASE | re.DOTALL,
        )

    def get_board_str(self):
        return create_board_str(
            player_resources=self.state.game_state["player_resources"],
            player_values=self.state.game_state["player_values"],
            inventory_values=self.state.game_state["inventory_value"],
            current_offer=self.state.game_state["current_offer"],
        )

    def reset(self, num_players: int, seed: Optional[int] = None):
        """Reset the Negotiation Game to its initial state"""
        self.state = ta.TwoPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)

        game_state = {
            "current_offer": None,
            "player_resources": {
                0: {"Wood": 10, "Gold": 10},
                1: {"Wood": 10, "Gold": 10},
            },
            "player_values": {},
            "trade_history": [],
        }

        game_state["player_values"][0] = {"Wood": 5, "Gold": 15}
        game_state["player_values"][1] = {"Wood": 15, "Gold": 5}

        for player_id in [0, 1]:
            initial_value = self._calculate_player_inventory_value(
                player_id, game_state
            )
            game_state.setdefault("inventory_value", {})[player_id] = {
                "initial": initial_value,
                "current": initial_value,
                "change": 0,
            }

        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._prompt,
        )

    def _prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """Generate the initial prompt for a player"""
        resource_value_list = "\n\t+ ".join(
            [
                f"{f'[{res}]':{' '}<8}  Qty: {game_state['player_resources'][player_id][res]:{' '}<2}   Value: {game_state['player_values'][player_id][res]}"
                for res in game_state["player_resources"][player_id].keys()
            ]
        )
        prompt = (
            f"You are Player {player_id} in the Negotiation Game.\n"
            "You have some resources, and your task is to trade such that the total value of your resources increases.\n"
            f"The resources and associated values you currently have are:\n\t+ "
            f"{resource_value_list}\n"
            "At each turn, you can talk to your opponent or make a trade offer.\n"
            "Use the following special tokens for actions:\n"
            "  - [Offer]: To make a trade offer.\n"
            "    Format: [Offer: Offered Resources -> Requested Resources]\n"
            "    Example: [Offer: 3 Wood -> 2 Gold]\n"
            "  - [Accept]: To accept an incoming offer.\n"
            "  - [Deny]: To deny an incoming offer (default).\n"
            "YOU CAN INCLUDE ADDITIONAL TEXT BEFORE AND/OR AFTER THESE TOKENS.\n"
            '    Example: "I\'m open to negotiation."\n'
        )
        if self.state.max_turns:
            prompt += f"The game lasts for {self.state.max_turns} turns in total.\n"
        else:
            prompt += "The game has no turn limit.\n"
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """Process the player's action"""
        self.state.add_observation(
            from_id=self.state.current_player_id,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION
        )

        self._check_and_execute_existing_offer(
            player_id=self.state.current_player_id, action=action
        )

        self._check_for_new_offer(player_id=self.state.current_player_id, action=action)

        if self.state.check_turn_limit():
            self._determine_winner()

        return self.state.step()

    def _check_and_execute_existing_offer(self, player_id: int, action: str) -> None:
        """
        Check if the player accepts or denies the current offer and execute accordingly.

        Args:
            player_id (int): ID of the player responding to the offer.
            action (str): The action string.
        """
        current_offer = self.state.game_state.get("current_offer")

        if not current_offer:
            return

        if player_id != current_offer["to_player"]:
            if self.accept_pattern.search(action):
                reason = f"Player {player_id} tried to accept an offer that was not made to them."
                self.state.set_invalid_move(reason=reason)
            return

        if self.accept_pattern.search(action):
            self._attempt_to_execute_trade(player_id=player_id, action=action)
        elif self.deny_pattern.search(action):
            self.state.add_observation(
                message=f"Player {player_id} denied the trade offer from Player {current_offer['from_player']}.",
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
            )
            self.state.game_state["trade_history"].append(
                {
                    "from_player": current_offer["from_player"],
                    "to_player": player_id,
                    "offered_resources": current_offer["offered_resources"],
                    "requested_resources": current_offer["requested_resources"],
                    "outcome": "Denied",
                }
            )
            self.state.game_state["current_offer"] = None

    def _attempt_to_execute_trade(self, player_id: int, action: str) -> None:
        """
        Attempt to execute the trade if both players have sufficient resources.

        Args:
            player_id (int): ID of the player accepting the offer.
            action (str): The action string.
        """
        current_offer = self.state.game_state["current_offer"]
        proposer_id = current_offer["from_player"]
        acceptor_id = player_id

        proposer_has_resources = self._check_if_sufficient_resources(
            trade_resources=current_offer["offered_resources"],
            player_resources=self.state.game_state["player_resources"][proposer_id],
        )
        acceptor_has_resources = self._check_if_sufficient_resources(
            trade_resources=current_offer["requested_resources"],
            player_resources=self.state.game_state["player_resources"][acceptor_id],
        )

        if proposer_has_resources and acceptor_has_resources:
            for resource, qty in current_offer["offered_resources"].items():
                self.state.game_state["player_resources"][proposer_id][resource] -= qty
                self.state.game_state["player_resources"][acceptor_id][resource] += qty
            for resource, qty in current_offer["requested_resources"].items():
                self.state.game_state["player_resources"][acceptor_id][resource] -= qty
                self.state.game_state["player_resources"][proposer_id][resource] += qty

            self.state.add_observation(
                message=f"Player {acceptor_id} accepted the trade offer from Player {proposer_id}.",
                observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
            )

            self.state.game_state["trade_history"].append(
                {
                    "from_player": proposer_id,
                    "to_player": acceptor_id,
                    "offered_resources": current_offer["offered_resources"],
                    "requested_resources": current_offer["requested_resources"],
                    "outcome": "Accepted",
                }
            )

            self._update_inventory_values()

            self.state.game_state["current_offer"] = None

        else:
            if not acceptor_has_resources and not proposer_has_resources:
                reason = f"Neither player has sufficient resources for the trade."
                self.state.add_observation(
                    message=f"Trade cancelled: {reason}",
                    observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                )
                self.state.game_state["current_offer"] = None
            elif not acceptor_has_resources:
                reason = f"Player {acceptor_id} tried accepting a trade without having the necessary resources."
                self.state.set_invalid_move(reason=reason)
            else:
                reason = f"Trade cancelled: Player {proposer_id} no longer has the resources they offered."
                self.state.add_observation(
                    message=reason,
                    observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                )
                self.state.game_state["current_offer"] = None

    def _check_if_sufficient_resources(
        self, trade_resources: Dict[str, int], player_resources: Dict[str, int]
    ) -> bool:
        """
        Check if a player has sufficient resources for a trade.

        Args:
            trade_resources (Dict[str, int]): Resources required for the trade.
            player_resources (Dict[str, int]): Player's current resources.

        Returns:
            bool: True if sufficient, False otherwise.
        """
        for resource, qty in trade_resources.items():
            if player_resources.get(resource, 0) < qty:
                return False
        return True

    def _check_for_new_offer(self, player_id: int, action: str):
        """
        Check if the player's action contains a new trade offer.

        Args:
            player_id (int): ID of the player making the offer.
            action (str): The action string.
        """
        if not self.state.done:
            offer_match = self.offer_pattern.search(action)
            if offer_match:
                matched_offer = offer_match.group(1).strip()
                parsed_offer = self._parse_offer(matched_offer)
                if parsed_offer:

                    if self._check_if_sufficient_resources(
                        trade_resources=parsed_offer["offered_resources"],
                        player_resources=self.state.game_state["player_resources"][
                            player_id
                        ],
                    ):
                        if self.state.game_state["current_offer"] is not None:
                            prev_offer = self.state.game_state["current_offer"]
                            if prev_offer["to_player"] == player_id:
                                self.state.add_observation(
                                    message=f"Player {player_id} implicitly denied the previous offer by making a counter-offer.",
                                    observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                                )
                                self.state.game_state["trade_history"].append(
                                    {
                                        "from_player": prev_offer["from_player"],
                                        "to_player": player_id,
                                        "offered_resources": prev_offer[
                                            "offered_resources"
                                        ],
                                        "requested_resources": prev_offer[
                                            "requested_resources"
                                        ],
                                        "outcome": "Implicitly Denied",
                                    }
                                )

                        self.state.game_state["current_offer"] = {
                            "from_player": player_id,
                            "to_player": 1 - player_id,
                            "offered_resources": parsed_offer["offered_resources"],
                            "requested_resources": parsed_offer["requested_resources"],
                        }

                        message = f"Player {player_id} made the following offer to Player {1 - player_id}: {self._offer_to_str(parsed_offer)}"
                        self.state.add_observation(
                            message=message,
                            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                        )
                    else:
                        reason = f"Player {player_id} tried to make a trade offer without having the necessary resources."
                        self.state.set_invalid_move(reason=reason)
                else:
                    reason = (
                        f"Player {player_id} made a trade offer in an incorrect format."
                    )
                    self.state.set_invalid_move(reason=reason)

    def _parse_offer(self, offer_str: str) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Parse a trade offer string into a structured dictionary.

        Args:
            offer_str (str): The offer string extracted from the action.

        Returns:
            Optional[Dict[str, Dict[str, int]]]: Parsed offer details or None if parsing fails.
        """
        offer_str = " ".join(offer_str.split())
        offer_str = re.sub(r"[.,!?]+$", "", offer_str)
        offer_str = re.sub(
            r"^(I\s+(?:give|offer)\s+)", "", offer_str, flags=re.IGNORECASE
        )
        offer_parts = re.split(r"\s*->\s*", offer_str)
        if len(offer_parts) != 2:
            return None

        offered_items_str = offer_parts[0].strip()
        requested_items_str = offer_parts[1].strip()

        offered_items = self._parse_resource_list(offered_items_str)
        requested_items = self._parse_resource_list(requested_items_str)

        if not offered_items or not requested_items:
            return None

        return {
            "offered_resources": offered_items,
            "requested_resources": requested_items,
        }

    def _parse_resource_list(self, resource_str: str) -> Optional[Dict[str, int]]:
        """
        Parse a string of resources and quantities into a dictionary.

        Args:
            resource_str (str): String containing resources, e.g., "2 Wood, 1 Gold".

        Returns:
            Optional[Dict[str, int]]: Parsed resources or None if parsing fails.
        """
        resource_list = re.split(r",\s*|\s+and\s+", resource_str, flags=re.IGNORECASE)
        resources = {}
        for item in resource_list:
            item = item.strip()
            if not item:
                continue
            match = re.match(r"^(\d+)\s+(.+)$", item)
            if not match:
                return None
            qty_str, resource_name = match.groups()
            qty = int(qty_str)
            resource_name = resource_name.strip().title()
            resource_aliases = {
                "Woods": "Wood",
                "Golds": "Gold",
            }
            resource_name = resource_aliases.get(resource_name, resource_name)
            if resource_name not in self.resource_names or qty <= 0:
                return None
            if resource_name in resources:
                resources[resource_name] += qty
            else:
                resources[resource_name] = qty
        return resources

    def _offer_to_str(self, parsed_offer: Dict[str, Dict[str, int]]) -> str:
        """
        Convert a parsed offer dictionary to a readable string format.

        Args:
            parsed_offer (Dict[str, Dict[str, int]]): Parsed offer details.

        Returns:
            str: Readable string representation of the offer.
        """
        offered = ", ".join(
            f"{qty} {res}" for res, qty in parsed_offer["offered_resources"].items()
        )
        requested = ", ".join(
            f"{qty} {res}" for res, qty in parsed_offer["requested_resources"].items()
        )
        return f"Offered items: {offered} -> Requested items: {requested}"

    def _determine_winner(self):
        """Determine the winner based on the change in inventory values"""
        if not self.state.done:
            self._update_inventory_values()

            if (
                self.state.game_state["inventory_value"][0]["change"]
                == self.state.game_state["inventory_value"][1]["change"]
            ):
                self.state.set_draw(
                    reason=f"Same change in inventory value for all players. Draw."
                )
            else:
                winner_id = (
                    0
                    if (
                        self.state.game_state["inventory_value"][0]["change"]
                        > self.state.game_state["inventory_value"][1]["change"]
                    )
                    else 1
                )
                reason = f"Player {winner_id} won by having a larger gain in inventory value."
                self.state.set_winner(player_id=winner_id, reason=reason)

    def _update_inventory_values(self):
        """Update the current inventory values and their changes for both players"""
        for player_id in range(self.state.num_players):
            current_inventory_value = self._calculate_player_inventory_value(
                player_id=player_id, game_state=self.state.game_state
            )

            self.state.game_state["inventory_value"][player_id][
                "current"
            ] = current_inventory_value
            self.state.game_state["inventory_value"][player_id]["change"] = (
                current_inventory_value
                - self.state.game_state["inventory_value"][player_id]["initial"]
            )

    def _calculate_player_inventory_value(
        self, player_id: int, game_state: Dict[str, Any]
    ) -> float:
        """
        Calculate the total inventory value for a player.

        Args:
            player_id (int): ID of the player.
            game_state (Dict[str, Any]): Current game state.

        Returns:
            float: Total inventory value.
        """
        resources = game_state["player_resources"][player_id]
        values = game_state["player_values"][player_id]
        inventory_value = sum([qty * values[res] for res, qty in resources.items()])
        return inventory_value

