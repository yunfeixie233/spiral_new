import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta
from textarena.envs.SimpleNegotiation.renderer import create_board_str


class SimpleNegotiationEnv(ta.Env):

    def __init__(self, max_turns: Optional[int] = 10):
        self.max_turns = max_turns
        # Simplified to 2 resources (from env_old)
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
        # Use TwoPlayerState interface from env_new
        self.state = ta.TwoPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)

        # Simplified starting resources - both players start with same quantities (from env_old)
        game_state = {
            "current_offer": None,
            "player_resources": {
                0: {"Wood": 10, "Gold": 10},
                1: {"Wood": 10, "Gold": 10},
            },
            "player_values": {},
            "trade_history": [],
        }

        # Generate opposite preferences for players (from env_old)
        # Player 0 prefers Gold (Wood=5, Gold=15)
        # Player 1 prefers Wood (Wood=15, Gold=5)
        game_state["player_values"][0] = {"Wood": 5, "Gold": 15}
        game_state["player_values"][1] = {"Wood": 15, "Gold": 5}

        # Keep track of the inventory (both initial and current)
        for player_id in [0, 1]:
            initial_value = self._calculate_player_inventory_value(
                player_id, game_state
            )
            game_state.setdefault("inventory_value", {})[player_id] = {
                "initial": initial_value,
                "current": initial_value,
                "change": 0,
            }

        # Use env_new interface - no seed parameter in state.reset(), use _prompt function name
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._prompt,
        )

    def _prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
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
        # Use env_new interface - observation_type instead of from_id/to_id
        self.state.add_observation(
            from_id=self.state.current_player_id,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION
        )

        # Check if the player is responding to an existing offer
        self._check_and_execute_existing_offer(
            player_id=self.state.current_player_id, action=action
        )

        # Check if the player's action contains a new trade offer
        self._check_for_new_offer(player_id=self.state.current_player_id, action=action)

        # Use env_new interface - check_turn_limit() method
        if self.state.check_turn_limit():
            self._determine_winner()

        return self.state.step()

    def _check_and_execute_existing_offer(self, player_id: int, action: str) -> None:
        # check if there is a current offer (from env_old)
        current_offer = self.state.game_state.get("current_offer")

        if not current_offer:
            return

        # Check if player is the recipient of the offer (from env_old)
        if player_id != current_offer["to_player"]:
            # Player cannot accept/deny an offer not made to them
            if self.accept_pattern.search(action):
                reason = f"Player {player_id} tried to accept an offer that was not made to them."
                # Use env_new interface - no player_id parameter
                self.state.set_invalid_move(reason=reason)
            return

        # Check if offer was accepted (from env_old)
        if self.accept_pattern.search(action):
            self._attempt_to_execute_trade(player_id=player_id, action=action)
        elif self.deny_pattern.search(action):
            # Explicit deny (from env_old)
            # Use env_new interface - observation_type instead of from_id/to_id
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
        current_offer = self.state.game_state["current_offer"]
        proposer_id = current_offer["from_player"]
        acceptor_id = player_id

        # Check if BOTH players have sufficient resources (from env_old)
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

            # Use env_new interface - observation_type
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

            # Reset trade offer
            self.state.game_state["current_offer"] = None

        else:
            # From env_old - handle different cases of insufficient resources
            if not acceptor_has_resources and not proposer_has_resources:
                reason = f"Neither player has sufficient resources for the trade."
                # This is a special case - the game state changed, so we just cancel the trade
                # Use env_new interface
                self.state.add_observation(
                    message=f"Trade cancelled: {reason}",
                    observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                )
                self.state.game_state["current_offer"] = None
            elif not acceptor_has_resources:
                reason = f"Player {acceptor_id} tried accepting a trade without having the necessary resources."
                # Use env_new interface - no player_id parameter
                self.state.set_invalid_move(reason=reason)
            else:
                # Proposer doesn't have resources - this shouldn't penalize the acceptor
                reason = f"Trade cancelled: Player {proposer_id} no longer has the resources they offered."
                # Use env_new interface
                self.state.add_observation(
                    message=reason,
                    observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                )
                self.state.game_state["current_offer"] = None

    def _check_if_sufficient_resources(
        self, trade_resources: Dict[str, int], player_resources: Dict[str, int]
    ) -> bool:
        for resource, qty in trade_resources.items():
            if player_resources.get(resource, 0) < qty:
                return False
        return True

    def _check_for_new_offer(self, player_id: int, action: str):
        # Check if the game has already done
        if not self.state.done:
            offer_match = self.offer_pattern.search(action)
            if offer_match:
                matched_offer = offer_match.group(1).strip()
                parsed_offer = self._parse_offer(matched_offer)
                if parsed_offer:

                    # check if necessary resources
                    if self._check_if_sufficient_resources(
                        trade_resources=parsed_offer["offered_resources"],
                        player_resources=self.state.game_state["player_resources"][
                            player_id
                        ],
                    ):
                        # From env_old - handle counter-offers
                        if self.state.game_state["current_offer"] is not None:
                            prev_offer = self.state.game_state["current_offer"]
                            if prev_offer["to_player"] == player_id:
                                # Use env_new interface
                                self.state.add_observation(
                                    message=f"Player {player_id} implicitly denied the previous offer by making a counter-offer.",
                                    observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                                )
                                # Add to trade history
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

                        # Add the offer to the game state with consistent keys
                        self.state.game_state["current_offer"] = {
                            "from_player": player_id,
                            "to_player": 1 - player_id,
                            "offered_resources": parsed_offer["offered_resources"],
                            "requested_resources": parsed_offer["requested_resources"],
                        }

                        message = f"Player {player_id} made the following offer to Player {1 - player_id}: {self._offer_to_str(parsed_offer)}"
                        # Use env_new interface
                        self.state.add_observation(
                            message=message,
                            observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION
                        )
                    else:
                        reason = f"Player {player_id} tried to make a trade offer without having the necessary resources."
                        # Use env_new interface - no player_id parameter
                        self.state.set_invalid_move(reason=reason)
                else:
                    # Erroneous offer
                    reason = (
                        f"Player {player_id} made a trade offer in an incorrect format."
                    )
                    # Use env_new interface - no player_id parameter
                    self.state.set_invalid_move(reason=reason)

    def _parse_offer(self, offer_str: str) -> Optional[Dict[str, Dict[str, int]]]:
        # From env_old - use the old parsing logic
        offer_str = " ".join(offer_str.split())
        offer_str = re.sub(r"[.,!?]+$", "", offer_str)
        offer_str = re.sub(
            r"^(I\s+(?:give|offer)\s+)", "", offer_str, flags=re.IGNORECASE
        )
        offer_parts = re.split(r"\s*->\s*", offer_str)
        if len(offer_parts) != 2:
            return None  # Erroneous offer

        offered_items_str = offer_parts[0].strip()
        requested_items_str = offer_parts[1].strip()

        offered_items = self._parse_resource_list(offered_items_str)
        requested_items = self._parse_resource_list(requested_items_str)

        if not offered_items or not requested_items:
            return None  # Erroneous offer

        return {
            "offered_resources": offered_items,
            "requested_resources": requested_items,
        }

    def _parse_resource_list(self, resource_str: str) -> Optional[Dict[str, int]]:
        # From env_old - use the old parsing logic
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
            resource_name = (
                resource_name.strip().title()
            )  # Ensure consistent casing
            # Handle resource aliases if any
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
        offered = ", ".join(
            f"{qty} {res}" for res, qty in parsed_offer["offered_resources"].items()
        )
        requested = ", ".join(
            f"{qty} {res}" for res, qty in parsed_offer["requested_resources"].items()
        )
        return f"Offered items: {offered} -> Requested items: {requested}"

    def _determine_winner(self):
        # From env_old - determine winner logic
        if not self.state.done:
            self._update_inventory_values()

            if (
                self.state.game_state["inventory_value"][0]["change"]
                == self.state.game_state["inventory_value"][1]["change"]
            ):
                # Draw
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
                # Use env_new interface - set_winner instead of set_winners
                self.state.set_winner(player_id=winner_id, reason=reason)

    def _update_inventory_values(self):
        for player_id in range(self.state.num_players):
            # Calculate current inventory value
            current_inventory_value = self._calculate_player_inventory_value(
                player_id=player_id, game_state=self.state.game_state
            )

            # Update
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
        resources = game_state["player_resources"][player_id]
        values = game_state["player_values"][player_id]
        inventory_value = sum([qty * values[res] for res, qty in resources.items()])
        return inventory_value

