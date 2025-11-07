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

import random
import re


def kuhn_poker_parse_available_actions(observation: str):
    # Parse the last line which contains the current available actions
    last_line = observation.strip().split("\n")[-1]
    available_actions = re.findall(r"\[(.*?)\]", last_line)
    # Add brackets
    available_actions = [f"[{action}]" for action in available_actions]
    # Remove [GAME]
    available_actions = [action for action in available_actions if action != "[GAME]"]
    return available_actions


def tic_tac_toe_parse_available_moves(observation: str):
    # Find the section after "Available Moves:" and before "Next Action:"
    moves_section_pattern = r"Available Moves:(.*?)Next Action:"
    moves_section = (
        re.search(moves_section_pattern, observation, re.DOTALL).group(1).strip()
    )

    # Now extract the moves from this section
    pattern = r"'\[(\d+)\]'"
    available_moves = re.findall(pattern, moves_section)

    available_moves = [f"[{move}]" for move in available_moves]

    return available_moves

def pig_dice_parse_available_actions(observation: str):
    """
    Parse available actions for Pig Dice game.
    
    The game sends messages like: "Available actions: '[roll]' or '[hold]'"
    PigDice always has the same two actions available: roll and hold.
    The environment only accepts the full forms via regex: r"\[(roll|hold)\]"
    
    Args:
        observation: The current game observation
        
    Returns:
        List of valid action strings that the environment accepts
    """
    # PigDice always has the same action space throughout the game
    # Return only the actions that match the environment's regex pattern
    return ["[roll]", "[hold]"]

def simple_negotiation_parse_available_actions(observation: str):
    valid_actions = []

    our_player_pattern = r"You are Player (\d+)"
    our_player_match = re.search(our_player_pattern, observation)
    if not our_player_match:
        return []

    our_player_id = int(our_player_match.group(1))

    # Parse current resources
    resources_pattern = r"\[(\w+)\]\s+Qty:\s+(\d+)"
    resources = {}

    for match in re.finditer(resources_pattern, observation):
        resource_name = match.group(1)
        quantity = int(match.group(2))
        resources[resource_name] = quantity

    # Check if there's a pending offer to us
    offer_to_us_pattern = r"Player (\d+) made the following offer to Player (\d+):"
    offer_matches = list(re.finditer(offer_to_us_pattern, observation))

    if offer_matches:
        last_offer = offer_matches[-1]
        from_player = int(last_offer.group(1))
        to_player = int(last_offer.group(2))

        # If the last offer is to us and hasn't been responded to
        if to_player == our_player_id:
            offer_position = last_offer.end()
            remaining_text = observation[offer_position:]

            # Check if this offer hasn't been responded to yet
            if not re.search(r"(accepted|denied|implicitly denied)", remaining_text):
                valid_actions.extend(
                    [
                        "[Accept]",
                        "[Deny]",
                    ]
                )

    # Generate ALL possible trade offers based on available resources
    if resources and len(resources) >= 2:
        resource_names = list(resources.keys())
        
        for offer_resource in resource_names:
            for request_resource in resource_names:
                if offer_resource == request_resource:
                    continue
                
                # Generate all possible quantities we can offer
                max_offer_qty = resources[offer_resource]
                for offer_qty in range(1, max_offer_qty + 1):
                    for request_qty in range(1, resources[request_resource] + 1):
                        offer_str = f"[Offer: {offer_qty} {offer_resource} -> {request_qty} {request_resource}]"
                        valid_actions.append(offer_str)

    valid_actions = list(dict.fromkeys(valid_actions))

    return valid_actions 


def briscola_parse_available_actions(observation: str):
    """
    Parse available actions for Briscola card game.
    
    Actions are always [play X] where X is 1 to hand_size.
    Hand size is typically 2-3 cards.
    """
    hand_pattern = r"Your hand:\s+((?:\s+\d+\.\s+[^\n]+\n?)+)"
    hand_match = re.search(hand_pattern, observation)
    
    if hand_match:
        hand_text = hand_match.group(1)
        card_lines = [line.strip() for line in hand_text.split('\n') if line.strip()]
        num_cards = len(card_lines)
        return [f"[play {i}]" for i in range(1, num_cards + 1)]
    
    return []


def colonel_blotto_parse_available_actions(observation: str):
    """
    Parse available actions for Colonel Blotto.
    
    Actions are allocations like [A4 B2 C14] where units must sum to total_units.
    This generates a reasonable subset of valid allocations.
    """
    units_pattern = r"Units to allocate:\s*(\d+)"
    fields_pattern = r"Available fields:\s*([A-Z, ]+)"
    
    units_match = re.search(units_pattern, observation)
    fields_match = re.search(fields_pattern, observation)
    
    total_units = int(units_match.group(1)) if units_match else 20
    fields_str = fields_match.group(1) if fields_match else "A, B, C"
    field_names = [f.strip() for f in fields_str.split(',')]
    
    valid_actions = []
    
    if len(field_names) == 3:
        for a in range(0, total_units + 1, 2):
            for b in range(0, total_units - a + 1, 2):
                c = total_units - a - b
                if c >= 0:
                    action = f"[{field_names[0]}{a} {field_names[1]}{b} {field_names[2]}{c}]"
                    valid_actions.append(action)
    
    valid_actions = valid_actions[::max(1, len(valid_actions) // 50)]
    
    return valid_actions


def indian_poker_parse_available_actions(observation: str):
    """
    Parse available actions for Indian Poker game.
    
    The game announces possible actions in one of two formats:
    - No bet to call: '[check]', '[bet X]'
    - Bet to call: '[call]' (cost X), '[raise X]', '[fold]'
    
    For '[bet X]' and '[raise X]', we generate reasonable amounts (1-10).
    
    Args:
        observation: The current game observation
        
    Returns:
        List of valid action strings
    """
    valid_actions = []
    
    # Find the line with "Your possible actions:"
    for line in observation.split('\n'):
        if 'possible actions' in line.lower():
            # Check which actions are available
            if '[check]' in line:
                valid_actions.append('[check]')
            
            if '[call]' in line:
                valid_actions.append('[call]')
            
            if '[fold]' in line:
                valid_actions.append('[fold]')
            
            # For '[bet X]' or '[raise X]', generate concrete amounts
            if '[bet X]' in line or '[bet' in line.lower():
                # Generate bet amounts from 1 to 10
                for amount in range(1, 11):
                    valid_actions.append(f'[bet {amount}]')
            
            if '[raise X]' in line or '[raise' in line.lower():
                # Generate raise amounts from 1 to 10
                for amount in range(1, 11):
                    valid_actions.append(f'[raise {amount}]')
            
            break
    
    return valid_actions

def truth_and_deception_parse_available_actions(observation: str):
    """
    Parse available actions for Truth and Deception game.
    
    The game has two phases:
    1. Conversation phase: Any text message is valid (return empty to skip validation)
    2. Guessing phase: Only [Fact 1] or [Fact 2] are valid
    
    Note: The model wraps messages in \boxed{...}, but extract_chat_action extracts the content.
    This function returns the valid extracted content (what's inside the box).
    The environment validates that the extracted content contains exactly [Fact 1] or [Fact 2].
    
    Args:
        observation: The current game observation
        
    Returns:
        List of valid action strings (extracted content, not including \boxed{})
    """
    # Check if we're in the guessing phase by looking for the guessing prompt
    if "Now guess which of the two facts are correct" in observation:
        # We're in the guessing phase - only these exact strings are valid
        # These are what should be INSIDE \boxed{}, i.e., model outputs \boxed{[Fact 1]}
        return ["[Fact 1]", "[Fact 2]"]
    else:
        # In conversation phase, return empty list (any message is valid)
        return []

def coup_parse_available_actions(observation: str):
    """
    Parse available actions for Coup game.
    
    Coup has multiple phases with different valid actions:
    1. Play phase: [income], [foreign aid], [coup X], [tax], [assassinate X], [steal X], [exchange]
    2. QueryForBlockOrChallenge: [PASS], [BULLSHIT], and possibly [block ...] actions
    3. QueryToChallengeTheBlocker: [PASS] or [BULLSHIT]
    4. QueryWhichToKeep: [keep card1 card2] or [keep card]
    
    Args:
        observation: The current game observation
        
    Returns:
        List of valid action strings
    """
    valid_actions = []
    
    # Extract only the current prompt (after last "It is now your turn.") to avoid matching history
    segments = observation.split('It is now your turn.')
    current_prompt = segments[-1] if len(segments) > 1 else observation
    
    # Phase 1: QueryWhichToKeep - after exchange
    if "You need to choose which" in current_prompt and "to keep" in current_prompt:
        # Parse which cards are available from "You now have: " line
        cards_pattern = r"You now have: (.+?)\."
        cards_match = re.search(cards_pattern, observation)
        if cards_match:
            cards_str = cards_match.group(1)
            # Extract card names
            card_names = re.findall(r"(Duke|Assassin|Ambassador|Captain|Contessa)", cards_str)
            
            # Determine if we need to keep 1 or 2 cards by checking the instruction
            # Look for "two cards to keep" or "[keep <card1> <card2>]" pattern
            if "two cards to keep" in observation or "<card2>" in observation:
                # Need to keep 2 cards - generate all pairs (including duplicates if player has them)
                from itertools import combinations_with_replacement
                # Use set of available cards to avoid impossible combinations
                unique_cards = sorted(set(card_names))
                for card1, card2 in combinations_with_replacement(unique_cards, 2):
                    valid_actions.append(f"[keep {card1} {card2}]")
            else:
                # Need to keep 1 card only - only unique cards
                for card in sorted(set(card_names)):
                    valid_actions.append(f"[keep {card}]")
        return valid_actions
    
    # Phase 2: QueryForBlockOrChallenge
    # Extract only the LAST question (after last "It is now your turn.") to avoid matching history
    # Split by "It is now your turn." and take the last segment
    segments = observation.split('It is now your turn.')
    current_prompt = segments[-1] if len(segments) > 1 else observation
    
    # Look for lines containing "Do you want to" - extract only up to the end of that line
    # This avoids matching error messages or previous actions
    do_you_want_match = re.search(r'(Player #\d+ is attempting[^\n]+).*?(Do you want to[^\?]*\?)', current_prompt, re.DOTALL)
    if do_you_want_match:
        # Combine the action description and the question
        question_text = do_you_want_match.group(1) + ' ' + do_you_want_match.group(2)
    elif "Do you want to" in current_prompt:
        # Fallback: find the line containing "Do you want to"
        for line in current_prompt.split('\n'):
            if "Do you want to" in line:
                question_text = line
                break
        else:
            question_text = current_prompt
    else:
        question_text = current_prompt
    
    if "Do you want to" in question_text:
        # Always can pass
        valid_actions.append("[PASS]")
        
        # Check if BULLSHIT is available - check in the question only
        if "[BULLSHIT]" in question_text or "call [BULLSHIT]" in question_text:
            valid_actions.append("[BULLSHIT]")
        
        # Check for block options - check in the question only
        if "[block foreign aid]" in question_text:
            valid_actions.append("[block foreign aid]")
        if "[block steal captain]" in question_text:
            valid_actions.append("[block steal captain]")
        if "[block steal ambassador]" in question_text:
            valid_actions.append("[block steal ambassador]")
        if "[block assassinate]" in question_text:
            valid_actions.append("[block assassinate]")
        
        return valid_actions
    
    # Phase 3: QueryToChallengeTheBlocker
    if "is blocking with" in current_prompt and "Do you want to call [BULLSHIT] or [PASS]" in current_prompt:
        return ["[BULLSHIT]", "[PASS]"]
    
    # Phase 4: Forced coup (10+ coins)
    if "You have 10 or more coins and must coup" in current_prompt:
        # Find all active players
        player_pattern = r"Player #(\d+) has (\d+) coins"
        players = []
        for match in re.finditer(player_pattern, observation):
            player_id = int(match.group(1))
            players.append(player_id)
        
        # Find our player id
        our_id_match = re.search(r"You are Player #(\d+)", observation)
        our_id = int(our_id_match.group(1)) if our_id_match else None
        
        # Generate coup actions for all other players
        for pid in players:
            if pid != our_id:
                valid_actions.append(f"[coup {pid}]")
        return valid_actions
    
    # Phase 5: Normal play phase
    if "What action do you want to take?" in current_prompt or current_prompt.strip().endswith("What action do you want to take?"):
        # Basic actions always available
        valid_actions.extend(["[income]", "[foreign aid]"])
        
        # Role-based actions
        valid_actions.extend(["[tax]", "[exchange]"])
        
        # Find all active players and our coins
        player_pattern = r"Player #(\d+) has (\d+) coins"
        players = []
        our_coins = 0
        
        our_id_match = re.search(r"You are Player #(\d+)\. You have (\d+) coins", observation)
        if our_id_match:
            our_id = int(our_id_match.group(1))
            our_coins = int(our_id_match.group(2))
        else:
            our_id = None
        
        for match in re.finditer(player_pattern, observation):
            player_id = int(match.group(1))
            if player_id != our_id:
                players.append(player_id)
        
        # Targeted actions - need at least 1 other player
        for pid in players:
            valid_actions.append(f"[steal {pid}]")
            if our_coins >= 3:
                valid_actions.append(f"[assassinate {pid}]")
            if our_coins >= 7:
                valid_actions.append(f"[coup {pid}]")
        
        return valid_actions
    
    # Default: return empty list if phase cannot be determined
    return []

def simple_tak_parse_available_actions(observation: str):
    """
    Parse available actions for SimpleTak game.
    
    SimpleTak is a connection game where players place stones on an NxN board.
    Actions are cell placements in format [N] where N is the cell index.
    
    The observation includes a line like:
    "Available Moves: [0], [1], [2], [5], ..."
    
    Args:
        observation: The current game observation
        
    Returns:
        List of valid action strings in format "[N]"
    """
    available_moves = []
    
    # Find the "Available Moves:" section
    moves_pattern = r"Available Moves:\s*(.+?)(?:\n|$)"
    moves_match = re.search(moves_pattern, observation)
    
    if moves_match:
        moves_section = moves_match.group(1)
        # Extract all moves in format [N]
        pattern = r"\[(\d+)\]"
        moves = re.findall(pattern, moves_section)
        available_moves = [f"[{move}]" for move in moves]
    
    return available_moves


_VALID_ACTION_PARSER = {
    "TicTacToe-v0": tic_tac_toe_parse_available_moves,
    "KuhnPoker-v1": kuhn_poker_parse_available_actions,
    "SimpleNegotiation-v1": simple_negotiation_parse_available_actions,
    "PigDice-v1": pig_dice_parse_available_actions,
    "TicTacToe-v1": tic_tac_toe_parse_available_moves,
    "Briscola-v1": briscola_parse_available_actions,
    "ColonelBlotto-v1": colonel_blotto_parse_available_actions,
    "IndianPoker-v1": indian_poker_parse_available_actions,
    "SimpleTak-v0": simple_tak_parse_available_actions,
    "TruthAndDeception-v2": truth_and_deception_parse_available_actions,
    "Coup-v0": coup_parse_available_actions,
}


def get_valid_action_parser(env_id: str):
    try:
        return _VALID_ACTION_PARSER[env_id]
    except KeyError:
        raise NotImplementedError(f"valid action parser not implemented for {env_id}")
