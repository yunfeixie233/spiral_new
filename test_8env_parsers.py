#!/usr/bin/env python3
"""
Test script for all parsing functions against the 8-environment dataset.
"""

import json
import os
import glob
from collections import defaultdict
from spiral.agents.utils import (
    tic_tac_toe_parse_available_moves,
    kuhn_poker_parse_available_actions,
    simple_negotiation_parse_available_actions,
    pig_dice_parse_available_actions,
    briscola_parse_available_actions,
    colonel_blotto_parse_available_actions
)

# Read all JSON files
game_state_dir = "/ephemeral/games-workspace/spiral_new/oat-output/spiral-qwen3-4b-base-8env-self-play_1018T03:51:06/game_state"
json_files = sorted(glob.glob(os.path.join(game_state_dir, "*.json")))

print("=" * 80)
print("TESTING PARSERS ON 8-ENVIRONMENT DATASET")
print("=" * 80)
print(f"Found {len(json_files)} JSON files\n")

# Detect game type from observation
def detect_game_type(observation):
    """Detect the game type from the observation text."""
    if "Tic Tac Toe" in observation:
        return "TicTacToe"
    elif "Kuhn Poker" in observation:
        return "KuhnPoker"
    elif "Pig Dice" in observation:
        return "PigDice"
    elif "Negotiation" in observation:
        return "SimpleNegotiation"
    elif "Briscola" in observation:
        return "Briscola"
    elif "Colonel" in observation or "Blotto" in observation or "Commander" in observation:
        return "ColonelBlotto"
    elif "Indian Poker" in observation:
        return "IndianPoker"
    return None

# Collect test cases by game type
test_cases = defaultdict(list)

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        for record in data:
            for entry in record['history']:
                observation = entry[1]
                actual_action = entry[2]
                game_type = detect_game_type(observation)
                
                if game_type:
                    test_cases[game_type].append((observation, actual_action, os.path.basename(json_file)))

# Define parsers for each game
from spiral.agents.utils import indian_poker_parse_available_actions

parsers = {
    'TicTacToe': tic_tac_toe_parse_available_moves,
    'KuhnPoker': kuhn_poker_parse_available_actions,
    'PigDice': pig_dice_parse_available_actions,
    'SimpleNegotiation': simple_negotiation_parse_available_actions,
    'Briscola': briscola_parse_available_actions,
    'ColonelBlotto': colonel_blotto_parse_available_actions,
    'IndianPoker': indian_poker_parse_available_actions,
}

# Test each game type
results = {}

for game_type in sorted(test_cases.keys()):
    print(f"{'=' * 80}")
    print(f"Testing {game_type}")
    print('=' * 80)
    
    cases = test_cases[game_type]
    print(f"Total test cases: {len(cases)}")
    
    # Check if we have a parser for this game
    if game_type not in parsers:
        print(f"  WARNING: No parser implemented for {game_type}")
        print()
        continue
    
    parser_func = parsers[game_type]
    
    total = 0
    valid = 0
    invalid = 0
    pass_count = 0
    fail_count = 0
    failures = []
    
    for obs, action, filename in cases:
        total += 1
        
        # Skip invalid actions
        if action in ["[INVALID_ACTION]", "[｜INVALID_ACTION｜]"]:
            invalid += 1
            continue
        
        valid += 1
        parsed_actions = parser_func(obs)
        
        if action in parsed_actions:
            pass_count += 1
        else:
            fail_count += 1
            failures.append((action, parsed_actions, obs[:300], filename))
    
    # Calculate pass rate
    pass_rate = 0 if valid == 0 else (pass_count / valid * 100)
    
    print(f"  Valid actions: {valid}")
    print(f"  Invalid actions: {invalid}")
    print(f"  Pass: {pass_count}")
    print(f"  Fail: {fail_count}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    
    # Show sample failures
    if failures:
        print(f"\n  Sample failures (first 3):")
        for i, (action, parsed, obs_snippet, filename) in enumerate(failures[:3]):
            print(f"\n    Failure {i+1} (from {filename}):")
            print(f"      Actual action: {action}")
            if len(parsed) > 10:
                print(f"      Num parsed actions: {len(parsed)}")
                print(f"      First 5 parsed: {parsed[:5]}")
            else:
                print(f"      Parsed actions: {parsed}")
            print(f"      Observation snippet: {obs_snippet}...")
    else:
        print(f"\n  SUCCESS: All valid actions parsed correctly!")
    
    results[game_type] = {
        'total': total,
        'valid': valid,
        'invalid': invalid,
        'pass': pass_count,
        'fail': fail_count,
        'pass_rate': pass_rate
    }
    
    print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)

for game_type in sorted(results.keys()):
    stats = results[game_type]
    status = "PASS" if stats['pass_rate'] == 100.0 else "FAIL"
    print(f"{game_type:20s} {status:6s} ({stats['pass_rate']:5.1f}% pass rate, {stats['fail']}/{stats['valid']} failures)")

# Check for games without parsers
games_without_parsers = set(test_cases.keys()) - set(parsers.keys())
if games_without_parsers:
    print(f"\nGames without parsers: {', '.join(sorted(games_without_parsers))}")

print("\n" + "=" * 80)
print("DETAILED OBSERVATIONS")
print("=" * 80)

# Analyze specific issues
for game_type, stats in sorted(results.items()):
    if stats['fail'] > 0 and game_type in test_cases:
        print(f"\n{game_type}:")
        
        # Sample failed actions
        failed_actions = []
        parser_func = parsers[game_type]
        
        for obs, action, filename in test_cases[game_type]:
            if action not in ["[INVALID_ACTION]", "[｜INVALID_ACTION｜]"]:
                parsed = parser_func(obs)
                if action not in parsed:
                    failed_actions.append(action)
                    if len(failed_actions) >= 3:
                        break
        
        if failed_actions:
            print(f"  Sample failed actions:")
            for action in failed_actions:
                print(f"    - {action}")

print()

