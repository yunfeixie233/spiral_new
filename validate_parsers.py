#!/usr/bin/env python3
"""
Validation script for game action parsers.

This script reads JSON game data and validates that the parsing functions
in spiral.agents.utils correctly extract available actions from observations.

Usage:
    python validate_parsers.py <path_to_game_state_directory>
"""

import json
import os
import glob
import sys
from collections import defaultdict
from spiral.agents.utils import (
    tic_tac_toe_parse_available_moves,
    kuhn_poker_parse_available_actions,
    simple_negotiation_parse_available_actions,
    pig_dice_parse_available_actions,
    briscola_parse_available_actions,
    colonel_blotto_parse_available_actions
)


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
    elif "Blotto" in observation:
        return "ColonelBlotto"
    return None


def get_parser_for_game(game_type):
    """Get the parser function for a given game type."""
    parsers = {
        'TicTacToe': tic_tac_toe_parse_available_moves,
        'KuhnPoker': kuhn_poker_parse_available_actions,
        'PigDice': pig_dice_parse_available_actions,
        'SimpleNegotiation': simple_negotiation_parse_available_actions,
        'Briscola': briscola_parse_available_actions,
        'ColonelBlotto': colonel_blotto_parse_available_actions
    }
    return parsers.get(game_type)


def validate_parsers(game_state_dir):
    """Validate all parsers against actual game data."""
    
    json_files = sorted(glob.glob(os.path.join(game_state_dir, "*.json")))
    
    if not json_files:
        print(f"ERROR: No JSON files found in {game_state_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print("=" * 80)
    
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
                        test_cases[game_type].append((observation, actual_action))
    
    # Validate each game type
    results = {}
    
    for game_type in sorted(test_cases.keys()):
        parser_func = get_parser_for_game(game_type)
        cases = test_cases[game_type]
        
        print(f"\n{'=' * 80}")
        print(f"Testing {game_type}")
        print('=' * 80)
        print(f"Test cases: {len(cases)}")
        
        total = 0
        valid = 0
        invalid = 0
        mismatches = []
        
        for obs, action in cases:
            total += 1
            
            # Skip invalid actions
            if action in ["[INVALID_ACTION]", "[｜INVALID_ACTION｜]"]:
                invalid += 1
                continue
            
            valid += 1
            parsed_actions = parser_func(obs)
            
            if action not in parsed_actions:
                mismatches.append((action, parsed_actions, obs[:200]))
        
        pass_rate = 0 if valid == 0 else ((valid - len(mismatches)) / valid * 100)
        
        print(f"  Valid actions: {valid}")
        print(f"  Invalid actions: {invalid}")
        print(f"  Mismatches: {len(mismatches)}")
        print(f"  Pass rate: {pass_rate:.1f}%")
        
        if len(mismatches) > 0:
            print(f"\n  Sample mismatches:")
            for i, (action, parsed, obs_snippet) in enumerate(mismatches[:3]):
                print(f"\n    {i+1}. Actual: {action}")
                print(f"       Parsed: {parsed[:5]}{'...' if len(parsed) > 5 else ''}")
        
        results[game_type] = {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'mismatches': len(mismatches),
            'pass_rate': pass_rate
        }
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print('=' * 80)
    
    for game_type, stats in sorted(results.items()):
        status = "PASS" if stats['pass_rate'] == 100.0 else "FAIL"
        print(f"{game_type:20s} {status:6s} ({stats['pass_rate']:5.1f}% pass rate, {stats['mismatches']}/{stats['valid']} mismatches)")
    
    print()
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_parsers.py <path_to_game_state_directory>")
        sys.exit(1)
    
    game_state_dir = sys.argv[1]
    
    if not os.path.isdir(game_state_dir):
        print(f"ERROR: {game_state_dir} is not a valid directory")
        sys.exit(1)
    
    validate_parsers(game_state_dir)

