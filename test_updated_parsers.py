#!/usr/bin/env python3
"""
Test script for updated Simple Negotiation and Kuhn Poker parsers.
"""

import json
import os
import glob
from spiral.agents.utils import (
    simple_negotiation_parse_available_actions,
    kuhn_poker_parse_available_actions,
)

# Read all JSON files
game_state_dir = "/ephemeral/games-workspace/spiral_new/oat-output/spiral-qwen3-4b-base-kp-ttt-4k-self-play_1018T00:54:38/game_state"
json_files = sorted(glob.glob(os.path.join(game_state_dir, "*.json")))

print("=" * 80)
print("TESTING UPDATED PARSERS")
print("=" * 80)
print(f"Found {len(json_files)} JSON files\n")

# Collect test cases
kuhn_poker_cases = []
negotiation_cases = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        for record in data:
            for entry in record['history']:
                observation = entry[1]
                actual_action = entry[2]
                
                if "Kuhn Poker" in observation:
                    kuhn_poker_cases.append((observation, actual_action))
                elif "Negotiation" in observation:
                    negotiation_cases.append((observation, actual_action))

# Test Kuhn Poker Parser
print("=" * 80)
print("KUHN POKER PARSER TEST")
print("=" * 80)
print(f"Total test cases: {len(kuhn_poker_cases)}")

kp_total = 0
kp_valid = 0
kp_invalid = 0
kp_pass = 0
kp_fail = 0
kp_failures = []

for obs, action in kuhn_poker_cases:
    kp_total += 1
    
    if action in ["[INVALID_ACTION]", "[｜INVALID_ACTION｜]"]:
        kp_invalid += 1
        continue
    
    kp_valid += 1
    parsed = kuhn_poker_parse_available_actions(obs)
    
    if action in parsed:
        kp_pass += 1
    else:
        kp_fail += 1
        kp_failures.append((action, parsed, obs[:300]))

print(f"  Total: {kp_total}")
print(f"  Valid actions: {kp_valid}")
print(f"  Invalid actions: {kp_invalid}")
print(f"  Pass: {kp_pass}")
print(f"  Fail: {kp_fail}")

if kp_valid > 0:
    print(f"  Pass rate: {kp_pass / kp_valid * 100:.1f}%")

if kp_failures:
    print(f"\n  Sample failures:")
    for i, (action, parsed, obs_snippet) in enumerate(kp_failures[:3]):
        print(f"\n    Failure {i+1}:")
        print(f"      Actual action: {action}")
        print(f"      Parsed actions: {parsed}")
        print(f"      Observation snippet: {obs_snippet}...")

# Test Simple Negotiation Parser
print("\n" + "=" * 80)
print("SIMPLE NEGOTIATION PARSER TEST")
print("=" * 80)
print(f"Total test cases: {len(negotiation_cases)}")

sn_total = 0
sn_valid = 0
sn_invalid = 0
sn_pass = 0
sn_fail = 0
sn_failures = []

for obs, action in negotiation_cases:
    sn_total += 1
    
    if action in ["[INVALID_ACTION]", "[｜INVALID_ACTION｜]"]:
        sn_invalid += 1
        continue
    
    sn_valid += 1
    parsed = simple_negotiation_parse_available_actions(obs)
    
    if action in parsed:
        sn_pass += 1
    else:
        sn_fail += 1
        sn_failures.append((action, parsed, obs[:300]))

print(f"  Total: {sn_total}")
print(f"  Valid actions: {sn_valid}")
print(f"  Invalid actions: {sn_invalid}")
print(f"  Pass: {sn_pass}")
print(f"  Fail: {sn_fail}")

if sn_valid > 0:
    print(f"  Pass rate: {sn_pass / sn_valid * 100:.1f}%")

if sn_failures:
    print(f"\n  Sample failures (first 5):")
    for i, (action, parsed, obs_snippet) in enumerate(sn_failures[:5]):
        print(f"\n    Failure {i+1}:")
        print(f"      Actual action: {action}")
        print(f"      Num parsed actions: {len(parsed)}")
        print(f"      First 5 parsed: {parsed[:5]}")
        print(f"      Observation snippet: {obs_snippet}...")

# Detailed analysis of failures
print("\n" + "=" * 80)
print("DETAILED FAILURE ANALYSIS - SIMPLE NEGOTIATION")
print("=" * 80)

if sn_failures:
    # Categorize failures
    offer_failures = []
    accept_failures = []
    other_failures = []
    
    for action, parsed, obs in sn_failures:
        if action.startswith("[Offer:"):
            offer_failures.append((action, parsed, obs))
        elif action == "[Accept]":
            accept_failures.append((action, parsed, obs))
        else:
            other_failures.append((action, parsed, obs))
    
    print(f"\nFailure breakdown:")
    print(f"  Offer failures: {len(offer_failures)}")
    print(f"  Accept failures: {len(accept_failures)}")
    print(f"  Other failures: {len(other_failures)}")
    
    # Analyze offer failures
    if offer_failures:
        print(f"\n  Analyzing offer failures:")
        print(f"  Sample offer failure:")
        action, parsed, obs = offer_failures[0]
        print(f"    Actual: {action}")
        
        # Check if it's a multi-resource offer
        if "," in action:
            print(f"    -> Multi-resource offer detected")
            print(f"    -> Parser may not support multi-resource offers")
        
        # Extract resources from observation
        import re
        resources_pattern = r"\[(\w+)\]\s+Qty:\s+(\d+)"
        resources_in_obs = dict(re.findall(resources_pattern, obs))
        print(f"    Resources in observation: {resources_in_obs}")
        
        # Extract resources from action
        offer_pattern = r"\[Offer: (.*?) -> (.*?)\]"
        match = re.search(offer_pattern, action)
        if match:
            offered = match.group(1)
            requested = match.group(2)
            print(f"    Offered: {offered}")
            print(f"    Requested: {requested}")
            
            # Check if resources exist
            for item in offered.split(","):
                item = item.strip()
                parts = item.split()
                if len(parts) >= 2:
                    resource = parts[1]
                    if resource not in resources_in_obs:
                        print(f"    WARNING: Resource '{resource}' not in observation (hallucinated)")
    
    # Analyze accept failures
    if accept_failures:
        print(f"\n  Analyzing accept failures:")
        for i, (action, parsed, obs) in enumerate(accept_failures[:2]):
            print(f"\n    Accept failure {i+1}:")
            
            # Check if there's an offer in the observation
            if "Player" in obs and "made the following offer" in obs:
                print(f"      -> Offer found in observation")
                
                # Check if offer was already responded to
                import re
                if re.search(r"(accepted|denied|implicitly denied)", obs):
                    print(f"      -> Offer was already responded to")
                elif re.search(r"\[Player \d+\] \[Offer:", obs):
                    print(f"      -> Counter-offer was made (implicit rejection)")
                else:
                    print(f"      -> Offer appears to be pending")
            else:
                print(f"      -> No pending offer found in observation")
            
            print(f"      '[Accept]' in parsed: {'[Accept]' in parsed}")
            print(f"      '[Deny]' in parsed: {'[Deny]' in parsed}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if kp_valid > 0:
    kp_status = "PASS" if kp_pass == kp_valid else f"FAIL ({kp_pass}/{kp_valid})"
else:
    kp_status = "NO VALID TEST CASES"

if sn_valid > 0:
    sn_status = "PASS" if sn_pass == sn_valid else f"FAIL ({sn_pass}/{sn_valid})"
    sn_rate = f"{sn_pass / sn_valid * 100:.1f}%"
else:
    sn_status = "NO VALID TEST CASES"
    sn_rate = "N/A"

print(f"Kuhn Poker:          {kp_status}")
print(f"Simple Negotiation:  {sn_status} - Pass rate: {sn_rate}")

if kp_valid > 0 and kp_pass == kp_valid:
    print("\nKuhn Poker parser: SUCCESS - All valid actions parsed correctly!")
else:
    print(f"\nKuhn Poker parser: {kp_fail} failures out of {kp_valid} valid actions")

if sn_valid > 0:
    improvement = (sn_pass / sn_valid * 100) - 16.2
    print(f"\nSimple Negotiation parser improvement: {improvement:+.1f}% from baseline (16.2%)")
    
    if sn_pass == sn_valid:
        print("Simple Negotiation parser: SUCCESS - All valid actions parsed correctly!")
    else:
        print(f"Simple Negotiation parser: {sn_fail} failures out of {sn_valid} valid actions")

print()

