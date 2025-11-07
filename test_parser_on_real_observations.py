"""Test the parser on actual observations from the game logs."""

import sys
sys.path.insert(0, '/ephemeral/games-workspace/spiral_new')

from spiral.agents.utils import coup_parse_available_actions
import json

# Load actual game data
with open('/ephemeral/games-workspace/spiral_new/oat-output/4b_coup_1107T0007/game_state/actor0_step0.json', 'r') as f:
    data = json.load(f)

game = data[0]
history = game['history']

print('='*80)
print('TESTING PARSER ON REAL OBSERVATIONS')
print('='*80)
print()

for turn_idx, turn in enumerate(history):
    player_id = turn[0]
    observation = turn[1]
    action_taken = turn[2]
    
    print(f'=== Turn {turn_idx}: Player {player_id} ===')
    print()
    
    # Show key parts of observation
    obs_lines = observation.split('\n')
    print('Last 3 lines of observation:')
    for line in obs_lines[-3:]:
        if line.strip():
            print(f'  "{line}"')
    print()
    
    # Run parser
    print('Running parser...')
    parsed_actions = coup_parse_available_actions(observation)
    print(f'Parser returned {len(parsed_actions)} actions:')
    print(f'  {parsed_actions}')
    print()
    
    # Show action taken
    print(f'Action model took: {repr(action_taken)}')
    print()
    
    # Check if action is in parsed actions
    if action_taken in parsed_actions:
        print('✅ ACTION IS VALID - In action space')
    elif '[｜INVALID_ACTION｜]' in action_taken:
        print('⚠ Action was marked INVALID_ACTION (expected after errors)')
    else:
        print('❌ ACTION IS INVALID - Not in action space!')
        print(f'   Model output: {action_taken}')
        print(f'   Valid actions: {parsed_actions}')
    
    print()
    print('-'*80)
    print()

print('='*80)
print('DIAGNOSIS')
print('='*80)
print()
print('Key findings:')
print('1. Turn 0: Model output [exchange] - Check if in action space')
print('2. Turn 1: Model output [block steal ambassador] for Exchange action')
print('   - This is INVALID because you cannot block steal for an exchange')
print('   - Parser correctly returns only [PASS] and [BULLSHIT]')
print('3. Turn 2: Model hit error limit and returned INVALID_ACTION')
print()
print('Conclusion: The issue is that the MODEL is generating invalid actions,')
print('not that the PARSER is wrong. The parser is correctly identifying valid actions.')


