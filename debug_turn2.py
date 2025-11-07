"""Debug Turn 2 specifically to understand the observation structure after errors."""

import sys
sys.path.insert(0, '/ephemeral/games-workspace/spiral_new')

import json

# Load actual game data
with open('/ephemeral/games-workspace/spiral_new/oat-output/4b_coup_1107T0007/game_state/actor0_step0.json', 'r') as f:
    data = json.load(f)

game = data[0]
history = game['history']

turn_2 = history[2]
observation = turn_2[1]

print('='*80)
print('TURN 2 OBSERVATION STRUCTURE')
print('='*80)
print()

# Show the full observation in structured way
print('Full observation (showing key segments):')
print()

# Split by markers
if 'It is now your turn.' in observation:
    segments = observation.split('It is now your turn.')
    print(f'Number of "It is now your turn." markers: {len(segments) - 1}')
    print()
    
    for idx, segment in enumerate(segments):
        if idx == 0:
            print(f'--- Segment {idx} (BEFORE first "It is now your turn.") ---')
        else:
            print(f'--- Segment {idx} (AFTER "It is now your turn." #{idx}) ---')
        
        # Show first and last 3 lines of each segment
        lines = segment.strip().split('\n')
        if len(lines) <= 10:
            for line in lines:
                print(f'  {line}')
        else:
            print(f'  [First 3 lines]')
            for line in lines[:3]:
                print(f'  {line}')
            print(f'  ... ({len(lines) - 6} lines omitted) ...')
            print(f'  [Last 3 lines]')
            for line in lines[-3:]:
                print(f'  {line}')
        print()

# Extract what parser would see as "current_prompt"
segments = observation.split('It is now your turn.')
current_prompt = segments[-1] if len(segments) > 1 else observation

print('='*80)
print('WHAT PARSER SEES AS current_prompt:')
print('='*80)
print(current_prompt[:500])
print()

# Check what triggers
print('='*80)
print('TRIGGER CHECKS:')
print('='*80)
print(f'"Do you want to" in current_prompt: {"Do you want to" in current_prompt}')
print(f'"[block" in current_prompt: {"[block" in current_prompt}')
print(f'"[BULLSHIT]" in current_prompt: {"[BULLSHIT]" in current_prompt}')
print(f'"[PASS]" in current_prompt: {"[PASS]" in current_prompt}')


