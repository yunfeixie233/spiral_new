# TwoDollar max_rounds Change: 20 → 10

## Change Made

```python
# In spiral/envs/__init__.py
register(
    id="TwoDollar-v1",
    entry_point="spiral.envs.TwoDollar.env:TwoDollarEnv",
    total_amount=2.00,
    max_rounds=10,  # Changed from 20 to 10
    error_allowance=3,
)
```

## Impact Analysis

### ✅ Automatic Adjustments (No code changes needed)

The TwoDollar environment is well-designed to handle `max_rounds` as a parameter. Everything adjusts automatically:

#### 1. Game Duration
- **Before**: Games could last up to 20 rounds
- **After**: Games end after 10 rounds maximum
- **Impact**: Faster games, less context to process

#### 2. x_rounds Role Deadline
The `x_rounds` role has a secret deadline to accept a deal.

**Code** (env.py line 119):
```python
self.player_deadline[player_id] = self.max_rounds // 2
```

- **Before**: Deadline = 20 // 2 = 10 rounds
- **After**: Deadline = 10 // 2 = 5 rounds
- **Impact**: x_rounds role must negotiate faster

#### 3. Prompt Text
**Code** (env.py line 156):
```python
f"There are {self.max_rounds} maximum rounds."
```

- **Before**: "There are 20 maximum rounds."
- **After**: "There are 10 maximum rounds."
- **Impact**: Players know correct round limit

#### 4. x_rounds Role Instructions
**Code** (env.py lines 184-186):
```python
deadline = self.player_deadline.get(player_id, self.max_rounds // 2)
prompt = prompt.replace("{deadline}", str(deadline))
prompt = prompt.replace("{total_rounds}", str(self.max_rounds))
```

**Before**:
```
"This game ends in 10 rounds for you (while your opponent gets the full 20 rounds)"
```

**After**:
```
"This game ends in 5 rounds for you (while your opponent gets the full 10 rounds)"
```

- **Impact**: x_rounds role sees correct deadline

#### 5. Round Display
**Code** (env.py line 545):
```python
round_info = f"=== ROUND {self.state.turn + 1} of {self.max_rounds} ==="
```

- **Before**: "=== ROUND 5 of 20 ==="
- **After**: "=== ROUND 5 of 10 ==="
- **Impact**: Correct progress display

### ✅ No Changes Needed

The following remain unchanged:
- `total_amount`: Still $2.00
- `error_allowance`: Still 3 invalid moves allowed
- Action space: Still `[Propose] $X.XX`, `[Accept]`, `[Reject]`
- Role mechanics: All roles work the same
- Reward calculation: Based on final amounts, not rounds

## Why This Change?

### Benefits of Shorter Games (10 rounds)

1. **Faster Training**: Each game completes in ~half the time
2. **Less Context**: Shorter observation history (better for LLM context limits)
3. **Quicker Convergence**: More games per training iteration
4. **Still Sufficient**: 10 rounds is enough for meaningful negotiation

### Negotiation is Still Viable

With 10 rounds, players can still:
- Make opening offers (rounds 1-2)
- Counter-negotiate (rounds 3-6)
- Converge on agreement (rounds 7-10)
- Use strategic timing

The original 20 rounds may have been excessive for training purposes.

## Validation

### Test Different Scenarios

The change affects all role combinations:

| Role Combination | Old Deadline | New Deadline | Still Viable? |
|------------------|--------------|--------------|---------------|
| both normal | 20 rounds | 10 rounds | ✓ Yes |
| one x_rounds | 10 rounds | 5 rounds | ✓ Yes - creates urgency |
| one high_tension | 20 rounds | 10 rounds | ✓ Yes |
| one say_little | 20 rounds | 10 rounds | ✓ Yes |

**x_rounds role** is most affected:
- Previously had 10 rounds to accept (50% of game)
- Now has 5 rounds to accept (50% of game)
- **Percentage is same**, absolute time is shorter
- This maintains game balance

## Recommendations

### If Games End Too Quickly
If 10 rounds proves insufficient during training (e.g., most games hit max_rounds without deals):

**Option 1**: Increase to 12 rounds
```python
max_rounds=12,  # Deadline becomes 6 rounds for x_rounds
```

**Option 2**: Increase to 15 rounds
```python
max_rounds=15,  # Deadline becomes 7 rounds for x_rounds
```

### If Games Are Too Long
If games typically end in 3-5 rounds anyway:

**Option 1**: Decrease to 8 rounds
```python
max_rounds=8,  # Deadline becomes 4 rounds for x_rounds
```

**Monitor** training metrics:
- Average game length
- Deal acceptance rate
- x_rounds role success rate

## Summary

✅ **Change is SAFE** - All adjustments are automatic

✅ **No other code changes needed** - Environment is well-parameterized

✅ **Game balance maintained** - Percentages stay the same

⚠️ **Monitor training** - Verify 10 rounds is sufficient for meaningful negotiation

The TwoDollar environment was designed with `max_rounds` as a configurable parameter, so changing from 20 to 10 requires **no additional modifications**. Everything adjusts automatically!

