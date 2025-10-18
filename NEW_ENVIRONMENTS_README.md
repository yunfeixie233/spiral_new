# New Environments Integration Guide

## Summary

Successfully integrated 4 new two-player game environments into the SPIRAL codebase:

1. **Briscola-v1** - Italian card game (finite action space)
2. **ColonelBlotto-v1** - Resource allocation strategy (finite action space) 
3. **IndianPoker-v1** - Betting card game (infinite action space)
4. **TwoDollar-v1** - Negotiation game (infinite action space)

## Environment Details

### 1. Briscola (Briscola-v1)

**Game Background**: Classic Italian card game using a 40-card deck. Players take turns playing cards to win tricks and collect points.

**Action Space**: FINITE
- `[play 1]` - Play first card in hand
- `[play 2]` - Play second card in hand  
- `[play 3]` - Play third card in hand (if available)

**Implementation**:
- Parse function extracts hand size from observation
- Typically 2-3 cards in hand at any time
- Action parser: `briscola_parse_available_actions()`

**Registration**:
```python
register(
    id="Briscola-v1",
    entry_point="spiral.envs.Briscola.env:BriscolaEnv",
)
```

---

### 2. Colonel Blotto (ColonelBlotto-v1)

**Game Background**: Military strategy game where commanders allocate units across battlefields. Win individual battles to win rounds.

**Action Space**: FINITE but LARGE
- Format: `[A4 B2 C14]` - Allocate units to fields A, B, C
- Must allocate exactly 20 units total across 3 fields
- Theoretical space: 231 combinations (stars and bars)
- **Practical implementation**: Samples ~50-66 representative allocations

**Implementation**:
- Parse function generates allocations in steps of 2 units
- Samples every Nth allocation to keep space manageable
- Action parser: `colonel_blotto_parse_available_actions()`

**Registration**:
```python
register(
    id="ColonelBlotto-v1",
    entry_point="spiral.envs.ColonelBlotto.env:ColonelBlottoEnv",
    num_fields=3,
    num_total_units=20,
    num_rounds=10,
)
```

---

### 3. Indian Poker (IndianPoker-v1)

**Game Background**: Poker variant where each player sees their opponent's card but not their own. Players bet based on what they see.

**Action Space**: INFINITE ❌
- `[check]` - Pass without betting (when no bet is facing you)
- `[fold]` - Surrender the hand
- `[call]` - Match current bet
- `[bet X]` - Open betting with amount X (X = 1 to chip count)
- `[raise X]` - Raise current bet by amount X (X = 1 to remaining chips)

**Why Infinite**: Betting amounts (X) can be any integer from 1 to player's chip count (starting with 100 chips). This creates a prohibitively large action space that cannot be enumerated.

**Implementation**:
- **No parse function** - uses natural language extraction
- Added to chat-based games in `train_spiral.py` line 499
- Uses `extract_chat_action()` method

**Registration**:
```python
register(
    id="IndianPoker-v1",
    entry_point="spiral.envs.IndianPoker.env:IndianPokerEnv",
    max_rounds=5,
    starting_chips=100,
)
```

---

### 4. Two Dollar (TwoDollar-v1)

**Game Background**: Negotiation game where two players must agree on how to split $2.00. Each player has secret role instructions with different objectives/constraints.

**Action Space**: INFINITE ❌
- `[Propose] $X.XX` - Propose a split where you get $X.XX (X.XX = 0.00 to 2.00)
- `[Accept]` - Accept current proposal
- `[Reject]` - Reject current proposal

**Why Infinite**: Proposal amounts can be any value with 2 decimal places from $0.00 to $2.00. This creates 201 possible proposal amounts, but combined with natural language negotiation, the action space is effectively infinite.

**Implementation**:
- **No parse function** - uses natural language extraction
- Added to chat-based games in `train_spiral.py` line 499
- Uses `extract_chat_action()` method

**Registration**:
```python
register(
    id="TwoDollar-v1",
    entry_point="spiral.envs.TwoDollar.env:TwoDollarEnv",
    total_amount=2.00,
    max_rounds=20,
    error_allowance=3,
)
```

---

## Files Modified

### 1. `/ephemeral/games-workspace/spiral_new/spiral/envs/__init__.py`
- Added 4 environment registrations
- Each configured with appropriate default parameters

### 2. `/ephemeral/games-workspace/spiral_new/spiral/agents/utils.py`
- Added `briscola_parse_available_actions()` - Extracts hand size and generates play actions
- Added `colonel_blotto_parse_available_actions()` - Generates sampled allocation actions
- Updated `_VALID_ACTION_PARSER` dictionary with:
  - `"Briscola-v1": briscola_parse_available_actions`
  - `"ColonelBlotto-v1": colonel_blotto_parse_available_actions`

### 3. `/ephemeral/games-workspace/spiral_new/train_spiral.py`
- Line 499: Added `"IndianPoker-v1"` and `"TwoDollar-v1"` to chat-based games list
- These environments bypass fixed action space parsing

---

## Usage

### Training with Fixed Action Space Games (Briscola, ColonelBlotto)

```bash
python train_spiral.py \
    --env_ids Briscola-v1 \
    --use_llm_obs_wrappers True \
    --num_envs 1
```

### Training with Infinite Action Space Games (IndianPoker, TwoDollar)

```bash
python train_spiral.py \
    --env_ids IndianPoker-v1 \
    --use_llm_obs_wrappers True \
    --num_envs 1
```

### Multi-Environment Training

```bash
python train_spiral.py \
    --env_ids Briscola-v1 ColonelBlotto-v1 \
    --use_llm_obs_wrappers True True \
    --num_envs 1
```

---

## Design Rationale

### Why Some Games Don't Have Parse Functions

**IndianPoker** and **TwoDollar** have infinite or very large action spaces:
- IndianPoker: Betting amounts can be 1 to chip_count (100+ possibilities)
- TwoDollar: Proposal amounts can be $0.00 to $2.00 in cents (201 possibilities)

For these games:
1. Enumerating all actions is impractical or impossible
2. LLMs generate actions via natural language reasoning
3. Action extraction uses `extract_chat_action()` which looks for `\boxed{}` notation
4. Similar to existing `SimpleNegotiation-v1` which also has unbounded offers

### Action Space Sampling (ColonelBlotto)

ColonelBlotto has 231 possible allocations (C(20+3-1, 3-1) = C(22,2) = 231).

To keep the action space manageable:
- Generate allocations in steps of 2 units
- Sample every Nth allocation (~50-66 actions)
- Still covers diverse strategies (balanced, aggressive, defensive)

---

## Verification

All environments have been tested and verified:

```
✓ Briscola-v1: 3 actions parsed
✓ ColonelBlotto-v1: 66 actions parsed  
✓ IndianPoker-v1: chat-based (infinite action space)
✓ TwoDollar-v1: chat-based (infinite action space)
```

---

## Action Space Summary

| Environment | Type | Size | Parse Function | Chat-Based |
|-------------|------|------|----------------|------------|
| Briscola | Finite | 2-3 | ✓ | ✗ |
| ColonelBlotto | Finite (Large) | ~66 | ✓ | ✗ |
| IndianPoker | Infinite | N/A | ✗ | ✓ |
| TwoDollar | Infinite | N/A | ✗ | ✓ |

---

## Next Steps

1. **Test with different opponents**: Try training against random and LLM opponents
2. **Tune hyperparameters**: Adjust reward scaling, learning rates for each game
3. **Evaluate performance**: Use eval modes to test learned policies
4. **Add to evaluation suite**: Include in `--eval_env_ids` for benchmarking

---

## Troubleshooting

### Import Errors
If you see import errors for `textarena`, ensure you're in the correct environment:
```bash
cd /ephemeral/games-workspace/spiral_new
export PYTHONPATH=/ephemeral/games-workspace/spiral_new:$PYTHONPATH
```

### Action Parsing Errors
For Briscola/ColonelBlotto, if actions fail to parse:
- Check observation format hasn't changed
- Verify regex patterns match game output
- Enable debug logging to see parsed actions

### Chat-Based Action Extraction
For IndianPoker/TwoDollar, ensure:
- LLM output uses `\boxed{action}` format
- Template includes action formatting instructions
- `use_llm_obs_wrapper=True` for proper context

