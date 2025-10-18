# Answers to Your Three Questions

## Question 1: Create run_4b_8env.sh based on run_4b_4env.sh

### ✅ COMPLETED

**Created**: `/ephemeral/games-workspace/spiral_new/script/scaling/run_4b_8env.sh`

**Changes from run_4b_4env.sh**:

```bash
# OLD (4 environments)
--env_ids KuhnPoker-v1 SimpleNegotiation-v1 TicTacToe-v1 PigDice-v1
--use_llm_obs_wrappers True True False False

# NEW (8 environments)
--env_ids KuhnPoker-v1 SimpleNegotiation-v1 TicTacToe-v1 PigDice-v1 Briscola-v1 ColonelBlotto-v1 IndianPoker-v1 TwoDollar-v1
--use_llm_obs_wrappers True True False False True True True True
```

**Also updated evaluation**:
```bash
# Added new environments to eval
--eval_env_ids TicTacToe-v1 KuhnPoker-v1 Briscola-v1 ColonelBlotto-v1
--eval_use_llm_obs_wrappers False True True True
```

**Updated wandb run name**:
```bash
--wb-run-name spiral-qwen3-4b-base-8env-self-play
```

---

## Question 2: Understanding use_llm_obs_wrappers

### 📚 Complete Analysis in `OBSERVATION_WRAPPER_GUIDE.md`

### Quick Summary

**What `use_llm_obs_wrappers` does**:
- Controls how game observations are formatted for the LLM
- `True` = Full conversation history (usually)
- `False` = Only first prompt + last state

### Wrapper Logic (from `spiral/envs/__init__.py`)

```python
envs_with_action_messages = ["KuhnPoker-v1"]

if use_llm_obs_wrapper:
    if env_id in envs_with_action_messages:
        env = ta.wrappers.FirstLastObservationWrapper(env=env)
    else:
        env = ta.wrappers.LLMObservationWrapper(env=env)
else:
    env = ta.wrappers.FirstLastObservationWrapper(env=env)
```

### Settings Verification

#### ✅ OLD Environments (All CORRECT)

| Environment | Setting | Wrapper | Correct? | Rationale |
|-------------|---------|---------|----------|-----------|
| KuhnPoker-v1 | True | FirstLast | ✅ | Appends actions each turn; avoid redundancy |
| SimpleNegotiation-v1 | True | LLM (full) | ✅ | Needs negotiation history |
| TicTacToe-v1 | False | FirstLast | ✅ | Board state is complete |
| PigDice-v1 | False | FirstLast | ✅ | Current state sufficient |

#### ✅ NEW Environments (All APPROPRIATE)

| Environment | Setting | Wrapper | Rationale |
|-------------|---------|---------|-----------|
| Briscola-v1 | True | LLM (full) | Track played cards for strategy |
| ColonelBlotto-v1 | True | LLM (full) | See opponent's allocation patterns |
| IndianPoker-v1 | True | LLM (full) | Betting history informs decisions |
| TwoDollar-v1 | True | LLM (full) | Negotiation requires full conversation |

### Decision Rules

**Use `True` (Full History) when**:
- ✓ Negotiation/conversation is core gameplay
- ✓ Historical actions inform future strategy
- ✓ Pattern recognition matters (card counting, betting)
- ✓ Games with bluffing, trading, or social deduction

**Use `False` (First+Last Only) when**:
- ✓ Current state is self-contained
- ✓ Board shows all needed information
- ✓ Simple games with complete state visibility
- ✓ Full history creates confusion

### Special Case: envs_with_action_messages

**Currently only**: `["KuhnPoker-v1"]`

**Why?** KuhnPoker appends the same action messages every turn:
```
Your available actions are: '[check]', '[bet]'
Your available actions are: '[check]', '[bet]'  # Redundant!
```

**Should new envs be added?**
- Briscola, ColonelBlotto, IndianPoker also append action messages
- However, the historical context outweighs redundancy
- **Recommendation**: Keep current list as-is

---

## Question 3: Change TwoDollar max_rounds from 20 to 10

### ✅ COMPLETED

**Changed in**: `/ephemeral/games-workspace/spiral_new/spiral/envs/__init__.py`

```python
# Before
register(
    id="TwoDollar-v1",
    max_rounds=20,  # Old value
)

# After
register(
    id="TwoDollar-v1",
    max_rounds=10,  # New value
)
```

### ✅ NO Other Changes Needed

The TwoDollar environment is well-designed with `max_rounds` as a parameter. **Everything adjusts automatically**:

#### Automatic Adjustments

1. **Game Duration**: 20 rounds → 10 rounds
2. **x_rounds Role Deadline**: 
   - Before: `20 // 2 = 10` rounds
   - After: `10 // 2 = 5` rounds
3. **Prompt Text**: Automatically shows "10 maximum rounds"
4. **Role Instructions**: x_rounds role sees correct deadline
5. **UI Display**: Round counter shows "X of 10"

#### What Stays the Same
- Total amount: $2.00
- Error allowance: 3
- Action space: Same
- All role mechanics: Same
- Reward calculation: Same

### Benefits of Shorter Games

1. **Faster Training**: Games complete in ~50% less time
2. **Less Context**: Shorter observation history
3. **More Games**: More training iterations per hour
4. **Still Sufficient**: 10 rounds is enough for negotiation:
   - Rounds 1-2: Opening offers
   - Rounds 3-6: Counter-negotiations  
   - Rounds 7-10: Convergence

### Monitoring Recommendations

Track these metrics during training:
- Average game length (should be < 10 rounds)
- Deal acceptance rate (should be > 50%)
- x_rounds role success rate (should be reasonable)

If games consistently hit the max_rounds limit without deals, consider increasing to 12-15 rounds.

---

## Summary

### ✅ All Three Tasks Completed

1. **run_4b_8env.sh**: Created with all 8 environments
2. **use_llm_obs_wrappers**: Analyzed and verified all settings are correct
3. **TwoDollar max_rounds**: Changed from 20 to 10, no other changes needed

### Files Modified

1. ✅ `script/scaling/run_4b_8env.sh` (new file)
2. ✅ `spiral/envs/__init__.py` (max_rounds: 20→10)

### Files Created (Documentation)

1. 📚 `OBSERVATION_WRAPPER_GUIDE.md` - Complete wrapper analysis
2. 📚 `TWODOLLAR_MAXROUNDS_CHANGE.md` - Impact analysis
3. 📚 `QUERY_ANSWERS_SUMMARY.md` - This file

### Ready to Use

You can now run training with 8 environments:
```bash
bash script/scaling/run_4b_8env.sh
```

Or customize the command with different settings:
```bash
python train_spiral.py \
    --env_ids Briscola-v1 ColonelBlotto-v1 IndianPoker-v1 TwoDollar-v1 \
    --use_llm_obs_wrappers True True True True \
    --num_envs 1 \
    ...
```

---

## Key Insights

### Observation Wrappers
- KuhnPoker is special (FirstLast despite True)
- New environments correctly use LLM (full history)
- No changes needed to existing settings

### TwoDollar Change
- Well-parameterized environment
- Automatic adjustments throughout
- No code changes needed beyond registration

### Training Efficiency
- 8 environments provide diverse training signal
- Shorter TwoDollar games improve throughput
- All observation wrappers optimized for each game type

