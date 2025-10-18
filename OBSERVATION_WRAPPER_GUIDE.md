# Observation Wrapper Guide

## Understanding `use_llm_obs_wrappers`

The `use_llm_obs_wrappers` parameter controls how game observations are formatted for the LLM. Different games need different observation strategies.

### Available Wrappers

#### 1. LLMObservationWrapper (`use_llm_obs_wrappers=True`)
**What it does**: Accumulates ALL observations with sender tags like `[GAME]` or `[Player 0]`

**Format**:
```
[GAME] Game started
[Player 0] I place my mark at [4]
[GAME] Player 0 placed X at position 4
[Player 1] I place my mark at [0]
[GAME] Player 1 placed O at position 0
...
```

**Use when**:
- Game requires full conversational/action history
- Negotiations, card counting, pattern recognition
- Examples: SimpleNegotiation, TwoDollar, IndianPoker

#### 2. FirstLastObservationWrapper (`use_llm_obs_wrappers=False` OR special cases)
**What it does**: Shows only the FIRST observation (prompt) and LAST observation (current state)

**Format**:
```
[First observation - game rules/prompt]

[Last observation - current game state]

Next Action:
```

**Use when**:
- Only current game state matters
- Full history causes confusion or redundancy
- Games that re-send available actions each turn
- Examples: TicTacToe, PigDice, KuhnPoker (special)

### Logic in `spiral/envs/__init__.py`

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

**Decision tree**:
- `use_llm_obs_wrappers=True` + in `envs_with_action_messages` → FirstLastObservationWrapper
- `use_llm_obs_wrappers=True` + NOT in list → LLMObservationWrapper (full history)
- `use_llm_obs_wrappers=False` → FirstLastObservationWrapper

---

## Environment-Specific Settings

### Existing Environments

| Environment | Setting | Wrapper Used | Rationale |
|-------------|---------|--------------|-----------|
| **KuhnPoker-v1** | True | FirstLast | Appends available actions each turn; full history would duplicate action lists |
| **SimpleNegotiation-v1** | True | LLM (full) | Negotiation requires full conversation history for context |
| **TicTacToe-v1** | False | FirstLast | Board state is complete; full history unnecessary |
| **PigDice-v1** | False | FirstLast | Current score and dice state is sufficient |

### New Environments

| Environment | Recommended | Wrapper Used | Rationale |
|-------------|-------------|--------------|-----------|
| **Briscola-v1** | True | LLM (full) | Card game where tracking played cards helps strategy |
| **ColonelBlotto-v1** | True | LLM (full) | Seeing opponent's past allocations helps predict strategy |
| **IndianPoker-v1** | True | LLM (full) | Betting game; betting patterns and history matter |
| **TwoDollar-v1** | True | LLM (full) | Negotiation game; full conversation history essential |

---

## Detailed Analysis

### KuhnPoker-v1: Special Case
**Setting**: `use_llm_obs_wrappers=True` + in `envs_with_action_messages`

**Why special?**
```python
# KuhnPoker appends available actions every turn:
"Your available actions are: '[check]', '[bet]'"
```

If we used LLMObservationWrapper (full history), the observation would accumulate:
```
Turn 1: Your available actions are: '[check]', '[bet]'
Turn 2: Your available actions are: '[check]', '[bet]'
Turn 3: Your available actions are: '[call]', '[fold]'
Turn 4: Your available actions are: '[call]', '[fold]'
```

This creates redundancy. FirstLastObservationWrapper shows:
```
[Prompt with rules]

[Current state + available actions]

Next Action:
```

**Should other envs be in this list?**

Let me check which new envs append available actions:

1. **Briscola**: Yes, `_announce_turn()` includes actions
2. **ColonelBlotto**: Yes, `_render_game_state()` includes format
3. **IndianPoker**: Yes, `_announce_actions()` each turn
4. **TwoDollar**: No, actions are contextual

**However**, these games benefit from history despite action messages:
- Briscola: Need to see cards played
- ColonelBlotto: Need to see past allocations
- IndianPoker: Need to see betting patterns

**Recommendation**: Keep them with LLMObservationWrapper (full history). The benefit of history outweighs redundant action messages.

### Verification of Current Settings

From `run_4b_4env.sh`:
```bash
--env_ids KuhnPoker-v1 SimpleNegotiation-v1 TicTacToe-v1 PigDice-v1
--use_llm_obs_wrappers True True False False
```

| Environment | Setting | Result | Correct? |
|-------------|---------|--------|----------|
| KuhnPoker-v1 | True | FirstLast (special) | ✓ Yes |
| SimpleNegotiation-v1 | True | LLM (full) | ✓ Yes |
| TicTacToe-v1 | False | FirstLast | ✓ Yes |
| PigDice-v1 | False | FirstLast | ✓ Yes |

**All current settings are CORRECT!**

---

## Recommended Settings for run_4b_8env.sh

```bash
--env_ids KuhnPoker-v1 SimpleNegotiation-v1 TicTacToe-v1 PigDice-v1 Briscola-v1 ColonelBlotto-v1 IndianPoker-v1 TwoDollar-v1
--use_llm_obs_wrappers True True False False True True True True
```

**Breakdown**:
1. KuhnPoker-v1: `True` (FirstLast via special list)
2. SimpleNegotiation-v1: `True` (LLM full history)
3. TicTacToe-v1: `False` (FirstLast)
4. PigDice-v1: `False` (FirstLast)
5. **Briscola-v1**: `True` (LLM full history) - track cards
6. **ColonelBlotto-v1**: `True` (LLM full history) - track allocations
7. **IndianPoker-v1**: `True` (LLM full history) - betting patterns
8. **TwoDollar-v1**: `True` (LLM full history) - negotiation

---

## When to Use Each Setting

### Use `True` (LLMObservationWrapper - full history) when:
- ✓ Negotiation/conversation is core gameplay
- ✓ Historical actions inform future strategy
- ✓ Pattern recognition across turns matters
- ✓ Game involves bluffing, betting, or trading

### Use `False` (FirstLastObservationWrapper) when:
- ✓ Current state is self-contained
- ✓ Board/game state shows all needed information
- ✓ Historical actions don't inform strategy
- ✓ Simpler games with complete state visibility

### Special consideration (add to `envs_with_action_messages`) when:
- ✓ Game appends same action list every turn
- ✓ Full history creates significant redundancy
- ✓ Current state + prompt is sufficient

---

## Potential Issue: Should Briscola/ColonelBlotto/IndianPoker be in `envs_with_action_messages`?

**Current code**:
```python
envs_with_action_messages = ["KuhnPoker-v1"]
```

**Analysis**:
- All three games (Briscola, ColonelBlotto, IndianPoker) DO append available actions
- However, unlike KuhnPoker where action messages are the main redundancy, these games benefit more from seeing full history

**Recommendation**: Keep current settings. The historical context outweighs the redundancy.

**Alternative approach** (if action redundancy becomes a problem):
```python
envs_with_action_messages = ["KuhnPoker-v1", "Briscola-v1", "ColonelBlotto-v1", "IndianPoker-v1"]
```

Then adjust to use different wrappers:
```python
envs_needing_history = ["SimpleNegotiation-v1", "TwoDollar-v1"]

if use_llm_obs_wrapper:
    if env_id in envs_needing_history:
        env = ta.wrappers.LLMObservationWrapper(env=env)
    else:
        env = ta.wrappers.FirstLastObservationWrapper(env=env)
else:
    env = ta.wrappers.FirstLastObservationWrapper(env=env)
```

**But this loses history benefits. Current approach is better.**

---

## Summary

✅ **All current `use_llm_obs_wrappers` settings are CORRECT**

✅ **New environment settings in run_4b_8env.sh are APPROPRIATE**:
- Briscola: True (needs card history)
- ColonelBlotto: True (needs allocation history)
- IndianPoker: True (needs betting history)
- TwoDollar: True (needs negotiation history)

✅ **No changes needed to `spiral/envs/__init__.py`** - current logic is sound

The only environment in `envs_with_action_messages` is KuhnPoker-v1, and that's appropriate because it's the only one where action message redundancy outweighs the value of history.

