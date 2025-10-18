# Updated Parser Test Results

## Test Setup

Tested the updated parsing functions against all JSON files in:
- `/ephemeral/games-workspace/spiral_new/oat-output/spiral-qwen3-4b-base-kp-ttt-4k-self-play_1018T00:54:38/game_state`
- 96 JSON files with game history data

## Results Summary

| Parser | Valid Actions | Pass | Fail | Pass Rate | Improvement |
|--------|--------------|------|------|-----------|-------------|
| **Kuhn Poker** | 0 | 0 | 0 | N/A | N/A (all invalid) |
| **Simple Negotiation** | 37 | 8 | 29 | 21.6% | +5.4% |

### Kuhn Poker

- **Status**: Cannot test - all 23 test cases had invalid actions
- **Parser Status**: Fixed (now correctly parses available actions from observations)
- **Note**: The lack of valid test cases doesn't mean the parser is broken; it means the model in this dataset never generated valid Kuhn Poker actions

### Simple Negotiation

- **Previous pass rate**: 16.2% (with random sampling)
- **Current pass rate**: 21.6% (with exhaustive generation)
- **Improvement**: +5.4 percentage points

However, this improvement revealed several critical issues:

## Critical Issues Found

### 1. Multi-Resource Offers (PRIMARY ISSUE)

**Problem**: The parser only generates single-resource offers, but the game supports multi-resource offers.

**Current parser generates**:
```
[Offer: 1 Wheat -> 1 Wood]
[Offer: 2 Sheep -> 3 Brick]
```

**Actual actions in data**:
```
[Offer: 5 Wheat, 3 Sheep -> 3 Ore, 2 Brick]  # 2 resources offered, 2 requested
[Offer: 3 Wheat, 2 Sheep -> 3 Ore, 1 Brick]  # 2 resources offered, 2 requested
```

**Impact**: This accounts for most of the failures (14 out of 29 failures)

### 2. Excessive Number of Actions

**Problem**: Generating all possible combinations creates too many actions.

**Current behavior**:
- With typical resources (Wheat:14, Wood:9, Sheep:12, Brick:11, Ore:7)
- Parser generates: **2,218 single-resource offers**
- For multi-resource offers, this would explode to tens of thousands

**Impact**: 
- Model has to choose from 2000-6000 actions
- Computationally expensive
- Makes it harder for model to find good actions

### 3. Resource Name Variations

**Problem**: The model sometimes uses plural forms that don't match the parser.

**Example**:
- Observation has: `[Brick]   Qty: 11`
- Actual action: `[Offer: 1 Ore -> 2 Bricks]`  (plural)
- Parser generates: `[Offer: 1 Ore -> 2 Brick]` (singular)

**Impact**: Minor, but causes mismatches

### 4. Accept/Deny Edge Cases

**Problem**: Some observations don't have clear pending offers, but action is [Accept].

**Example from data**:
```
[Player 1] rejected the trade offer.
[Player 1] made the following offer to Player 0: ...
[Player 0] [Accept]  # <-- This was already responded to
```

**Impact**: 4 out of 29 failures

## Detailed Statistics

### Failure Breakdown
- **Offer failures**: 14 (mostly multi-resource offers)
- **Accept failures**: 4 (edge cases in detection logic)
- **Other failures**: 11 (resource name variations, hallucinated resources)

### Sample Failures

**Failure 1** (Multi-resource):
- Actual: `[Offer: 5 Wheat, 3 Sheep -> 3 Ore, 2 Brick]`
- Parser generated: 2,218 single-resource offers (but not this one)

**Failure 2** (Resource hallucination):
- Actual: `[Offer: 3 Cotton, 5 Wheat -> 4 Land, 2 Coal]`
- Resources in observation: Wheat, Wood, Sheep, Brick, Ore
- Problem: Cotton, Land, and Coal don't exist (model hallucinated them)

**Failure 3** (Plural variation):
- Actual: `[Offer: 1 Ore -> 2 Bricks]`
- Parser generates: `[Offer: 1 Ore -> 2 Brick]`
- Problem: "Bricks" vs "Brick"

## Recommendations

### Short-term: Practical Improvements

1. **Support multi-resource offers** (up to 2 resources each side)
   ```python
   # Generate offers with 1-2 resources on each side
   for num_offer in [1, 2]:
       for num_request in [1, 2]:
           # Generate combinations
   ```

2. **Limit total actions** to ~100-200 offers
   - Sample intelligently based on heuristics
   - E.g., prioritize offers that increase total value
   - Or randomly sample from all possibilities

3. **Add resource name normalization**
   ```python
   # Handle both singular and plural
   resource_variations = {
       'Brick': ['Brick', 'Bricks'],
       'Ore': ['Ore', 'Ores'],
       # ...
   }
   ```

### Long-term: Architectural Improvements

1. **Parse actual available offers from observation**
   - Some game variants may list all valid offers
   - Would be more accurate than generation

2. **Use value-based filtering**
   - Only generate offers that could reasonably increase value
   - Reduces action space significantly

3. **Better state tracking**
   - Track pending offers more carefully
   - Handle counter-offers correctly

## Conclusion

**Kuhn Poker**: ✅ Parser is fixed and working correctly (tested with manual examples)

**Simple Negotiation**: ⚠️ Parser improved from 16.2% to 21.6%, but has fundamental limitations:
- Missing multi-resource offer support
- Generates too many actions (2000+)
- Some edge cases in Accept/Deny detection

**Next Steps**:
1. Decide if 21.6% pass rate is acceptable for your use case
2. If not, implement multi-resource offer support
3. Add intelligent sampling to limit action space
4. Consider whether generating all possible offers is the right approach

**Note**: Some failures are due to model behavior (hallucinating resources), not parser issues.
