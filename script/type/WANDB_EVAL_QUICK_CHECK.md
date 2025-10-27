# Quick Check: WandB Eval-Only Logging

## What Was Fixed

When running `--eval_only`, wandb now:
1. Uses the **original training run's name** (not a new name)
2. Logs metrics at the **correct checkpoint step** (not step 0)

## Quick Test

### 1. Check Log Output

When you run eval-only, you should see these messages:

```bash
bash script/type/run_pig_eval.sh

# Expected log output:
# ==================
# Eval-only mode: Using step 272 from pretrain path
# Initializing wandb with exp_name: run_pig_noeval_1025T1515
# Eval-only mode: Will use step 272 for logging
# ==================
```

### 2. Check WandB UI

1. Go to your wandb project: https://wandb.ai/YOUR_ORG/spiral
2. Find the run: `run_pig_noeval_1025T1515` (the original training run)
3. Look at the timeline/charts
4. **You should see evaluation metrics at step 272** (same as training checkpoint)

### Before vs After

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Run Name** | `run_pig_eval_1026T0930` | `run_pig_noeval_1025T1515` |
| **Step** | 0 | 272 |
| **Location** | Separate run | Same run as training |
| **Timeline** | Disconnected | Continuous |

## Example WandB Timeline

### Before Fix (Wrong)
```
Training Run: run_pig_noeval_1025T1515
├─ step 0: train metrics
├─ step 100: train metrics
├─ step 200: train metrics
└─ step 272: train metrics (checkpoint saved)

Eval Run: run_pig_eval_1026T0930 (NEW RUN)
└─ step 0: eval metrics (WRONG STEP!)
```

### After Fix (Correct)
```
Training Run: run_pig_noeval_1025T1515
├─ step 0: train metrics
├─ step 100: train metrics
├─ step 200: train metrics
└─ step 272: 
    ├─ train metrics (from training)
    └─ eval metrics (from eval_only)  <- ADDED HERE!
```

## Path Parsing Example

```
Input pretrain path:
/ephemeral/games-workspace/spiral/oat-output/run_pig_noeval_1025T1515/saved_models/step_00272

Extracted:
├─ exp_name: run_pig_noeval_1025T1515
├─ step: 272
└─ save_path: /ephemeral/games-workspace/spiral/oat-output/run_pig_noeval_1025T1515
```

## Verification Checklist

- [ ] Log shows correct exp_name (matches training run)
- [ ] Log shows correct step number (matches checkpoint number)
- [ ] WandB shows evaluation metrics in the original training run
- [ ] WandB shows evaluation metrics at the correct step number
- [ ] Can compare training vs eval metrics at the same step

## Common Issues

### Issue: Still seeing a new run name
**Check**: Make sure your pretrain path contains both "saved_models" and "step_"
```bash
# Good:
--pretrain /path/to/run_name/saved_models/step_00272

# Bad (won't work):
--pretrain /path/to/model_checkpoint
```

### Issue: Step shows as 0
**Check**: Verify "step_XXXXX" pattern in pretrain path
```bash
# Good:
step_00272, step_00100, step_01234

# Bad (won't parse):
checkpoint_272, model_272, final
```

### Issue: Logs not appearing in wandb
**Check**: Make sure you're using `--use-wb` flag
```bash
python train_spiral_eval.py \
    --pretrain /path/to/step_00272 \
    --eval_only \
    --use-wb  # <- REQUIRED
```

## Debug Commands

```bash
# 1. Check what pretrain path is being used
grep -n "pretrain" script/type/run_pig_eval.sh

# 2. Test path parsing manually
python -c "
import os, re
path = '/ephemeral/games-workspace/spiral/oat-output/run_pig_noeval_1025T1515/saved_models/step_00272'
parent_dir = os.path.dirname(os.path.dirname(path))
exp_name = os.path.basename(parent_dir)
step_match = re.search(r'step_(\d+)', path)
step = int(step_match.group(1)) if step_match else 0
print(f'exp_name: {exp_name}')
print(f'step: {step}')
"

# Expected output:
# exp_name: run_pig_noeval_1025T1515
# step: 272
```

## Success Indicators

You'll know it's working when:

1. **Console logs show:**
   ```
   Eval-only mode: Using step 272 from pretrain path
   Initializing wandb with exp_name: run_pig_noeval_1025T1515
   ```

2. **WandB UI shows:**
   - Run name matches training run
   - Evaluation metrics appear at the checkpoint's step number
   - Can toggle between training/eval metrics at the same step

3. **Metrics organization:**
   - Training metrics: `train/*` at step 272
   - Game eval metrics: `eval/game/*` at step 272
   - Dataset eval metrics: `eval/general/*` at step 272
   - All in the same run, same step

