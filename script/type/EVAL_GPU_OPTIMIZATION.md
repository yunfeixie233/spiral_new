# Evaluation GPU Optimization Guide

## Problem Identified

When running evaluation with `--eval_only` flag using 8 GPUs:
- **Only GPU 0 (rank 0)** performs actual evaluation work
- **GPUs 1-7** are blocked at `dist.barrier()` in `oat/oat/learners/base.py:706`
- GPUs 1-7 show 100% utilization due to busy-waiting at barriers, but do no useful work
- This wastes 7 out of 8 GPUs during evaluation

## Root Cause

In `oat/oat/learners/base.py`, the `evaluate()` method (lines 635-722):

```python
def evaluate(self, dataloader, steps):
    # ... setup code ...
    
    if self.strategy.is_rank_0():
        # ALL evaluation work happens here (lines 649-702)
        # - Loading data
        # - Dispatching to actors
        # - Collecting results
        # - Computing metrics
        pass
    
    # Ranks 1-7 skip all work above and jump directly here:
    dist.barrier(group=self._same_actor_group)  # Line 704
    logging.info(f"rank {self.strategy.get_rank()} cpubarrier done")
    dist.barrier()  # Line 706 - WHERE RANKS 1-7 ARE STUCK
    
    # Broadcast results from rank 0 to all ranks
    win_rate = self.strategy.broadcast(win_rate)
    # ... more broadcasts ...
```

## Solution

For `--eval_only` runs, reduce GPU count to 1:

### Before (Inefficient)
```bash
--gpus 8          # 7 GPUs wasted waiting at barriers
--collocate
--eval_only
```

### After (Optimized)
```bash
--gpus 1          # Only 1 GPU needed since only rank 0 works
--collocate
--eval_only
```

## Implementation

### Updated Scripts

1. **`run_pig_eval.sh`** - Changed `--gpus 8` to `--gpus 1`
2. **`run_pig_eval_gpu3.sh`** - Example showing how to run on specific GPU

### Using a Specific GPU

To run on a specific GPU (e.g., GPU 3):

```bash
export CUDA_VISIBLE_DEVICES=3
bash script/type/run_pig_eval.sh
```

Or use the dedicated script:
```bash
bash script/type/run_pig_eval_gpu3.sh
```

**How it works** (after the fix in `oat/oat/interface.py`):
1. You set `export CUDA_VISIBLE_DEVICES=3` before running the script
2. The framework detects this and sets `gpu_offset = 3`
3. With `--gpus 1`, it calculates `learner_gpus = [0 + 3] = [3]`
4. The learner and actor processes are launched on GPU 3

**Before the fix**: The framework would always use GPU 0, ignoring `CUDA_VISIBLE_DEVICES`

## GPU Allocation Details

With `--gpus 1` and `--collocate`:
- 1 Learner process (rank 0) on GPU 0
- 1 Actor process on GPU 0 (collocated)
- Total: 1 GPU utilized

With `--gpus 8` and `--collocate` (previous setup):
- 8 Learner processes (ranks 0-7) on GPUs 0-7
- 8 Actor processes on GPUs 0-7 (collocated)
- Result: 7 GPUs waste resources waiting at barriers

## Performance Comparison

| Configuration | GPUs Used | GPUs Working | GPU Waste | Notes |
|--------------|-----------|--------------|-----------|-------|
| `--gpus 8` | 8 | 1 | 87.5% | GPUs 1-7 busy-wait at barriers |
| `--gpus 1` | 1 | 1 | 0% | Optimal for eval-only |

## When to Use Each Configuration

### Use `--gpus 1` (Optimized)
- When running `--eval_only` (pure evaluation)
- When only dataset evaluation is needed
- When only game evaluation is needed
- Saves 7 GPUs for other tasks

### Use `--gpus 8` (Full Setup)
- When training AND evaluating (normal RL training)
- During active learning/self-play
- When learner processes need distributed training

## Verification

To verify the optimization is working:

```bash
# In another terminal, monitor GPU usage:
watch -n 1 nvidia-smi

# Expected for --gpus 1:
# - Only GPU 0 (or selected GPU) shows activity
# - Other GPUs remain free

# Expected for --gpus 8 (old behavior):
# - GPU 0 shows fluctuating utilization (doing work)
# - GPUs 1-7 show 100% utilization (busy-waiting)
```

## Code References

- **Evaluation orchestration**: `oat/oat/learners/base.py:635-722` (evaluate method)
- **Barrier location**: `oat/oat/learners/base.py:706`
- **GPU allocation logic**: `oat/oat/interface.py:30-132` (get_program function)
- **SelfPlay evaluation**: `train_spiral_eval.py:932-1066` (evaluate method override)

## Fix Applied: CUDA_VISIBLE_DEVICES Respect

### Problem
The original code in `oat/oat/interface.py` always calculated `gpu_offset` based on `args.gpus` and ignored the `CUDA_VISIBLE_DEVICES` environment variable set by the user. This meant:
```bash
export CUDA_VISIBLE_DEVICES=3  # User wants GPU 3
# But code always used GPU 0!
```

### Solution (Lines 40-56 in oat/oat/interface.py)
```python
# Check if CUDA_VISIBLE_DEVICES is set to use specific GPUs
cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if cuda_visible_devices is not None and cuda_visible_devices.strip():
    # Parse the specified GPUs
    specified_gpus = [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]
    if specified_gpus:
        # Use the first GPU as the base offset
        gpu_offset = specified_gpus[0]
        # ...
```

Now the framework respects user's GPU selection via `CUDA_VISIBLE_DEVICES`.

## Future Improvements

To utilize all GPUs during evaluation, the `evaluate()` method would need to:
1. Distribute evaluation batches across all ranks (not just rank 0)
2. Use `DistributedSampler` for the evaluation dataloader
3. Gather results from all ranks at the end

This would require modifying `oat/oat/learners/base.py:evaluate()` method.

