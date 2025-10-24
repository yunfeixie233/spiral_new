# NCCL Timeout Fix

## Problem

The distributed training was experiencing NCCL timeout errors:
```
[Rank 4] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=5, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=3600000) ran for 3631340 milliseconds before timing out.
```

The timeout was always 1 hour (3600000ms) even when `NCCL_TIMEOUT` environment variable was set in the shell script.

## Root Cause

The issue was in `/oat/oat/utils/deepspeed.py` at line 241:

```python
def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
    ...
    deepspeed.init_distributed(timeout=timeout)
```

The timeout was hardcoded to 60 minutes and didn't read the `NCCL_TIMEOUT` environment variable.

## Solution

### 1. Code Fix (REQUIRED)

Modified `/oat/oat/utils/deepspeed.py` to read `NCCL_TIMEOUT` environment variable:

```python
def setup_distributed(self, timeout=None) -> None:
    ...
    # Check for NCCL_TIMEOUT environment variable
    if timeout is None:
        nccl_timeout_ms = os.environ.get('NCCL_TIMEOUT', None)
        if nccl_timeout_ms is not None:
            timeout_seconds = int(nccl_timeout_ms) / 1000.0
            timeout = timedelta(seconds=timeout_seconds)
            logging.info(f"[DeepSpeed NCCL Config] Using NCCL_TIMEOUT from environment: {nccl_timeout_ms}ms = {timeout}")
        else:
            timeout = timedelta(minutes=60)
            logging.info(f"[DeepSpeed NCCL Config] Using default timeout: {timeout}")
    
    logging.info(f"[DeepSpeed NCCL Config] Initializing distributed with timeout: {timeout}")
    deepspeed.init_distributed(timeout=timeout)
```

### 2. Environment Variable Configuration

All run scripts have been updated to set `NCCL_TIMEOUT=21600000` (6 hours):

- `/run.sh`
- `/script/reproduce/run_KuhnPoker.sh`
- `/script/reproduce/run_KuhnPoker_eval.sh`
- `/script/scaling/run_4b_4env.sh`

### 3. Additional Safety Measures in train_spiral.py

Added timeout handling for evaluation futures and max turn limits:

- Evaluation futures now have 600s (10 minute) timeout per game
- Evaluation episodes are truncated after 200 turns
- Invalid futures results are handled gracefully

## Recommended Timeout Values

- **NCCL_TIMEOUT**: 21600000ms (6 hours) - For long-running distributed operations
- **Evaluation future timeout**: 600s (10 minutes) - Per game evaluation
- **Max eval turns**: 200 - Maximum turns before truncating a game

## How to Verify

When running training, you should see log messages like:
```
[DeepSpeed NCCL Config] Using NCCL_TIMEOUT from environment: 21600000ms = 6:00:00
[DeepSpeed NCCL Config] Initializing distributed with timeout: 6:00:00
```

If you don't see these messages, the environment variable isn't being read correctly.
