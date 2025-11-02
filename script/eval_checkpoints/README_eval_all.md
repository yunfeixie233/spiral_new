# Evaluation Scheduler for SPIRAL Checkpoints

This script evaluates all checkpoints from a training run across multiple GPUs.

## How It Works

The scheduler follows the same pattern as `eval_all_6c_iter_mathvista.py`:

1. Discovers all checkpoint directories under the input path
2. Splits checkpoints into N groups (default: 8 for 8 GPUs)
3. Launches N parallel processes, each assigned to a different GPU
4. Each process evaluates its checkpoints sequentially

## Usage

```bash
python script/eval_checkpoints/eval_all_checkpoints.py INPUT_PATH [OPTIONS]
```

### Required Arguments

- `INPUT_PATH`: Path to the saved_models directory containing checkpoints

### Optional Arguments

- `--num_gpus N`: Number of GPUs to use (default: 8)
- `--workspace_root PATH`: Workspace root directory (default: /ephemeral/games-workspace/spiral_new)
- `--keyword PREFIX`: Checkpoint directory prefix to match (default: step_)

## Example

Evaluate all checkpoints from a training run:

```bash
python script/eval_checkpoints/eval_all_checkpoints.py \
    /ephemeral/games-workspace/spiral_new/oat-output/run_4b_4env_noresume_randenv_1028T0731/saved_models
```

Evaluate using only 4 GPUs:

```bash
python script/eval_checkpoints/eval_all_checkpoints.py \
    /ephemeral/games-workspace/spiral_new/oat-output/run_4b_4env_noresume_randenv_1028T0731/saved_models \
    --num_gpus 4
```

## Scheduling Details

If you have 24 checkpoints and 8 GPUs:
- Each GPU gets 3 checkpoints to evaluate sequentially
- GPU 0 evaluates checkpoints 1-3
- GPU 1 evaluates checkpoints 4-6
- ...
- GPU 7 evaluates checkpoints 22-24

When a GPU finishes its assigned checkpoints, it stops. All 8 processes run in parallel.

## Evaluation Command

Each checkpoint is evaluated using the same command as `eval_step_080_gpu7.sh`, with only the `--pretrain` path and `CUDA_VISIBLE_DEVICES` varying per checkpoint.

