# 4env Checkpoint Evaluation Scripts

This directory contains scripts to evaluate checkpoints from the 4-environment training run.

## Structure

- `eval_step_XXX_gpuY.sh`: Individual evaluation scripts for each checkpoint
  - 8 scripts total, covering steps 192 down to 80 (intervals of 16)
  - Each script runs on a dedicated GPU (GPU 0-7)
  
- `run_all_evals.sh`: Master script to launch all evaluations in parallel
  - Launches scripts with 60-second delays between each
  - Redirects output to individual log files
  
- `logs/`: Directory containing execution logs for each evaluation

## Checkpoints Evaluated

From `/ephemeral/games-workspace/spiral/oat-output/run_4b_4env_1026T1829/saved_models`:

- Step 192 on GPU 0
- Step 176 on GPU 1
- Step 160 on GPU 2
- Step 144 on GPU 3
- Step 128 on GPU 4
- Step 112 on GPU 5
- Step 96 on GPU 6
- Step 80 on GPU 7

## Usage

To run all evaluations:

```bash
cd /ephemeral/games-workspace/spiral/script/eval_checkpoints/4env
./run_all_evals.sh
```

To run a single evaluation:

```bash
./eval_step_192_gpu0.sh
```

## Monitoring

Monitor progress of all evaluations:

```bash
tail -f logs/eval_step_*.log
```

Check running processes:

```bash
ps aux | grep train_spiral_eval
```

Check specific evaluation:

```bash
tail -f logs/eval_step_192_gpu0.log
```

## Configuration

Each evaluation script:
- Uses the 4-environment setup (KuhnPoker, SimpleNegotiation, TicTacToe, PigDice)
- Evaluates against google/gemini-2.0-flash-lite-001
- Runs in eval_only mode with skip_game_eval
- Logs to wandb with run name matching the script name

