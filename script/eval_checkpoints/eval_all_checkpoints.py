#!/usr/bin/env python3

import os
import subprocess
import re
import argparse
import sys
import multiprocessing
import math
import time

def extract_step_number(path):
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0

def run_evaluation_on_checkpoint(checkpoint_path, gpu_id, workspace_root):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    env["LD_LIBRARY_PATH"] = subprocess.run(
        ["python", "-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"],
        capture_output=True,
        text=True
    ).stdout.strip() + ":" + env.get("LD_LIBRARY_PATH", "")
    
    env["NCCL_CUMEM_ENABLE"] = "0"
    env["NCCL_TIMEOUT"] = "7200000"
    env["LP_DEBUG"] = "1"
    env["LP_LOG_LEVEL"] = "DEBUG"
    
    script_name = f"eval_{os.path.basename(checkpoint_path)}_gpu{gpu_id}"
    
    cmd = [
        "python", "train_spiral_eval.py",
        "--env_ids", "SimpleNegotiation-v1",
        "--use_llm_obs_wrappers", "True",
        "--eval_env_ids", "SimpleTak-v0", "IndianPoker-v1",
        "--eval_use_llm_obs_wrappers", "False", "True",
        "--eval_opponent_names", "google/gemini-2.0-flash-lite-001",
        "--eval_split", "all",
        "--gamma", "1",
        "--gpus", "1",
        "--gradient-checkpointing",
        "--num_samples", "1",
        "--rollout_batch_size", "128",
        "--dump_game_state_every", "1",
        "--num_envs", "1",
        "--rollout_batch_size_per_device", "16",
        "--pi_buffer_maxlen_per_device", "16",
        "--pretrain", checkpoint_path,
        "--enable_prefix_caching",
        "--collocate",
        "--vllm_sleep",
        "--vllm_gpu_ratio", "0.45",
        "--rnd-seed",
        "--learning_rate", "0.000001",
        "--lr_scheduler", "constant",
        "--lr_warmup_ratio", "0",
        "--num_ppo_epochs", "2",
        "--train_batch_size", "128",
        "--train_batch_size_per_device", "1",
        "--beta", "0",
        "--max_model_len", "12800",
        "--generate_max_length", "4096",
        "--max_context_length", "32768",
        "--temperature", "1.0",
        "--top_p", "1",
        "--eval_steps", "32",
        "--save_steps", "16",
        "--eval_games", "16",
        "--eval_temperature", "0.6",
        "--eval_top_p", "0.95",
        "--eval_generate_max_length", "4096",
        "--max_train", "51200",
        "--max_ckpt_save_num", "2",
        "--max_weight_save_num", "200",
        "--use-wb",
        "--wb-run-name", script_name,
        "--wb-project", "spiral",
        "--save-ckpt",
        "--eval_only"
        ]
    
    print(f"GPU {gpu_id}: Running evaluation for {checkpoint_path}")
    print(f"GPU {gpu_id}: CUDA_VISIBLE_DEVICES={gpu_id}")
    print(f"GPU {gpu_id}: Command: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, env=env, cwd=workspace_root)
    
    if process.returncode == 0:
        print(f"GPU {gpu_id}: Completed evaluation for {checkpoint_path}")
    else:
        print(f"GPU {gpu_id}: Failed evaluation for {checkpoint_path} (return code: {process.returncode})", file=sys.stderr)
    
    return process.returncode

def process_checkpoint_group(checkpoints, gpu_id, workspace_root, delay):
    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\nGPU {gpu_id}: Processing checkpoint {i+1}/{len(checkpoints)}: {checkpoint_path}")
        run_evaluation_on_checkpoint(checkpoint_path, gpu_id, workspace_root)
        if i < len(checkpoints) - 1:
            print(f"GPU {gpu_id}: Waiting {delay} seconds before next checkpoint...")
            time.sleep(delay)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints across multiple GPUs"
    )
    parser.add_argument(
        "input_path",
        help="Path containing checkpoint subdirectories (e.g., saved_models/)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)",
    )
    parser.add_argument(
        "--workspace_root",
        default="/ephemeral/games-workspace/spiral_new",
        help="Workspace root directory (default: /ephemeral/games-workspace/spiral_new)",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default="step_",
        help="Checkpoint directory prefix to match (default: step_)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=120,
        help="Time delay in seconds between each checkpoint evaluation (default: 120)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input path '{args.input_path}' does not exist.", file=sys.stderr)
        return 1
    
    checkpoint_dirs = [
        os.path.join(args.input_path, d)
        for d in os.listdir(args.input_path)
        if os.path.isdir(os.path.join(args.input_path, d)) and d.startswith(args.keyword)
    ]
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories starting with '{args.keyword}' found in '{args.input_path}'", file=sys.stderr)
        return 1
    
    checkpoint_dirs.sort(key=extract_step_number)
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories:")
    for i, d in enumerate(checkpoint_dirs, 1):
        print(f"  {i}. {d}")
    
    num_gpus = args.num_gpus
    group_size = math.ceil(len(checkpoint_dirs) / num_gpus)
    groups = [
        checkpoint_dirs[i * group_size : (i + 1) * group_size]
        for i in range(num_gpus)
    ]
    
    groups = [g for g in groups if g]
    
    print(f"\nSplitting {len(checkpoint_dirs)} checkpoints into {len(groups)} groups:")
    for i, group in enumerate(groups):
        print(f"  GPU {i}: {len(group)} checkpoints")
    
    processes = []
    for gpu_id, group_checkpoints in enumerate(groups):
        p = multiprocessing.Process(
            target=process_checkpoint_group,
            args=(group_checkpoints, gpu_id, args.workspace_root, args.delay),
        )
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("\nAll evaluations completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

