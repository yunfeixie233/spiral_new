#!/usr/bin/env python3

import os
import re
import math

def extract_step_number(path):
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0

def test_checkpoint_discovery(input_path, num_gpus=8):
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist.")
        return
    
    keyword = "step_"
    checkpoint_dirs = [
        os.path.join(input_path, d)
        for d in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, d)) and d.startswith(keyword)
    ]
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in '{input_path}'")
        return
    
    checkpoint_dirs.sort(key=extract_step_number)
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories:")
    for i, d in enumerate(checkpoint_dirs, 1):
        step = extract_step_number(d)
        print(f"  {i:2d}. {os.path.basename(d)} (step {step})")
    
    group_size = math.ceil(len(checkpoint_dirs) / num_gpus)
    groups = [
        checkpoint_dirs[i * group_size : (i + 1) * group_size]
        for i in range(num_gpus)
    ]
    
    groups = [g for g in groups if g]
    
    print(f"\nScheduling across {len(groups)} GPUs (group size: {group_size}):")
    for gpu_id, group in enumerate(groups):
        print(f"\n  GPU {gpu_id}: {len(group)} checkpoints")
        for ckpt in group:
            print(f"    - {os.path.basename(ckpt)}")
    
    print(f"\nTotal evaluations to run: {len(checkpoint_dirs)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_discovery.py INPUT_PATH [NUM_GPUS]")
        print("\nExample:")
        print("  python test_discovery.py /path/to/saved_models")
        print("  python test_discovery.py /path/to/saved_models 4")
        sys.exit(1)
    
    input_path = sys.argv[1]
    num_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    test_checkpoint_discovery(input_path, num_gpus)

