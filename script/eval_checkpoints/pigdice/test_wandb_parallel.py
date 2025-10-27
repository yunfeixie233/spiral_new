#!/usr/bin/env python3
"""
Test script to verify if wandb can handle multiple processes
writing to the same run at different steps.

This simulates what happens when running multiple eval scripts in parallel.
"""

import os
import sys
import time
import wandb

def test_wandb_write(exp_name, step, process_id, delay=0):
    """
    Test writing to wandb at a specific step.
    
    Args:
        exp_name: The experiment name (and ID) for wandb
        step: The step number to log at
        process_id: ID of this process (for identification)
        delay: Delay in seconds before starting
    """
    if delay > 0:
        print(f"Process {process_id}: Waiting {delay} seconds before starting...")
        time.sleep(delay)
    
    print(f"Process {process_id}: Starting at step {step}")
    
    try:
        # Initialize wandb with same ID (simulating eval_only mode)
        run = wandb.init(
            project="spiral_test",
            name=exp_name,
            id=exp_name,  # Same ID for all processes
            resume="allow",
            reinit=True,
        )
        
        print(f"Process {process_id}: WandB initialized with run ID: {run.id}")
        
        # Log some dummy metrics at the specified step
        metrics = {
            f"test/metric_a": step * 1.1 + process_id,
            f"test/metric_b": step * 2.2 + process_id,
            f"test/process_id": process_id,
        }
        
        wandb.log(metrics, step=step)
        print(f"Process {process_id}: Logged metrics at step {step}")
        
        # Simulate some work
        time.sleep(2)
        
        # Finish the run
        wandb.finish()
        print(f"Process {process_id}: Finished successfully")
        
        return True
        
    except Exception as e:
        print(f"Process {process_id}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequential_with_gaps():
    """Test sequential writes with time gaps (like staggered parallel runs)"""
    print("=" * 80)
    print("TEST 1: Sequential writes with 10-second gaps")
    print("=" * 80)
    
    exp_name = "test_wandb_sequential"
    steps = [272, 240, 208, 176, 144, 112, 80, 48]
    
    results = []
    for i, step in enumerate(steps):
        delay = i * 10  # 0s, 10s, 20s, 30s, etc.
        print(f"\n--- Starting process {i} for step {step} (after {delay}s delay) ---")
        success = test_wandb_write(exp_name, step, i, delay=delay)
        results.append((step, success))
        
    print("\n" + "=" * 80)
    print("TEST 1 RESULTS:")
    for step, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  Step {step}: {status}")
    print("=" * 80)


def test_truly_parallel():
    """Test truly parallel writes (multiple processes at same time)"""
    print("\n" + "=" * 80)
    print("TEST 2: Truly parallel writes using multiprocessing")
    print("=" * 80)
    
    import multiprocessing
    
    exp_name = "test_wandb_parallel"
    steps = [272, 240, 208, 176]
    
    # Start all processes at the same time
    processes = []
    for i, step in enumerate(steps):
        p = multiprocessing.Process(
            target=test_wandb_write,
            args=(exp_name, step, i, 0)  # No delay
        )
        processes.append(p)
        p.start()
        print(f"Started process {i} for step {step}")
    
    # Wait for all to complete
    for i, p in enumerate(processes):
        p.join()
        print(f"Process {i} completed")
    
    print("=" * 80)


def test_same_run_different_times():
    """Test writing to same run at different times (realistic scenario)"""
    print("\n" + "=" * 80)
    print("TEST 3: Same run, different times (realistic eval scenario)")
    print("=" * 80)
    
    exp_name = "test_wandb_realistic"
    
    # First write (simulating step 272 evaluation)
    print("\n--- First evaluation (step 272) ---")
    test_wandb_write(exp_name, 272, 1, delay=0)
    
    # Wait before second write
    print("\n--- Waiting 15 seconds before next evaluation ---")
    time.sleep(15)
    
    # Second write (simulating step 240 evaluation)
    print("\n--- Second evaluation (step 240) ---")
    test_wandb_write(exp_name, 240, 2, delay=0)
    
    # Wait before third write
    print("\n--- Waiting 15 seconds before next evaluation ---")
    time.sleep(15)
    
    # Third write (simulating step 208 evaluation)
    print("\n--- Third evaluation (step 208) ---")
    test_wandb_write(exp_name, 208, 3, delay=0)
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WandB parallel writing")
    parser.add_argument(
        "--test",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Which test to run: 1=sequential with gaps, 2=truly parallel, 3=realistic scenario"
    )
    
    args = parser.parse_args()
    
    # Check if wandb is configured
    if not os.getenv("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set. Wandb might require manual login.")
        print("You can set it with: export WANDB_API_KEY=your_key")
        print()
    
    if args.test == 1:
        test_sequential_with_gaps()
    elif args.test == 2:
        test_truly_parallel()
    elif args.test == 3:
        test_same_run_different_times()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("Check wandb dashboard at: https://wandb.ai/your_org/spiral_test")
    print("=" * 80)

