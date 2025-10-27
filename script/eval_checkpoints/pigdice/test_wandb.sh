#!/bin/bash
# Simple test script for wandb parallel writing

cd /ephemeral/games-workspace/spiral

echo "Testing WandB parallel writing behavior..."
echo ""
echo "This will test if wandb can handle multiple processes writing to"
echo "the same run ID at different steps with time gaps between them."
echo ""

# Test 3: Realistic scenario (most similar to actual eval scripts)
echo "Running Test 3: Realistic eval scenario"
echo "This simulates running 3 evaluations sequentially with 15-second gaps"
echo ""

python script/eval_checkpoints/test_wandb_parallel.py --test 3

echo ""
echo "Test complete!"
echo "Check the wandb dashboard to verify if all steps were logged correctly."

