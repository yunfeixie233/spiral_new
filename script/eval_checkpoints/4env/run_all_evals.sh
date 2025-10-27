#!/bin/bash
# Master script to launch all 8 evaluation scripts in parallel
# Each script runs on a separate GPU with a staggered start

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "Starting all evaluation scripts with staggered delays..."
echo "=================================================="

# Launch step 192 on GPU 0 immediately
echo "Launching step 192 evaluation on GPU 0..."
bash eval_step_192_gpu0.sh > logs/eval_step_192_gpu0.log 2>&1 &
PID_192=$!
echo "Started with PID: $PID_192"

# Wait 60 seconds before launching next
sleep 60

# Launch step 176 on GPU 1
echo "Launching step 176 evaluation on GPU 1..."
bash eval_step_176_gpu1.sh > logs/eval_step_176_gpu1.log 2>&1 &
PID_176=$!
echo "Started with PID: $PID_176"

# Wait 60 seconds before launching next
sleep 60

# Launch step 160 on GPU 2
echo "Launching step 160 evaluation on GPU 2..."
bash eval_step_160_gpu2.sh > logs/eval_step_160_gpu2.log 2>&1 &
PID_160=$!
echo "Started with PID: $PID_160"

# Wait 60 seconds before launching next
sleep 60

# Launch step 144 on GPU 3
echo "Launching step 144 evaluation on GPU 3..."
bash eval_step_144_gpu3.sh > logs/eval_step_144_gpu3.log 2>&1 &
PID_144=$!
echo "Started with PID: $PID_144"

# Wait 60 seconds before launching next
sleep 60

# Launch step 128 on GPU 4
echo "Launching step 128 evaluation on GPU 4..."
bash eval_step_128_gpu4.sh > logs/eval_step_128_gpu4.log 2>&1 &
PID_128=$!
echo "Started with PID: $PID_128"

# Wait 60 seconds before launching next
sleep 60

# Launch step 112 on GPU 5
echo "Launching step 112 evaluation on GPU 5..."
bash eval_step_112_gpu5.sh > logs/eval_step_112_gpu5.log 2>&1 &
PID_112=$!
echo "Started with PID: $PID_112"

# Wait 60 seconds before launching next
sleep 60

# Launch step 96 on GPU 6
echo "Launching step 96 evaluation on GPU 6..."
bash eval_step_096_gpu6.sh > logs/eval_step_096_gpu6.log 2>&1 &
PID_096=$!
echo "Started with PID: $PID_096"

# Wait 60 seconds before launching next
sleep 60

# Launch step 80 on GPU 7
echo "Launching step 80 evaluation on GPU 7..."
bash eval_step_080_gpu7.sh > logs/eval_step_080_gpu7.log 2>&1 &
PID_080=$!
echo "Started with PID: $PID_080"

echo "=================================================="
echo "All evaluation scripts have been launched!"
echo ""
echo "Process IDs:"
echo "  Step 192 (GPU 0): $PID_192"
echo "  Step 176 (GPU 1): $PID_176"
echo "  Step 160 (GPU 2): $PID_160"
echo "  Step 144 (GPU 3): $PID_144"
echo "  Step 128 (GPU 4): $PID_128"
echo "  Step 112 (GPU 5): $PID_112"
echo "  Step 96  (GPU 6): $PID_096"
echo "  Step 80  (GPU 7): $PID_080"
echo ""
echo "Logs are being written to:"
echo "  $SCRIPT_DIR/logs/"
echo ""
echo "To monitor progress, use:"
echo "  tail -f logs/eval_step_*.log"
echo ""
echo "To check running processes:"
echo "  ps aux | grep train_spiral_eval"
echo ""
echo "=================================================="

# Wait for all background jobs to complete
wait

echo "All evaluation scripts have completed!"

