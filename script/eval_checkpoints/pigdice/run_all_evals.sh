#!/bin/bash
# Master script to launch all 8 evaluation scripts in parallel
# Each script runs on a separate GPU with a staggered start

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "Starting all evaluation scripts with staggered delays..."
echo "=================================================="

# Launch step 272 on GPU 0 immediately
echo "Launching step 272 evaluation on GPU 0..."
bash eval_step_272_gpu0.sh > logs/eval_step_272_gpu0.log 2>&1 &
PID_272=$!
echo "Started with PID: $PID_272"

# Wait 60 seconds before launching next
sleep 60

# Launch step 240 on GPU 1
echo "Launching step 240 evaluation on GPU 1..."
bash eval_step_240_gpu1.sh > logs/eval_step_240_gpu1.log 2>&1 &
PID_240=$!
echo "Started with PID: $PID_240"

# Wait 60 seconds before launching next
sleep 60

# Launch step 208 on GPU 2
echo "Launching step 208 evaluation on GPU 2..."
bash eval_step_208_gpu2.sh > logs/eval_step_208_gpu2.log 2>&1 &
PID_208=$!
echo "Started with PID: $PID_208"

# Wait 60 seconds before launching next
sleep 60

# Launch step 176 on GPU 3
echo "Launching step 176 evaluation on GPU 3..."
bash eval_step_176_gpu3.sh > logs/eval_step_176_gpu3.log 2>&1 &
PID_176=$!
echo "Started with PID: $PID_176"

# Wait 60 seconds before launching next
sleep 60

# Launch step 144 on GPU 4
echo "Launching step 144 evaluation on GPU 4..."
bash eval_step_144_gpu4.sh > logs/eval_step_144_gpu4.log 2>&1 &
PID_144=$!
echo "Started with PID: $PID_144"

# Wait 60 seconds before launching next
sleep 60

# Launch step 112 on GPU 5
echo "Launching step 112 evaluation on GPU 5..."
bash eval_step_112_gpu5.sh > logs/eval_step_112_gpu5.log 2>&1 &
PID_112=$!
echo "Started with PID: $PID_112"

# Wait 60 seconds before launching next
sleep 60

# Launch step 80 on GPU 6
echo "Launching step 80 evaluation on GPU 6..."
bash eval_step_080_gpu6.sh > logs/eval_step_080_gpu6.log 2>&1 &
PID_080=$!
echo "Started with PID: $PID_080"

# Wait 60 seconds before launching next
sleep 60

# Launch step 48 on GPU 7
echo "Launching step 48 evaluation on GPU 7..."
bash eval_step_048_gpu7.sh > logs/eval_step_048_gpu7.log 2>&1 &
PID_048=$!
echo "Started with PID: $PID_048"

echo "=================================================="
echo "All evaluation scripts have been launched!"
echo ""
echo "Process IDs:"
echo "  Step 272 (GPU 0): $PID_272"
echo "  Step 240 (GPU 1): $PID_240"
echo "  Step 208 (GPU 2): $PID_208"
echo "  Step 176 (GPU 3): $PID_176"
echo "  Step 144 (GPU 4): $PID_144"
echo "  Step 112 (GPU 5): $PID_112"
echo "  Step 80  (GPU 6): $PID_080"
echo "  Step 48  (GPU 7): $PID_048"
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

