#!/bin/bash
# Evaluation script for step 112 on GPU 5

export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export NCCL_TIMEOUT=7200000
export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG

# GPU Selection
export CUDA_VISIBLE_DEVICES=5

SCRIPT_NAME=$(basename "$0" .sh)

# Change to workspace root
cd /ephemeral/games-workspace/spiral

python train_spiral_eval.py \
    --env_ids PigDice-v1 \
    --use_llm_obs_wrappers False \
    --eval_env_ids SimpleTak-v0 IndianPoker-v1 \
    --eval_use_llm_obs_wrappers False True \
    --eval_opponent_names google/gemini-2.0-flash-lite-001 \
    --eval_split all \
    --gamma 1 \
    --gpus 1 \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --dump_game_state_every 1 \
    --num_envs 1 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain /ephemeral/games-workspace/spiral/oat-output/run_pig_noeval_1025T1515/saved_models/step_00112 \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.45 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --max_context_length 32768 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 32 \
    --save_steps 16 \
    --eval_games 16 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 51200 \
    --max_ckpt_save_num 2 \
    --max_weight_save_num 200 \
    --use-wb \
    --wb-run-name $SCRIPT_NAME \
    --wb-project spiral \
    --save-ckpt \
    --eval_only

