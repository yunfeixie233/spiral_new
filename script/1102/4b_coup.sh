# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Common =========
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export NCCL_TIMEOUT=3600000

# Verify NCCL timeout is set
echo "NCCL_TIMEOUT is set to: $NCCL_TIMEOUT milliseconds"

export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG
SCRIPT_NAME=$(basename "$0" .sh)

# Notes ==========
# Coup is a 2-6 player bluffing game with hidden roles and deception mechanics.
# Key aspects requiring full history:
#   - Bluffing patterns: Track who claims what cards and when
#   - Card counting: Remember revealed cards to estimate remaining possibilities
#   - Challenge history: Past challenges inform future BULLSHIT decisions
#   - Block patterns: Track blocking behavior to predict card holdings
# 
# Therefore: `--use_llm_obs_wrappers True` to enable full history
# 
# Setting `--save_steps 16` to save checkpoints every 16 policy iteration steps.
# Set `--eval_opponent_names google/gemini-2.0-flash-lite-001` if you have OpenRouter access.
# `--env_sampling_mode`: Controls multi-env trajectory collection strategy
#   - "split" (default): Split trajectories evenly across all envs (balanced multi-task)
#   - "random": Randomly pick ONE env per step for all trajectories (curriculum learning)
# 
# REINFORCE Configuration (RECOMMENDED for self-play):
# `--critic_type reinforce`: Simple REINFORCE algorithm (no baseline in advantage computation)
# `--num_samples 1`: One trajectory per game (natural for self-play)
# `--use_role_baseline True`: Use role-specific baseline for variance reduction
# 
# This configuration is optimal for self-play because:
#   - Role baseline provides variance reduction without conflicting with advantage computation
#   - No grouping required (works naturally with num_samples=1)
#   - Clean separation: role baseline in actor, REINFORCE in learner
pkill -u ubuntu python
python train_spiral_eval.py \
    --critic_type reinforce \
    --env_ids Coup-v0 \
    --use_llm_obs_wrappers True \
    --eval_env_ids Coup-v0 \
    --eval_use_llm_obs_wrappers True \
    --eval_opponent_names random google/gemini-2.0-flash-lite-001 \
    --eval_split all \
    --env_sampling_mode random \
    --gamma 1 \
    --gpus 8 \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --dump_game_state_every 1 \
    --num_envs 1 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain ./checkpoint/Qwen3-4B-Base \
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
    --save_steps 32 \
    --eval_games 16 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 128000 \
    --max_ckpt_save_num 2 \
    --max_weight_save_num 200 \
    --use-wb \
    --wb-run-name $SCRIPT_NAME \
    --wb-project spiral \
    --save-ckpt \
    --debug \
    --skip_game_eval \
    --skip_dataset_eval 



