# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 8 GPUs
# Note: `generate_max_length` is set to a small value to speed up the real generation
python -m oat.experiment.run_math_rl \
    --critic_type drgrpo \
    --gpus 8 \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward \
    --oracle math \
    --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --prompt_template r1_distill_qwen \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data ./data/train/math_12k \
    --train_split train \
    --input_key problem \
    --output_key answer \
    --max-train 9999999 \
    --num_prompt_epoch 20 \
    --prompt_max_length 1024 \
    --num_samples 8 \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 10 \
    --save_steps -1 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 128 \
    --eval_batch_size 200 \
    --eval_steps -1 \
    --eval_temperature 0 \
    --eval_top_p 0.95 \
    --eval_n 1 \
    --eval_generate_max_length 4096 \
    --eval_data ./data/evaluation_suite \
    --eval_input_key input \
    --debug \
    --dry_run \
    --dry_run_prompt_len 1024 \
    --dry_run_response_len 16384
