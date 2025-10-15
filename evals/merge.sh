BASE_MODEL="Qwen/Qwen3-4B-base"
LORA_PATH="LeonGuertler/qwen3-4b-game-tuned-lora"
TARGET_PATH=the-acorn-ai/Qwen3-4B-Leon-0515

export CUDA_VISIBLE_DEVICES=0

python3 merge_lora.py \
    --base-model $BASE_MODEL \
    --lora-path $LORA_PATH \
    --target-path $TARGET_PATH \
    --device "cuda"