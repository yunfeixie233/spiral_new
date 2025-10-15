#!/bin/bash
MODELS=(
  # Add models here in order of steps, otherwise, the wandb curve can't be plotted.
)

for MODEL in ${MODELS[@]}; do
  MODEL_GROUP=$(echo $MODEL | sed -E 's/-step[_]?[0-9]+$//')
  MODEL_GROUP=$(basename $MODEL_GROUP)
  WANDB_ID=${MODEL_GROUP}-games
  echo "WANDB_ID: $WANDB_ID"

  bash evals/game/run.sh \
    --model_path $MODEL \
    --opponent google/gemini-2.0-flash-001 \
    --gpus 1,2 \
    --episodes 512 \
    --base_port 8050 \
    --wandb_id $WANDB_ID
done
