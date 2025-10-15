#!/bin/bash

MODELS=(
  # Add models here in order of steps, otherwise, the wandb curve can't be plotted.
)

export CUDA_VISIBLE_DEVICES=4,5
SERVER_PORT=7000
for MODEL in ${MODELS[@]}; do
    MODEL_GROUP=$(echo $MODEL | sed -E 's/-step[_]?[0-9]+$//')
    echo $MODEL_GROUP
    MODEL_GROUP=$(basename $MODEL_GROUP)
    WANDB_ID=${MODEL_GROUP}-benchmarks
    echo "Running evaluation for model: $MODEL"
    echo "WANDB_ID: $WANDB_ID"
    bash eval.sh $MODEL $WANDB_ID $SERVER_PORT
done