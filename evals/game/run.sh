#!/bin/bash
# Defaults
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OPPONENT_MODEL="google/gemini-2.0-flash-001"
GPUS="0,1"
RUN_NAME_SUFFIX=""
TEMPERATURE=0.7
TOP_P=0.9
ENV_IDS="TicTacToe-v0"
EPISODES=512
MAX_SEQ_LEN=8192
OUTPUT_DIR="evals/game/data/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"

usage() {
  echo "Usage: $0 [--model_path PATH] [--opponent MODEL] [--gpus GPUS] [--run_name_suffix SUFFIX] [--temperature TEMP] [--top_p TOPP] [--env_ids IDS] [--episodes N] [--max_seq_len N] [--output_dir DIR]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model_path)
      MODEL_PATH="$2"; shift; shift;;
    --opponent)
      OPPONENT_MODEL="$2"; shift; shift;;
    --gpus)
      GPUS="$2"; shift; shift;;
    --run_name_suffix)
      RUN_NAME_SUFFIX="$2"; shift; shift;;
    --temperature)
      TEMPERATURE="$2"; shift; shift;;
    --top_p)
      TOP_P="$2"; shift; shift;;
    --env_ids)
      ENV_IDS="$2"; shift; shift;;
    --episodes)
      EPISODES="$2"; shift; shift;;
    --max_seq_len)
      MAX_SEQ_LEN="$2"; shift; shift;;
    --output_dir)
      OUTPUT_DIR="$2"; shift; shift;;
    --base_port)
      BASE_PORT="$2"; shift; shift;;
    --wandb_id)
      WANDB_ID="$2"; shift; shift;;
    -h|--help)
      usage;;
    *)
      echo "Unknown option $1"; usage;;
  esac
 done

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Extract step number from model name if it follows the pattern of having "step" followed by numbers
if [[ $MODEL_PATH =~ step[_-]?([0-9]+) ]]; then
    STEP="${BASH_REMATCH[1]}"
    echo "Extracted step: $STEP"
else
    STEP="None"
    echo "No step number found in model name"
fi


RUN_NAME=$(basename "$MODEL_PATH")
echo "Extracted run name: $RUN_NAME"

if [[ $MODEL_PATH =~ ^(.+)[_-]?step[_-]?[0-9]+$ ]]; then
    WANDB_GROUP="${BASH_REMATCH[1]}"
    WANDB_GROUP="${WANDB_GROUP%-}"
    WANDB_GROUP="${WANDB_GROUP%_}"
else
    WANDB_GROUP="$MODEL_PATH"
fi

echo "Running evaluation with:"
echo "Model: $MODEL_PATH"
echo "Opponent: $OPPONENT_MODEL"
echo "Environment: $ENV_IDS"
echo "Episodes: $EPISODES"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Run name suffix: $RUN_NAME_SUFFIX"
echo "Temperature: $TEMPERATURE"
echo "Top P: $TOP_P"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "WANDB_GROUP: $WANDB_GROUP"
echo "STEP: $STEP"

EXTRA_ARGS=""
if [[ -n "$STEP" ]]; then
  EXTRA_ARGS="--step $STEP"
fi

if [[ -n "$WANDB_ID" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --wandb_id $WANDB_ID"
fi

# TODO: Add support to KuhnPoker-v1 and LiarsDice-v1 and more environments.
# (spatial reasoning) Snake
# (spatial reasoning) Connect Four
# (spatial reasoning) SimpleTak
# (strategic reasoning) Poker
# (strategic reasoning) Liars dice
# (strategic reasoning) Nim
# (ToM reasoning) 20 questions
# (ToM reasoning) SimpleNegotiation -> DontSayIt
# (ToM reasoning) Auction Games

set -x
python3 evals/game/eval.py \
    --model_path "$MODEL_PATH" \
    --opponent $OPPONENT_MODEL \
    --env-ids TicTacToe-v0 KuhnPoker-v1 Nim-v0 \
    --episodes "$EPISODES" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --output-dir "$OUTPUT_DIR" \
    --gpus "$GPUS" \
    --run_name_suffix "$RUN_NAME_SUFFIX" \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --base_port $BASE_PORT \
    --wandb_group "$WANDB_GROUP" \
    $EXTRA_ARGS