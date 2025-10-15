## Eval Game
### Parameters
- `MODEL_PATH`($1): The path to the model to evaluate.
- `OPPONENT_MODEL`($2): The path to the opponent model.
- `ENV_IDS`($3): The environment ids to evaluate on.
- `EPISODES`($4): The number of episodes to evaluate on.
- `MAX_SEQ_LEN`($5): The maximum sequence length.
- `OUTPUT_DIR`($6): The output directory.
- `GPUS`($7): The GPUs to use.
### Prerequisites
Before running the evaluation, you need to set up your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=<your-openrouter-api-key>
```
### Run with default values
```bash
./run.sh
```

### Run with meta-llama/Llama-3-8B as model and anthropic/claude-3-opus-20240229 as opponent
```bash
./run.sh "meta-llama/Llama-3-8B" "anthropic/claude-3-opus-20240229"
```

### Run with all custom parameters
```bash
./run.sh "meta-llama/Llama-3-8B" "anthropic/claude-3-opus-20240229" "0,1" "Chess-v0" 100 4096 "data/custom"
```