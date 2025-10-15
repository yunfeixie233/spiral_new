import argparse
import os

from evals.game.collectors import VLLMCollector, get_openrouter_model_ids
from evals.game.wandb_summary import analyze


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Collect RL data using a model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The model used for data collection",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="The opponent model used for data collection",
    )
    parser.add_argument(
        "--env-ids", nargs="+", required=True, help="List of environment IDs", type=str
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="The number games to be evaluated",
    )
    parser.add_argument(
        "--max-seq-len", type=int, required=True, help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Base directory for storing data"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top-p", type=float, required=False, default=0.9, help="Generation top-p"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        required=False,
        default=512,
        help="The number of envs running in parallel",
    )
    parser.add_argument(
        "--vllm-max-num-seq",
        type=int,
        required=False,
        default=64,
        help="The number of seq being generated in parallel on vllm",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=False,
        default=None,
        help="The GPUs to use for data collection",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        required=False,
        default=1,
        help="The number of GPUs to use for data collection",
    )
    parser.add_argument(
        "--base_port",
        type=int,
        required=False,
        default=8000,
        help="The base port for the vllm servers",
    )
    parser.add_argument(
        "--run_name_suffix",
        type=str,
        required=False,
        default="",
        help="The suffix to add to the run name",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        required=False,
        default="",
        help="The group to add to the run name",
    )
    parser.add_argument(
        "--step",
        required=False,
        default=None,
        help="The step to use for the model",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        required=False,
        default=None,
        help="The id to use for the run",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="If set, do not use wandb for logging",
    )

    return parser.parse_args()


def collect_data(args):
    if args.step is not None:
        args.step = int(args.step)

    if args.gpus is not None:
        gpus = [int(gpu) for gpu in args.gpus.split(",")]
    else:
        gpus = None

    if isinstance(args.opponents, str):
        args.opponents = [args.opponents]

    # Define storage paths
    data_folder = os.path.join(args.output_dir, "data")
    os.makedirs(data_folder, exist_ok=True)

    logging_folder = os.path.join(args.output_dir, "logging")
    os.makedirs(logging_folder, exist_ok=True)

    # Run the data collection and optionally evaluation
    openrouter_model_ids = get_openrouter_model_ids()
    checkpoint_paths = []
    if args.model_path not in openrouter_model_ids:
        checkpoint_paths.append(args.model_path)
    if args.opponents is not None:
        for opponent in args.opponents:
            if (opponent not in openrouter_model_ids) and (opponent != "random"):
                checkpoint_paths.append(opponent)

    with VLLMCollector(
        env_ids=args.env_ids,
        checkpoint_paths=checkpoint_paths,
        output_dir=args.output_dir,
        max_new_tokens=args.max_seq_len,
        max_workers=args.max_workers,
        vllm_max_num_seq=args.vllm_max_num_seq,
        tensor_parallel_size=args.tensor_parallel_size,
        base_port=args.base_port,
        gpus=gpus,
    ) as collector:
        print(f"[Evaluation] running {args.episodes} of evaluation on {args.env_ids}")
        collector.evaluate(
            env_ids=args.env_ids,
            num_episodes=args.episodes,
            model_name=args.model_path,
            opponent_names=args.opponents,
        )


def analyze_data(args):
    if args.wandb_group != "":
        wandb_run_name = args.wandb_group + args.run_name_suffix
    else:
        wandb_run_name = args.model_path + args.run_name_suffix

    if args.wandb_id is not None:
        wandb_id = args.wandb_id
    else:
        wandb_id = wandb_run_name.split("/")[-1]

    analyze(
        data_folder=args.output_dir,
        wandb_org="stlm",
        wandb_project="oat-game-eval",
        wandb_run_name=wandb_id,
        wandb_id=wandb_id,
        step=args.step,
        no_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    args = parse_args()
    collect_data(args)
    analyze_data(args)
