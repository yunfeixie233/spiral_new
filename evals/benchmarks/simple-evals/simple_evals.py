import argparse
import json

import pandas as pd

from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .knowlogic_eval import KnowLogicEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .mmlu_pro_eval import MMLUProEval
from .sampler.chat_completion_sampler import ChatCompletionSampler
from .simpleqa_eval import SimpleQAEval


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    # =====
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--base_url", type=str, help="Base URL")
    parser.add_argument("--max_tokens", type=int, help="Max tokens")
    parser.add_argument("--tasks", type=str, nargs="+", help="Tasks to run")

    parser.add_argument("--output_path", type=str, help="Path to output metrics.csv")
    args = parser.parse_args()

    models = {}

    if args.model_name_or_path:
        model_name = args.model_name_or_path.split("/")[-1].lower()
        args.model = model_name
        models[model_name] = ChatCompletionSampler(
            model=args.model_name_or_path,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
        )

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        if args.model not in models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: models[args.model]}

    grading_sampler = ChatCompletionSampler(model="gpt-4o")
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu_pro":
                return MMLUProEval(num_examples=1 if debug_mode else num_examples)
            # case "mmlu":
            #     return MMLUEval(num_examples=1 if debug_mode else num_examples)
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else 10,
                )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if debug_mode else 10, num_examples=num_examples
                )
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug_mode else 250)
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "knowlogic":
                return KnowLogicEval(num_examples=10 if debug_mode else num_examples)
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug)
        # for eval_name in ["simpleqa", "mmlu", "math", "gpqa", "mgsm", "drop", "humaneval"]
        for eval_name in args.tasks
    }
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        score = result.get("f1_score", result.get("score", None))
        if "mmlu_pro" in eval_model_name:
            eval_name = "mmlu_pro"
            model_name = eval_model_name[eval_model_name.find("mmlu_pro") + 9 :]
        else:
            eval_name = eval_model_name[: eval_model_name.find("_")]
            model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        metrics = {}
        for key, value in result.items():
            if "std" not in key:
                metrics[key] = value
        merge_metrics.append(
            {
                "eval_name": eval_name,
                "model_name": model_name,
                "metric": metrics,
                "score": score,
            }
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())

    print("\nWriting to csv...")
    write_to_csv(args, merge_metrics)
    return merge_metrics


def write_to_csv(args, merge_metrics):
    output_path = args.output_path
    metrics_data = []
    for row in merge_metrics:
        for key, value in row["metric"].items():
            keyMap = {
                "score": "Score",
                "chars": "Avg Tokens",
                "invalid_count": "Invalid Format",
            }
            metrics_data.append(
                {
                    "key": f'{row["eval_name"]}/{keyMap[key] if key in keyMap else key}',
                    "value": value,
                }
            )
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
