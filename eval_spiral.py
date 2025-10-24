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

"""Evaluation script for trained SPIRAL models."""

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
import numpy as np
import textarena as ta
import torch
import vllm
from oat.args import get_default_args
from oat.utils.data import load_data_from_disk_or_hf
from torch.utils.data import DataLoader
from tqdm import tqdm

load_dotenv()

from spiral.agents.random import RandomAgent
from spiral.agents.utils import get_valid_action_parser
from spiral.components import MATHOracle
from spiral.envs import make_env
from spiral.metrics import EvaluationMetrics
from spiral.template import TEMPLATE_FACTORY
from spiral.utils import extract_boxed_answer

logging.basicConfig(level=logging.DEBUG)

INVALID_ACTION = "[｜INVALID_ACTION｜]"


@dataclass
class EvalArgs:
    """Arguments for evaluation."""
    
    # Model settings
    pretrain: str = ""
    checkpoint_path: str = ""
    
    # Game evaluation
    eval_games: int = 16
    eval_env_ids: List[str] = field(
        default_factory=lambda: ["TicTacToe-v0", "KuhnPoker-v1", "SimpleNegotiation-v1"]
    )
    eval_use_llm_obs_wrappers: List[bool] = field(default_factory=lambda: [False, True, True])
    eval_opponent_names: List[str] = field(
        default_factory=lambda: ["random", "google/gemini-2.0-flash-lite-001"]
    )
    eval_prompt_template: Literal["qwen3_general", "r1_general", "llama_instruct_general"] = "qwen3_general"
    
    # Template settings
    prompt_template: Literal["qwen3", "r1", "llama_instruct"] = "qwen3"
    prompt_template_overrides: str = ""
    
    # Math reasoning evaluation
    eval_data: Optional[str] = "./data"
    eval_input_key: str = "input"
    eval_output_key: str = "answer"
    eval_split: str = "all"
    eval_batch_size: int = 32
    
    # Evaluation control
    skip_game_eval: bool = False
    skip_dataset_eval: bool = False
    
    # Generation parameters
    eval_temperature: float = 0.0
    eval_top_p: float = 1.0
    eval_top_k: int = -1
    eval_generate_max_length: int = 2048
    
    # vLLM settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    
    # Output settings
    output_dir: str = "./eval_results"
    save_individual_results: bool = True


class SpiralEvaluator:
    """Evaluator for trained SPIRAL models."""
    
    def __init__(self, args: EvalArgs):
        self.args = args
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize model
        logging.info(f"Loading model from {args.checkpoint_path or args.pretrain}")
        self.llm = vllm.LLM(
            model=args.checkpoint_path or args.pretrain,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        # Initialize oracle for dataset evaluation
        self.oracle = MATHOracle(
            args.eval_prompt_template, "fast", correct_reward=1, incorrect_reward=0
        )
        
        # Set up sampling parameters
        self.eval_sampling_params = vllm.SamplingParams(
            temperature=args.eval_temperature,
            top_p=args.eval_top_p,
            top_k=args.eval_top_k,
            max_tokens=args.eval_generate_max_length,
            n=1,
            logprobs=True,
        )
        
        # Parse template overrides
        self._template_overrides = self._parse_template_overrides(
            args.prompt_template_overrides
        )
        
        # Load dataset for evaluation if needed
        if not args.skip_dataset_eval and args.eval_data:
            self.eval_dataset_dict = load_data_from_disk_or_hf(args.eval_data)
            if args.eval_split != "all":
                self.eval_dataset_dict = {
                    k: v
                    for k, v in self.eval_dataset_dict.items()
                    if k in args.eval_split.split(",")
                }
        else:
            self.eval_dataset_dict = {}
    
    def _parse_template_overrides(self, override_str: str) -> Dict[str, str]:
        """Parse template overrides from string format 'env1:template1,env2:template2'."""
        if not override_str:
            return {}
        
        overrides = {}
        for pair in override_str.split(","):
            if ":" in pair:
                env, template = pair.split(":")
                overrides[env.strip()] = template.strip()
        return overrides
    
    def generate(self, prompts, sampling_params):
        """Generate responses using vLLM."""
        return self.llm.generate(prompts, sampling_params)
    
    def agent_act(self, observation: str, env_id: str) -> str:
        """Use the LLM as a policy to act.
        
        Args:
            observation: Observation from TextArena environment.
            env_id: Environment ID.
            
        Returns:
            Action string.
        """
        # Get template for this specific environment
        template_name = self._template_overrides.get(
            env_id, self.args.prompt_template
        )
        
        formatted_observation = TEMPLATE_FACTORY[template_name](
            observation, system_prompt=None
        )
        
        outputs = self.generate([formatted_observation], self.eval_sampling_params)
        raw_action = outputs[0].outputs[0].text
        
        # Chat-based extraction for environments with infinite/unbounded action spaces
        if env_id in ["DontSayIt-v0", "SimpleNegotiation-v1", "IndianPoker-v1", "TwoDollar-v1"]:
            clean_action = self.extract_chat_action(raw_action)
        else:
            action_space = get_valid_action_parser(env_id)(observation)
            clean_action = self.extract_action(raw_action, action_space)
        
        return clean_action
    
    def extract_chat_action(self, text: str) -> str:
        """Extract action for chat-based environments."""
        answer_match = extract_boxed_answer(text)
        
        if answer_match is not None:
            raw_action = answer_match.strip()
            if raw_action.strip("\n ") == "":
                return INVALID_ACTION
            return raw_action
        else:
            return INVALID_ACTION
    
    def extract_action(self, text: str, action_space: list) -> str:
        """
        Extract and format the actual action from the model's output.

        This method handles different template formats and ensures the action
        is properly formatted for the environment.

        Args:
            text: Raw text output from the model

        Returns:
            Cleaned and formatted action string ready for the environment
        """
        if not text:
            return ""  # Handle empty text case

        try:
            # First extract the raw action based on template format
            raw_action = ""

            if self.args.prompt_template == "r1":
                # Extract content from <answer> tags
                answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

                if answer_match:
                    # Found answer tags
                    raw_action = answer_match.group(1).strip()
                else:
                    # Fallback: try to find content after </think> if no answer tags
                    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
                    if think_match:
                        # Get position after </think> tag
                        think_end_pos = text.find("</think>") + len("</think>")
                        # Extract everything after the closing think tag
                        raw_action = text[think_end_pos:].strip()
                    else:
                        # No tags found, use the whole text
                        raw_action = text.strip()

            elif self.args.prompt_template == "deepseek_r1_distill_qwen":
                # Extract content from \boxed{} notation
                boxed_match = re.search(r"\\boxed\{([^}]*)\}", text, re.DOTALL)

                if boxed_match:
                    # Found boxed content
                    raw_action = boxed_match.group(1).strip()
                else:
                    # Fallback: try to find content after </think> tag
                    think_match = re.search(r"</think>(.*)", text, re.DOTALL)
                    if think_match:
                        raw_action = think_match.group(1).strip()
                    else:
                        # No tags found, use the whole text
                        raw_action = text.strip()

            elif self.args.prompt_template in ["qwen", "qwen3", "llama_instruct"]:
                raw_action = extract_boxed_answer(text)
                if raw_action is None:
                    raw_action = text.strip()
                    
            elif self.args.prompt_template in ["octothinker", "octothinker_enforce_thinking"]:
                # OctoThinker templates use \boxed{} format for actions
                raw_action = extract_boxed_answer(text)
                if raw_action is None:
                    # Fallback: if enforce_thinking, try to get content after </think>
                    if "octothinker_enforce_thinking" in self.args.prompt_template:
                        think_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
                        if think_match:
                            raw_action = think_match.group(1).strip()
                        else:
                            raw_action = text.strip()
                    else:
                        raw_action = text.strip()

            else:
                raise NotImplementedError

            # Now apply any necessary formatting to make the action valid for the environment

            # 1. Convert \boxed{} format to [content] format if found in the action
            formatted_action = re.sub(
                r"\\boxed\{([^}]*)\}",  # Match \boxed{...} capturing everything up to the matching }
                r"[\1]",  # Replace with brackets around the captured content
                raw_action,
            )

            # 2. If there are no brackets but we should have them, add them
            if "[" not in formatted_action and "]" not in formatted_action:
                # Check if this is a short action that likely needs brackets
                words = formatted_action.split()
                if (
                    len(words) <= 5
                ):  # Heuristic for a short action that might need brackets
                    formatted_action = f"[{formatted_action}]"

            # 3. Additional cleaning to ensure valid formatting
            # Remove any extra newlines, tabs, or multiple spaces
            formatted_action = re.sub(r"\s+", " ", formatted_action).strip()

            # NOTE(zc): ad-hoc postprocessing, strictly enforcing action space.
            if formatted_action not in action_space:
                formatted_action = INVALID_ACTION

            return formatted_action

        except Exception as e:
            logging.error(f"Error in extract_action: {e}")
            # Return invalid action if extraction fails.
            return INVALID_ACTION
    
    def run_eval_episode(self, env_id: str, opponent_name: str, player_id: int = 0) -> Dict:
        """Run a single evaluation episode.
        
        Args:
            env_id: Environment ID
            opponent_name: Name of opponent agent
            player_id: Player ID for the model (0 or 1)
            
        Returns:
            Dictionary of metrics for this episode
        """
        logging.info(
            f"Eval on {env_id} against {opponent_name} agent as player{player_id}"
        )
        
        opponent_id = 1 - player_id
        agents = {
            player_id: lambda obs: self.agent_act(obs, env_id),
            opponent_id: (
                RandomAgent(env_id)
                if opponent_name == "random"
                else ta.agents.OpenRouterAgent(opponent_name)
            ),
        }
        
        _use_llm_obs_wrapper = dict(
            zip(self.args.eval_env_ids, self.args.eval_use_llm_obs_wrappers)
        )[env_id]
        env = make_env(env_id, _use_llm_obs_wrapper)
        env.reset(num_players=2, seed=int(time.time_ns()))
        env.state.error_allowance = 0
        
        turn_counter = 0
        done = False
        invalid_rewards = None
        game_history = []
        
        while not done:
            pid, observation = env.get_observation()
            action = agents[pid](observation)
            game_history.append({
                "turn": turn_counter,
                "player_id": pid,
                "observation": observation,
                "action": action,
            })
            done, info = env.step(action)
            if action == INVALID_ACTION:
                done = True
            turn_counter += 1
            if done and action == INVALID_ACTION:
                invalid_rewards = {0: 1, 1: 1}
                invalid_rewards[pid] = -1
                rewards = {0: 1, 1: 1}
                rewards[pid] = -1
        
        if "rewards" not in locals():
            rewards_dict, game_info = env.close()
            rewards = rewards_dict
        
        if invalid_rewards:
            invalid_move = (invalid_rewards[0] == 1 and invalid_rewards[1] == -1) or (
                invalid_rewards[0] == -1 and invalid_rewards[1] == -1
            )
        else:
            invalid_move = False
        
        if rewards[player_id] > rewards[opponent_id]:
            outcome = "win"
        elif rewards[player_id] < rewards[opponent_id]:
            outcome = "loss"
        else:
            outcome = "draw"
        
        metrics = {
            "outcome": outcome,
            "invalid_move": invalid_move,
            "reason": info.get("reason", ""),
            "num_turns": turn_counter,
            "opponent_reward": rewards[opponent_id],
            "model_reward": rewards[player_id],
            "env_id": env_id,
            "opponent_name": opponent_name,
            "model_pid": player_id,
            "game_history": game_history if self.args.save_individual_results else None,
        }
        
        return metrics
    
    def evaluate_games(self) -> Dict:
        """Evaluate model on games against various opponents.
        
        Returns:
            Dictionary of aggregated game metrics
        """
        logging.info("Starting game evaluation")
        t0 = time.time()
        
        eval_env_ids = self.args.eval_env_ids
        eval_opponent_names = self.args.eval_opponent_names
        game_metrics = EvaluationMetrics(eval_env_ids, eval_opponent_names)
        
        total_games = self.args.eval_games
        
        # Generate evaluation runs
        eval_runs_list = []
        for env_id in eval_env_ids:
            for opponent_name in eval_opponent_names:
                if opponent_name == "random":
                    try:
                        RandomAgent(env_id)
                    except NotImplementedError:
                        logging.warning(
                            f"Random opponent is not supported for {env_id}, skipping"
                        )
                        continue
                
                for game_nr in range(total_games):
                    # Alternate player position
                    player_id = game_nr % 2
                    eval_runs_list.append((env_id, opponent_name, game_nr, player_id))
        
        # Run evaluation
        random.shuffle(eval_runs_list)
        all_results = []
        
        progress_bar = tqdm(eval_runs_list, desc="Evaluating games")
        for env_id, opponent_name, game_nr, player_id in progress_bar:
            result = self.run_eval_episode(env_id, opponent_name, player_id)
            game_metrics.add_result(result)
            all_results.append(result)
            progress_bar.set_postfix({
                "env": env_id,
                "opponent": opponent_name[:20],
            })
        
        game_metrics.aggregate()
        
        metrics_dict = game_metrics.to_dict()
        metrics_dict["eval/game_eval_time"] = time.time() - t0
        
        # Save individual results if requested
        if self.args.save_individual_results:
            results_path = os.path.join(self.args.output_dir, "game_results.json")
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=4)
            logging.info(f"Saved individual game results to {results_path}")
        
        return metrics_dict
    
    def eval_dataloader_collate_fn(self, item_list):
        """Collate function for dataset evaluation."""
        problems = []
        formatted_problems = []
        answers = []
        for item in item_list:
            problems.append(item["problem"])
            formatted_problems.append(
                TEMPLATE_FACTORY[self.args.eval_prompt_template](item["problem"])
            )
            answers.append(item["answer"])
        return formatted_problems, problems, answers
    
    def evaluate_dataset(self, dataset, benchmark_name: str) -> Dict:
        """Evaluate model on a reasoning dataset.
        
        Args:
            dataset: Dataset to evaluate on
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary of metrics
        """
        logging.info(f"Evaluating on {benchmark_name}")
        
        eval_prompts_dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.eval_dataloader_collate_fn,
        )
        
        all_responses = []
        all_answers = []
        all_problems = []
        total_tokens = 0
        
        for formatted_problems, problems, answers in tqdm(eval_prompts_dataloader, desc=f"Eval {benchmark_name}"):
            outputs = self.generate(formatted_problems, self.eval_sampling_params)
            
            for output, problem, answer in zip(outputs, problems, answers):
                response = output.outputs[0].text
                num_tokens = len(output.outputs[0].token_ids)
                total_tokens += num_tokens
                
                all_responses.append(response)
                all_answers.append(answer)
                all_problems.append({
                    "problem": problem,
                    "answer": answer,
                    "response": response,
                })
        
        # Compute accuracy using oracle (same as train_spiral.py)
        rewards, infos = self.oracle.get_reward(
            inputs=[""] * len(all_responses),
            responses=all_responses,
            references=all_answers,
            batch_size=self.args.eval_batch_size,
        )
        
        accuracy = rewards.float().mean().item()
        num_correct = int(rewards.sum().item())
        
        metrics = {
            "accuracy": accuracy,
            "score": accuracy,
            "response_tok_len": total_tokens / len(all_answers) if all_answers else 0.0,
            "num_samples": len(all_answers),
            "num_correct": num_correct,
        }
        
        # Save individual results if requested
        if self.args.save_individual_results:
            results_path = os.path.join(
                self.args.output_dir, f"dataset_results_{benchmark_name}.json"
            )
            with open(results_path, "w") as f:
                json.dump(all_problems, f, indent=4)
            logging.info(f"Saved {benchmark_name} results to {results_path}")
        
        return metrics
    
    def evaluate_datasets(self) -> Dict:
        """Evaluate model on all reasoning datasets.
        
        Returns:
            Dictionary of aggregated dataset metrics
        """
        logging.info("Starting dataset evaluation")
        t0 = time.time()
        
        all_metrics = {}
        accuracies = []
        scores = []
        lens = []
        
        for benchmark_name, dataset in self.eval_dataset_dict.items():
            metrics = self.evaluate_dataset(dataset, benchmark_name)
            
            # Add to aggregated metrics
            for k, v in metrics.items():
                all_metrics[f"eval/general/{benchmark_name}/{k}"] = v
            
            accuracies.append(metrics["accuracy"])
            scores.append(metrics["score"])
            lens.append(metrics["response_tok_len"])
        
        # Compute averages
        if accuracies:
            all_metrics["eval/general/average/accuracy"] = np.mean(accuracies)
            all_metrics["eval/general/average/score"] = np.mean(scores)
            all_metrics["eval/general/average/response_tok_len"] = np.mean(lens)
        
        all_metrics["eval/dataset_eval_time"] = time.time() - t0
        
        return all_metrics
    
    def evaluate(self) -> Dict:
        """Run full evaluation pipeline.
        
        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        
        # Game evaluation
        if not self.args.skip_game_eval:
            game_metrics = self.evaluate_games()
            all_metrics.update(game_metrics)
        else:
            logging.info("Skipping game evaluation")
        
        # Dataset evaluation
        if not self.args.skip_dataset_eval and self.eval_dataset_dict:
            dataset_metrics = self.evaluate_datasets()
            all_metrics.update(dataset_metrics)
        else:
            logging.info("Skipping dataset evaluation")
        
        # Save all metrics
        metrics_path = os.path.join(self.args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        logging.info(f"Saved metrics to {metrics_path}")
        
        # Print summary
        self.print_summary(all_metrics)
        
        return all_metrics
    
    def print_summary(self, metrics: Dict):
        """Print evaluation summary."""
        logging.info("=" * 80)
        logging.info("EVALUATION SUMMARY")
        logging.info("=" * 80)
        
        # Game metrics
        game_keys = [k for k in metrics.keys() if k.startswith("eval/game/")]
        if game_keys:
            logging.info("\nGame Evaluation:")
            for key in sorted(game_keys):
                logging.info(f"  {key}: {metrics[key]:.4f}")
        
        # Dataset metrics
        dataset_keys = [k for k in metrics.keys() if k.startswith("eval/general/")]
        if dataset_keys:
            logging.info("\nDataset Evaluation:")
            for key in sorted(dataset_keys):
                value = metrics[key]
                if isinstance(value, float):
                    logging.info(f"  {key}: {value:.4f}")
                else:
                    logging.info(f"  {key}: {value}")
        
        logging.info("=" * 80)


if __name__ == "__main__":
    args: EvalArgs = get_default_args(EvalArgs)
    
    # Validate arguments
    assert args.checkpoint_path or args.pretrain, \
        "Must specify either --checkpoint_path or --pretrain"
    
    assert len(args.eval_env_ids) == len(args.eval_use_llm_obs_wrappers), \
        f"Number of eval_env_ids ({len(args.eval_env_ids)}) must match number of eval_use_llm_obs_wrappers ({len(args.eval_use_llm_obs_wrappers)})"
    
    # Create evaluator and run evaluation
    evaluator = SpiralEvaluator(args)
    metrics = evaluator.evaluate()

