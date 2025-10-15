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

import numpy as np


class EvaluationMetrics:
    """Class to track and aggregate evaluation metrics for self-play with hierarchical keys."""

    def __init__(self, eval_env_ids, opponent_names):
        """Initialize metrics tracking structure."""
        self.metrics = {}

        # Basic metric types
        self.metric_types = [
            "win",
            "loss",
            "draw",
            "invalid",
            "game_lengths",
            "model_rewards",
            "opponent_rewards",
        ]

        # Add summary metrics
        for metric in self.metric_types:
            self.metrics[f"eval/{metric}"] = 0.0

        # Add env-specific metrics
        for env_id in eval_env_ids:
            for opponent_name in opponent_names:
                # if (opponent_name == "random") and (env_id != "TicTacToe-v0"):
                #     continue

                edited_opponent_name = opponent_name.replace("/", "-")
                eval_prefix = f"eval/{env_id}-{edited_opponent_name}"
                for metric in self.metric_types:
                    self.metrics[f"{eval_prefix}/{metric}"] = 0.0

                # Add player-specific metrics for this env and opponent
                for pid in [0, 1]:
                    pid_prefix = f"eval/{env_id}-{edited_opponent_name}/pid-{pid}"
                    for metric in self.metric_types:
                        self.metrics[f"{pid_prefix}/{metric}"] = 0.0

        # Tracking data for calculating metrics
        self.tracking_data = {
            "overall": self._new_tracking_dict(),
        }

        # Track by env
        for env_id in eval_env_ids:
            eval_prefix = f"eval/{env_id}"
            self.tracking_data[eval_prefix] = self._new_tracking_dict()

            # Track by env-opponent
            for opponent_name in opponent_names:
                # if (opponent_name == "random") and (env_id != "TicTacToe-v0"):
                #     continue

                edited_opponent_name = opponent_name.replace("/", "-")
                eval_prefix = f"eval/{env_id}-{edited_opponent_name}"
                self.tracking_data[eval_prefix] = self._new_tracking_dict()

                # Track by player ID
                for pid in [0, 1]:
                    eval_prefix = f"eval/{env_id}-{edited_opponent_name}/pid-{pid}"
                    self.tracking_data[eval_prefix] = self._new_tracking_dict()

    def _new_tracking_dict(self):
        """Create a new dictionary for tracking raw metrics data."""
        return {f"{key}-list": [] for key in self.metric_types}

    def add_result(self, result):
        """Add a single evaluation result to the metrics."""
        env_id = result["env_id"]
        opponent_name = result["opponent_name"]
        edited_opponent_name = opponent_name.replace("/", "-")
        pid = result["model_pid"]

        # Extract metrics
        is_win = 1 if result["outcome"] == "win" else 0
        is_loss = 1 if result["outcome"] == "loss" else 0
        is_draw = 1 if result["outcome"] == "draw" else 0
        is_invalid = int(result["invalid_move"])

        # Update overall metrics
        self._add_to_tracking(
            key="overall",
            win=is_win,
            loss=is_loss,
            draw=is_draw,
            invalid=is_invalid,
            game_lengths=result["num_turns"],
            model_rewards=result["model_reward"],
            opponent_rewards=result["opponent_reward"],
        )

        # Update env-opponent metrics
        env_opp_key = f"eval/{env_id}-{edited_opponent_name}"
        self._add_to_tracking(
            key=env_opp_key,
            win=is_win,
            loss=is_loss,
            draw=is_draw,
            invalid=is_invalid,
            game_lengths=result["num_turns"],
            model_rewards=result["model_reward"],
            opponent_rewards=result["opponent_reward"],
        )

        # Update pid-env-opponent metrics
        pid_env_opp_key = f"eval/{env_id}-{edited_opponent_name}/pid-{pid}"
        self._add_to_tracking(
            key=pid_env_opp_key,
            win=is_win,
            loss=is_loss,
            draw=is_draw,
            invalid=is_invalid,
            game_lengths=result["num_turns"],
            model_rewards=result["model_reward"],
            opponent_rewards=result["opponent_reward"],
        )

    def _add_to_tracking(self, key, **kwargs):
        """Add metrics to the tracking dictionary for the given key."""
        for metric_name, value in kwargs.items():
            metric_key = f"{metric_name}-list"
            if metric_key in self.tracking_data[key]:
                self.tracking_data[key][metric_key].append(value)

    def aggregate(self):
        """Aggregate all tracked metrics and update the metrics dictionary."""

        # Aggregate metrics for each tracking key
        for key, data in self.tracking_data.items():
            if not data["win-list"]:  # Skip if no data
                continue

            # Determine the prefix for the metrics
            prefix = key
            if key == "overall":
                prefix = "eval"

            # Calculate and store each metric
            for metric_name in self.metric_types:
                metric_key = f"{prefix}/{metric_name}"
                if metric_key in self.metrics:  # Ensure the key exists
                    self.metrics[metric_key] = np.mean(
                        data[f"{metric_name}-list"]
                    ).item()

    def to_dict(self):
        """Convert metrics to a dictionary for broadcasting."""
        return self.metrics

    @classmethod
    def from_dict(cls, metrics_dict, eval_env_ids=None, opponent_names=None):
        """Create a metrics object from a dictionary (for non-rank-0 processes)."""
        metrics = cls(eval_env_ids or [], opponent_names or [])
        metrics.metrics = metrics_dict
        return metrics
