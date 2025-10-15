"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re

import pandas
import datasets
from . import common
from .common import (ANSWER_PATTERN_MULTICHOICE_SELF_PLAY, HTML_JINJA,
                     format_multichoice_question,
                     format_multichoice_question_few_shot)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

SHOTS = 0

class GPQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: (
            int | None
        ) = None,  # restrict to a subset of the data for debugging
    ):
        rng = random.Random(0)
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if SHOTS > 0:
            few_shot_examples = datasets.load_dataset("Idavidrein/gpqa", "gpqa_main", split="train").to_pandas()

            few_shot_examples = few_shot_examples.to_dict(orient="records")
            # filter based on record ID that shouldn't be in the main eval
            few_shot_examples = [
                example | {"permutation": rng.sample(range(4), 4)}
                for example in few_shot_examples
                if example["Record ID"] not in df["Record ID"].values
            ]
            few_shot_examples = [
                format_multichoice_question_few_shot(GPQAEval.process_example(example)) for example in few_shot_examples[:SHOTS]
            ]
            few_shot_examples = "\n\n".join(few_shot_examples)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [
            example | {"permutation": rng.sample(range(4), 4)} for example in examples
        ]
        if SHOTS > 0:
            for example in examples:
                example["few_shot_examples"] = few_shot_examples
        else:
            for example in examples:
                example["few_shot_examples"] = ""
        self.examples = examples
        self.n_repeats = n_repeats

    @staticmethod
    def process_example(example: dict) -> dict:
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in example["permutation"]]
        correct_index = choices.index(example["Correct Answer"])
        correct_answer = "ABCD"[correct_index]
        choices_dict = dict(
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
            question=example["Question"],
            correct_answer=correct_answer,
            few_shot_examples=example["few_shot_examples"] if "few_shot_examples" in example else "",
            explanation=example["Explanation"] if "Explanation" in example else "",
        )
        return choices_dict
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices_dict = GPQAEval.process_example(row)
            correct_answer = choices_dict["correct_answer"]
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            match = re.search(ANSWER_PATTERN_MULTICHOICE_SELF_PLAY, response_text)
            extracted_answer = match.group(1) if match else None
            invalid_answer = extracted_answer is None or extracted_answer == ""
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"chars": len(response_text), "invalid_count": invalid_answer},
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
