"""
KnowLogic: A Benchmark for Commonsense Reasoning via Knowledge-Driven Data Synthesis
Weidong Zhan, Yue Wang, Nan Hu, Liming Xiao, Jingyuan Ma, Yuhang Qin, Zheng Li, Yixin Yang, Sirui Deng, Jinkun Ding, Wenhan Ma, Rui Li, Weilin Luo, Qun Liu, Zhifang Sui
https://arxiv.org/abs/2503.06218
"""

import random
import re

import datasets

from . import common
from .common import HTML_JINJA
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult


class KnowLogicEval(Eval):
    def __init__(
        self,
        num_examples: int | None = None,
    ):
        ds = datasets.load_dataset("Pokerwf/KnowLogic", split="test")
        # Filter to only keep English examples
        ds = ds.filter(lambda example: example["language"] == "en")
        df = ds.to_pandas()
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=common.QUERY_TEMPLATE_MULTICHOICE_SELF_PLAY_KNOWLOGIC.format(
                        **row
                    ),
                    role="user",
                )
            ]
            response_text = sampler(prompt_messages)
            match = re.search(
                common.ANSWER_PATTERN_MULTICHOICE_SELF_PLAY_KNOWLOGIC, response_text
            )
            extracted_answer = match.group(1) if match else None
            invalid_answer = extracted_answer is None or extracted_answer == ""
            extracted_answer_list = (
                [
                    answer.strip()
                    for answer in extracted_answer.split(",")
                    if answer.strip(" )")
                ]
                if extracted_answer
                else []
            )
            # Check if extracted answers match the correct answers
            extracted_answers = set(extracted_answer_list)
            correct_answers = set(row["answer"])
            score = 1.0 if correct_answers == extracted_answers else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
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
