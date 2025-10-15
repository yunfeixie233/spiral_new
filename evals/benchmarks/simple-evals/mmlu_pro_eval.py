"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re

import datasets

from . import common
from .common import (ANSWER_PATTERN_MULTICHOICE_SELF_PLAY_PRO, HTML_JINJA,
                     format_multichoice_question_pro,
                     format_multichoice_question_pro_with_answers,
                     normalize_extracted_answer)
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

SHOTS = 0

class MMLUProEval(Eval):
    def __init__(self, num_examples: int | None = None, language: str = "EN-US"):
        if SHOTS > 0:
            few_shot_examples = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

            few_shot_df = few_shot_examples.to_pandas()
            # group by category into dictionary
            few_shot_examples = few_shot_df.groupby("category").apply(lambda x: x.to_dict(orient="records")).to_dict()
            few_shot_examples = {k: v[:SHOTS] for k, v in few_shot_examples.items()}
            few_shot_examples = {k: "\n\n".join(format_multichoice_question_pro_with_answers(ex) for ex in v) for k, v in few_shot_examples.items()}
        
        ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        df = ds.to_pandas()
        examples = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Convert options from numpy array to list
            if "options" in row_dict and hasattr(row_dict["options"], "tolist"):
                row_dict["options"] = row_dict["options"].tolist()
            if SHOTS > 0:
                row_dict["few_shot_examples"] = few_shot_examples[row_dict["category"]]
            examples.append(row_dict)

        print(examples[0])
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question_pro(row), role="user"
                )
            ]
            # response_text = normalize_response(sampler(prompt_messages))
            response_text = sampler(prompt_messages)
            extracted_answer = None
            # for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            #     regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE_PRO.format(answer_regex)
            #     match = re.search(regex, response_text)
            #     if match:
            #         extracted_answer = normalize_extracted_answer(match.group(1))
            #         break
            match = re.search(ANSWER_PATTERN_MULTICHOICE_SELF_PLAY_PRO, response_text)
            extracted_answer = match.group(1) if match else None
            invalid_answer = extracted_answer is None or extracted_answer == ""
            if extracted_answer:
                extracted_answer = normalize_extracted_answer(extracted_answer)
            score = 1.0 if extracted_answer == row["answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = row["category"]
            return SingleEvalResult(
                html=html,
                score=score,
                metrics={"invalid_count": invalid_answer, "chars": len(response_text)},
                convo=convo,
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
