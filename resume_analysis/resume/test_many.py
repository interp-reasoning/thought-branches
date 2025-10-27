import sys
import os
import numpy as np
import asyncio
from rollouts import RolloutsClient


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resume.simple_prompt_maker import (
    get_resume_by_id,
    make_prompts_for_resume_group,
)

JOBS = ["basic", "meta", "meta_simple", "gm", "palantir"]
RESUME_IDS = list(range(100))
ANTI_BIAS = ["none", "v1", "v2", "simple"]
VARIANT_IDS = list(range(4))


def get_yes_no(content):
    if "yes" in content.lower():
        return 1.0
    elif "no" in content.lower():
        return 0.0
    else:
        return np.nan


def get_prompt(resume_id, job, variant_idx, anti_bias):
    resume_group = get_resume_by_id(resume_id)
    prompts = make_prompts_for_resume_group(
        resume_group, job_name=job, anti_bias=anti_bias, prompt_type="yes_no"
    )
    prompt_data = prompts[variant_idx]
    prompt = prompt_data["prompt"]
    return prompt


async def gen_prompts(prompts, client, num_responses=50):
    all_coroutines = []
    for prompt in prompts:
        coroutine = client.agenerate(prompt, n_samples=num_responses)
        all_coroutines.append(coroutine)
    return await asyncio.gather(*all_coroutines)


def test_combinations(model, num_responses=50):
    client = RolloutsClient(model=model, temperature=0.7, max_tokens=16384)
    for job in JOBS:
        for resume_id in RESUME_IDS:
            variant_scores = f"{job} | {resume_id:<3} | "
            prompts = []
            for variant_idx in VARIANT_IDS:
                prompt = get_prompt(
                    resume_id, job, variant_idx, anti_bias="none"
                )
                prompts.append(prompt)
            # print(f"{len(prompts)=}")
            # quit()

            results = asyncio.run(
                gen_prompts(prompts, client, num_responses=num_responses)
            )
            for rollouts, variant_idx in zip(results, VARIANT_IDS):
                # rollouts = client.generate(prompt, n_samples=num_responses)
                answers = [r.content for r in rollouts]
                yn_answers = [get_yes_no(a) for a in answers]
                p_yes = np.nanmean(yn_answers)
                variant_score = f"({variant_idx}): {p_yes=:.1%}, "
                variant_scores += variant_score
            variant_scores = variant_scores[:-2]
            print(variant_scores)
            # print(f"{resume_id=} {job=} {variant_idx=} {anti_bias=} {p_yes=:.1%}")


if __name__ == "__main__":
    MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
    test_combinations(model=MODEL_NAME)
    # make_combination(0, "basic", 0, "none")
