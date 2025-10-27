import asyncio
import copy
import random
import time
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from pkld import pkld
from rollouts import RolloutsClient
from tqdm import tqdm

from analyze_all_resumes import get_variant_info
from simple_prompt_maker import get_resume_by_id, make_prompts_for_resume_group


# @pkld(overwrite=False)
def get_case_responses(
    resume_id=3,
    job="meta",
    variant_idx=1,
    num_responses=5000,
    correct=True,
    model="qwen/qwen3-30b-a3b",
    anti_bias="none",
):

    # Get the resume and prompt
    resume_group = get_resume_by_id(resume_id)
    if not resume_group:
        raise ValueError(f"Resume {resume_id} not found")

    prompts = make_prompts_for_resume_group(
        resume_group, job_name=job, anti_bias=anti_bias, prompt_type="yes_no"
    )

    prompt_data = prompts[variant_idx]
    prompt = prompt_data["prompt"]

    # Create client with default settings
    client = RolloutsClient(model=model, temperature=0.7, max_tokens=16384)

    # Generate multiple responses (one prompt sampled concurrently). This runs on seeds from 0 to n_samples (e.g., 0, 1, 2, 3, 4)
    rollouts = client.generate(prompt, n_samples=num_responses)

    # Get variant info for cleaning
    name, pronouns, email = get_variant_info(resume_id, job, variant_idx)

    # Return result with metadata
    return {
        "prompt": prompt,
        "responses": rollouts,
        "resume_id": resume_id,
        "job": job,
        "variant_idx": variant_idx,
        "variant_demographics": prompt_data.get("variant_demographics"),
        "name": name,
        "pronouns": pronouns,
        "email": email,
    }
