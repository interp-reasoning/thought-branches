from pprint import pprint
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resampler.sentence_splitter import (
    split_into_paragraphs,
    string_to_sentences,
)
from resampler.sentence_splitter import split_into_paragraphs_safe

# Add parent directory to path to allow imports when running directly

from pkld import pkld

# from resampler.base_generator import (
#     get_prompt_responses,
#     limited_async_gather,
# )
# from resampler.generate_chunk_rollouts_openrouter_seed import generate_multiple_responses_openrouter
from tqdm import tqdm
from collections import defaultdict
import asyncio
import re
from typing import List
from rollouts import RolloutsClient

from dataclasses import dataclass, field


def clean_python_string_literal(text: str) -> str:
    """
    Clean a Python string literal that may contain quotes, escaped characters, etc.
    """
    # Remove outer parentheses if present
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    # Handle multiline Python string concatenation
    # Remove quotes and string concatenation operators
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove leading/trailing quotes and concatenation
        # line = re.sub(r'^["\']', "", line)  # Remove leading quote
        # line = re.sub(r'["\']$', "", line)  # Remove trailing quote
        # line = re.sub(r'^["\']', "", line)  # Remove any remaining leading quote
        # line = re.sub(r'["\']$', "", line)  # Remove any remaining trailing quote

        cleaned_lines.append(line)

    # Join all lines with spaces
    result = " ".join(cleaned_lines)

    # Replace escaped characters
    result = result.replace("\\n", " ")
    result = result.replace('\\"', '"')
    result = result.replace("\\'", "'")
    result = result.replace("\\\\", "\\")

    # Clean up extra whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


async def limited_async_gather(
    coroutines: list, max_concurrent: int = 500, desc: str = "Processing"
) -> list:
    """Process coroutines with a limit on concurrent executions and retry logic."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore_and_retry(
        idx: int, coro, max_retries: int = 3
    ):
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    result = await coro
                    return idx, result
                except (asyncio.TimeoutError, Exception) as e:
                    if "timeout" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2  # Exponential backoff
                            print(
                                f"\nTimeout for task {idx}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                            )
                            await asyncio.sleep(wait_time)
                            continue
                    # If not a timeout or last attempt, return error
                    return idx, {"error": str(e)}
            return idx, {"error": "Max retries exceeded"}

    # Create tasks with indices to maintain order
    tasks = [
        run_with_semaphore_and_retry(i, coro)
        for i, coro in enumerate(coroutines)
    ]

    # Process all tasks with progress bar
    from tqdm.asyncio import tqdm as async_tqdm

    results = []
    for task in async_tqdm.as_completed(tasks, total=len(tasks), desc=desc):
        try:
            result = await task
            results.append(result)
        except Exception as e:
            # Don't crash, just continue
            pass

    # Sort by index and extract results
    results.sort(key=lambda x: x[0])
    return [result for _, result in results]


def get_sentence2prompts(
    reasoning, prompt, kill_extra_space=True, do_paragraphs=False
):
    sentence2prompts = defaultdict(list)
    sentence2downstream_reasoning = defaultdict(list)
    sentnece2upstream_reasoning = defaultdict(list)
    if do_paragraphs:
        paragraphs, paragraph_positions = split_into_paragraphs_safe(reasoning)
        sentences, positions = paragraphs, paragraph_positions
    else:
        sentences, positions = string_to_sentences(reasoning)

    sentences_cnt = []

    for i, (sentence, idx_alt) in enumerate(zip(sentences, positions)):
        idx = idx_alt

        prompt_reasoning = f"{prompt}\n<think>\n{reasoning[:idx]}"
        if kill_extra_space and prompt_reasoning.endswith(" "):
            prompt_reasoning = prompt_reasoning[:-1]
            upstream_reasoning = reasoning[: idx - 1]
            downstream_reasoning = " " + reasoning[idx:]
        else:
            upstream_reasoning = reasoning[:idx]
            downstream_reasoning = reasoning[idx:]

        sentence2prompts[sentence].append(prompt_reasoning)
        sentence2downstream_reasoning[sentence].append(
            downstream_reasoning
        )  # Includes sentence itself
        sentnece2upstream_reasoning[sentence].append(upstream_reasoning)
        cnt = len(sentence2prompts[sentence]) - 1
        sentences_cnt.append((sentence, cnt))
    return (
        sentence2prompts,
        sentence2downstream_reasoning,
        sentnece2upstream_reasoning,
        sentences_cnt,
    )


# print(
#     get_sentence2prompts(
#         "What is the capital of France? I think its Paris.",
#         "",
#         do_paragraphs=False,
#     )[0]
# )
# quit()


@dataclass
class ResamplePrompt:
    count: int = 0
    original_content: list = field(default_factory=list)
    sentences: list = field(default_factory=list)
    original_responses: list = field(default_factory=list)
    original_upstream_reasoning: list = field(default_factory=list)
    original_downstream_reasoning: list = field(default_factory=list)
    original_resp_idx: list = field(default_factory=list)
    sentence_positions: list = field(default_factory=list)
    sentence_norm_positions: list = field(default_factory=list)
    resample_idx: list = field(default_factory=list)
    next_resample_idxs: list = field(default_factory=list)
    original_length: list = field(default_factory=list)


@dataclass
class ResampleSentence:
    # original_content: list = field(default_factory=list)
    new_responses: list = field(default_factory=list)
    original_responses: list = field(default_factory=list)
    original_resp_idx: list = field(default_factory=list)
    # original_lengths: list = field(default_factory=list)
    count: int = 0
    positions: list = field(default_factory=list)
    norm_positions: list = field(default_factory=list)
    mirror: list = field(default_factory=list)
    next_sentence: list = field(default_factory=list)
    variants: list = field(default_factory=list)
    resume_ids: list = field(default_factory=list)
    original_lengths: list = field(default_factory=list)
    # new_lengths: list = field(default_factory=list)

    def merge(self, other):
        # self.original_content.extend(other.original_content)
        self.new_responses.extend(other.new_responses)
        self.original_responses.extend(other.original_responses)
        self.original_resp_idx.extend(other.original_resp_idx)
        self.count += other.count
        self.positions.extend(other.positions)
        self.norm_positions.extend(other.norm_positions)
        self.mirror.extend(other.mirror)


@pkld(overwrite=False)
def mad_resample(
    responses,
    prompt,
    num_responses=1,
    model="deepseek/deepseek-r1-0528-qwen3-8b",
    max_concurrent=500,
    include_original_resp_idx=True,
    do_paragraphs=False,
    resample_idx_approach=True,
):
    print("Running mad_resample")
    assert include_original_resp_idx, "include_original_resp_idx must be True"
    all_coroutines = []
    all_sentences = []

    original_contents = []
    original_reasoning = []

    resample_data = defaultdict(ResamplePrompt)

    idx = 0
    idx2next = {}
    for resp_idx, response in tqdm(
        enumerate(responses), desc="Prepping coroutines"
    ):
        content = response.content
        reasoning = response.reasoning
        (
            sentence2prompts,
            sentence2downstream_reasoning,
            sentnece2upstream_reasoning,
            sentences_cnt,
        ) = get_sentence2prompts(reasoning, prompt, do_paragraphs=do_paragraphs)

        for i, (sentence, cnt) in enumerate(sentences_cnt):
            p = sentence2prompts[sentence][cnt]
            resample_data[p].count += 1
            resample_data[p].original_content.append(content)
            resample_data[p].sentences.append(sentence)
            resample_data[p].original_responses.append(response)
            response.upstream_reasoning = sentnece2upstream_reasoning[sentence][
                cnt
            ]
            response.downstream_reasoning = sentence2downstream_reasoning[
                sentence
            ][cnt]

            # Redundant with above
            resample_data[p].original_upstream_reasoning.append(
                sentnece2upstream_reasoning[sentence][cnt]
            )
            resample_data[p].original_downstream_reasoning.append(
                sentence2downstream_reasoning[sentence][cnt]
            )

            resample_data[p].original_resp_idx.append(resp_idx)
            resample_data[p].sentence_positions.append(i)
            resample_data[p].sentence_norm_positions.append(
                i / len(sentence2prompts)
            )
            resample_data[p].original_length.append(len(sentences_cnt))

            resample_data[p].resample_idx.append(idx)
            if i == len(sentences_cnt) - 1:
                resample_data[p].next_resample_idxs.append(content)
            else:
                next_sentence, next_cnt = sentences_cnt[i + 1]
                next_p = sentence2prompts[next_sentence][next_cnt]
                if next_p in resample_data:
                    count = resample_data[next_p].count
                else:
                    count = 0
                resample_data[p].next_resample_idxs.append((next_p, count))
                # print(f"{p=}")
                # print(f"{next_p=}")
                # print("-------")
                # quit()
            idx += 1
        # quit()

    prompts_sorted = list(resample_data.keys())
    prompts_sorted.sort(key=lambda x: resample_data[x].count, reverse=True)

    resample_idxs_sorted = []
    # next_idxs_resorted = []
    p2it = {}
    for i, p in enumerate(prompts_sorted):
        p2it[p] = i
        all_sentences.extend(resample_data[p].sentences)
        original_contents.extend(resample_data[p].original_content)
        original_reasoning.extend(resample_data[p].original_responses)
        resample_idxs_sorted.extend(resample_data[p].resample_idx)
        # next_idxs_resorted.extend(resample_data[p].next_idxs)

    client = RolloutsClient(model=model, verbose=False)
    for p in prompts_sorted:
        count = resample_data[p].count
        all_coroutines.append(
            client.agenerate(p, n_samples=count * num_responses)
        )
    new_rollouts_all = asyncio.run(
        limited_async_gather(
            all_coroutines,
            max_concurrent=max_concurrent,
            desc=f"Generating responses (max {max_concurrent} concurrent)",
        )
    )

    # new_rollouts_all = asyncio.run(gen_w_client())

    assert len(new_rollouts_all) == len(
        prompts_sorted
    ), f"Length mismatch: {len(new_rollouts_all)} != {len(prompts_sorted)}"

    sentence2data = defaultdict(ResampleSentence)
    sentence2data_mirror = defaultdict(ResampleSentence)
    for i in range(len(prompts_sorted)):
        new_rollouts = new_rollouts_all[i]
        p = prompts_sorted[i]
        count = resample_data[p].count
        assert (
            len(new_rollouts) == count
        ), f"Length mismatch: {len(new_rollouts)} != {count}"

        for j in range(len(new_rollouts)):
            response = new_rollouts[j]
            sentence = resample_data[p].sentences[j]
            resp_idx = resample_data[p].original_resp_idx[j]

            new_answer = response.content
            response.downstream_reasoning = response.reasoning

            response.upstream_reasoning = resample_data[
                p
            ].original_upstream_reasoning[j]
            # print(
            #     f"{response.upstream_reasoning=}:{response.downstream_reasoning=}"
            # )
            # try:
            if response.downstream_reasoning is None:
                response.reasoning = response.upstream_reasoning
            else:
                response.reasoning = (
                    response.upstream_reasoning + response.downstream_reasoning
                )

            assert (
                response.upstream_reasoning in p
            ), f"{sentence=}\n\n{response.upstream_reasoning=}\n\n{p=}"

            if resample_idx_approach:
                next_resample_idx = resample_data[p].next_resample_idxs[j]
                if isinstance(next_resample_idx, str):
                    original_response = resample_data[p].original_responses[j]
                    original_contents = next_resample_idx
                    if original_contents == "" or new_answer == "":
                        continue
                else:
                    next_p, next_j = next_resample_idx
                    original_contents = resample_data[p].original_content[
                        next_j
                    ]
                    # original_conten
                    if original_contents == "" or new_answer == "":
                        continue
                    next_p_i = p2it[next_p]
                    # print("good")
                    try:
                        assert (
                            response.upstream_reasoning in next_p
                        ), f"{response.upstream_reasoning=}\n{next_p=}\n{next_p[-len(response.upstream_reasoning):]=}"
                        original_response = new_rollouts_all[next_p_i][next_j]
                        original_contents = original_response.content
                        assert (
                            sentence in next_p[-len(sentence) - 10 :]
                        ), f"{sentence=}\n{next_p=}\n{next_p[-len(sentence):]=}"
                    except AssertionError as e:
                        print("BAD SENTENCE?")
                        continue
                    # Warning, this new response has as .reasoning only the downstream reasoning
                    # original_response.full = response.upstream_reasoning + original_response.full
                    # original_response.reasoning = response.upstream_reasoning + original_response.reasoning
                    # original_response = resample_data[
                    #     next_p
                    # ].original_responses[next_j]
                    # next_sentence = resample_data[next_p].sentences[next_j]
                    # sentence2data[sentence].next_sentence.append(
                    #     (next_sentence, next_j)
                    # )
            else:
                original_contents = resample_data[p].original_content[j]
                if original_contents == "" or new_answer == "":
                    continue
                original_response = resample_data[p].original_responses[j]

            sentence2data[sentence].original_lengths.append(
                resample_data[p].original_length[j]
            )

            sentence2data[sentence].new_responses.append(response)
            sentence2data[sentence].original_responses.append(original_response)
            sentence2data[sentence].original_resp_idx.append(resp_idx)
            sentence2data[sentence].count += 1
            sentence2data[sentence].positions.append(
                resample_data[p].sentence_positions[j]
            )
            sentence2data[sentence].norm_positions.append(
                resample_data[p].sentence_norm_positions[j]
            )

            if do_paragraphs:
                downstream_sentences, _ = split_into_paragraphs_safe(
                    response.downstream_reasoning
                )
            else:
                downstream_sentences, _ = string_to_sentences(
                    response.downstream_reasoning
                )
            if len(downstream_sentences) == 0:
                new_sentence = ""
            else:
                new_sentence = downstream_sentences[0]
            sentence2data[sentence].mirror.append(new_sentence)

            sentence2data_mirror[new_sentence].original_lengths.append(
                resample_data[p].original_length[j]
            )
            sentence2data_mirror[new_sentence].new_responses.append(
                original_response
            )
            sentence2data_mirror[new_sentence].original_responses.append(
                response
            )
            sentence2data_mirror[new_sentence].original_resp_idx.append(
                resp_idx
            )
            sentence2data_mirror[new_sentence].count += 1
            sentence2data_mirror[new_sentence].positions.append(
                resample_data[p].sentence_positions[j]
            )
            sentence2data_mirror[new_sentence].norm_positions.append(
                resample_data[p].sentence_norm_positions[j]
            )
            sentence2data_mirror[new_sentence].mirror.append(sentence)

    return sentence2data, sentence2data_mirror
    # else:
    #     raise ValueError("No coroutines to run")
