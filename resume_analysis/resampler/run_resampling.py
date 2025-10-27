import json
from pathlib import Path
import sys
import os
import asyncio
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from resampler.analyze_delta import analyze_sentence2verdicts
from resampler.mirror import add_verdict_to_base, final_prep_sentence2verdict
from resampler.weighted_kmeans import get_all2most_cluster2commons, get_cluster_mappers
from resampler.sentence_splitter import split_into_paragraphs_safe, string_to_sentences
from resampler.word_analysis import make_word2verdict, make_word_stats_csv

# Add parent directory to path to allow imports when running directly

# from blackmail_classifier_openai import eval_sentence2post_verdicts
# from resampler.base_generator import get_prompt_responses
from resampler.sentence_resampler import mad_resample





def check_numbered_list(text):
    req_strings = ["1.", "2.", "3."]
    for req_string in req_strings:
        if req_string not in text:
            return False
    return True


def check_dashed_list(text):
    if "- " in text:
        return True
    return False


def add_verdict_to_sentence2new_responses(sentence2new_responses, sentence2verdicts):
    for sentence, verdicts in sentence2verdicts.items():
        new_responses = sentence2new_responses[sentence]
        for verdict, new_response in zip(verdicts, new_responses):
            new_response["verdict"] = verdict[1]
    return sentence2new_responses


def drop_no_post_base(base_responses):
    pruned_base_responses = []
    n_no_post = 0
    for response in base_responses:
        if response["post"] == "":
            n_no_post += 1
            continue
        pruned_base_responses.append(response)
    print(f"Number of responses with no post: {n_no_post}")
    return pruned_base_responses


def print_paragraph_stats(base_responses):
    verdict_yes_base = 0
    verdict_no_base = 0
    cnt_paragraphs_yes = []
    cnt_paragraphs_no = []
    for response in base_responses:
        if not isinstance(response["verdict"], bool):
            continue
        paragraphs, pos_in_paragraph = split_into_paragraphs_safe(response["reasoning"])
        if response["verdict"]:
            verdict_yes_base += 1
            cnt_paragraphs_yes.append(len(paragraphs))
        else:
            verdict_no_base += 1
            cnt_paragraphs_no.append(len(paragraphs))
    print(f"Mean number of paragraphs in yes: {np.mean(cnt_paragraphs_yes)}")
    print(f"Mean number of paragraphs in no: {np.mean(cnt_paragraphs_no)}")

    p_yes_base = verdict_yes_base / (verdict_yes_base + verdict_no_base)
    print(f"Baseline stats: {verdict_yes_base=}, {verdict_no_base=} ({p_yes_base=:.1%})")


def make_position_df(
    sentence2positions,
    sentence2norm_positions,
    sentence2verdicts,
    sentence2original_response_idx,
    model_name,
):
    df_as_l = []
    for sentence, verdicts in sentence2verdicts.items():
        for verdict, position, norm_position, original_response_idx in zip(
            verdicts,
            sentence2positions[sentence],
            sentence2norm_positions[sentence],
            sentence2original_response_idx[sentence],
        ):
            n01_rspl = 1 if (not verdict[0] and verdict[1]) else 0
            n10_rspl = 1 if (verdict[0] and not verdict[1]) else 0
            n00_rspl = 1 if (not verdict[0] and not verdict[1]) else 0
            n11_rspl = 1 if (verdict[0] and verdict[1]) else 0
            row = {
                "sentence": sentence,
                "position": position,
                "norm_position": norm_position,
                "n01_rspl": n01_rspl,
                "n10_rspl": n10_rspl,
                "n00_rspl": n00_rspl,
                "n11_rspl": n11_rspl,
                "resp_idx": original_response_idx,
            }
            df_as_l.append(row)
    df = pd.DataFrame(df_as_l)
    model_long2short = {"qwen/qwq-32b": "qwq-32b", "qwen/qwen3-235b-a22b": "qwen3-235b-a22b"}
    df.to_csv(f"position_df_{model_long2short[model_name]}.csv", index=False)
    return df

    p_yes = verdict_yes / (verdict_yes + verdict_no)
    return p_yes


if __name__ == "__main__":
    N_CLUSTERS = 50
    UZAY_CHUNKS_CHECK = 10
    DO_PARAGRAPHS = True
    DROP_BY_THEREFORE = False
    DROP_ONES_I_DONT_LIKE = True
    WORD_ANALYSIS = True

    fp_prompt = "extracted_prompt.txt"
    # fp_prompt = "uzay_blackmail_prompt.txt"
    # fp_prompt = "extracted_prompt_w_scratchpad.txt"

    with open(fp_prompt, "r") as f:
        prompt = f.read()
    model_name = r"qwen/qwq-32b"
    # model_name = r"qwen/qwen3-235b-a22b"
    # model_name = r"deepseek/deepseek-r1-0528:free"
    # model_name = r"deepseek/deepseek-r1-0528:free"
    provider_config = {
        "max_price": {"completion": 2.2},
        "quantizations": [
            "fp16",
            "fp8",
        ],  #  "bf16","bf8"
    }
    # model_name = r"deepseek/deepseek-r1-0528"
    if model_name == r"qwen/qwen3-235b-a22b":
        provider_config["max_price"] = {"completion": 0.61}
    # prompt += "TEST"
    base_responses = get_prompt_responses(
        prompt,
        num_responses=1500 if model_name == r"qwen/qwq-32b" else 20_000,  # 20_000
        max_concurrent=250,
        model=model_name,
        temperature=0.7,
        top_p=0.95,
        max_tokens=16384,
        max_retries=100,
        provider_config=provider_config,
    )
    # print(f"Initial base responses: {len(base_responses)}")
    # quit()
    if model_name == r"qwen/qwen3-235b-a22b":
        pre_drop_ones_i_dont_like = len(base_responses)
        base_responses = drop_ones_i_dont_like(base_responses)
        post_drop_ones_i_dont_like = len(base_responses)
        print(
            f"Dropped {pre_drop_ones_i_dont_like - post_drop_ones_i_dont_like} responses that I don't like"
        )
    # print(f"Base responses after dropping: {len(base_responses)}")
    # quit()

    if DROP_BY_THEREFORE:
        pre_drop_therefore = len(base_responses)
        base_responses = drop_by_therefore(base_responses)
        post_drop_therefore = len(base_responses)
        print(f"Dropped {pre_drop_therefore - post_drop_therefore} responses with therefore")

    pre_drop_no_post = len(base_responses)
    base_responses = drop_no_post_base(base_responses)
    post_drop_no_post = len(base_responses)
    print(f"Dropped {pre_drop_no_post - post_drop_no_post} responses with no post")
    base_responses = add_verdict_to_base(base_responses)

    model_long2short = {"qwen/qwq-32b": "qwq-32b", "qwen/qwen3-235b-a22b": "qwen3-235b-a22b"}

    out_l = []
    for i, response in enumerate(base_responses):
        reasoning = response["reasoning"]
        prompt_reasoning = f"{prompt}\n<think>\n{reasoning}"
        sentences, positions = string_to_sentences(reasoning)
        paragraphs, paragraph_positions = split_into_paragraphs_safe(reasoning)
        verdict = response["verdict"]
        out = {
            "prompt": prompt_reasoning,
            "reasoning": reasoning,
            "sentences": sentences,
            "paragraphs": paragraphs,
            "paragraph_positions": paragraph_positions,
            "verdict": verdict,
        }
        out_l.append(out)

    fp_out = f"blackbox/for_blackbox/{model_long2short[model_name]}/responses_{len(out_l)}.json"
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    # with open(fp_out, "w") as f:
    #     json.dump(out_l, f)
    # quit()

    # print_paragraph_stats(base_responses)
    # quit()
    # quit()

    res = mad_resample(
        {"responses": base_responses},
        prompt,
        num_responses=1,
        model=model_name,
        max_concurrent=500,
        strict_req_post=False,
        include_original_resp_idx=True,
        do_paragraphs=DO_PARAGRAPHS,
        provider_config=provider_config,
    )
    sentence2post = res[0]
    sentence2new_responses = res[1]
    sentence2original_responses = res[2]
    sentence2original_resp_idx = res[3]
    sentence2cnt_base = res[4]
    sentence2positions = res[5]
    sentence2norm_positions = res[6]

    # Evaluate all posts for blackmail
    print("\n" + "=" * 80)
    print("EVALUATING POSTS FOR BLACKMAIL CONTENT")
    print("=" * 80)

    sentence2verdicts = eval_sentence2post_verdicts(sentence2post)
    make_position_df(
        sentence2positions,
        sentence2norm_positions,
        sentence2verdicts,
        sentence2original_resp_idx,
        model_name,
    )
    quit()

    sentence2new_responses = add_verdict_to_sentence2new_responses(
        sentence2new_responses, sentence2verdicts
    )
    print("Prepping equalization and mirroring...")

    if WORD_ANALYSIS:
        print("Making word2verdicts...")
        word2verdicts_ctfl, word2verdicts_rspl, word2cnt_ctfl, word2cnt_rspl = make_word2verdict(
            sentence2verdicts,
            sentence2new_responses,
            do_paragraphs=DO_PARAGRAPHS,
            include_mirror=True,
        )
        print("Done prepping equalization and mirroring.")
        make_word_stats_csv(
            word2verdicts_ctfl, word2verdicts_rspl, word2cnt_ctfl, word2cnt_rspl, model_name
        )
    else:
        # Drop to equalize and also add mirror data
        # shouldn't do both this and the word analysis
        sentence2verdicts, sentence2cnt_base = final_prep_sentence2verdict(
            base_responses, sentence2verdicts, sentence2new_responses, sentence2cnt_base
        )
    quit()

    if model_name == "qwen/qwq-32b":
        UZAY_CHUNKS_CHECK = None
    all2cluster, sentence2cnt = get_cluster_mappers(
        sentence2new_responses,
        n_clusters=N_CLUSTERS,
        chunks_check=UZAY_CHUNKS_CHECK,
        do_paragraphs=DO_PARAGRAPHS,
        base_responses=base_responses if model_name != "qwen/qwq-32b" else None,
    )
    # print(list(all2cluster.keys())[:10])
    # quit()
    all2most, most2commons = get_all2most_cluster2commons(all2cluster, sentence2cnt_base)
    first_most = list(all2most.values())[0]
    print(f"First most: {first_most}")
    # print(f"First most: {len(most2commons[first_most])=}")
    # quit()

    # Create output directory for analysis results
    analysis_output_dir = "analysis_outputs"
    print(f"\nSaving analysis results to: {analysis_output_dir}/")

    most2delta, most2pre_post, most2logit, df = analyze_sentence2verdicts(
        sentence2verdicts,
        all2most,
        most2commons,
        sentence2cnt_base,
        output_dir=analysis_output_dir,
        n_clusters=N_CLUSTERS,
        min_cluster_cnt=50,
    )

    M_delta = np.mean(list(most2delta.values()))
    print(f"Mean delta: {M_delta=:.1%}")

    # TODO: Look at how often a different sentence is mentioned

