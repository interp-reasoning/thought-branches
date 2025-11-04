import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pprint import pprint
from time import time

import numpy as np
from pkld import pkld

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BERT.BERT_embedding_cache import get_bert_embeddings_cached
from base_responses import get_case_responses
from resampler.sentence_resampler import ResampleSentence, mad_resample
from resampler.weighted_kmeans import get_cluster_mappers
from resume.name_standardize import clean_name_from_str


DO_PARAGRAPHS = False
MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b"
N_CLUSTERS = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RESAMPLE_IDX_APPROACH = False
EQUALIZE = False
ANTI_BIAS = "none"
INCLUDE_MIRROR = True


def get_yes_no(content):
    if "yes" in content.lower():
        return 1.0
    elif "no" in content.lower():
        return 0.0
    else:
        return np.nan


def contrast_new_v_original(
    sentence2data, normalize=False, norm_by_lengths=False, variant=None
):

    # quit()

    overall_yn = []
    new_yn = []
    sentenc2data_variant = {}

    for sentence, data in sentence2data.items():
        original_responses = data.original_responses
        if variant is not None:
            original_responses = [
                r
                for r, v in zip(data.original_responses, data.variants)
                if v == variant
            ]

        original_answers = [r.content for r in original_responses]
        original_answers = np.array(list(map(get_yes_no, original_answers)))
        overall_yn.extend(list(original_answers))
        # original_p_yes = np.nanmean(original_answers)

        original_lengths = np.array(data.original_lengths)
        # print(f"{len(original_lengths)=}")
        # print(f"{len(original_responses)=}")
        # print(f"{data.count=}")
        if variant is None:
            assert len(original_lengths) == len(
                original_responses
            ), f"{len(original_lengths)=} {len(original_responses)=} {data.count=}"
            assert len(original_lengths) == data.count

        denom = np.nansum(1 / original_lengths)

        if norm_by_lengths:
            original_lengths = data.original_lengths
            original_p_yes = (
                np.nansum(original_answers / original_lengths) / denom
            )
        else:
            original_p_yes = np.nanmean(original_answers)

        new_responses = data.new_responses
        if variant is not None:
            new_responses = [
                r for r, v in zip(new_responses, data.variants) if v == variant
            ]
        new_answers = [r.content for r in new_responses]
        new_answers = np.array(list(map(get_yes_no, new_answers)))
        new_yn.extend(list(new_answers))
        if variant is None:
            assert len(original_lengths) == len(new_answers)
        if norm_by_lengths:
            new_p_yes = np.nansum(new_answers / original_lengths) / denom
        else:
            new_p_yes = np.nanmean(new_answers)
        delta = original_p_yes - new_p_yes
        delta_any = np.nanmean(np.abs(original_answers - new_answers))

        if variant is not None:
            sentenc2data_variant[sentence] = {
                "original_yn": original_answers,
                "new_yn": new_answers,
                "original_p_yes": original_p_yes,
                "new_p_yes": new_p_yes,
                "delta": delta,
                "delta_any": delta_any,
            }
        else:
            sentence2data[sentence].original_yn = original_answers
            sentence2data[sentence].new_yn = new_answers
            sentence2data[sentence].original_p_yes = original_p_yes
            sentence2data[sentence].new_p_yes = new_p_yes
            sentence2data[sentence].delta = delta
            sentence2data[sentence].delta_any = delta_any

    print(f"{np.nanmean(overall_yn)=:.3%} {np.nanmean(new_yn)=:.3%}")
    # print(f"{len(overall_yn)=} {len(new_yn)=}")
    # quit()

    if normalize:
        deltas = [data.delta for data in sentence2data.values()]
        weights = [data.count for data in sentence2data.values()]
        weighted_delta = np.average(deltas, weights=weights)
        for data in sentence2data.values():
            data.delta -= weighted_delta

    if variant is not None:
        return sentenc2data_variant
    else:
        return sentence2data


def collapse_sentence2data(sentence2data, all2most):
    most_sentences = []
    n_missing = 0
    for sentence, data in sentence2data.items():
        if sentence not in all2most:
            raise ValueError(f"Sentence {sentence} not in all2most")
            # n_missing += 1
            # continue
        most_sentence = all2most[sentence]
        if sentence == most_sentence:
            continue
        most_sentences.append(most_sentence)
        assert most_sentence in sentence2data
        most_data = sentence2data[most_sentence]
        most_data.original_responses.extend(data.original_responses)
        most_data.new_responses.extend(data.new_responses)
        most_data.original_resp_idx.extend(data.original_resp_idx)
        most_data.count += data.count
        most_data.positions.extend(data.positions)
        most_data.norm_positions.extend(data.norm_positions)
        most_data.original_lengths.extend(data.original_lengths)
        if len(data.variants) > 0:
            most_data.variants.extend(data.variants)
    print(f"Number of sentences missing from all2most: {n_missing}")

    most2data = {}
    for most_sentence in most_sentences:
        most2data[most_sentence] = sentence2data[most_sentence]
    return most2data


def merge_w_mirror(sentence2data, sentence2data_mirror, true_mirror=False):
    for sentence, data in sentence2data_mirror.items():
        for i in range(data.count):
            if true_mirror:
                mirror_mirror_s = data.mirror[i]
                sentence2data[mirror_mirror_s].original_responses.append(
                    data.new_responses[i]
                )
                sentence2data[mirror_mirror_s].new_responses.append(
                    data.original_responses[i]
                )
                sentence2data[mirror_mirror_s].original_resp_idx.append(
                    data.original_resp_idx[i]
                )
                sentence2data[mirror_mirror_s].count += 1
                sentence2data[mirror_mirror_s].positions.append(
                    data.positions[i]
                )
                sentence2data[mirror_mirror_s].norm_positions.append(
                    data.norm_positions[i]
                )
                sentence2data[mirror_mirror_s].mirror.append(data.mirror[i])
                sentence2data[mirror_mirror_s].original_lengths.append(
                    data.original_lengths[i]
                )
            else:
                mirror_mirror_s = sentence
                sentence2data[mirror_mirror_s].original_responses.append(
                    data.original_responses[i]
                )
                sentence2data[mirror_mirror_s].new_responses.append(
                    data.new_responses[i]
                )
                sentence2data[mirror_mirror_s].original_resp_idx.append(
                    data.original_resp_idx[i]
                )
                sentence2data[mirror_mirror_s].count += 1
                sentence2data[mirror_mirror_s].positions.append(
                    data.positions[i]
                )
                sentence2data[mirror_mirror_s].norm_positions.append(
                    data.norm_positions[i]
                )
                sentence2data[mirror_mirror_s].mirror.append(data.mirror[i])
                sentence2data[mirror_mirror_s].original_lengths.append(
                    data.original_lengths[i]
                )

            if len(data.variants) > 0:
                sentence2data[mirror_mirror_s].variants.append(data.variants[i])

    return sentence2data


def rando_drop_to_equalize(sentence2data, target=0.5, tol=0.005):
    weights = np.array([data.count for data in sentence2data.values()])
    baseline_s_yes = np.array(
        [data.original_p_yes for data in sentence2data.values()]
    )
    not_nan = np.array([not np.isnan(y) for y in baseline_s_yes])
    m_s_yes = np.average(baseline_s_yes[not_nan], weights=weights[not_nan])
    # m_s_yes = np.nansum(baseline_s_yes * weights) / np.nansum(weights * ~np.isnan(baseline_s_yes))
    # print(f"{target=:.1%}")
    # print(f"{m_s_yes=:.1%}")
    print("--- Equalizing ---")
    print(f"\tTarget yes percentage: {target:.1%}")
    print(f"\tSentence yes percentage: {m_s_yes:.1%}")
    # quit()
    if m_s_yes > target:
        drop_target = True
        target = 1 - target
        m_s_yes = 1 - m_s_yes
        print("\tDropping Yeses")
    else:
        print("\tDropping Nos")
        drop_target = False

    prop_drop = (target - m_s_yes) / (
        target * (1 - m_s_yes)
    )  # odds of dropping a No
    print(f"{m_s_yes=:.1%} {target=:.1%}")
    if drop_target:
        print(f"\tPercentage of [YESes] to drop: {prop_drop=:.1%}")
    else:
        print(f"\tPercentage of [NOs] to drop: {prop_drop=:.1%}")
    num_dropped = 0
    num_total = 0

    for sentence, data in sentence2data.items():
        for i in range(data.count - 1, -1, -1):
            if np.isnan(data.original_yn[i]):
                continue
            num_total += 1
            if data.original_yn[i] != drop_target:
                continue
            if random.random() < prop_drop:
                data.original_responses.pop(i)
                data.new_responses.pop(i)
                data.original_resp_idx.pop(i)
                data.count -= 1
                data.positions.pop(i)
                data.original_lengths.pop(i)
                if len(data.variants) > 0:
                    data.variants.pop(i)
                # print(f"{len(data.variants)=}")
                # data.variants.pop(i)
                num_dropped += 1
    contrast_new_v_original(sentence2data)
    weights = np.array([data.count for data in sentence2data.values()])
    baseline_s_yes = np.array(
        [data.original_p_yes for data in sentence2data.values()]
    )
    not_nan = np.array([not np.isnan(y) for y in baseline_s_yes])
    # print(f'{np.sum(np.isnan(baseline_s_yes))=}')
    m_s_yes = np.average(baseline_s_yes[not_nan], weights=weights[not_nan])
    if drop_target:  # revert it back to original target
        target = 1 - target
    assert (
        np.abs(m_s_yes - target) < tol
    ), f"Fail to equalize: {m_s_yes=:.1%} {target=}"
    print(
        f"\tEqualized ({m_s_yes=:.1%}; {target=}): {num_dropped=} {num_total=}"
    )
    print("--- ------- ---")
    return sentence2data


@pkld(overwrite=False)
def get_variant_sentence2data(
    variant_idx,
    resume_id,
    job,
    num_responses,
    model_name=MODEL_NAME,
    anti_bias=ANTI_BIAS,
    do_paragraphs=DO_PARAGRAPHS,
    resample_idx_approach=RESAMPLE_IDX_APPROACH,
    include_mirror=INCLUDE_MIRROR,
    equalize=EQUALIZE,
):
    result = get_case_responses(
        resume_id=resume_id,
        job=job,
        variant_idx=variant_idx,
        num_responses=num_responses,
        model=model_name,
        anti_bias=anti_bias,
    )
    yn = [get_yes_no(r.content) for r in result["responses"]]
    p_yes_base = np.nanmean(yn)
    print(f"{p_yes_base=:.1%}")
    # p_yes_base = 0.5

    sentence2data, sentence2data_mirror = mad_resample(
        result["responses"],
        result["prompt"],
        num_responses=1,
        model=model_name,
        max_concurrent=400,
        do_paragraphs=do_paragraphs,
        resample_idx_approach=resample_idx_approach,
    )
    key0 = list(sentence2data.keys())[0]
    # print(sentence2data[key0].original_responses[0])
    # quit()

    name = result["name"]
    first_name, last_name = name.split(" ")
    assert len(name.split(" ")) == 2, f"Bad name: {name=}"
    pronouns = result["pronouns"]
    email = result["email"]

    if include_mirror:
        sentence2data = merge_w_mirror(
            sentence2data, sentence2data_mirror, true_mirror=True
        )

    if equalize:
        sentence2data = contrast_new_v_original(sentence2data)
        sentence2data = rando_drop_to_equalize(sentence2data, target=p_yes_base)

    sentence2data = {
        clean_name_from_str(
            s, first_name, last_name, email, target="male"
        ): data
        for s, data in sentence2data.items()
    }
    # print(list(sentence2data.keys())[:100])
    # quit()
    # if equalize:
    for sentence, data in sentence2data.items():
        for i in range(data.count):
            data.mirror[i] = clean_name_from_str(
                data.mirror[i], first_name, last_name, email, target="male"
            )

    return sentence2data, p_yes_base


@pkld(overwrite=False)
def agg_multiple_variants(
    variant_idxs,
    resume_id,
    job,
    num_responses,
    do_paragraphs=DO_PARAGRAPHS,
    resample_idx_approach=RESAMPLE_IDX_APPROACH,
    include_mirror=INCLUDE_MIRROR,
    equalize=EQUALIZE,
):
    sentence2data = defaultdict(ResampleSentence)
    p_bases = []
    for variant_idx in variant_idxs:
        print(f"Processing variant: {variant_idx}")
        # try:
        sentence2data_variant, p_yes_base_variant = get_variant_sentence2data(
            variant_idx,
            resume_id,
            job,
            num_responses,
            do_paragraphs=do_paragraphs,
            resample_idx_approach=resample_idx_approach,
            include_mirror=include_mirror,
            equalize=equalize,
        )
        for sentence, data in sentence2data_variant.items():
            sentence2data_variant[sentence].variants.extend(
                [variant_idx] * data.count
            )
        # print(f"{p_yes_base_variant=:.1%}")

        contrast_new_v_original(sentence2data_variant, normalize=False)
        sentence2data = merge_w_mirror(sentence2data, sentence2data_variant)
        for sentence, data in sentence2data.items():
            assert len(data.original_lengths) == data.count
            assert (
                len(data.variants) == data.count
            ), f"{len(data.variants)=} {data.count=}"

        p_bases.append(p_yes_base_variant)
        # except TypeError as e:
        #     print(f"TypeError processing variant: {variant_idx}")
        #     print(e)
        #     continue
        # except AssertionError as e:
        #     print(f"AssertionError processing variant: {variant_idx}")
        #     print(e)
        #     continue
    return sentence2data, p_bases


def run_multiple_variants(resume_id, job, num_responses):
    variant_idxs = list(range(4))
    # variant_idxs = [0]
    # variant_idxs = variant_idxs[::-1]
    print(f"{RESUME_ID=}")
    print(f"{JOB=}")
    sentence2data, p_bases = agg_multiple_variants(
        variant_idxs, resume_id, job, num_responses
    )

    all2most = get_cluster_mappers(
        sentence2data,
        n_clusters=N_CLUSTERS,
    )
    for most in all2most.values():
        assert most in sentence2data

    most2data = collapse_sentence2data(sentence2data, all2most)
    most2data = contrast_new_v_original(most2data, normalize=False)
    # if EQUALIZE:
    #     most2data = rando_drop_to_equalize(
    #         most2data, target=np.nanmean(p_bases)
    #     )

    weights = [data.count for data in most2data.values()]

    deltas = [data.delta for data in most2data.values()]
    weighted_delta = np.average(deltas, weights=weights)
    print("----------------")
    print(f"Weighted_delta: {weighted_delta:+.3f}")
    print("----------------")

    most_sorted = sorted(most2data.items(), key=lambda x: x[1].delta)

    for most, data in most_sorted:
        print(
            f"{most:>100}: {data.delta=:+.3f} ({data.count})) | {data.original_p_yes:0.2f} -> {data.new_p_yes:0.2f}"
        )
    print("--------")
    print(f"{JOB=}")
    print(f"{RESUME_ID=}")
    return sentence2data


if __name__ == "__main__":

    NUM_RESPONSES = 500
    # NUM_RESPONSES = 250
    # NUM_RESPONSES = 100

    # qwen3-30b-a3b is not racist...

    # RESUME_ID = 21  # Bad resume, rec letter for unrelated person

    # RESUME_ID = 13
    # RESUME_ID = 39
    # RESUME_ID = 0 # focus on inexperience
    # RESUME_ID = 7
    JOB = "basic"
    # RESUME_ID = 54  # pro male
    # RESUME_ID = 69  # very pro female
    # RESUME_ID = 73  # also pro male
    # RESUME_ID = 83  # also pro male

    RESUME_ID = 61  # mild female bias
    # RESUME_ID = 67  # mild male bias

    # meta, 21 # pro-male
    # meta, 25 # pro white male but high percentage
    # meta, 32 # very pro-Black
    # meta, 35 # pro-Black
    # meta, 31 # pro-Black

    # JOB = "meta"
    # RESUME_ID = 21 # bad prompt? "But wait, the candidate's name in the resume is John, but the recommendation letters mention William Hale"
    # RESUME_ID = 25
    # RESUME_ID = 32
    # RESUME_ID = 31
    # RESUME_ID = 35

    run_multiple_variants(RESUME_ID, JOB, NUM_RESPONSES)

    # TODO: Look at the 3 biggest boosters and detractors for each job
    # TODO: Then try and link it to whether there is a white vs. black bias
    # TODO: Based on the factors with a biggest causal impact for one resume, can we predict the degree of bias?
