import sys
import os
import pickle
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from collections import defaultdict
from resampler.sentence_resampler import ResampleSentence
from resampler.weighted_kmeans import get_cluster_mappers
from resume.gen_resume_data import (
    agg_multiple_variants,
    get_yes_no,
    merge_w_mirror,
    collapse_sentence2data,
    contrast_new_v_original,
    rando_drop_to_equalize,
)
import numpy as np
from collections import Counter
from scipy import stats


def cnt_variant(sentence2data):
    for sentence, data in sentence2data.items():
        assert len(data.variants) == data.count
        print(f"{sentence:>100}: {len(data.variants)=}")
    quit()


def get_variant_rate(sentence2data):
    variant_overall = defaultdict(int)
    for sentence, data in sentence2data.items():
        for variant in data.variants:
            variant_overall[variant] += 1
    # variant_overall = {k: v / len(sentence2data) for k, v in variant_overall.items()}
    # print(f"{variant_overall=}")
    # quit()

    for sentence, data in sentence2data.items():
        assert (
            len(data.variants) == data.count
        ), f"{len(data.variants)=} {data.count=}"
        cnt = Counter(data.variants)
        # baseline = (cnt[1] + cnt[2]) / 2
        data.variant_cnt = {k: v / variant_overall[k] for k, v in cnt.items()}
        for v in range(4):
            if v not in data.variant_cnt:
                data.variant_cnt[v] = 0
    return sentence2data


def str_variant_rates(variant_cnt):
    num2key = {0: "WF", 1: "BF", 2: "WM", 3: "BM"}
    s_out = ""
    for num in range(4):
        if num not in variant_cnt:
            continue
        s_out += f"{num2key[num]}: {variant_cnt[num]:.1%}, "
    s_out = s_out[:-2]
    return s_out


def corr_variant(most2data, variant_idx):
    variant_freqs = []
    deltas = []
    for sentence, data in most2data.items():
        variant_freqs.append(data.variant_cnt[variant_idx])
        deltas.append(data.delta)

    r, p = stats.spearmanr(variant_freqs, deltas)
    return r, p


def correlate_frequency_with_bias(sentence2data, p_bases):
    # TODO: Note, all of the data here is fucking flipped... due to the merge with mirror

    fp_cluster_mapper = f"cluster_mapper/cluster_mapper_{JOB}_{p_bases}_{N_CLUSTERS}_{INCLUDE_MIRROR}.pkl"
    Path(fp_cluster_mapper).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(fp_cluster_mapper) and not OVERWRITE_CLUSTER_MAPPER:
        with open(fp_cluster_mapper, "rb") as f:
            all2most = pickle.load(f)
    else:
        all2most = get_cluster_mappers(
            sentence2data,
            n_clusters=N_CLUSTERS,
        )
        with open(fp_cluster_mapper, "wb") as f:
            pickle.dump(all2most, f)
    # all2most = get_cluster_mappers(
    #     sentence2data,
    #     n_clusters=N_CLUSTERS,
    # )
    most2data = collapse_sentence2data(sentence2data, all2most)
    most2data = contrast_new_v_original(most2data, normalize=False)
    # most2data_l = defaultdict(lambda: defaultdict(list))

    # for variant_idx in range(4):
    #     most2data_variant = contrast_new_v_original(
    #         most2data, normalize=False, variant=variant_idx
    #     )
    #     for sentence, data in most2data_variant.items():
    #         # most2data_l[sentence].append(data)
    #         most2data_l[sentence]["original_yn"].append(data["original_yn"])
    #         most2data_l[sentence]["new_yn"].append(data["new_yn"])
    #         most2data_l[sentence]["original_p_yes"].append(
    #             data["original_p_yes"]
    #         )
    #         most2data_l[sentence]["new_p_yes"].append(data["new_p_yes"])
    #         most2data_l[sentence]["delta"].append(data["delta"])
    #         most2data_l[sentence]["delta_any"].append(data["delta_any"])

    # for sentence, data in most2data_l.items():
    #     # print(data['original_yn'])
    #     # most2data[sentence].original_yn = np.nanmean(data['original_yn'])
    #     # most2data[sentence].new_yn = np.nanmean(data['new_yn'])
    #     most2data[sentence].original_p_yes = np.nanmean(data["original_p_yes"])
    #     most2data[sentence].new_p_yes = np.nanmean(data["new_p_yes"])
    #     most2data[sentence].delta = np.nanmean(data["delta"])
    #     most2data[sentence].delta_any = np.nanmean(data["delta_any"])

    # most2data = contrast_new_v_original(most2data, normalize=False)
    most2data = get_variant_rate(most2data)
    make_sentence_df(most2data)

    BF_WM = p_bases[1] - p_bases[2]
    BF_WM_freqs = []
    deltas = []
    mosts_sorted = sorted(most2data.items(), key=lambda x: x[1].delta)
    df_as_l = []
    for most, data in mosts_sorted:
        row = {}
        for variant_idx in range(4):
            row[f"variant_{variant_idx}"] = data.variant_cnt[variant_idx]
        row["delta"] = data.delta
        row["BF_WM_freq"] = data.variant_cnt[1] - data.variant_cnt[2]
        row["text"] = most
        df_as_l.append(row)

        BF_WM_freq = data.variant_cnt[1] - data.variant_cnt[2]
        deltas.append(data.delta)
        BF_WM_freqs.append(BF_WM_freq)
        s = str_variant_rates(data.variant_cnt)
        print(f"{most:>100}: {data.delta=:.3f} ({s})")
    r_bf_wm, p_bf_wm = stats.spearmanr(BF_WM_freqs, deltas)
    print(f"Delta x BF - WM frequency: {r_bf_wm=:.3f} {p_bf_wm=:.3f}")
    df = pd.DataFrame(df_as_l)
    df.to_csv(
        rf"bias_csvs/bias_delta_freq_{JOB}_{RESUME_ID}_{N_CLUSTERS}_{INCLUDE_MIRROR}.csv",
        index=False,
    )

    p_bases_l = []
    rs = []
    for variant_idx in range(4):
        p_base = p_bases[variant_idx] - np.nanmean(p_bases)
        r, p = corr_variant(most2data, variant_idx)
        print(f"Variant {variant_idx} ({p_base:.1%}): {r:.3f} {p:.3f}")
        # if variant_idx in [1, 2]:
        p_bases_l.append(p_base)
        rs.append(r)
    r_mini, p_mini = stats.spearmanr(p_bases_l, rs)
    return p_bases_l, rs, r_mini, p_mini, r_bf_wm, p_bf_wm


def make_sentence_df(most2data):
    original_resp2sentences = defaultdict(list)
    original_resp2decision = {}
    unique_sentences = set()
    for sentence, data in most2data.items():
        for i in range(data.count):
            original_resp2sentences[
                (data.original_resp_idx[i], data.variants[i])
            ].append(sentence)
            original_resp2decision[
                (data.original_resp_idx[i], data.variants[i])
            ] = data.original_yn[i]
            unique_sentences.add(sentence)

    sentence2dim = {sentence: i for i, sentence in enumerate(unique_sentences)}
    df_as_l = []
    for (idx, variant), sentences in original_resp2sentences.items():
        row = {
            "idx": idx,
            "variant": variant,
            "decision": original_resp2decision[(idx, variant)],
        }
        for i in range(len(sentence2dim)):
            row[f"x_{i}"] = 0
        for sentence in sentences:
            key = sentence2dim[sentence]
            row[f"x_{key}"] = 1
        df_as_l.append(row)

    df = pd.DataFrame(df_as_l)
    df.to_csv(
        f"bias_csvs/sentence_{JOB}_{N_CLUSTERS}_{INCLUDE_MIRROR}_{RESUME_ID}.csv",
        index=False,
    )
    # quit()
    # for original_resp_idx, sentences in original_resp2sentences.items():
    # for sentence in sentences:


def correlate_delta_with_frequency(sentence2data):

    fp_cluster_mapper = f"cluster_mapper/cluster_mapper_{JOB}_{p_bases}_{N_CLUSTERS}_{INCLUDE_MIRROR}.pkl"
    Path(fp_cluster_mapper).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(fp_cluster_mapper) and not OVERWRITE_CLUSTER_MAPPER:
        with open(fp_cluster_mapper, "rb") as f:
            all2most = pickle.load(f)
    else:
        all2most = get_cluster_mappers(
            sentence2data,
            n_clusters=N_CLUSTERS,
        )
        with open(fp_cluster_mapper, "wb") as f:
            pickle.dump(all2most, f)
    most2data = collapse_sentence2data(sentence2data, all2most)
    most2data = contrast_new_v_original(most2data, normalize=False)
    most2data = get_variant_rate(most2data)

    freqs = []
    deltas = []
    mosts_sorted = sorted(most2data.items(), key=lambda x: x[1].delta)
    for most, data in mosts_sorted:
        freq = (
            data.variant_cnt[0]
            + data.variant_cnt[1]
            + data.variant_cnt[2]
            + data.variant_cnt[3]
        ) / 4
        freqs.append(freq)
        deltas.append(data.delta)
        original_p_yes = data.original_p_yes
        new_p_yes = data.new_p_yes
        print(
            f"{most:>100}: {data.delta=:.3f} ({original_p_yes=:.1%} -> {new_p_yes=:.1%}) ({freq=:.1%})"
        )
    r, p = stats.spearmanr(freqs, deltas)
    print(f"Delta x frequency: {r=:.3f} {p=:.3f}")
    return r, p


if __name__ == "__main__":
    NUM_RESPONSES = 500
    JOB = "basic"
    # JOB = "meta"
    # 54 Anti-male
    RESUME_IDS = [13, 39, 0, 7, 69, 73, 83, 61]
    # , 6754,
    # RESUME_IDS = [39, 0, 69, 73, 83]  # 61, 67
    if JOB == "meta":
        RESUME_IDS = [25, 32, 35, 31]
        NUM_RESPONSES = 250

    # RESUME_IDS = RESUME_IDS[::-1]
    # RESUME_IDS = [0]
    # RESUME_IDS = [13]
    # RESUME_IDS = [69]
    # RESUME_IDS = [39, 7, 73, 83, 0]
    N_CLUSTERS = 512
    N_CLUSTERS = 128
    N_CLUSTERS = 16
    # N_CLUSTERS = 256
    # N_CLUSTERS = 32
    EQUALIZE = False
    INCLUDE_MIRROR = True
    RESAMPLE_IDX_APPROACH = True

    OVERWRITE_CLUSTER_MAPPER = False

    VARIANT_IDXS = list(range(4))
    DO_CORR_X_FREQ = False
    DO_CORR_BIAS = True
    # RESUME_IDS = [39]
    # VARIANT_IDXS = [2]
    # VARIANT_IDXS = list(range(4))[::-1]
    # print("TEST TEST")
    # print(f"{VARIANT_IDXS=}")
    # quit()
    # quit()

    all_p_bases = []
    all_rs = []

    sentence2data_super = defaultdict(ResampleSentence)
    delta_x_freqs = []
    r_minis = []
    all_r_bf_wms = []
    yes_rates = []
    for RESUME_ID in RESUME_IDS:
        print(f"Processing resume: {RESUME_ID}")
        sentence2data, p_bases = agg_multiple_variants(
            VARIANT_IDXS,
            RESUME_ID,
            JOB,
            NUM_RESPONSES,
            include_mirror=INCLUDE_MIRROR,
            resample_idx_approach=RESAMPLE_IDX_APPROACH,
            equalize=EQUALIZE,
        )
        variant_yn = defaultdict(list)
        for sentence, data in sentence2data.items():
            for i in range(data.count):
                original_response = data.original_responses[i]
                original_answer = get_yes_no(original_response.content)
                variant_yn[data.variants[i]].append(original_answer)
                new_response = data.new_responses[i]
                new_answer = get_yes_no(new_response.content)
                variant_yn[data.variants[i]].append(new_answer)
        p_bases = [np.nanmean(variant_yn[variant]) for variant in VARIANT_IDXS]
        m_p_base = np.nanmean(p_bases)
        yes_rates.append(m_p_base)
        print(f"{np.mean(yes_rates)=:.1%} {np.std(yes_rates)=:.1%}")
        print(f"| {RESUME_ID} | {p_bases=}")
        # quit()

        # for variant in VARIANT_IDXS:
        #     p_base = p_bases[variant] - np.nanmean(p_bases)
        #     r, p = stats.spearmanr(variant_yn[variant], p_base)
        #     print(f"Variant {variant} ({p_base:.1%}): {r:.3f} {p:.3f}")
        #     all_p_bases.append(p_base)
        #     all_rs.append(r)
        # for variant in VARIANT_IDXS:

        if DO_CORR_BIAS:
            p_bases, rs, r_mini, _, r_bf_wm, _ = correlate_frequency_with_bias(
                sentence2data, p_bases
            )
            all_p_bases.extend(p_bases)
            all_rs.extend(rs)
            all_r_bf_wms.append(r_bf_wm)
            r_meta, p_meta = stats.spearmanr(all_p_bases, all_rs)
            r_minis.append(r_mini)
            print(f"Meta ({len(all_p_bases)}): {r_meta=:.3f} {p_meta=:.3f}")
            if len(r_minis) > 2:
                t, p = stats.ttest_1samp(r_minis, 0)
            else:
                t = 0
                p = 0
            print(
                f"Mini ({len(r_minis)}): {np.mean(r_minis)=:.3f} {np.std(r_minis)=:.3f} {t=:.3f} {p=:.3f}"
            )

            if len(all_r_bf_wms) > 2:
                t, p = stats.ttest_1samp(all_r_bf_wms, 0)
            else:
                t = 0
                p = 0
            p_positive = np.mean(np.array(all_r_bf_wms) > 0)
            print(
                f"BF_WM ({p_positive=:.1%}): {np.mean(all_r_bf_wms)=:.3f} {np.std(all_r_bf_wms)=:.3f} {t=:.3f} {p=:.3f}"
            )
        elif DO_CORR_X_FREQ:
            r, p = correlate_delta_with_frequency(sentence2data)
            delta_x_freqs.append(r)
            mean_delta_x_freq = np.mean(delta_x_freqs)
            if len(delta_x_freqs) > 2:
                t, p = stats.ttest_1samp(delta_x_freqs, 0)
            else:
                t = 0
                p = 0
            p_positive = np.mean(np.array(delta_x_freqs) > 0)
            print(
                f"Delta x frequency ({p_positive=:.1%}): {mean_delta_x_freq=:.3f} {t=:.3f} {p=:.3f}"
            )
        else:

            sentence2data_super = merge_w_mirror(
                sentence2data_super, sentence2data
            )
            for sentence, data in sentence2data.items():
                assert len(data.original_lengths) == data.count
                assert len(data.variants) == data.count
                sentence2data_super[sentence].resume_ids.extend(
                    [RESUME_ID] * data.count
                )
        # contrast_new_v_original(sentence2data_super, normalize=False)
        # quit()

        # break
    # quit()
    if DO_CORR_X_FREQ or DO_CORR_BIAS:
        quit()
    total_count = sum([data.count for data in sentence2data_super.values()])

    fp_cluster_mapper = (
        f"cluster_mapper_{JOB}_{RESUME_IDS}_{N_CLUSTERS}_{INCLUDE_MIRROR}.pkl"
    )
    if os.path.exists(fp_cluster_mapper) and not OVERWRITE_CLUSTER_MAPPER:
        with open(fp_cluster_mapper, "rb") as f:
            all2most = pickle.load(f)
    else:
        all2most = get_cluster_mappers(
            sentence2data_super,
            n_clusters=N_CLUSTERS,
        )
        with open(fp_cluster_mapper, "wb") as f:
            pickle.dump(all2most, f)
    for most in all2most.values():
        assert most in sentence2data_super

    most2data = collapse_sentence2data(sentence2data_super, all2most)
    # most2data = sentence2data_super

    most2data = contrast_new_v_original(most2data, normalize=False)
    most2data = get_variant_rate(most2data)
    # if EQUALIZE:
    #     most2data = rando_drop_to_equalize(
    #         most2data, target=np.nanmean(p_bases)
    #     )

    weights = np.array([data.count for data in most2data.values()])
    # weights_inv_len = 1 / np.array([data.original_lengths for data in most2data.values()])
    # weights = [1 /data.count for data in most2data.values()]

    deltas = [data.delta for data in most2data.values()]

    weighted_delta = np.average(deltas, weights=weights)
    print("----------------")
    print(f"Weighted_delta: {weighted_delta:+.3f}")
    print("----------------")
    # quit()

    most_sorted = sorted(most2data.items(), key=lambda x: x[1].delta)

    deltas = []
    num_nans = 0
    num_total = 0
    variant_freqs = defaultdict(list)
    for most, data in most_sorted:
        if len(most_sorted) < 1025:
            s_variant_rates = str_variant_rates(data.variant_cnt)
            print(
                f"{most:>100}: {data.delta=:+.3f} ({data.count})) | {data.original_p_yes:0.2f} -> {data.new_p_yes:0.2f} ({s_variant_rates})"
            )
        if np.isnan(data.delta):
            num_nans += 1
            continue
        num_total += 1
        deltas.append(data.delta)
        for variant_idx in range(4):
            variant_freqs[variant_idx].append(data.variant_cnt[variant_idx])
        variant_freqs[4].append(sum(variant_freqs[i][-1] for i in range(4)))
    # print(f"{np.mean(deltas)=:.3f}")
    print(f"{num_nans=}/{num_total=}")
    for variant_idx in range(5):
        r, p = stats.spearmanr(variant_freqs[variant_idx], deltas)
        print(f"Variant {variant_idx}: {r=:.3f} {p=:.3f}")
        # print(f"Variant {variant_idx}: {np.mean(variant_freqs[variant_idx])=:.3f}")
    print("--------")
