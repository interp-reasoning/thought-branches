import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from pkld import pkld


# @pkld(overwrite=True)
def analyze_sentence2verdicts(
    sentence2verdicts,
    all2most,
    most2commons,
    sentences_cnt_base,
    get_both=True,
    output_dir="analysis_outputs",
    n_clusters=None,
    min_cluster_cnt=50,
):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include number of clusters in filename if provided
    if n_clusters is not None:
        output_file = os.path.join(output_dir, f"analysis_{n_clusters}clusters_{timestamp}.txt")
        csv_file = os.path.join(output_dir, f"analysis_{n_clusters}clusters_{timestamp}.csv")
    else:
        output_file = os.path.join(output_dir, f"analysis_{timestamp}.txt")
        csv_file = os.path.join(output_dir, f"analysis_{timestamp}.csv")

    # Open file for writing
    with open(output_file, "w", encoding="utf-8") as f:

        def print_both(text=""):
            """Print to console and write to file"""
            print(text)
            f.write(text + "\n")

        print_both("=" * 80)
        print_both(f"Sentence Verdicts Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_both("=" * 80)
        print_both()
        print_both("Analyzing sentence2verdicts")

        most2verdicts = defaultdict(list)
        missing_sentence_bad = 0
        sentence_in_good = 0
        for sentence, verdicts in sentence2verdicts.items():
            if sentence not in all2most:
                # print(f"Sentence not in all2most: {sentence}")
                missing_sentence_bad += 1
                continue
            sentence_in_good += 1
            # if sentence not in all2most:  # Sometimes happens for mirror sentences
            #     raise ValueError(f"Sentence {sentence} not in all2most")
            #     continue
            most = most2commons[all2most[sentence]][0]
            most2verdicts[most].extend(verdicts)
        sentence2verdicts = most2verdicts

        print_both(f"Failure to capture {missing_sentence_bad} sentences")
        print_both(f"Captured {sentence_in_good} sentences")
        p_bad = missing_sentence_bad / (missing_sentence_bad + sentence_in_good)
        print_both(f"Percentage of sentences not captured: {p_bad:.1%}")
        print_both()
        # print(list(sentence2verdicts.keys())[:10])
        # quit()

        df_as_l = []

        for sentence, answers in sentence2verdicts.items():

            original_l_binary = [ans[0] for ans in answers]
            new_l_binary = [ans[1] for ans in answers]

            # assert None not in original_l_binary, f"None in original_l_binary"
            # assert None not in new_l_binary, f"None in new_l_binary"
            prop_yes = np.nanmean(original_l_binary)
            n_yes = np.nansum(original_l_binary)
            n_no = np.sum(original_l_binary == 0)
            # n_resamples = len(original_l_binary)
            prop_yes_new = np.nanmean(new_l_binary)
            n_yes_new = np.nansum(new_l_binary)
            n_no_new = np.sum(new_l_binary == 0)
            delta = prop_yes - prop_yes_new

            prop_yes_smoothed = (n_yes + 1) / (n_yes + n_no + 2)
            prop_yes_new_smoothed = (n_yes_new + 1) / (n_yes_new + n_no_new + 2)

            # Clip probabilities to avoid log(0) or log(inf)
            eps = 1e-10
            prop_yes_smoothed = np.clip(prop_yes_smoothed, eps, 1 - eps)
            prop_yes_new_smoothed = np.clip(prop_yes_new_smoothed, eps, 1 - eps)

            smoothed_logit = np.log(prop_yes_smoothed / (1 - prop_yes_smoothed)) - np.log(
                prop_yes_new_smoothed / (1 - prop_yes_new_smoothed)
            )

            row = {
                "sentence": sentence,
                "prop_yes": prop_yes,
                "prop_yes_new": prop_yes_new,
                "prop_yes_smoothed": prop_yes_smoothed,
                "prop_yes_new_smoothed": prop_yes_new_smoothed,
                "delta": delta,
                "n_yes": n_yes,
                "n_no": n_no,
                "n_yes_new": n_yes_new,
                "n_no_new": n_no_new,
                "n_resamples": n_yes + n_no,
                "n_resamples_new": n_yes_new + n_no_new,
                "smoothed_logit": smoothed_logit,
            }
            df_as_l.append(row)

        df = pd.DataFrame(df_as_l)

        # Save DataFrame to CSV
        df.to_csv(csv_file, index=False)
        print_both(f"\nSaved detailed results to: {csv_file}")

        # print(df["n_resamples"].describe())
        # print(df["n_resamples"].sum())
        # quit()
        print_both("\nOriginal prop_yes statistics:")
        print_both(str(df["prop_yes"].describe()))
        print_both("\n" + "-" * 40)
        print_both("\nNew prop_yes statistics:")
        print_both(str(df["prop_yes_new"].describe()))
        print_both("\n" + "-" * 40)
        # print(df["delta"].describe())
        # return
        df.sort_values(by="delta", ascending=False, inplace=True)
        print(df["n_resamples"].describe())
        num_below_threshold = (df["n_resamples"] < min_cluster_cnt).sum()
        print(
            f"Number of sentences below threshold (under {min_cluster_cnt} resamples): {num_below_threshold}"
        )
        df = df[df["n_resamples"] >= min_cluster_cnt]
        print(f"Number of sentences after filtering: {len(df)}")

        print_both("\nDetailed Results (sorted by delta, descending):")
        print_both("=" * 80)

        most2delta = {}
        most2pre_post = {}
        most2logit = {}
        for i, row in df.iterrows():
            prop_yes = row["prop_yes"]
            prop_yes_new = row["prop_yes_new"]
            delta = row["delta"]
            sentence = row["sentence"]
            n_resamples = row["n_resamples"]
            smoothed_logit = row["smoothed_logit"]
            prop_yes_smoothed = row["prop_yes_smoothed"]
            prop_yes_new_smoothed = row["prop_yes_new_smoothed"]
            # commons = most2commons[sentence]
            commons = most2commons[sentence]
            commons = [c for c in commons if c in sentences_cnt_base]
            random.shuffle(commons)
            commons_sorted = sorted(commons, key=lambda x: sentences_cnt_base[x], reverse=True)

            print_both(
                f"\n{prop_yes=:.1%} {prop_yes_new=:.1%} ({delta:+.1%}; {smoothed_logit=:.2f}; {n_resamples} resamples): {sentence=}"
            )
            for common in commons_sorted[:10]:
                print_both(f"  {sentences_cnt_base[common]}: {common}")
            most2delta[sentence] = delta
            most2logit[sentence] = smoothed_logit
            most2pre_post[sentence] = (prop_yes_new_smoothed, prop_yes_smoothed)

        print_both("\n" + "=" * 80)
        print_both(f"Analysis complete. Results saved to:")
        print_both(f"  - Text output: {output_file}")
        print_both(f"  - CSV data: {csv_file}")
        print_both("=" * 80)

    return most2delta, most2pre_post, most2logit, df
