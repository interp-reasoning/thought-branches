import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from resampler.sentence_splitter import split_into_paragraphs_safe, string_to_sentences
from functools import cache
from collections import defaultdict
from pkld import pkld
from tqdm import tqdm


@cache
def remove_punctuation(text):
    return "".join(c for c in text if c.isalnum() or c.isspace())


@pkld(overwrite=True)
def make_word2verdict(
    sentence2verdicts, sentence2new_responses, do_paragraphs=False, include_mirror=True
):
    print(
        f"Making word2verdicts with do_paragraphs={do_paragraphs} and include_mirror={include_mirror}"
    )
    word2verdicts_ctfl = defaultdict(list)
    word2verdicts_rspl = defaultdict(list)
    word2cnt_ctfl = defaultdict(int)
    word2cnt_rspl = defaultdict(int)
    rare_pkld_errors = 0
    rare_sentence_para_assertion_errors = 0
    rare_empty_new_sentences = 0
    rare_verdicts_new_responses_mismatch = 0
    for sentence, verdicts in tqdm(sentence2verdicts.items(), desc="Making word2verdicts"):
        original_words = set(remove_punctuation(sentence).split())
        if len(sentence2new_responses[sentence]) != len(verdicts):
            rare_verdicts_new_responses_mismatch += 1
            continue
        # assert len(sentence2new_responses[sentence]) == len(verdicts)
        for new_response, verdict in zip(sentence2new_responses[sentence], verdicts):
            reasoning = new_response["downstream_reasoning"]
            try:
                if do_paragraphs:
                    paragraphs, paragraph_positions = split_into_paragraphs_safe(reasoning)
                    sentences, _ = paragraphs, paragraph_positions
                else:
                    sentences, _ = string_to_sentences(reasoning)
            except OSError as e:  # for short sentences with a "\n"
                rare_pkld_errors += 1
                continue
            except AssertionError as e:
                rare_sentence_para_assertion_errors += 1
                continue
            if len(sentences) == 0:
                rare_empty_new_sentences += 1
                continue
            new_sentence = sentences[0]
            new_words = set(remove_punctuation(new_sentence).split())
            only_original_words = original_words - new_words
            only_new_words = new_words - original_words
            for word in only_original_words:
                word2verdicts_ctfl[word].append((verdict[0], verdict[1]))
                word2cnt_ctfl[word] += 1
            for word in original_words:
                word2verdicts_rspl[word].append((verdict[0], verdict[1]))
                word2cnt_rspl[word] += 1
            if include_mirror:
                for word in only_new_words:
                    word2verdicts_ctfl[word].append((verdict[1], verdict[0]))
                    word2cnt_ctfl[word] += 1
                for word in new_words:
                    word2verdicts_rspl[word].append((verdict[1], verdict[0]))
                    word2cnt_rspl[word] += 1
    num_words = len(word2verdicts_ctfl)
    print(f"Completed word2verdicts! ({num_words} words)")
    print(f"\tNumber of rare pkld errors: {rare_pkld_errors}")
    if do_paragraphs:
        print(
            f"\tNumber of rare sentence-paragraph assertion errors: {rare_sentence_para_assertion_errors}"
        )
    print(f"\tNumber of rare empty new sentences: {rare_empty_new_sentences}")
    print(
        f"\tNumber of rare verdicts-new_responses mismatch: {rare_verdicts_new_responses_mismatch}"
    )
    return word2verdicts_ctfl, word2verdicts_rspl, word2cnt_ctfl, word2cnt_rspl


def make_word_stats_csv(
    word2verdicts_ctfl, word2verdicts_rspl, word2cnt_ctfl, word2cnt_rspl, model_name
):
    print("Making word_stats.csv...")
    words = list(word2verdicts_ctfl.keys())
    words.sort(key=lambda x: word2cnt_rspl[x], reverse=True)
    df_as_l = []
    for word in words:
        n_yes_ctfl = sum(1 for verdict in word2verdicts_ctfl[word] if verdict[0])
        n_no_ctfl = sum(1 for verdict in word2verdicts_ctfl[word] if not verdict[0])
        n_yes_ctfl_base = sum(1 for verdict in word2verdicts_ctfl[word] if verdict[1])
        n_no_ctfl_base = sum(1 for verdict in word2verdicts_ctfl[word] if not verdict[1])
        n_yes_rspl = sum(1 for verdict in word2verdicts_rspl[word] if verdict[0])
        n_no_rspl = sum(1 for verdict in word2verdicts_rspl[word] if not verdict[0])
        n_yes_rspl_base = sum(1 for verdict in word2verdicts_rspl[word] if verdict[1])
        n_no_rspl_base = sum(1 for verdict in word2verdicts_rspl[word] if not verdict[1])
        p_ctfl = n_yes_ctfl / (n_yes_ctfl + n_no_ctfl)
        p_rspl = n_yes_rspl / (n_yes_rspl + n_no_rspl)
        p_ctfl_base = n_yes_ctfl_base / (n_yes_ctfl_base + n_no_ctfl_base)
        p_rspl_base = n_yes_rspl_base / (n_yes_rspl_base + n_no_rspl_base)
        delta_ctfl = p_ctfl - p_ctfl_base
        delta_rspl = p_rspl - p_rspl_base

        n_00_ctfl = sum(
            1 for verdict in word2verdicts_ctfl[word] if not verdict[0] and not verdict[1]
        )
        n_01_ctfl = sum(1 for verdict in word2verdicts_ctfl[word] if not verdict[0] and verdict[1])
        n_10_ctfl = sum(1 for verdict in word2verdicts_ctfl[word] if verdict[0] and not verdict[1])
        n_11_ctfl = sum(1 for verdict in word2verdicts_ctfl[word] if verdict[0] and verdict[1])
        n_00_rspl = sum(
            1 for verdict in word2verdicts_rspl[word] if not verdict[0] and not verdict[1]
        )
        n_01_rspl = sum(1 for verdict in word2verdicts_rspl[word] if not verdict[0] and verdict[1])
        n_10_rspl = sum(1 for verdict in word2verdicts_rspl[word] if verdict[0] and not verdict[1])
        n_11_rspl = sum(1 for verdict in word2verdicts_rspl[word] if verdict[0] and verdict[1])

        row = {
            "word": word,
            "n_yes_ctfl": n_yes_ctfl,
            "n_no_ctfl": n_no_ctfl,
            "n_yes_rspl": n_yes_rspl,
            "n_no_rspl": n_no_rspl,
            "n_yes_ctfl_base": n_yes_ctfl_base,
            "n_no_ctfl_base": n_no_ctfl_base,
            "n_yes_rspl_base": n_yes_rspl_base,
            "n_no_rspl_base": n_no_rspl_base,
            "p_ctfl": p_ctfl,
            "p_rspl": p_rspl,
            "p_ctfl_base": p_ctfl_base,
            "p_rspl_base": p_rspl_base,
            "delta_ctfl": delta_ctfl,
            "delta_rspl": delta_rspl,
            "cnt_ctfl": word2cnt_ctfl[word],
            "cnt_rspl": word2cnt_rspl[word],
            "n_00_ctfl": n_00_ctfl,
            "n_01_ctfl": n_01_ctfl,
            "n_10_ctfl": n_10_ctfl,
            "n_11_ctfl": n_11_ctfl,
            "n_00_rspl": n_00_rspl,
            "n_01_rspl": n_01_rspl,
            "n_10_rspl": n_10_rspl,
            "n_11_rspl": n_11_rspl,
        }
        df_as_l.append(row)
    df = pd.DataFrame(df_as_l)
    model_name_str = model_name.replace("/", "_")
    fp_out = os.path.join("analysis_outputs", f"word_stats_{model_name_str}.csv")
    df.to_csv(fp_out, index=False)
    print(f"Saved word_stats to {fp_out}")


def get_word_variance(row, cr="ctfl"):
    if cr == "rspl":
        n_neg = row["n_01_rspl"]
        n_pos = row["n_10_rspl"]
        n_neu = row["n_00_rspl"] + row["n_11_rspl"]
    elif cr == "ctfl":
        n_neg = row["n_01_ctfl"]
        n_pos = row["n_10_ctfl"]
        n_neu = row["n_00_ctfl"] + row["n_11_ctfl"]
    else:
        raise ValueError(f"Invalid context ratio: {cr}")
    vals = [-1] * n_neg + [0] * n_neu + [1] * n_pos
    return np.var(vals)


def analyze_word_stats_csv(
    model_name=r"qwen/qwen3-235b-a22b", n_words=500, x_key="p_rspl_base", y_key="delta_rspl"
):
    model_name_str = model_name.replace("/", "_")
    fp_in = os.path.join("analysis_outputs", f"word_stats_{model_name_str}.csv")
    df = pd.read_csv(fp_in)
    print(f"Analyzing word_stats for {model_name_str}...")
    print(f"\tNumber of words: {len(df)}")
    print(f"\tNumber of words in ctfl: {len(df[df['n_yes_ctfl'] > 0])}")
    print(f"\tNumber of words in rspl: {len(df[df['n_yes_rspl'] > 0])}")
    print(f"\tNumber of words in ctfl: {len(df[df['n_no_ctfl'] > 0])}")
    print(f"\tNumber of words in rspl: {len(df[df['n_no_rspl'] > 0])}")

    df.sort_values(by="cnt_rspl", ascending=False, inplace=True)
    df = df.head(n_words)

    df["word_variance"] = df.apply(get_word_variance, axis=1)

    df = calc_outliers(df, x_key, y_key)

    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df[x_key], df[y_key], alpha=0.7)
    df.sort_values(by=y_key, ascending=True, inplace=True)

    # Add word labels next to each point
    for i, row in df.iterrows():
        print(f"{row["word"]}: y = {row[y_key]:.3f}")
        plt.text(row[x_key], row[y_key], row["word"], fontsize=8, alpha=0.8, ha="left", va="bottom")

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"Word Analysis: {x_key} vs {y_key} ({model_name_str}, top {n_words} words)")
    plt.grid(True, alpha=0.3)

    # Create pics directory if it doesn't exist
    os.makedirs("pics", exist_ok=True)

    # Save the plot
    filename = f"word_analysis_{model_name_str}_{x_key}_vs_{y_key}_top{n_words}.png"
    filepath = os.path.join("pics", filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved scatter plot to {filepath}")
    plt.close()


def calc_outliers(df, x_key, y_key):
    from scipy import signal

    df.sort_values(by=x_key, ascending=True, inplace=True)
    df["y_smoothed"] = signal.savgol_filter(df[y_key], window_length=len(df) // 10, polyorder=3)
    # plt.plot(df[x_key], df[y_key], label="original")
    # plt.plot(df[x_key], df["y_smoothed"], label="smoothed")
    # plt.legend()
    # plt.show()
    # quit()
    df[y_key] = df[y_key] - df["y_smoothed"]
    return df


def plot_most_negative_delta_ctfl(
    model_name=r"qwen/qwen3-235b-a22b", n_words=500, k_words=50, key="word_variance"
):
    """Plot bar chart of k_words with most negative delta_ctfl from top n_words."""
    model_name_str = model_name.replace("/", "_")
    fp_in = os.path.join("analysis_outputs", f"word_stats_{model_name_str}.csv")
    df = pd.read_csv(fp_in)
    df_top = df.head(n_words)

    df_top["word_variance"] = df_top.apply(get_word_variance, axis=1)

    df_top[f"{key}_abs"] = df_top[key].abs()
    df_top[f"{key}_rank"] = stats.rankdata(df_top[f"{key}_abs"])

    df_print = df_top.sort_values(by=f"{key}_rank", ascending=True)

    # TODO: look at variation of words' effects in different contexts somehow

    for idx, row in df_print.iterrows():
        rank = int(row[f"{key}_rank"])
        delta = row[f"{key}_abs"]
        print(f"#{rank:<3}: {row['word']}: {key} = {delta:.3f}")
    # quit()

    print(f"Plotting most negative delta_ctfl for {model_name_str}...")
    print(f"\tTotal words in dataset: {len(df)}")

    # Get top n_words by cnt_rspl (same as analyze_word_stats_csv)
    df.sort_values(by="cnt_rspl", ascending=False, inplace=True)
    df_top = df.head(n_words)
    print(f"\tUsing top {n_words} words by cnt_rspl")

    # From those, get k_words with most negative delta_ctfl
    df_negative = df_top.sort_values(by="delta_ctfl", ascending=True).head(k_words)
    print(f"\tFound {len(df_negative)} words with most negative delta_ctfl")

    # Create bar plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(df_negative)), df_negative["delta_ctfl"], alpha=0.7, color="red")

    # Set x-axis labels to be the words (rotated for readability)
    plt.xticks(range(len(df_negative)), df_negative["word"], rotation=45, ha="right")

    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, df_negative["delta_ctfl"])):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Words")
    plt.ylabel("Delta CTFL")
    plt.title(
        f"Top {k_words} Words with Most Negative Delta CTFL\n({model_name_str}, from top {n_words} by count)"
    )
    plt.grid(True, alpha=0.3, axis="y")

    # Create pics directory if it doesn't exist
    os.makedirs("pics", exist_ok=True)

    # Save the plot
    filename = f"most_negative_delta_ctfl_{model_name_str}_top{k_words}_from{n_words}.png"
    filepath = os.path.join("pics", filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved bar plot to {filepath}")
    plt.close()




if __name__ == "__main__":
    # analyze_word_stats_csv()
    analyze_word_stats_csv(y_key="word_variance", n_words=500)

    # plot_most_negative_delta_ctfl()
