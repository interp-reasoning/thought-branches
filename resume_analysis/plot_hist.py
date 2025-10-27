import sys
import os
import pickle
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
import scipy.stats as stats


if __name__ == "__main__":
    N_CLUSTERS = 16
    INCLUDE_MIRROR = True
    JOB = "basic"

    # 54,
    RESUME_IDS = [13, 39, 0, 7, 69, 73, 83]

    df_as_l = []
    for RESUME_ID in RESUME_IDS:
        df = pd.read_csv(
            rf"bias_csvs/bias_delta_freq_{JOB}_{RESUME_ID}_{N_CLUSTERS}_{INCLUDE_MIRROR}.csv"
        )
        df["BF_WM_freq"] = np.log(df["variant_1"] / df["variant_2"])
        print(f"----- {RESUME_ID} -----")
        for idx, row in df.head(5).iterrows():
            print(
                f'{row["text"]=} {row["delta"]=:.3f} {row["BF_WM_freq"]=:.3f}'
            )
        for idx, row in df.tail(5).iterrows():
            print(
                f'{row["text"]=} {row["delta"]=:.3f} {row["BF_WM_freq"]=:.3f}'
            )

        df_as_l.append(df)
    df = pd.concat(df_as_l)
    df = df[df["BF_WM_freq"].abs() < 0.6]
    # df.to_csv(rf"bias_csvs/bias_delta_freq_{JOB}_{RESUME_IDS}_{N_CLUSTERS}_{INCLUDE_MIRROR}.csv", index=False)

    print(f"{df.columns=}")

    vabs_max = df["delta"].abs().max()

    # plt.hist(df["delta"], bins=25, range=(-vabs_max, vabs_max))
    # plt.show()
    # # plt.savefig(rf"bias_csvs/bias_delta_freq_{JOB}_{RESUME_IDS}_{N_CLUSTERS}_{INCLUDE_MIRROR}.png")
    # plt.close()
    r, p = stats.spearmanr(df["delta"], df["BF_WM_freq"])
    print(f"{r=:.3f} {p=:.3f}")

    plt.scatter(df["delta"], df["BF_WM_freq"])
    plt.show()
    # plt.savefig(rf"bias_csvs/bias_delta_freq_{JOB}_{RESUME_IDS}_{N_CLUSTERS}_{INCLUDE_MIRROR}.png")
    plt.close()
