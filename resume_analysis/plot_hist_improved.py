import os
import pickle
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set font size globally - use default DejaVu Sans
matplotlib.rc("font", size=14)

if __name__ == "__main__":
    N_CLUSTERS = 16
    INCLUDE_MIRROR = True
    JOB = "basic"

    # 54,
    RESUME_IDS = [13, 39, 0, 7, 69, 73, 83, 61]

    # Create color map from turbo colormap
    cmap = plt.cm.turbo
    colors = [cmap(i / len(RESUME_IDS)) for i in range(len(RESUME_IDS))]

    df_as_l = []
    for i, RESUME_ID in enumerate(RESUME_IDS):
        df = pd.read_csv(
            rf"bias_csvs/bias_delta_freq_{JOB}_{RESUME_ID}_{N_CLUSTERS}_{INCLUDE_MIRROR}.csv"
        )
        df["BF_WM_freq"] = np.log(df["variant_1"] / df["variant_2"])
        df["resume_id"] = RESUME_ID
        df["color_idx"] = i
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

    print(f"{df.columns=}")

    vabs_max = df["delta"].abs().max()

    # Calculate correlation
    r, p = stats.spearmanr(df["delta"], df["BF_WM_freq"])
    print(f"{r=:.3f} {p=:.3f}")

    # Create a beautiful scatterplot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each resume with different color
    for i, RESUME_ID in enumerate(RESUME_IDS):
        df_resume = df[df["resume_id"] == RESUME_ID]
        ax.scatter(
            df_resume["delta"] * 100,  # Convert to percentage
            df_resume["BF_WM_freq"],
            alpha=0.7,
            s=80,  # Larger dots
            color=colors[i],
            edgecolors="black",
            linewidth=0.5,
        )

    # Add trend line (dashed black)
    x = df["delta"].values * 100  # Convert to percentage
    y = df["BF_WM_freq"].values
    z = np.polyfit(x, y, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        p_line(x_line),
        "--",  # Dashed line
        color="black",
        linewidth=2.5,
        alpha=0.8,
    )

    # Add correlation text near the trend line (no box)
    # Position it at about 70% along the trend line
    x_text_pos = x.min() + (x.max() - x.min()) * 0.7
    y_text_pos = p_line(x_text_pos)

    # Offset the text slightly above the line
    y_offset = (df["BF_WM_freq"].max() - df["BF_WM_freq"].min()) * 0.08

    # ax.text(
    #     x_text_pos,
    #     y_text_pos + y_offset,
    #     f"$r = {r:.3f}$",
    #     fontsize=20,
    #     color="black",
    # )

    # Remove all spines (box)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add only X=0 and Y=0 lines
    ax.axhline(y=0, color="black", linewidth=1.5, alpha=0.7)
    ax.axvline(x=0, color="black", linewidth=1.5, alpha=0.7)

    # Set labels (no bold, LaTeX style for y-label, larger font)
    ax.set_xlabel(r"Effect on yes/no rate (ΔYes %)", fontsize=22)
    ax.set_ylabel(
        r"Bias in uttering: $\log(p_{\mathrm{BF}} / p_{\mathrm{WM}})$",
        fontsize=22,
    )
    # No title

    # Add subtle grid for better readability
    ax.grid(True, alpha=0.15, linestyle="--")

    # Set tick parameters with visible tick marks
    ax.tick_params(
        axis="both", which="major", labelsize=18, length=6, width=1.5
    )

    # Add some padding to the axes limits for better visualization
    x_padding = (x.max() - x.min()) * 0.05
    y_padding = (df["BF_WM_freq"].max() - df["BF_WM_freq"].min()) * 0.05
    ax.set_xlim(x.min() - x_padding, x.max() + x_padding)
    ax.set_ylim(
        df["BF_WM_freq"].min() - y_padding, df["BF_WM_freq"].max() + y_padding
    )

    # No legend

    plt.tight_layout()

    # Save plot with high DPI for better quality
    Path("plots").mkdir(exist_ok=True)
    plt.savefig(
        "plots/bias_delta_freq_improved.png", dpi=300, bbox_inches="tight"
    )

    # Show plot
    plt.show()

    plt.close()
