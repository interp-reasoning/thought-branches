import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np


def format_text(text, max_length=65):
    """Escape LaTeX special characters and handle long text"""
    # Escape special LaTeX characters
    text = text.replace("\\", "\\textbackslash{}")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("$", "\\$")
    text = text.replace("#", "\\#")
    text = text.replace("_", "\\_")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")
    text = text.replace("'", "'")  # Handle apostrophes

    # Handle long text - split into lines if needed
    if len(text) > max_length:
        # Find a good break point (space, comma) near max_length
        break_point = max_length
        for i in range(max_length, max(0, max_length - 20), -1):
            if text[i] in [' ', ',', '.']:
                break_point = i
                break

        first_line = text[:break_point].rstrip()
        remainder = text[break_point:].lstrip()

        # Check if we need to truncate the remainder
        if len(remainder) > 30:
            remainder = remainder[:27] + "\\ldots$^b$"

        return first_line, remainder

    return text, None


def create_latex_table(df, resume_id, n_rows=5):
    """Create a LaTeX table for a single resume with professional formatting"""

    latex = []
    has_truncated = False  # Track if any text was truncated

    latex.append("\\begin{table}[ht]")
    latex.append("\\centering")
    latex.append(
        f"\\caption{{Resume {resume_id}: Most positive and negative bias effects on interview recommendations}}"
    )
    latex.append(f"\\label{{tab:resume_{resume_id}}}")
    latex.append("\\begin{tabular}{@{}p{9.5cm}cc@{}}")
    latex.append("\\toprule")
    latex.append(
        "\\textbf{Text Snippet} & \\textbf{$\\boldsymbol{\\Delta}$Yes (\\%)} & \\textbf{Log Ratio}$^{\\mathbf{a}}$ \\\\"
    )
    latex.append("\\midrule")
    latex.append("\\multicolumn{3}{@{}l}{\\textsc{Most Negative Effects}} \\\\")
    latex.append("\\midrule")

    # Add head rows (most negative effects)
    for idx, row in df.head(n_rows).iterrows():
        text, remainder = format_text(row["text"])
        delta_pct = row["delta"] * 100
        bf_wm = row["BF_WM_freq"]

        # Format delta with sign
        delta_str = f"${delta_pct:+.1f}$"

        # Format log ratio with sign
        bf_wm_str = f"${bf_wm:+.3f}$"

        if remainder:
            has_truncated = True if "\\ldots" in remainder else has_truncated
            # First line of text
            latex.append(f"{text} & {delta_str} & {bf_wm_str} \\\\")
            # Second line with indent
            latex.append(f"\\quad {remainder} & & \\\\")
        else:
            latex.append(f"{text} & {delta_str} & {bf_wm_str} \\\\")

    latex.append("\\midrule")
    latex.append("\\multicolumn{3}{@{}l}{\\textsc{Most Positive Effects}} \\\\")
    latex.append("\\midrule")

    # Add tail rows (most positive effects)
    for idx, row in df.tail(n_rows).iterrows():
        text, remainder = format_text(row["text"])
        delta_pct = row["delta"] * 100
        bf_wm = row["BF_WM_freq"]

        # Format delta with sign
        delta_str = f"${delta_pct:+.1f}$"

        # Format log ratio with sign
        bf_wm_str = f"${bf_wm:+.3f}$"

        if remainder:
            has_truncated = True if "\\ldots" in remainder else has_truncated
            # First line of text
            latex.append(f"{text} & {delta_str} & {bf_wm_str} \\\\")
            # Second line with indent
            latex.append(f"\\quad {remainder} & & \\\\")
        else:
            latex.append(f"{text} & {delta_str} & {bf_wm_str} \\\\")

    latex.append("\\bottomrule")

    # Add footnotes
    latex.append("\\multicolumn{3}{l}{\\footnotesize $^a$Log ratio: $\\log(p_{\\text{BF}}/p_{\\text{WM}})$} \\\\")
    if has_truncated:
        latex.append("\\multicolumn{3}{l}{\\footnotesize $^b$Text truncated for space} \\\\")

    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


if __name__ == "__main__":
    N_CLUSTERS = 16
    INCLUDE_MIRROR = True
    JOB = "basic"

    # RESUME_IDS = [13, 39, 0, 7, 69, 73, 83]
    RESUME_IDS = [13, 39, 0, 7, 69, 73, 83, 61]

    # Create output directory
    output_dir = Path("plots/latex_tables")
    output_dir.mkdir(exist_ok=True, parents=True)

    all_tables = []

    for RESUME_ID in RESUME_IDS:
        # Read the data
        df = pd.read_csv(
            rf"bias_csvs/bias_delta_freq_{JOB}_{RESUME_ID}_{N_CLUSTERS}_{INCLUDE_MIRROR}.csv"
        )
        df["BF_WM_freq"] = np.log(df["variant_1"] / df["variant_2"])

        # Filter out extreme values
        df = df[df["BF_WM_freq"].abs() < 0.6]

        # Sort by delta to get most negative and positive
        df = df.sort_values("delta")

        # Create LaTeX table
        latex_table = create_latex_table(df, RESUME_ID)

        # Save individual table
        individual_file = output_dir / f"resume_{RESUME_ID}_table.tex"
        with open(individual_file, "w") as f:
            f.write(latex_table)

        print(f"Created table for Resume {RESUME_ID}")

        # Add to combined list
        all_tables.append(latex_table)

    # Create combined document
    combined_doc = []
    combined_doc.append("% LaTeX tables for all resumes")
    combined_doc.append("% Copy these into your LaTeX document")
    combined_doc.append("")

    for table in all_tables:
        combined_doc.append(table)
        combined_doc.append("")
        combined_doc.append("\\clearpage")
        combined_doc.append("")

    # Save combined document
    combined_file = output_dir / "all_tables_combined.tex"
    with open(combined_file, "w") as f:
        f.write("\n".join(combined_doc))

    print(f"\nAll tables saved to {output_dir}/")
    print(f"Combined document: {combined_file}")

    # Also create a minimal complete LaTeX document for testing
    complete_doc = []
    complete_doc.append("\\documentclass{article}")
    complete_doc.append("\\usepackage{booktabs}")  # For professional tables
    complete_doc.append("\\usepackage{array}")      # For column specifications
    complete_doc.append("\\usepackage{amsmath}")    # For boldsymbol
    complete_doc.append("\\usepackage[margin=1in]{geometry}") # Better margins
    complete_doc.append("\\begin{document}")
    complete_doc.append("")

    for table in all_tables:
        complete_doc.append(table)
        complete_doc.append("")
        complete_doc.append("\\clearpage")
        complete_doc.append("")

    complete_doc.append("\\end{document}")

    # Save complete document
    complete_file = output_dir / "complete_document.tex"
    with open(complete_file, "w") as f:
        f.write("\n".join(complete_doc))

    print(f"Complete LaTeX document: {complete_file}")
