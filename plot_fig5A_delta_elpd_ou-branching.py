
"""
Read ArviZ compare output (model_compare_loo.csv) and generate Figure 5:
a publication-ready bar plot of ΔELPD (± dSE), ordered RW → OU → OU-Branching.

Interpretation:
- ΔELPD is computed relative to the best model (lowest 'rank' in az.compare output).
- Higher ΔELPD = worse than best model; 0 = best model.

Usage:
  python plot_fig5_delta_elpd.py \
    --csv "/path/to/model_compare_loo_ou-branching.csv" \
    --out "/path/to/Figure5_deltaELPD.pdf"

Optional:
  --title "Figure 5. PSIS-LOO benchmark"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _standardize_model_name(s: str) -> str:
    s = str(s).strip()
    s_up = s.upper()

    if s_up in {"RW", "RANDOMWALK", "RANDOM_WALK", "DIFFUSION"}:
        return "RW"
    if s_up in {"OU", "OU-ONLY", "OU_ONLY"}:
        return "OU"

    # map any branch variants to OU-Branching
    if s_up in {
        "OU-BRANCH", "OU_BRANCH", "OU–BRANCH", "OUBRANCH",
        "OU-BRANCHING", "OU_BRANCHING", "OU–BRANCHING", "OUBRANCHING",
    }:
        return "OU-Branching"

    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to model_compare_loo_ou-branching.csv from az.compare")
    ap.add_argument("--out", required=True, help="Output file path (.pdf or .png recommended)")
    ap.add_argument("--title", default="Benchmarking by PSIS-LOO (ΔELPD ± dSE)", help="Plot title")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"rank", "elpd_loo", "dse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found columns: {df.columns.tolist()}")

    # Detect model-name column: either an explicit 'model' column or unnamed index column
    model_col = None
    for c in df.columns:
        if c.lower() in {"model", "name", "models"}:
            model_col = c
            break
    if model_col is None:
        # common case: first column is unnamed index holding model names
        model_col = df.columns[0]

    df = df.copy()
    df["Model"] = df[model_col].map(_standardize_model_name)

    # Build ΔELPD relative to best model (rank==0). az.compare already uses that convention.
    # If rank column is present but best isn't 0 (rare), use minimum rank.
    best_rank = df["rank"].min()
    df_best = df.loc[df["rank"] == best_rank].copy()
    if df_best.shape[0] != 1:
        raise ValueError("Could not uniquely identify best model from 'rank' column.")
    best_elpd = float(df_best["elpd_loo"].iloc[0])

    # ΔELPD = best_elpd - model_elpd (>=0; 0 for best model)
    df["delta_elpd"] = best_elpd - df["elpd_loo"].astype(float)

    # Error bars: use dse as the SE of ΔELPD difference vs best model.
    # For the best model, set dse=0 for display.
    df["dse_plot"] = df["dse"].astype(float)
    df.loc[df["rank"] == best_rank, "dse_plot"] = 0.0

    # Order: RW → OU → OU-Branching
    order = ["RW", "OU", "OU-Branching"]
    # Keep only those if present, but don't crash if a model is missing
    present = [m for m in order if m in set(df["Model"])]
    df_plot = df.set_index("Model").reindex(present).reset_index()

    if df_plot["Model"].isna().any():
        raise ValueError("Model name mapping failed for at least one model; check model names in the CSV.")

    # Plot
    plt.figure(figsize=(6.5, 3.8))
    x = np.arange(len(df_plot))
    y = df_plot["delta_elpd"].to_numpy()
    yerr = df_plot["dse_plot"].to_numpy()

    plt.bar(x, y)
    plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=4)

    plt.xticks(x, df_plot["Model"].tolist())
    plt.ylabel("ΔELPD (relative to best; 0 = best)")
    plt.title(args.title)

    # After plotting bars + errorbars
    #for xi, yi, ei in zip(x, y, yerr):
        #if yi == 0:
           # txt = "0"
        #else:
            #txt = f"{yi:.1f}±{ei:.1f}"
       # plt.text(xi, yi + max(yerr)*0.05, txt, ha="center", va="bottom", fontsize=10)
    
    #plt.ylabel("ΔELPD_LOO (vs best; 0 = best)")
    #plt.ylim(0, max(y + yerr) * 1.15)
   
    # y = bar heights (ΔELPD), yerr = dSE
   # y = bar heights, yerr = dSE
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    
    pad = 0.03 * (y + yerr).max()
    
    # Make sure the plot is tall enough to include labels above error bars
    ymax = (y + yerr + pad).max()
    plt.ylim(0, ymax * 1.10)
    
    # Now add labels at the top of the error bars
    for xi, yi, ei in zip(x, y, yerr):
        label = "0" if yi == 0 else f"{yi:.1f}±{ei:.1f}"
        plt.text(xi, yi + ei + pad, label, ha="center", va="bottom", fontsize=10, clip_on=False)


    # Add a subtle grid for readability
    plt.grid(axis="y", alpha=0.25)

    # Tight layout for publication
    plt.tight_layout()

    # Save
    plt.savefig(out_path, dpi=300)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
