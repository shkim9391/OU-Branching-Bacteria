from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEFAULT_BASE_DIR = Path(
    "/Figure_5"
)
BASE_DIR = Path(os.environ.get("OU_FIG5_DIR", DEFAULT_BASE_DIR))

LOO_CSV = BASE_DIR / "model_comparison_loo.csv"
OUT_PREFIX = BASE_DIR / "Figure5_benchmarking_ablation_real"

MODEL_ORDER = ["RW", "OU-only", "OU-Branching"]
MODEL_DISPLAY = {
    "RW": "RW",
    "OU-only": "OU-only",
    "OU-Branching": "OU-Branching",
}
COLOR_MAP = {
    "RW": "#9e9e9e",
    "OU-only": "#4c78a8",
    "OU-Branching": "#d62728",
}

PSIS_COLS = [
    "n_obs",
    "n_pareto_k_gt_0_7",
    "n_pareto_k_gt_1_0",
    "max_pareto_k",
    "psis_warning",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def add_panel_label(ax, label):
    ax.text(
        -0.12, 1.08, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def normalize_model_name(x):
    s = str(x).strip()
    low = s.lower().replace("_", "-").replace(" ", "")
    if low in ["rw", "randomwalk", "random-walk"]:
        return "RW"
    if low in ["ou", "ou-only", "ouonly"]:
        return "OU-only"
    if low in ["ou-branching", "oubranching", "full", "hybrid", "ou-branching-full"]:
        return "OU-Branching"
    return s


def ensure_optional_psis_columns(df):
    """
    Add PSIS diagnostic columns if absent.
    Values remain NaN until supplied by a real ArviZ diagnostic summary.
    """
    for col in PSIS_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_or_make_loo_table():
    if LOO_CSV.exists():
        df = pd.read_csv(LOO_CSV)

        if "model" not in df.columns or "elpd_loo" not in df.columns:
            raise ValueError(
                "model_comparison_loo.csv must contain at least model and elpd_loo columns."
            )

        df["model"] = df["model"].map(normalize_model_name)
        df = df[df["model"].isin(MODEL_ORDER)].copy()

        if df.empty:
            raise ValueError("No recognized models found in model_comparison_loo.csv.")

        if "se_elpd" not in df.columns:
            df["se_elpd"] = 0.0
        if "p_loo" not in df.columns:
            df["p_loo"] = np.nan

        df = ensure_optional_psis_columns(df)

        best_elpd = df["elpd_loo"].max()
        best_model = df.loc[df["elpd_loo"].idxmax(), "model"]

        # This is ELPD loss relative to the best model.
        # Positive values mean worse predictive performance than the best model.
        df["elpd_loss_relative_to_best"] = best_elpd - df["elpd_loo"]

        # Backward-compatible column name for plotting.
        df["delta_elpd"] = df["elpd_loss_relative_to_best"]

        if "se_delta" not in df.columns:
            df["se_delta"] = df["se_elpd"].fillna(0.0)

        df.loc[df["model"] == best_model, "se_delta"] = 0.0

    else:
        df = pd.DataFrame({
            "model": ["RW", "OU-only", "OU-Branching"],

            # These are placeholder display values if the real LOO table is absent.
            # In the manuscript table, use the real elpd_loo values when available.
            "elpd_loo": [-1005.76, -926.59, -908.54],
            "se_elpd": [14.4, 10.2, 0.0],
            "p_loo": [np.nan, np.nan, np.nan],

            # ELPD loss relative to OU-Branching.
            "elpd_loss_relative_to_best": [97.2, 18.0, 0.0],
            "delta_elpd": [97.2, 18.0, 0.0],
            "se_delta": [14.4, 10.2, 0.0],

            # Unknown until computed from ArviZ pointwise PSIS-LOO.
            "n_obs": [np.nan, np.nan, np.nan],
            "n_pareto_k_gt_0_7": [np.nan, np.nan, np.nan],
            "n_pareto_k_gt_1_0": [np.nan, np.nan, np.nan],
            "max_pareto_k": [np.nan, np.nan, np.nan],
            "psis_warning": [np.nan, np.nan, np.nan],
        })

    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)

    df.to_csv(BASE_DIR / "figure5_model_comparison_table_used.csv", index=False)
    return df


def write_psis_template_if_needed(df):
    """
    If Pareto-k diagnostics are missing, write a small template file.
    Fill this file later after computing diagnostics from ArviZ.
    """
    missing = df["n_pareto_k_gt_0_7"].isna().any()

    if missing:
        template = pd.DataFrame({
            "model": MODEL_ORDER,
            "n_obs": [np.nan, np.nan, np.nan],
            "n_pareto_k_gt_0_7": [np.nan, np.nan, np.nan],
            "n_pareto_k_gt_1_0": [np.nan, np.nan, np.nan],
            "max_pareto_k": [np.nan, np.nan, np.nan],
            "psis_warning": [np.nan, np.nan, np.nan],
        })
        template_path = BASE_DIR / "figure5_psis_diagnostics_template.csv"
        template.to_csv(template_path, index=False)
        print(f"\nPareto-k diagnostics not found. Template written to:\n{template_path}")


def make_psis_sentence(df):
    """
    Write a manuscript-ready sentence for Section 3.5.
    If diagnostics are missing, write a placeholder sentence.
    """
    diag = {}
    for model in MODEL_ORDER:
        row = df[df["model"].astype(str) == model]
        if row.empty:
            diag[model] = None
        else:
            value = row.iloc[0]["n_pareto_k_gt_0_7"]
            diag[model] = None if pd.isna(value) else int(value)

    out_path = BASE_DIR / "figure5_manuscript_psis_sentence.txt"

    if any(v is None for v in diag.values()):
        sentence = (
            "Pareto-k diagnostics should be added after computing pointwise PSIS-LOO. "
            "Suggested manuscript sentence: "
            "The number of observations with Pareto-k > 0.7 was X for the random-walk model, "
            "Y for the OU-only model, and Z for the OU-Branching model."
        )
    else:
        sentence = (
            f"The number of observations with Pareto-k > 0.7 was "
            f"{diag['RW']} for the random-walk model, "
            f"{diag['OU-only']} for the OU-only model, and "
            f"{diag['OU-Branching']} for the OU-Branching model."
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(sentence + "\n")

    print(f"\nManuscript PSIS sentence written to:\n{out_path}")
    print(sentence)


def component_matrix():
    return pd.DataFrame({
        "model": MODEL_ORDER,
        "Unconstrained diffusion": [1, 1, 1],
        "OU mean reversion": [0, 1, 1],
        "Hierarchical pooling": [0, 1, 1],
        "Branching/count layer": [0, 0, 1],
    })


def panel_a_elpd(ax, df):
    x = np.arange(len(df))
    colors = [COLOR_MAP[str(m)] for m in df["model"]]

    ax.bar(
        x,
        df["elpd_loo"],
        yerr=df["se_elpd"].fillna(0.0),
        color=colors,
        alpha=0.82,
        capsize=4,
        linewidth=0.8,
        edgecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[str(m)] for m in df["model"]])
    ax.set_ylabel("ELPD-LOO")
    ax.set_title("Approximate out-of-sample predictive fit", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    best_i = int(np.argmax(df["elpd_loo"].values))
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.06 * (y_max - y_min)

    ax.text(
        best_i,
        y_text,
        "best",
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="bold",
    )


def panel_b_delta(ax, df):
    x = np.arange(len(df))
    colors = [COLOR_MAP[str(m)] for m in df["model"]]

    ax.bar(
        x,
        df["delta_elpd"],
        yerr=df["se_delta"].fillna(0.0),
        color=colors,
        alpha=0.82,
        capsize=4,
        linewidth=0.8,
        edgecolor="black",
    )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[str(m)] for m in df["model"]])

    # Updated wording: these are positive losses relative to the best model.
    ax.set_ylabel("ELPD loss relative to OU-Branching")
    ax.set_title("Predictive loss relative to OU-Branching", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    upper = (df["delta_elpd"] + df["se_delta"].fillna(0.0)).max()
    if not np.isfinite(upper) or upper <= 0:
        upper = 1.0
    ax.set_ylim(0, upper * 1.22)

    max_delta = max(float(df["delta_elpd"].max()), 1.0)
    label_offset = max_delta * 0.045

    for i, row in df.iterrows():
        val = float(row["delta_elpd"])
        se = float(row["se_delta"]) if np.isfinite(row["se_delta"]) else 0.0
        label = "0" if abs(val) < 1e-9 else f"{val:.1f} ± {se:.1f}"

        y_text = val + se + label_offset

        ax.text(
            i,
            y_text,
            label,
            ha="center",
            va="bottom",
            fontsize=7.3,
        )


def panel_c_ablation(ax, comp):
    components = comp.columns[1:].tolist()
    models = comp["model"].tolist()

    ax.set_xlim(0, len(components))
    ax.set_ylim(0, len(models))
    ax.invert_yaxis()

    for i, model in enumerate(models):
        for j, component in enumerate(components):
            val = int(comp.loc[i, component])
            color = COLOR_MAP[model] if val else "#f0f0f0"

            rect = Rectangle(
                (j, i),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.85 if val else 1.0,
            )
            ax.add_patch(rect)

            ax.text(
                j + 0.5,
                i + 0.5,
                "yes" if val else "no",
                ha="center",
                va="center",
                fontsize=7.8,
                fontweight="bold" if val else "normal",
                color="white" if val else "#777777",
            )

    ax.set_xticks(np.arange(len(components)) + 0.5)
    ax.set_xticklabels([
        "Stochastic\ndiffusion",
        "OU mean\nreversion",
        "Hierarchical\npooling",
        "Branching/count\nlayer",
    ], fontsize=7.2)

    ax.set_yticks(np.arange(len(models)) + 0.5)
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in models])
    ax.set_title("Model ablation matrix", fontweight="bold")
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


def format_k_diag(df):
    """
    Return short Pareto-k diagnostic text for Panel D.
    """
    if df["n_pareto_k_gt_0_7"].isna().any():
        return "Pareto-k diagnostics: add counts in text after ArviZ pointwise LOO."

    pieces = []
    for model in MODEL_ORDER:
        row = df[df["model"].astype(str) == model].iloc[0]
        pieces.append(f"{MODEL_DISPLAY[model]} k>0.7: {int(row['n_pareto_k_gt_0_7'])}")

    return "; ".join(pieces)


def panel_d_interpretation(ax, df):
    ax.axis("off")

    row_rw = df[df["model"].astype(str) == "RW"].iloc[0]
    row_ou = df[df["model"].astype(str) == "OU-only"].iloc[0]

    k_text = format_k_diag(df)

    lines = [
        ("Benchmark interpretation", 0.96, 10.2, "bold"),
        ("1. RW tests unconstrained stochastic diffusion.", 0.82, 8.1, "normal"),
        ("2. OU-only adds mean reversion and hierarchical pooling.", 0.70, 8.1, "normal"),
        ("3. OU-Branching adds the discrete mutation/count observation layer.", 0.58, 8.1, "normal"),
        (
            f"ELPD loss: RW={row_rw['delta_elpd']:.1f}; "
            f"OU-only={row_ou['delta_elpd']:.1f}; OU-Branching=0.",
            0.41,
            8.1,
            "normal",
        ),
        (k_text, 0.29, 7.7, "normal"),
        ("Interpret together with posterior predictive checks.", 0.16, 8.2, "bold"),
    ]

    for text, y, fs, weight in lines:
        ax.text(
            0.02,
            y,
            text,
            ha="left",
            va="top",
            fontsize=fs,
            fontweight=weight,
            wrap=True,
        )


def main():
    print(f"Using BASE_DIR: {BASE_DIR}")
    print(f"Looking for LOO table: {LOO_CSV}")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_make_loo_table()
    comp = component_matrix()

    comp.to_csv(BASE_DIR / "figure5_model_ablation_matrix.csv", index=False)

    write_psis_template_if_needed(df)
    make_psis_sentence(df)

    print("\nModel comparison table used:")
    print(df.to_string(index=False))

    fig = plt.figure(figsize=(8.6, 6.7))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.0, 1.0],
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    panel_a_elpd(axA, df)
    panel_b_delta(axB, df)
    panel_c_ablation(axC, comp)
    panel_d_interpretation(axD, df)

    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Benchmarking and model ablation by PSIS-LOO",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    fig.savefig(f"{OUT_PREFIX}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_PREFIX}.svg", bbox_inches="tight")

    print("\nSaved:")
    print(f"{OUT_PREFIX}.png")
    print(f"{OUT_PREFIX}.pdf")
    print(f"{OUT_PREFIX}.svg")
    print(BASE_DIR / "figure5_model_comparison_table_used.csv")
    print(BASE_DIR / "figure5_model_ablation_matrix.csv")
    print(BASE_DIR / "figure5_manuscript_psis_sentence.txt")


if __name__ == "__main__":
    main()
