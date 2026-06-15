from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D


# =============================================================================
# User-configurable defaults
# =============================================================================

DEFAULT_OUTPUT_DIR = Path(
    "/Fig_S1"
)

PROJECT_ROOT = Path(
    "/Fig_S1"
)

FIG_BASENAME = "supp_figure_s1_dataset_overview"

# Offset used in the Methods section.
LOG10_OFFSET = 1e-9

# Expected genotype order.
GENOTYPE_ORDER = ["WT", "priA", "recG"]

# Consistent genotype colors across panels.
GENOTYPE_COLORS = {
    "WT": "#D62728",
    "priA": "#7F7F7F",
    "recG": "#FF7F0E",
}

# Common candidate column names.
COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "genotype": [
        "genotype", "strain", "background", "mutant", "condition", "group",
        "genetic_background", "ecoli_strain", "E_coli_strain"
    ],
    "time": [
        "time", "timepoint", "time_point", "t", "day", "days", "generation",
        "generations", "passage", "passage_number"
    ],
    "replicate": [
        "replicate", "rep", "replicate_id", "sample", "sample_id", "id",
        "well", "clone", "biological_replicate"
    ],
    "mutation_frequency": [
        "mutation_frequency", "mut_frequency", "mutation_freq", "mut_freq",
        "frequency", "freq", "mutation_rate", "mut_rate", "observed_frequency",
        "x", "value"
    ],
    "count": [
        "count", "counts", "mutation_count", "mut_count", "K", "k",
        "num_mutants", "n_mutants", "events", "mutation_events"
    ],
    "trials": [
        "trials", "trial", "N", "n", "total", "total_count", "total_cells",
        "colonies", "cfu", "screened", "assayed", "denominator"
    ],
}


# =============================================================================
# Helpers
# =============================================================================

def normalize_name(name: str) -> str:
    """Normalize a column name for lenient matching."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(
    df: pd.DataFrame,
    role: str,
    exclude: Optional[Iterable[str]] = None
) -> Optional[str]:
    """
    Find a dataframe column corresponding to a semantic role.

    This version is intentionally stricter for count/trial columns to avoid
    false matches such as assigning the genotype column "background" to "N"
    or "n" simply because it contains the letter n.

    Parameters
    ----------
    df
        Input dataframe.
    role
        One of COLUMN_CANDIDATES keys.
    exclude
        Optional columns that should not be considered.

    Returns
    -------
    str or None
        Original column name if found.
    """
    exclude_norm = {normalize_name(c) for c in (exclude or []) if c is not None}
    norm_to_original = {
        normalize_name(c): c
        for c in df.columns
        if normalize_name(c) not in exclude_norm
    }
    candidates = [normalize_name(c) for c in COLUMN_CANDIDATES[role]]

    # Exact normalized match first.
    for cand in candidates:
        if cand in norm_to_original:
            return norm_to_original[cand]

    # For count/trial columns, avoid permissive single-letter substring matches.
    if role in {"count", "trials"}:
        strong_terms = {
            "count": [
                "mutation_count", "mut_count", "num_mutants", "n_mutants",
                "events", "mutation_events", "counts"
            ],
            "trials": [
                "trials", "total_count", "total_cells", "colonies", "cfu",
                "screened", "assayed", "denominator"
            ],
        }[role]
        strong_terms = [normalize_name(s) for s in strong_terms]
        for norm_col, original in norm_to_original.items():
            for term in strong_terms:
                if term and (term in norm_col or norm_col in term):
                    return original
        return None

    # Soft contains match for non-count roles only.
    for norm_col, original in norm_to_original.items():
        for cand in candidates:
            if len(cand) <= 1:
                continue
            if cand and (cand in norm_col or norm_col in cand):
                return original

    return None


def coerce_time(values: pd.Series) -> pd.Series:
    """
    Convert time column to sortable numeric or ordered string values.

    If values are numeric, return numeric.
    If values contain digits, extract the first number.
    Otherwise return categorical codes preserving sorted unique labels.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.75:
        return numeric

    extracted = values.astype(str).str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    extracted_num = pd.to_numeric(extracted, errors="coerce")
    if extracted_num.notna().mean() >= 0.75:
        return extracted_num

    # Fallback: preserve lexical order as integer codes.
    labels = sorted(values.astype(str).dropna().unique())
    mapping = {lab: i + 1 for i, lab in enumerate(labels)}
    return values.astype(str).map(mapping)


def standardize_genotype_label(x: object) -> str:
    """Map common genotype labels to WT, priA, recG when possible."""
    s = str(x).strip()
    low = s.lower()

    if low in {"wt", "wildtype", "wild_type", "wild-type", "wild type", "control"}:
        return "WT"
    if "pria" in low or "pri a" in low or "pri-a" in low:
        return "priA"
    if "recg" in low or "rec g" in low or "rec-g" in low:
        return "recG"

    return s


def ordered_genotypes(values: Iterable[str]) -> List[str]:
    """Return genotype order with WT, priA, recG first if present."""
    unique = list(pd.Series(list(values)).dropna().astype(str).unique())
    out = [g for g in GENOTYPE_ORDER if g in unique]
    out += sorted([g for g in unique if g not in out])
    return out


def genotype_color(genotype: str) -> str:
    """Return a stable color for a genotype."""
    return GENOTYPE_COLORS.get(str(genotype), "0.35")


def numeric_column_is_usable(df: pd.DataFrame, col: Optional[str], min_valid_fraction: float = 0.5) -> bool:
    """
    Return True if a candidate count/trial column is numerically usable.

    This protects against false detections such as assigning a text genotype
    column to count or trials.
    """
    if col is None or col not in df.columns:
        return False
    numeric = pd.to_numeric(df[col], errors="coerce")
    return bool(numeric.notna().mean() >= min_valid_fraction)


def find_candidate_csvs(search_dirs: List[Path]) -> List[Path]:
    """Find plausible mutation-frequency CSV files."""
    patterns = [
        "*mutation*frequency*.csv",
        "*mutation*freq*.csv",
        "*mut*frequency*.csv",
        "*mut*freq*.csv",
        "*processed*.csv",
        "*raw*.csv",
        "*.csv",
    ]

    seen = set()
    candidates: List[Path] = []

    for base in search_dirs:
        if not base.exists():
            continue
        for pattern in patterns:
            for p in base.rglob(pattern):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    candidates.append(p)

    # Prefer filenames that sound directly relevant.
    def score_path(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        score = 0
        for token in ["mutation", "mut", "frequency", "freq", "processed", "dataset", "data"]:
            if token in name:
                score += 1
        # smaller file depth preferred after score
        return (-score, len(p.parts))

    candidates = sorted(candidates, key=score_path)
    return candidates


def read_csv_lenient(path: Path) -> pd.DataFrame:
    """Read CSV/TSV with a few common separators."""
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def detect_input_file(explicit_input: Optional[Path], output_dir: Path) -> Path:
    """
    Select input file.

    Search order:
    1. Explicit --input
    2. CSV files in output directory
    3. Nearby project directories
    4. Entire Bioinformatics_Advances subtree
    """
    if explicit_input is not None:
        if not explicit_input.exists():
            raise FileNotFoundError(f"Explicit input file does not exist: {explicit_input}")
        return explicit_input

    search_dirs = [
        output_dir,
        output_dir.parent,
        output_dir.parent.parent,
        PROJECT_ROOT / "Bioinformatics_Advances",
        PROJECT_ROOT,
    ]

    candidates = find_candidate_csvs(search_dirs)

    usable: List[Tuple[Path, Dict[str, Optional[str]]]] = []

    for p in candidates:
        try:
            df = read_csv_lenient(p)
        except Exception:
            continue

        if df.empty or len(df.columns) < 3:
            continue

        genotype_col = find_column(df, "genotype")
        time_col = find_column(df, "time")
        mutation_frequency_col = find_column(df, "mutation_frequency")
        reserved_cols = [genotype_col, time_col, mutation_frequency_col]

        colmap = {
            "genotype": genotype_col,
            "time": time_col,
            "mutation_frequency": mutation_frequency_col,
            "count": find_column(df, "count", exclude=reserved_cols),
            "trials": find_column(df, "trials", exclude=reserved_cols),
        }

        # Runtime sanity check for optional count/trial columns.
        # Text columns such as "background" must never be retained as K/N.
        if not numeric_column_is_usable(df, colmap["count"]):
            colmap["count"] = None
        if not numeric_column_is_usable(df, colmap["trials"]):
            colmap["trials"] = None

        # Minimum needed for S1: genotype, time, mutation frequency.
        if colmap["genotype"] and colmap["time"] and colmap["mutation_frequency"]:
            usable.append((p, colmap))

    if not usable:
        msg = [
            "Could not auto-detect a mutation-frequency CSV.",
            "",
            "Please run with an explicit input file, for example:",
            "    python generate_supp_figure_s1_dataset_overview.py --input /path/to/data.csv",
            "",
            "Required columns, using any common naming variant:",
            "    genotype/strain/background",
            "    time/timepoint/day/generation",
            "    mutation_frequency/mut_freq/frequency",
            "",
            "Optional columns:",
            "    replicate/sample_id",
            "    count/mutation_count/K",
            "    trials/N/total/colonies",
        ]
        raise FileNotFoundError("\n".join(msg))

    chosen = usable[0][0]
    print(f"[INFO] Auto-detected input dataset: {chosen}")
    return chosen


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Standardize input dataset to canonical columns."""
    genotype_col = find_column(df, "genotype")
    time_col = find_column(df, "time")
    replicate_col = find_column(df, "replicate")
    mutation_frequency_col = find_column(df, "mutation_frequency")

    reserved_cols = [
        genotype_col,
        time_col,
        replicate_col,
        mutation_frequency_col,
    ]

    colmap: Dict[str, Optional[str]] = {
        "genotype": genotype_col,
        "time": time_col,
        "replicate": replicate_col,
        "mutation_frequency": mutation_frequency_col,
        "count": find_column(df, "count", exclude=reserved_cols),
        "trials": find_column(df, "trials", exclude=reserved_cols),
    }

    # Runtime sanity check for optional count/trial columns.
    # Text columns such as "background" must never be retained as K/N.
    if not numeric_column_is_usable(df, colmap["count"]):
        colmap["count"] = None
    if not numeric_column_is_usable(df, colmap["trials"]):
        colmap["trials"] = None

    missing = [k for k in ["genotype", "time", "mutation_frequency"] if colmap[k] is None]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["genotype"] = df[colmap["genotype"]].map(standardize_genotype_label)
    out["time_raw"] = df[colmap["time"]]
    out["time"] = coerce_time(df[colmap["time"]])

    if colmap["replicate"] is not None:
        out["replicate"] = df[colmap["replicate"]].astype(str)
    else:
        # If no replicate ID exists, make deterministic row-level IDs.
        out["replicate"] = (
            out["genotype"].astype(str)
            + "_rep"
            + (df.groupby([colmap["genotype"], colmap["time"]]).cumcount() + 1).astype(str)
        )

    out["mutation_frequency"] = pd.to_numeric(df[colmap["mutation_frequency"]], errors="coerce")

    if colmap["count"] is not None:
        out["count"] = pd.to_numeric(df[colmap["count"]], errors="coerce")
    else:
        out["count"] = np.nan

    if colmap["trials"] is not None:
        out["trials"] = pd.to_numeric(df[colmap["trials"]], errors="coerce")
    else:
        out["trials"] = np.nan

    # Drop unusable rows.
    before = len(out)
    out = out.dropna(subset=["genotype", "time", "mutation_frequency"]).copy()
    after = len(out)

    if after < before:
        print(f"[INFO] Dropped {before - after} rows with missing genotype/time/frequency.")

    # Enforce nonnegative frequencies.
    bad = out["mutation_frequency"] < 0
    if bad.any():
        print(f"[WARN] Removing {bad.sum()} rows with negative mutation frequency.")
        out = out.loc[~bad].copy()

    out["log10_mutation_frequency"] = np.log10(out["mutation_frequency"] + LOG10_OFFSET)

    # Stable sorting.
    genotype_order = ordered_genotypes(out["genotype"])
    out["genotype"] = pd.Categorical(out["genotype"], categories=genotype_order, ordered=True)
    out = out.sort_values(["genotype", "replicate", "time"]).reset_index(drop=True)

    return out, colmap


def add_panel_label(ax, label: str) -> None:
    """Add bold panel label."""
    ax.text(
        -0.08, 1.08, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )


def style_axis(ax) -> None:
    """Apply simple consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def safe_sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


# =============================================================================
# Plotting functions
# =============================================================================

def plot_panel_a(ax, data: pd.DataFrame) -> None:
    """Dataset schematic."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    add_panel_label(ax, "A")

    n_genotypes = data["genotype"].nunique()
    n_time = data["time"].nunique()
    n_reps = data[["genotype", "replicate"]].drop_duplicates().shape[0]
    n_obs = len(data)

    title = "Dataset structure"
    ax.text(
        0.5, 0.94, title,
        ha="center", va="center",
        fontsize=11, fontweight="bold"
    )

    # Put n values in the title line to avoid overlap inside boxes.
    boxes = [
        (0.08, 0.62, 0.23, 0.18, f"Genotypes (n={n_genotypes})", "WT, priA, recG"),
        (0.385, 0.62, 0.23, 0.18, f"Time points (n={n_time})", r"$\Delta t = 1$"),
        (0.69, 0.62, 0.23, 0.18, f"Replicates (n={n_reps})", "genotype-replicates"),
        (0.23, 0.25, 0.23, 0.18, f"Observations (n={n_obs})", "frequency values"),
        (0.54, 0.25, 0.23, 0.18, "Transform", r"$\log_{10}(x + 10^{-9})$"),
    ]

    for x, y, w, h, head, body in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.0,
            edgecolor="0.25",
            facecolor="0.96"
        )
        ax.add_patch(patch)

        ax.text(
            x + w / 2, y + h * 0.66,
            head,
            ha="center", va="center",
            fontsize=8.0,
            fontweight="bold"
        )

        ax.text(
            x + w / 2, y + h * 0.34,
            body,
            ha="center", va="center",
            fontsize=7.6
        )

    arrows = [
        ((0.31, 0.71), (0.385, 0.71)),
        ((0.615, 0.71), (0.69, 0.71)),
        ((0.78, 0.62), (0.63, 0.43)),
        ((0.43, 0.62), (0.35, 0.43)),
        ((0.46, 0.34), (0.54, 0.34)),
    ]

    for start, end in arrows:
        arr = FancyArrowPatch(
            start, end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="0.35"
        )
        ax.add_patch(arr)

    ax.text(
        0.5, 0.06,
        "Complete longitudinal dataset used as input for OU–Branching inference",
        ha="center", va="center",
        fontsize=8,
        color="0.25"
    )

def plot_panel_b(ax, data: pd.DataFrame) -> None:
    """Raw mutation-frequency trajectories."""
    add_panel_label(ax, "B")
    genotypes = ordered_genotypes(data["genotype"].astype(str))

    for genotype in genotypes:
        color = genotype_color(genotype)
        sub = data[data["genotype"].astype(str) == genotype]

        for _, rep_df in sub.groupby("replicate", observed=True):
            # A log y-axis cannot display exact zeros. For display only, zeros are
            # placed at the numerical floor used in the Methods transformation.
            y_plot = rep_df["mutation_frequency"].clip(lower=LOG10_OFFSET)
            ax.plot(
                rep_df["time"],
                y_plot,
                marker="o",
                linewidth=0.8,
                alpha=0.32,
                markersize=2.5,
                color=color
            )

        mean_df = (
            sub.groupby("time", observed=True)["mutation_frequency"]
            .mean()
            .reset_index(name="mean")
        )
        ax.plot(
            mean_df["time"],
            mean_df["mean"].clip(lower=LOG10_OFFSET),
            marker="o",
            linewidth=2.1,
            markersize=4.0,
            color=color,
            label=genotype
        )

    ax.set_title("Raw mutation-frequency trajectories", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time point", fontsize=9)
    ax.set_ylabel("Mutation frequency", fontsize=9)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="best")
    style_axis(ax)

def plot_panel_c(ax, data: pd.DataFrame) -> None:
    """Log10-transformed trajectories."""
    add_panel_label(ax, "C")
    genotypes = ordered_genotypes(data["genotype"].astype(str))

    for genotype in genotypes:
        color = genotype_color(genotype)
        sub = data[data["genotype"].astype(str) == genotype]

        for _, rep_df in sub.groupby("replicate", observed=True):
            ax.plot(
                rep_df["time"],
                rep_df["log10_mutation_frequency"],
                marker="o",
                linewidth=0.8,
                alpha=0.35,
                markersize=2.5,
                color=color
            )

        mean_df = (
            sub.groupby("time", observed=True)["log10_mutation_frequency"]
            .agg(["mean", safe_sem])
            .reset_index()
        )
        sem_col = "safe_sem" if "safe_sem" in mean_df.columns else "sem"
        ax.errorbar(
            mean_df["time"],
            mean_df["mean"],
            yerr=mean_df[sem_col],
            marker="o",
            linewidth=2.1,
            markersize=4.0,
            capsize=2.5,
            color=color,
            label=genotype
        )

    ax.set_title(r"Processed $\log_{10}$ mutation frequencies", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time point", fontsize=9)
    ax.set_ylabel(r"$\log_{10}$(mutation frequency + $10^{-9}$)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="best")
    style_axis(ax)

def plot_panel_d(ax, data: pd.DataFrame) -> None:
    """Replicate-level heatmap."""
    add_panel_label(ax, "D")

    heat_df = data.copy()
    heat_df["row_id"] = heat_df["genotype"].astype(str) + " | " + heat_df["replicate"].astype(str)

    pivot = heat_df.pivot_table(
        index="row_id",
        columns="time",
        values="log10_mutation_frequency",
        aggfunc="mean",
        observed=True
    )

    # Order rows by genotype then replicate.
    row_order = (
        heat_df[["row_id", "genotype", "replicate"]]
        .drop_duplicates()
        .sort_values(["genotype", "replicate"])
        ["row_id"]
        .tolist()
    )
    pivot = pivot.reindex(row_order)

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest"
    )

    ax.set_title("Replicate-level processed-data overview", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time point", fontsize=9)
    ax.set_ylabel("Genotype | replicate", fontsize=9)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=0, fontsize=7)

    if pivot.shape[0] <= 18:
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=6.5)
    else:
        ax.set_yticks([])

    # Mark missing values with a light rectangle overlay.
    missing = np.isnan(pivot.values)
    for i in range(missing.shape[0]):
        for j in range(missing.shape[1]):
            if missing[i, j]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1, 1,
                    facecolor="white",
                    edgecolor="0.75",
                    hatch="///",
                    linewidth=0.2
                )
                ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(r"$\log_{10}$ frequency", fontsize=8)

    style_axis(ax)


def plot_panel_e(ax, data: pd.DataFrame) -> None:
    """Distribution by genotype."""
    add_panel_label(ax, "E")
    genotypes = ordered_genotypes(data["genotype"].astype(str))
    groups = [
        data.loc[data["genotype"].astype(str) == g, "log10_mutation_frequency"].dropna().values
        for g in genotypes
    ]
    positions = np.arange(1, len(genotypes) + 1)

    parts = ax.violinplot(
        groups,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for body, genotype in zip(parts["bodies"], genotypes):
        body.set_facecolor(genotype_color(genotype))
        body.set_alpha(0.22)
        body.set_edgecolor("0.25")
        body.set_linewidth(0.6)

    ax.boxplot(
        groups,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.3, "color": "black"},
        boxprops={"linewidth": 0.9, "facecolor": "white", "alpha": 0.85},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8}
    )

    # Jittered points.
    rng = np.random.default_rng(7)
    for pos, vals, genotype in zip(positions, groups, genotypes):
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=13,
            alpha=0.65,
            linewidth=0.2,
            edgecolor="0.25",
            color=genotype_color(genotype)
        )

    ax.set_title("Genotype-stratified processed values", fontsize=10, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(genotypes, fontsize=8)
    ax.set_ylabel(r"$\log_{10}$(mutation frequency + $10^{-9}$)", fontsize=9)
    style_axis(ax)

def plot_panel_f(ax, data: pd.DataFrame) -> None:
    """Count-layer overview or fallback observation coverage matrix."""
    add_panel_label(ax, "F")
    has_counts = data["count"].notna().any() and data["trials"].notna().any()

    if has_counts:
        valid = data.dropna(subset=["count", "trials"]).copy()
        valid = valid[valid["trials"] > 0].copy()
        valid["observed_probability"] = valid["count"] / valid["trials"]

        genotypes = ordered_genotypes(valid["genotype"].astype(str))
        for genotype in genotypes:
            color = genotype_color(genotype)
            sub = valid[valid["genotype"].astype(str) == genotype]
            summary = (
                sub.groupby("time", observed=True)
                .agg(
                    mean_count=("count", "mean"),
                    sem_count=("count", safe_sem),
                    mean_trials=("trials", "mean"),
                    mean_prob=("observed_probability", "mean"),
                )
                .reset_index()
            )
            ax.errorbar(
                summary["time"],
                summary["mean_count"],
                yerr=summary["sem_count"],
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                capsize=2.5,
                color=color,
                label=genotype
            )

        ax.set_title("Observed count-layer summary", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time point", fontsize=9)
        ax.set_ylabel("Observed mutation-event count K", fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="best")
        style_axis(ax)

        n_valid = len(valid)
        ax.text(
            0.02, 0.96,
            f"{n_valid} rows with K and N",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.75", alpha=0.85)
        )

    else:
        # Fallback: table-style observation coverage matrix by genotype and time.
        coverage = (
            data.groupby(["genotype", "time"], observed=True)
            .size()
            .reset_index(name="n_observations")
        )

        genotypes = ordered_genotypes(coverage["genotype"].astype(str))
        times = sorted(coverage["time"].dropna().unique())

        pivot = coverage.pivot_table(
            index="genotype",
            columns="time",
            values="n_observations",
            aggfunc="sum",
            fill_value=0,
            observed=True
        )
        pivot = pivot.reindex(index=genotypes, columns=times, fill_value=0)

        ax.set_title("Observation coverage by genotype and time", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time point", fontsize=9)
        ax.set_ylabel("Genotype", fontsize=9)

        ax.set_xlim(-0.5, len(times) - 0.5)
        ax.set_ylim(len(genotypes) - 0.5, -0.5)

        # Draw a calm table-like matrix rather than a saturated heatmap.
        max_val = max(float(np.nanmax(pivot.values)), 1.0)
        for i, genotype in enumerate(genotypes):
            for j, time in enumerate(times):
                val = int(pivot.loc[genotype, time])
                intensity = 0.96 - 0.16 * (val / max_val)
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=str(max(0.78, intensity)),
                    edgecolor="0.72",
                    linewidth=0.8
                )
                ax.add_patch(rect)
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8.5, color="0.10")

        ax.set_xticks(np.arange(len(times)))
        ax.set_xticklabels([str(t) for t in times], fontsize=7)
        ax.set_yticks(np.arange(len(genotypes)))
        ax.set_yticklabels(genotypes, fontsize=8)

        ax.text(
            0.5, -0.20,
            "Values indicate the number of replicate observations available at each genotype–time combination.",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color="0.25"
        )

        style_axis(ax)


# =============================================================================
# Main figure generation
# =============================================================================

def make_figure(data: pd.DataFrame, output_dir: Path, dpi: int = 600) -> Tuple[Path, Path]:
    """Create and save Supplementary Fig. S1."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    fig = plt.figure(figsize=(12.8, 8.8), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.92, 1.0, 1.0],
        wspace=0.28,
        hspace=0.42
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    plot_panel_a(ax_a, data)
    plot_panel_b(ax_b, data)
    plot_panel_c(ax_c, data)
    plot_panel_d(ax_d, data)
    plot_panel_e(ax_e, data)
    plot_panel_f(ax_f, data)

    fig.suptitle(
        "Supplementary Fig. S1. Full raw and processed mutation-frequency dataset overview",
        fontsize=13,
        fontweight="bold",
        y=0.985
    )

    png_path = output_dir / f"{FIG_BASENAME}.png"
    pdf_path = output_dir / f"{FIG_BASENAME}.pdf"

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def save_processed_snapshot(data: pd.DataFrame, output_dir: Path) -> Path:
    """Save the standardized processed dataset used for plotting."""
    path = output_dir / "supp_figure_s1_standardized_input_snapshot.csv"
    data.to_csv(path, index=False)
    return path


def print_dataset_summary(data: pd.DataFrame, colmap: Dict[str, Optional[str]], input_file: Path) -> None:
    """Print useful run summary."""
    print("\n========== Supplementary Fig. S1 dataset summary ==========")
    print(f"Input file: {input_file}")
    print("Detected columns:")
    for role, col in colmap.items():
        print(f"  {role:20s}: {col}")

    print("\nStandardized dataset:")
    print(f"  rows                  : {len(data)}")
    print(f"  genotypes             : {list(ordered_genotypes(data['genotype'].astype(str)))}")
    print(f"  number of time points : {data['time'].nunique()}")
    print(f"  time values           : {list(sorted(data['time'].dropna().unique()))}")
    print(f"  genotype-replicates   : {data[['genotype', 'replicate']].drop_duplicates().shape[0]}")
    print(f"  frequency min/max     : {data['mutation_frequency'].min():.4g} / {data['mutation_frequency'].max():.4g}")
    print(f"  log10 min/max         : {data['log10_mutation_frequency'].min():.4g} / {data['log10_mutation_frequency'].max():.4g}")

    if data["count"].notna().any() and data["trials"].notna().any():
        valid = data.dropna(subset=["count", "trials"])
        print(f"  count-layer rows      : {len(valid)}")
        print(f"  count min/max         : {valid['count'].min():.4g} / {valid['count'].max():.4g}")
        print(f"  trials min/max        : {valid['trials'].min():.4g} / {valid['trials'].max():.4g}")
    else:
        print("  count-layer rows      : not detected")
    print("==========================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Fig. S1 dataset overview."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to mutation-frequency CSV/TSV. If omitted, the script auto-detects a suitable file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PNG/PDF outputs will be saved."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    explicit_input = Path(args.input).expanduser().resolve() if args.input else None

    input_file = detect_input_file(explicit_input, output_dir)
    raw = read_csv_lenient(input_file)
    data, colmap = prepare_dataset(raw)

    print_dataset_summary(data, colmap, input_file)

    snapshot_path = save_processed_snapshot(data, output_dir)
    png_path, pdf_path = make_figure(data, output_dir, dpi=args.dpi)

    print("[DONE] Saved Supplementary Fig. S1 outputs:")
    print(f"  PNG : {png_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  standardized input snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
