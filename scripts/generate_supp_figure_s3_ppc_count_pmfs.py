from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import arviz as az
except Exception as exc:
    az = None
    ARVIZ_IMPORT_ERROR = exc
else:
    ARVIZ_IMPORT_ERROR = None


# =============================================================================
# Defaults
# =============================================================================

PROJECT_ROOT = Path("/Fig_S3")

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "Fig_S3"
)

DEFAULT_IDATA = PROJECT_ROOT / "Figure_2" / "trace_core.nc"
DEFAULT_DATA = PROJECT_ROOT / "Figure_2" / "mut_freq_data.csv"

FIG_BASENAME = "supp_figure_s3_ppc_count_pmfs"
GENOTYPE_ORDER = ["WT", "priA", "recG"]
GENOTYPE_COLORS = {"WT": "#D62728", "priA": "#7F7F7F", "recG": "#FF7F0E"}
PPC_VAR_CANDIDATES = ["Y_transition", "y_transition", "Y_pred", "y_pred", "Y_ppc", "obs"]

COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "genotype": ["genotype", "strain", "background", "mutant", "condition", "group"],
    "time": ["time", "timepoint", "time_point", "t", "day", "generation", "passage"],
    "replicate": ["replicate", "rep", "replicate_id", "sample", "sample_id", "well", "clone"],
    "mutation_frequency": ["mutation_frequency", "mut_frequency", "mutation_freq", "mut_freq", "frequency", "freq", "mutation_rate", "mut_rate", "x", "value"],
    "count": ["count", "counts", "mutation_count", "mut_count", "K", "k", "num_mutants", "n_mutants", "events", "mutation_events"],
    "trials": ["trials", "trial", "N", "n", "total", "total_count", "total_cells", "colonies", "cfu", "screened", "assayed", "denominator"],
}


# =============================================================================
# General helpers
# =============================================================================

def require_arviz() -> None:
    if az is None:
        raise ImportError(
            "ArViZ is required but could not be imported.\n"
            f"Original import error: {ARVIZ_IMPORT_ERROR}\n\n"
            "Install it in the active environment, for example:\n"
            "    conda install -c conda-forge arviz\n"
            "or:\n"
            "    pip install arviz"
        )


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")


def standardize_genotype_label(x: object) -> str:
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
    unique = list(pd.Series(list(values)).dropna().astype(str).unique())
    out = [g for g in GENOTYPE_ORDER if g in unique]
    out += sorted([g for g in unique if g not in out])
    return out


def genotype_color(genotype: str) -> str:
    return GENOTYPE_COLORS.get(str(genotype), "0.35")


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.10, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def read_csv_lenient(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def find_column(df: pd.DataFrame, role: str, exclude: Optional[Iterable[str]] = None) -> Optional[str]:
    exclude_norm = {normalize_name(c) for c in (exclude or []) if c is not None}
    norm_to_original = {normalize_name(c): c for c in df.columns if normalize_name(c) not in exclude_norm}
    candidates = [normalize_name(c) for c in COLUMN_CANDIDATES[role]]

    for cand in candidates:
        if cand in norm_to_original:
            return norm_to_original[cand]

    if role in {"count", "trials"}:
        strong_terms = {
            "count": ["mutation_count", "mut_count", "num_mutants", "n_mutants", "events", "mutation_events", "counts"],
            "trials": ["trials", "total_count", "total_cells", "colonies", "cfu", "screened", "assayed", "denominator"],
        }[role]
        strong_terms = [normalize_name(s) for s in strong_terms]
        for norm_col, original in norm_to_original.items():
            for term in strong_terms:
                if term and (term in norm_col or norm_col in term):
                    return original
        return None

    for norm_col, original in norm_to_original.items():
        for cand in candidates:
            if len(cand) <= 1:
                continue
            if cand and (cand in norm_col or norm_col in cand):
                return original
    return None


def numeric_column_is_usable(df: pd.DataFrame, col: Optional[str], min_valid_fraction: float = 0.5) -> bool:
    if col is None or col not in df.columns:
        return False
    numeric = pd.to_numeric(df[col], errors="coerce")
    return bool(numeric.notna().mean() >= min_valid_fraction)


def coerce_time(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.75:
        return numeric
    extracted = values.astype(str).str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    extracted_num = pd.to_numeric(extracted, errors="coerce")
    if extracted_num.notna().mean() >= 0.75:
        return extracted_num
    labels = sorted(values.astype(str).dropna().unique())
    mapping = {lab: i + 1 for i, lab in enumerate(labels)}
    return values.astype(str).map(mapping)


# =============================================================================
# Input loading
# =============================================================================

def detect_idata_file(explicit_idata: Optional[Path]) -> Path:
    if explicit_idata is not None:
        if not explicit_idata.exists():
            raise FileNotFoundError(f"Explicit InferenceData file does not exist: {explicit_idata}")
        return explicit_idata
    preferred = [
        DEFAULT_IDATA,
        PROJECT_ROOT / "Bioinformatics_Advances" / "Figure_3" / "trace_core.nc",
        PROJECT_ROOT / "Bioinformatics_Advances" / "Figure_4" / "trace_core.nc",
        PROJECT_ROOT / "Methodology" / "trace_core.nc",
    ]
    for p in preferred:
        if p.exists():
            print(f"[INFO] Auto-detected core InferenceData file: {p}")
            return p
    candidates = list(PROJECT_ROOT.rglob("trace_core.nc"))
    if candidates:
        chosen = sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]
        print(f"[INFO] Auto-detected core InferenceData file: {chosen}")
        return chosen
    raise FileNotFoundError("Could not find trace_core.nc automatically. Use --idata /full/path/to/trace_core.nc")


def detect_data_file(explicit_data: Optional[Path]) -> Path:
    if explicit_data is not None:
        if not explicit_data.exists():
            raise FileNotFoundError(f"Explicit data file does not exist: {explicit_data}")
        return explicit_data
    preferred = [DEFAULT_DATA, PROJECT_ROOT / "Bioinformatics_Advances" / "Figure_2" / "mut_freq_data.csv"]
    for p in preferred:
        if p.exists():
            return p
    candidates = list(PROJECT_ROOT.rglob("*mut*freq*.csv")) + list(PROJECT_ROOT.rglob("*mutation*frequency*.csv"))
    if candidates:
        return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]
    raise FileNotFoundError("Could not find mutation-frequency data automatically. Use --data /full/path/to/mut_freq_data.csv")


def prepare_dataset(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    genotype_col = find_column(raw, "genotype")
    time_col = find_column(raw, "time")
    replicate_col = find_column(raw, "replicate")
    freq_col = find_column(raw, "mutation_frequency")
    reserved = [genotype_col, time_col, replicate_col, freq_col]

    colmap = {
        "genotype": genotype_col,
        "time": time_col,
        "replicate": replicate_col,
        "mutation_frequency": freq_col,
        "count": find_column(raw, "count", exclude=reserved),
        "trials": find_column(raw, "trials", exclude=reserved),
    }
    if not numeric_column_is_usable(raw, colmap["count"]):
        colmap["count"] = None
    if not numeric_column_is_usable(raw, colmap["trials"]):
        colmap["trials"] = None

    missing = [k for k in ["genotype", "time", "mutation_frequency"] if colmap[k] is None]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {list(raw.columns)}")

    out = pd.DataFrame()
    out["genotype"] = raw[colmap["genotype"]].map(standardize_genotype_label)
    out["time_raw"] = raw[colmap["time"]]
    out["time"] = coerce_time(raw[colmap["time"]])
    out["replicate"] = raw[colmap["replicate"]].astype(str) if colmap["replicate"] else "rep1"
    out["mutation_frequency"] = pd.to_numeric(raw[colmap["mutation_frequency"]], errors="coerce")
    out["count"] = pd.to_numeric(raw[colmap["count"]], errors="coerce") if colmap["count"] else np.nan
    out["trials"] = pd.to_numeric(raw[colmap["trials"]], errors="coerce") if colmap["trials"] else np.nan

    out = out.dropna(subset=["genotype", "time", "mutation_frequency"]).copy()
    out = out[out["mutation_frequency"] >= 0].copy()
    out["genotype"] = pd.Categorical(out["genotype"], categories=ordered_genotypes(out["genotype"]), ordered=True)
    out = out.sort_values(["genotype", "replicate", "time"]).reset_index(drop=True)
    return out, colmap


def load_bg_categories(idata_path: Path, data: pd.DataFrame) -> List[str]:
    candidates = [
        idata_path.parent / "bg_categories.npy",
        DEFAULT_IDATA.parent / "bg_categories.npy",
        PROJECT_ROOT / "Methodology" / "bg_categories.npy",
    ]
    for p in candidates:
        if p.exists():
            try:
                return [standardize_genotype_label(x) for x in np.load(p, allow_pickle=True).tolist()]
            except Exception:
                pass
    return ordered_genotypes(data["genotype"].astype(str))


def load_time_values(idata_path: Path, data: pd.DataFrame) -> List[float]:
    data_times = sorted([float(x) for x in data["time"].dropna().unique()])
    candidates = [idata_path.parent / "times.npy", DEFAULT_IDATA.parent / "times.npy", PROJECT_ROOT / "Methodology" / "times.npy"]
    for p in candidates:
        if p.exists():
            try:
                vals = [float(x) for x in np.asarray(np.load(p, allow_pickle=True)).ravel().tolist()]
                if len(vals) == len(data_times) and vals == list(range(len(vals))):
                    return data_times
                return vals
            except Exception:
                pass
    return data_times


# =============================================================================
# Posterior predictive extraction
# =============================================================================

def choose_ppc_var(idata) -> str:
    if not hasattr(idata, "posterior_predictive"):
        raise ValueError("InferenceData has no posterior_predictive group. S3 requires Y_transition or equivalent PPC draws.")
    names = list(idata.posterior_predictive.data_vars)
    for cand in PPC_VAR_CANDIDATES:
        if cand in names:
            return cand
    norm_to_name = {normalize_name(n): n for n in names}
    for cand in PPC_VAR_CANDIDATES:
        nc = normalize_name(cand)
        for norm, name in norm_to_name.items():
            if nc in norm or norm in nc:
                return name
    for name in names:
        if {"chain", "draw"}.issubset(set(idata.posterior_predictive[name].dims)):
            return name
    raise ValueError(f"No usable posterior predictive variable found. Available: {names}")


def ppc_values_and_dims(idata, var_name: str) -> Tuple[np.ndarray, List[str], List[int]]:
    arr = idata.posterior_predictive[var_name]
    dims_order = ["chain", "draw"] + [d for d in arr.dims if d not in {"chain", "draw"}]
    arr = arr.transpose(*dims_order)
    values = np.asarray(arr.values)
    non_sample_dims = [d for d in arr.dims if d not in {"chain", "draw"}]
    sizes = [int(arr.sizes[d]) for d in non_sample_dims]
    return values, non_sample_dims, sizes


def build_vector_metadata(data: pd.DataFrame, bg_order: List[str], time_values: List[float], n_obs: int) -> Optional[pd.DataFrame]:
    full = data.copy()
    full["genotype_str"] = full["genotype"].astype(str).map(standardize_genotype_label)
    rank = {g: i for i, g in enumerate(bg_order)}
    full["bg_rank"] = full["genotype_str"].map(rank).fillna(999).astype(int)
    full = full.sort_values(["bg_rank", "replicate", "time"]).reset_index(drop=True)
    if n_obs == len(full):
        return full

    transition = (
        full.groupby(["genotype_str", "replicate"], observed=True, group_keys=False)
        .apply(lambda d: d.sort_values("time").iloc[1:])
        .reset_index(drop=True)
    )
    if n_obs == len(transition):
        return transition

    reps_by_bg = full.groupby("genotype_str", observed=True)["replicate"].apply(lambda x: list(pd.Series(x).drop_duplicates())).to_dict()
    for times in [time_values, time_values[1:]]:
        rows = []
        for bg in bg_order:
            for rep in reps_by_bg.get(bg, []):
                for t in times:
                    rows.append({"genotype_str": bg, "replicate": rep, "time": float(t)})
        meta = pd.DataFrame(rows)
        if n_obs == len(meta):
            return meta
    return None


def extract_log10_samples(
    idata,
    ppc_var: str,
    data: pd.DataFrame,
    genotype: str,
    time_value: float,
    bg_order: List[str],
    time_values: List[float],
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values, dims, sizes = ppc_values_and_dims(idata, ppc_var)
    sample_shape = values.shape[2:]
    genotype = standardize_genotype_label(genotype)

    if len(sample_shape) == 1:
        meta = build_vector_metadata(data, bg_order, time_values, sample_shape[0])
        if meta is None:
            raise ValueError(f"Could not map vectorized PPC variable {ppc_var} with length {sample_shape[0]} to dataset.")
        meta = meta.copy()
        meta["genotype_str"] = meta["genotype_str"].astype(str).map(standardize_genotype_label)
        meta["time_float"] = pd.to_numeric(meta["time"], errors="coerce").astype(float)
        mask = (meta["genotype_str"] == genotype) & (np.abs(meta["time_float"] - float(time_value)) <= 1e-8)
        if not mask.any():
            sub = meta[meta["genotype_str"] == genotype]
            nearest = sub.iloc[np.argmin(np.abs(sub["time_float"].values - float(time_value)))]["time_float"]
            mask = (meta["genotype_str"] == genotype) & (np.abs(meta["time_float"] - nearest) <= 1e-8)
        idx = np.where(mask.values)[0]
        flat = values[:, :, idx].reshape(-1)
    else:
        dims_norm = [normalize_name(d) for d in dims]
        bg_dim = next((i for i, d in enumerate(dims_norm) if any(k in d for k in ["bg", "background", "genotype", "strain"])), 0)
        time_dim = next((i for i, d in enumerate(dims_norm) if d in {"time", "t", "timepoint"} or "time" in d), len(sample_shape) - 1)
        if genotype not in bg_order:
            raise ValueError(f"Genotype {genotype} not in model background order {bg_order}")
        bg_idx = bg_order.index(genotype)
        time_len = sample_shape[time_dim]
        if time_len == len(time_values):
            avail_times = time_values
        elif time_len == len(time_values) - 1:
            avail_times = time_values[1:]
        else:
            avail_times = list(range(time_len))
        t_idx = int(np.argmin(np.abs(np.asarray(avail_times, dtype=float) - float(time_value))))
        slicer = [slice(None), slice(None)] + [slice(None)] * len(sample_shape)
        slicer[2 + bg_dim] = bg_idx
        slicer[2 + time_dim] = t_idx
        flat = values[tuple(slicer)].reshape(-1)

    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        raise ValueError(f"No finite posterior predictive samples for genotype={genotype}, time={time_value}")
    if len(flat) > max_samples:
        flat = rng.choice(flat, size=max_samples, replace=False)
    return flat


# =============================================================================
# Count PMFs
# =============================================================================

def choose_times(data: pd.DataFrame, requested: Optional[List[float]] = None) -> List[float]:
    if requested is not None:
        return requested
    times = sorted([float(x) for x in data["time"].dropna().unique()])
    if len(times) >= 3:
        return [times[0], times[len(times) // 2], times[-1]]
    return times


def observed_counts(data: pd.DataFrame, genotype: str, time_value: float, default_trials: int) -> Tuple[np.ndarray, int, bool]:
    d = data.copy()
    d["genotype_str"] = d["genotype"].astype(str).map(standardize_genotype_label)
    d["time_float"] = pd.to_numeric(d["time"], errors="coerce").astype(float)
    genotype = standardize_genotype_label(genotype)
    mask = (d["genotype_str"] == genotype) & (np.abs(d["time_float"] - float(time_value)) <= 1e-8)
    if not mask.any():
        sub = d[d["genotype_str"] == genotype]
        nearest = sub.iloc[np.argmin(np.abs(sub["time_float"].values - float(time_value)))]["time_float"]
        mask = (d["genotype_str"] == genotype) & (np.abs(d["time_float"] - nearest) <= 1e-8)
    sub = d.loc[mask].copy()

    has_real = sub["count"].notna().any() and sub["trials"].notna().any() and (sub["trials"].dropna() > 0).any()
    if has_real:
        valid = sub.dropna(subset=["count", "trials"])
        valid = valid[valid["trials"] > 0]
        if len(valid):
            n_trials = int(round(float(valid["trials"].median())))
            return valid["count"].round().astype(int).values, n_trials, True

    n_trials = int(default_trials)
    counts = np.rint(np.clip(sub["mutation_frequency"].values, 0, 1) * n_trials).astype(int)
    return counts, n_trials, False


def simulated_counts_from_log10(log10_samples: np.ndarray, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    p = np.power(10.0, log10_samples)
    p = np.clip(p, 0.0, 1.0)
    return rng.binomial(n=int(n_trials), p=p).astype(int)


def pmf_for_plot(counts: np.ndarray, obs: np.ndarray, max_span: int = 90) -> pd.DataFrame:
    counts = np.asarray(counts, dtype=int)
    obs = np.asarray(obs, dtype=int)
    lo = int(max(0, np.floor(np.quantile(counts, 0.005))))
    hi = int(np.ceil(np.quantile(counts, 0.995)))
    if len(obs):
        lo = min(lo, int(obs.min()))
        hi = max(hi, int(obs.max()))
    if hi <= lo:
        hi = lo + 1
    if hi - lo > max_span:
        center = int(np.round(np.median(counts)))
        lo = max(0, center - max_span // 2)
        hi = lo + max_span
        if len(obs):
            if obs.min() < lo:
                lo = max(0, int(obs.min()) - 5)
                hi = lo + max_span
            if obs.max() > hi:
                hi = int(obs.max()) + 5
                lo = max(0, hi - max_span)
    grid = np.arange(lo, hi + 1, dtype=int)
    vals, freq = np.unique(counts[(counts >= lo) & (counts <= hi)], return_counts=True)
    prob = {int(v): float(f) / len(counts) for v, f in zip(vals, freq)}
    return pd.DataFrame({"count": grid, "probability": [prob.get(int(k), 0.0) for k in grid]})


def summarize(genotype: str, time_value: float, n_trials: int, real_counts: bool, obs: np.ndarray, pp: np.ndarray) -> Dict[str, object]:
    return {
        "genotype": genotype,
        "time": float(time_value),
        "n_trials": int(n_trials),
        "observed_counts_are_real": bool(real_counts),
        "n_observed_counts": int(len(obs)),
        "observed_count_mean": float(np.mean(obs)) if len(obs) else np.nan,
        "observed_count_min": int(np.min(obs)) if len(obs) else np.nan,
        "observed_count_max": int(np.max(obs)) if len(obs) else np.nan,
        "pp_count_mean": float(np.mean(pp)),
        "pp_count_median": float(np.median(pp)),
        "pp_count_q05": float(np.quantile(pp, 0.05)),
        "pp_count_q95": float(np.quantile(pp, 0.95)),
        "pp_count_q025": float(np.quantile(pp, 0.025)),
        "pp_count_q975": float(np.quantile(pp, 0.975)),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_panel(ax, result: Dict[str, object], label: str) -> None:
    add_panel_label(ax, label)
    genotype = str(result["genotype"])
    time_value = float(result["time"])
    pmf = result["pmf"]
    pp_counts = np.asarray(result["pp_counts"], dtype=int)
    obs = np.asarray(result["obs_counts"], dtype=int)
    n_trials = int(result["n_trials"])
    real_counts = bool(result["real_counts"])

    color = genotype_color(genotype)
    x = pmf["count"].values
    y = pmf["probability"].values
    q05, q50, q95 = np.quantile(pp_counts, [0.05, 0.50, 0.95])

    ax.axvspan(q05, q95, color="0.85", alpha=0.50, linewidth=0)
    ax.bar(x, y, width=0.85, color=color, alpha=0.72, edgecolor="white", linewidth=0.25)
    ax.axvline(q50, color="black", linestyle="-", linewidth=1.0, alpha=0.75)

    if len(obs):
        obs_mean = float(np.mean(obs))
        ax.axvline(obs_mean, color="black", linestyle="--", linewidth=1.2)
        ymax = max(float(y.max()), 1e-12)
        for oc in obs:
            ax.plot([oc, oc], [0, ymax * 0.10], color="black", linewidth=0.75, alpha=0.75)

    count_label = "observed K" if real_counts else "implied K"
    ax.set_title(f"{genotype}, time {time_value:g}", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Mutation count K", fontsize=8.5)
    ax.set_ylabel("Posterior predictive probability", fontsize=8.5)
    ax.text(0.98, 0.92, f"N = {n_trials:,}\n{count_label}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="0.75", alpha=0.88))
    style_axis(ax)


def make_figure(results: List[Dict[str, object]], output_dir: Path, default_trials_used: bool, dpi: int) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(3, 3, figsize=(12.8, 9.2), constrained_layout=False)
    plt.subplots_adjust(left=0.065, right=0.985, top=0.905, bottom=0.095, wspace=0.26, hspace=0.42)

    for ax, result, label in zip(axes.ravel(), results, list("ABCDEFGHI")):
        plot_panel(ax, result, label)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.55", alpha=0.45, edgecolor="none", label="90% predictive interval"),
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.0, label="Predictive median"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Observed/implied mean"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("Supplementary Fig. S3. Additional posterior predictive mutation-count distributions",
                 fontsize=13, fontweight="bold", y=0.975)

    if default_trials_used:
        fig.text(0.5, 0.025,
                 "No explicit count/trial columns were detected; observed counts are implied from mutation frequency using the effective binomial trial size.",
                 ha="center", va="center", fontsize=8, color="0.25")

    png = output_dir / f"{FIG_BASENAME}.png"
    pdf = output_dir / f"{FIG_BASENAME}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# =============================================================================
# Main workflow
# =============================================================================

def parse_times_arg(times_arg: Optional[str]) -> Optional[List[float]]:
    if times_arg is None or str(times_arg).strip() == "":
        return None
    return [float(x.strip()) for x in str(times_arg).split(",") if x.strip()]


def build_results(idata, ppc_var: str, data: pd.DataFrame, idata_path: Path, args) -> Tuple[List[Dict[str, object]], pd.DataFrame, pd.DataFrame, bool]:
    rng = np.random.default_rng(args.seed)
    bg_order = load_bg_categories(idata_path, data)
    time_values = load_time_values(idata_path, data)
    times = choose_times(data, parse_times_arg(args.times))
    genotypes = [g for g in GENOTYPE_ORDER if g in set(data["genotype"].astype(str))]
    if len(genotypes) < 3:
        genotypes = ordered_genotypes(data["genotype"].astype(str))[:3]

    results: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    pmf_tables: List[pd.DataFrame] = []
    default_trials_used = False

    for genotype in genotypes:
        for time_value in times:
            log10_samples = extract_log10_samples(
                idata=idata,
                ppc_var=ppc_var,
                data=data,
                genotype=genotype,
                time_value=float(time_value),
                bg_order=bg_order,
                time_values=time_values,
                max_samples=args.max_samples,
                rng=rng,
            )
            obs, n_trials, real_counts = observed_counts(data, genotype, float(time_value), args.default_trials)
            if not real_counts:
                default_trials_used = True
            pp_counts = simulated_counts_from_log10(log10_samples, n_trials, rng)
            pmf = pmf_for_plot(pp_counts, obs)

            results.append({
                "genotype": genotype,
                "time": float(time_value),
                "pmf": pmf,
                "pp_counts": pp_counts,
                "obs_counts": obs,
                "n_trials": int(n_trials),
                "real_counts": bool(real_counts),
            })
            summary_rows.append(summarize(genotype, float(time_value), n_trials, real_counts, obs, pp_counts))
            pmf2 = pmf.copy()
            pmf2.insert(0, "time", float(time_value))
            pmf2.insert(0, "genotype", genotype)
            pmf_tables.append(pmf2)

    summary_df = pd.DataFrame(summary_rows)
    pmf_df = pd.concat(pmf_tables, ignore_index=True) if pmf_tables else pd.DataFrame()
    return results, summary_df, pmf_df, default_trials_used


def print_summary(idata_path: Path, data_path: Path, ppc_var: str, colmap: Dict[str, Optional[str]], summary_df: pd.DataFrame, default_trials: int) -> None:
    print("\n========== Supplementary Fig. S3 PPC count-PMF summary ==========")
    print(f"InferenceData file   : {idata_path}")
    print(f"Mutation data file   : {data_path}")
    print(f"PPC variable         : {ppc_var}")
    print("Detected data columns:")
    for k, v in colmap.items():
        print(f"  {k:20s}: {v}")
    print(f"Default/effective N  : {default_trials:,}")
    print(f"Panels generated     : {len(summary_df)}")
    if not summary_df.empty:
        cols = ["genotype", "time", "n_trials", "observed_counts_are_real", "observed_count_mean", "pp_count_median", "pp_count_q05", "pp_count_q95"]
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print("\nPanel summary:")
            print(summary_df[cols].to_string(index=False))
    print("=================================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Supplementary Fig. S3 posterior predictive count PMFs.")
    parser.add_argument("--idata", type=str, default=None, help="Path to trace_core.nc.")
    parser.add_argument("--data", type=str, default=None, help="Path to mutation-frequency CSV.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--default-trials", type=int, default=1_000_000, help="Effective N when explicit K/N columns are absent.")
    parser.add_argument("--max-samples", type=int, default=12000, help="Maximum posterior predictive samples per panel.")
    parser.add_argument("--times", type=str, default=None, help="Comma-separated time points, e.g. '6,15,24'.")
    parser.add_argument("--seed", type=int, default=20260613, help="Random seed.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    require_arviz()
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    idata_path = detect_idata_file(Path(args.idata).expanduser().resolve() if args.idata else None)
    data_path = detect_data_file(Path(args.data).expanduser().resolve() if args.data else None)

    idata = az.from_netcdf(idata_path)
    data, colmap = prepare_dataset(read_csv_lenient(data_path))
    ppc_var = choose_ppc_var(idata)

    results, summary_df, pmf_df, default_trials_used = build_results(idata, ppc_var, data, idata_path, args)
    print_summary(idata_path, data_path, ppc_var, colmap, summary_df, args.default_trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "supp_figure_s3_ppc_count_pmf_summary.csv"
    pmf_path = output_dir / "supp_figure_s3_ppc_count_pmf_values.csv"
    summary_df.to_csv(summary_path, index=False)
    pmf_df.to_csv(pmf_path, index=False)

    png, pdf = make_figure(results, output_dir, default_trials_used, args.dpi)
    print("[DONE] Saved Supplementary Fig. S3 outputs:")
    print(f"  PNG        : {png}")
    print(f"  PDF        : {pdf}")
    print(f"  Summary CSV: {summary_path}")
    print(f"  PMF CSV    : {pmf_path}")


if __name__ == "__main__":
    main()
