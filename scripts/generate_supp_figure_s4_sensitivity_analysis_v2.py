from __future__ import annotations

import argparse
import itertools
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


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

PROJECT_ROOT = Path("/Fig_S4")

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "Fig_S4"
)

DEFAULT_IDATA = (
    PROJECT_ROOT
    / "Figure_2"
    / "trace_core.nc"
)

DEFAULT_DATA = PROJECT_ROOT / "Figure_2" / "mut_freq_data.csv"

FIG_BASENAME = "supp_figure_s4_sensitivity_analysis"

LOG10_OFFSET = 1e-9

GENOTYPE_ORDER = ["WT", "priA", "recG"]

GENOTYPE_COLORS = {
    "WT": "#D62728",
    "priA": "#7F7F7F",
    "recG": "#FF7F0E",
}

CORE_VAR_NAMES = [
    "mu_bg",
    "mu_hyper",
    "tau_mu",
    "theta",
    "sigma_bg",
    "sigma_obs",
]

PPC_VAR_CANDIDATES = [
    "Y_transition",
    "y_transition",
    "Y_pred",
    "y_pred",
    "Y_ppc",
    "posterior_predictive",
    "obs",
]

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
# General helpers
# =============================================================================

def require_arviz() -> None:
    if az is None:
        raise ImportError(
            "ArviZ is required but could not be imported.\n"
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
    ax.text(
        -0.10, 1.08, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def parameter_display_name(label: str) -> str:
    out = str(label)
    replacements = {
        "mu_hyper": r"$\mu_{\mathrm{hyper}}$",
        "tau_mu": r"$\tau_{\mu}$",
        "theta": r"$\theta$",
        "sigma_obs": r"$\sigma_{\mathrm{obs}}$",
        "sigma_bg": r"$\sigma_{\mathrm{bg}}$",
        "mu_bg": r"$\mu_{\mathrm{bg}}$",
    }
    for key, value in replacements.items():
        out = re.sub(key, lambda match, v=value: v, out, flags=re.IGNORECASE)
    out = out.replace("_", " ")
    out = out.replace("[0]", "[priA]")
    out = out.replace("[1]", "[recG]")
    out = out.replace("[2]", "[WT]")
    out = out.replace("[wt]", "[WT]")
    return out


# =============================================================================
# Data loading
# =============================================================================

def find_column(
    df: pd.DataFrame,
    role: str,
    exclude: Optional[Iterable[str]] = None,
) -> Optional[str]:
    exclude_norm = {normalize_name(c) for c in (exclude or []) if c is not None}
    norm_to_original = {
        normalize_name(c): c
        for c in df.columns
        if normalize_name(c) not in exclude_norm
    }

    candidates = [normalize_name(c) for c in COLUMN_CANDIDATES[role]]

    for cand in candidates:
        if cand in norm_to_original:
            return norm_to_original[cand]

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


def read_csv_lenient(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


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


def detect_data_file(explicit_data: Optional[Path]) -> Path:
    if explicit_data is not None:
        if not explicit_data.exists():
            raise FileNotFoundError(f"Explicit data file does not exist: {explicit_data}")
        return explicit_data

    preferred = [
        DEFAULT_DATA,
        PROJECT_ROOT / "Figure_2" / "mut_freq_data.csv",
    ]

    for p in preferred:
        if p.exists():
            return p

    candidates = list(PROJECT_ROOT.rglob("*mut*freq*.csv")) + list(PROJECT_ROOT.rglob("*mutation*frequency*.csv"))
    if candidates:
        return sorted(candidates, key=lambda p: (len(p.parts), str(p)))[0]

    raise FileNotFoundError("Could not find mutation-frequency data. Please use --data.")


def detect_idata_file(explicit_idata: Optional[Path]) -> Path:
    if explicit_idata is not None:
        if not explicit_idata.exists():
            raise FileNotFoundError(f"Explicit InferenceData file does not exist: {explicit_idata}")
        return explicit_idata

    preferred = [
        DEFAULT_IDATA,
        PROJECT_ROOT / "Figure_2" / "trace_core.nc",
    ]

    for p in preferred:
        if p.exists():
            print(f"[INFO] Auto-detected core InferenceData file: {p}")
            return p

    candidates = sorted(PROJECT_ROOT.rglob("trace_core.nc"), key=lambda p: (len(p.parts), str(p)))
    if candidates:
        print(f"[INFO] Auto-detected core InferenceData file: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError("Could not find trace_core.nc. Please use --idata.")


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    genotype_col = find_column(df, "genotype")
    time_col = find_column(df, "time")
    replicate_col = find_column(df, "replicate")
    mutation_frequency_col = find_column(df, "mutation_frequency")

    reserved_cols = [genotype_col, time_col, replicate_col, mutation_frequency_col]

    colmap: Dict[str, Optional[str]] = {
        "genotype": genotype_col,
        "time": time_col,
        "replicate": replicate_col,
        "mutation_frequency": mutation_frequency_col,
        "count": find_column(df, "count", exclude=reserved_cols),
        "trials": find_column(df, "trials", exclude=reserved_cols),
    }

    if not numeric_column_is_usable(df, colmap["count"]):
        colmap["count"] = None
    if not numeric_column_is_usable(df, colmap["trials"]):
        colmap["trials"] = None

    missing = [k for k in ["genotype", "time", "mutation_frequency"] if colmap[k] is None]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}; available columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["genotype"] = df[colmap["genotype"]].map(standardize_genotype_label)
    out["time_raw"] = df[colmap["time"]]
    out["time"] = coerce_time(df[colmap["time"]])

    if colmap["replicate"] is not None:
        out["replicate"] = df[colmap["replicate"]].astype(str)
    else:
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

    out = out.dropna(subset=["genotype", "time", "mutation_frequency"]).copy()
    out = out[out["mutation_frequency"] >= 0].copy()
    out["log10_mutation_frequency"] = np.log10(out["mutation_frequency"] + LOG10_OFFSET)

    genotype_order = ordered_genotypes(out["genotype"])
    out["genotype"] = pd.Categorical(out["genotype"], categories=genotype_order, ordered=True)
    out = out.sort_values(["genotype", "replicate", "time"]).reset_index(drop=True)

    return out, colmap


def load_bg_categories(idata_path: Path, data: pd.DataFrame) -> List[str]:
    candidates = [
        idata_path.parent / "bg_categories.npy",
        DEFAULT_IDATA.parent / "bg_categories.npy",
        PROJECT_ROOT / "Figure_2" / "bg_categories.npy",
    ]

    for p in candidates:
        if p.exists():
            try:
                arr = np.load(p, allow_pickle=True)
                labels = [standardize_genotype_label(x) for x in arr.tolist()]
                return labels
            except Exception:
                pass

    return ordered_genotypes(data["genotype"].astype(str))


def load_time_values(idata_path: Path, data: pd.DataFrame) -> List[float]:
    candidates = [
        idata_path.parent / "times.npy",
        DEFAULT_IDATA.parent / "times.npy",
        PROJECT_ROOT / "Figure_2" / "times.npy",
    ]

    for p in candidates:
        if p.exists():
            try:
                arr = np.load(p, allow_pickle=True)
                vals = [float(x) for x in np.asarray(arr).ravel().tolist()]
                data_times = sorted([float(x) for x in data["time"].dropna().unique()])
                if len(vals) == len(data_times) and set(vals) == set(range(len(vals))):
                    return data_times
                return vals
            except Exception:
                pass

    return sorted([float(x) for x in data["time"].dropna().unique()])


# =============================================================================
# Model manifest and posterior summaries
# =============================================================================

def load_manifest(
    manifest_path: Optional[Path],
    primary_idata_path: Path,
) -> pd.DataFrame:
    """
    Load optional model sensitivity manifest.

    If no manifest is provided, use the primary model only.
    """
    if manifest_path is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        required = {"model_label", "idata_path"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing required column(s): {missing}")
        if "model_class" not in manifest.columns:
            manifest["model_class"] = "sensitivity"
        manifest["idata_path"] = manifest["idata_path"].astype(str)
        return manifest

    rows = [
        {
            "model_label": "Primary",
            "idata_path": str(primary_idata_path),
            "model_class": "primary",
        }
    ]

    # Auto-detect optional sensitivity traces if the user later adds them.
    search_dirs = [
        DEFAULT_OUTPUT_DIR,
        DEFAULT_OUTPUT_DIR.parent,
        PROJECT_ROOT / "Fig_S4",
    ]

    optional_patterns = [
        ("Weak prior", ["weak_prior", "weak-prior", "prior_weak"]),
        ("Regularizing prior", ["regularizing_prior", "regularized", "strong_prior", "prior_regularizing"]),
        ("Alternative prior", ["alt_prior", "alternative_prior"]),
    ]

    found_paths = {str(primary_idata_path.resolve())}
    for label, tokens in optional_patterns:
        for base in search_dirs:
            if not base.exists():
                continue
            for p in base.rglob("trace_core.nc"):
                p_str = str(p).lower()
                if any(tok in p_str for tok in tokens) and str(p.resolve()) not in found_paths:
                    rows.append({
                        "model_label": label,
                        "idata_path": str(p),
                        "model_class": "prior",
                    })
                    found_paths.add(str(p.resolve()))
                    break

    return pd.DataFrame(rows)


def flatten_variable(idata, var_name: str, bg_labels: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Extract scalar posterior arrays as flat posterior samples.
    """
    if var_name not in idata.posterior:
        return {}

    arr = idata.posterior[var_name].transpose("chain", "draw", ...)
    values = np.asarray(arr.values)

    out: Dict[str, np.ndarray] = {}

    if values.ndim == 2:
        out[var_name] = values.reshape(-1)
        return out

    chains, draws = values.shape[:2]
    flat = values.reshape(chains, draws, -1)

    non_sample_dims = [d for d in arr.dims if d not in {"chain", "draw"}]
    coord_labels: List[str] = []

    if non_sample_dims:
        coords = []
        for d in non_sample_dims:
            if d in arr.coords:
                coords.append([str(x) for x in arr.coords[d].values])
            else:
                coords.append([str(i) for i in range(arr.sizes[d])])
        for combo in itertools.product(*coords):
            coord_labels.append(",".join(combo))

    for idx in range(flat.shape[2]):
        if var_name in {"mu_bg", "sigma_bg"} and bg_labels is not None and idx < len(bg_labels):
            label = f"{var_name}[{bg_labels[idx]}]"
        elif idx < len(coord_labels):
            label = f"{var_name}[{coord_labels[idx]}]"
        else:
            label = f"{var_name}[{idx}]"
        out[label] = flat[:, :, idx].reshape(-1)

    return out


def posterior_parameter_summary(
    manifest: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize selected posterior parameters across available fits.
    """
    rows = []

    for _, row in manifest.iterrows():
        label = str(row["model_label"])
        model_class = str(row.get("model_class", "sensitivity"))
        path = Path(str(row["idata_path"])).expanduser()

        if not path.exists():
            print(f"[WARN] Skipping missing idata path for {label}: {path}")
            continue

        try:
            idata = az.from_netcdf(path)
        except Exception as exc:
            print(f"[WARN] Could not load {path}: {exc}")
            continue

        bg_labels = load_bg_categories(path, data)

        for var in ["mu_bg", "sigma_bg", "mu_hyper", "tau_mu", "theta", "sigma_obs"]:
            components = flatten_variable(idata, var, bg_labels=bg_labels)
            for param_label, samples in components.items():
                samples = np.asarray(samples, dtype=float)
                samples = samples[np.isfinite(samples)]
                if len(samples) == 0:
                    continue

                rows.append({
                    "model_label": label,
                    "model_class": model_class,
                    "idata_path": str(path),
                    "parameter": param_label,
                    "parameter_display": parameter_display_name(param_label),
                    "mean": float(np.mean(samples)),
                    "median": float(np.median(samples)),
                    "q025": float(np.quantile(samples, 0.025)),
                    "q05": float(np.quantile(samples, 0.05)),
                    "q95": float(np.quantile(samples, 0.95)),
                    "q975": float(np.quantile(samples, 0.975)),
                    "n_samples": int(len(samples)),
                })

    return pd.DataFrame(rows)


# =============================================================================
# Posterior predictive mapping and coverage
# =============================================================================

def choose_ppc_var(idata) -> Optional[str]:
    if not hasattr(idata, "posterior_predictive"):
        return None

    names = list(idata.posterior_predictive.data_vars)

    for cand in PPC_VAR_CANDIDATES:
        if cand in names:
            return cand

    norm_to_name = {normalize_name(n): n for n in names}
    for cand in PPC_VAR_CANDIDATES:
        norm_cand = normalize_name(cand)
        for norm, name in norm_to_name.items():
            if norm_cand in norm or norm in norm_cand:
                return name

    for name in names:
        dims = set(idata.posterior_predictive[name].dims)
        if {"chain", "draw"}.issubset(dims):
            return name

    return None


def ppc_array(idata, var_name: str) -> Tuple[np.ndarray, List[str], List[int]]:
    arr = idata.posterior_predictive[var_name].transpose("chain", "draw", ...)
    values = np.asarray(arr.values)
    non_sample_dims = [d for d in arr.dims if d not in {"chain", "draw"}]
    sizes = [int(arr.sizes[d]) for d in non_sample_dims]
    return values, non_sample_dims, sizes


def build_transition_metadata(
    data: pd.DataFrame,
    bg_model_order: List[str],
    time_values: List[float],
    n_ppc_obs: int,
) -> Optional[pd.DataFrame]:
    full = data.copy()
    full["genotype_str"] = full["genotype"].astype(str).map(standardize_genotype_label)

    bg_rank = {g: i for i, g in enumerate(bg_model_order)}
    full["bg_rank"] = full["genotype_str"].map(bg_rank).fillna(999).astype(int)
    full = full.sort_values(["bg_rank", "replicate", "time"]).reset_index(drop=True)

    if n_ppc_obs == len(full):
        return full

    transition = (
        full.sort_values(["bg_rank", "replicate", "time"])
        .groupby(["genotype_str", "replicate"], observed=True, group_keys=False)
        .apply(lambda d: d.iloc[1:])
        .reset_index(drop=True)
    )
    if n_ppc_obs == len(transition):
        return transition

    return None


def coverage_for_idata(
    idata,
    idata_path: Path,
    data: pd.DataFrame,
    interval_mass: float = 0.90,
) -> Dict[str, object]:
    """
    Compute posterior predictive interval coverage on the log10 frequency scale.
    """
    ppc_var = choose_ppc_var(idata)
    if ppc_var is None:
        return {
            "coverage": np.nan,
            "mean_interval_width": np.nan,
            "n_observations": 0,
            "ppc_var": None,
        }

    values, non_sample_dims, sizes = ppc_array(idata, ppc_var)
    sample_shape = values.shape[2:]
    bg_order = load_bg_categories(idata_path, data)
    time_values = load_time_values(idata_path, data)

    if len(sample_shape) == 1:
        meta = build_transition_metadata(data, bg_order, time_values, sample_shape[0])
        if meta is None:
            return {
                "coverage": np.nan,
                "mean_interval_width": np.nan,
                "n_observations": 0,
                "ppc_var": ppc_var,
            }

        obs = meta["log10_mutation_frequency"].values.astype(float)
        draws = values.reshape(values.shape[0] * values.shape[1], values.shape[2])

    else:
        # Fallback: if array is not observation-vectorized, skip coverage.
        return {
            "coverage": np.nan,
            "mean_interval_width": np.nan,
            "n_observations": 0,
            "ppc_var": ppc_var,
        }

    alpha = (1.0 - interval_mass) / 2.0
    lo = np.quantile(draws, alpha, axis=0)
    hi = np.quantile(draws, 1.0 - alpha, axis=0)

    valid = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    if valid.sum() == 0:
        return {
            "coverage": np.nan,
            "mean_interval_width": np.nan,
            "n_observations": 0,
            "ppc_var": ppc_var,
        }

    covered = (obs[valid] >= lo[valid]) & (obs[valid] <= hi[valid])
    widths = hi[valid] - lo[valid]

    return {
        "coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(widths)),
        "n_observations": int(valid.sum()),
        "ppc_var": ppc_var,
    }


def posterior_predictive_coverage_summary(
    manifest: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in manifest.iterrows():
        label = str(row["model_label"])
        model_class = str(row.get("model_class", "sensitivity"))
        path = Path(str(row["idata_path"])).expanduser()

        if not path.exists():
            continue

        try:
            idata = az.from_netcdf(path)
        except Exception:
            continue

        cov = coverage_for_idata(idata, path, data, interval_mass=0.90)
        rows.append({
            "model_label": label,
            "model_class": model_class,
            "idata_path": str(path),
            **cov,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Count-layer sensitivity
# =============================================================================

def extract_ppc_log10_samples_for_panel(
    idata,
    idata_path: Path,
    data: pd.DataFrame,
    genotype: str,
    time_value: float,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Extract posterior predictive log10 samples for a genotype-time panel.
    """
    ppc_var = choose_ppc_var(idata)
    if ppc_var is None:
        raise ValueError("No posterior predictive variable found.")

    values, non_sample_dims, sizes = ppc_array(idata, ppc_var)
    sample_shape = values.shape[2:]

    bg_order = load_bg_categories(idata_path, data)
    time_values = load_time_values(idata_path, data)
    genotype = standardize_genotype_label(genotype)

    if len(sample_shape) == 1:
        meta = build_transition_metadata(data, bg_order, time_values, sample_shape[0])
        if meta is None:
            raise ValueError("Could not map posterior predictive vector to observations.")

        meta = meta.copy()
        meta["genotype_str"] = meta["genotype"].astype(str).map(standardize_genotype_label)
        meta["time_float"] = pd.to_numeric(meta["time"], errors="coerce").astype(float)

        mask = (
            (meta["genotype_str"] == genotype)
            & (np.abs(meta["time_float"] - float(time_value)) <= 1e-8)
        )

        if not mask.any():
            sub = meta[meta["genotype_str"] == genotype]
            nearest = sub.iloc[np.argmin(np.abs(sub["time_float"].values - float(time_value)))]["time_float"]
            mask = (
                (meta["genotype_str"] == genotype)
                & (np.abs(meta["time_float"] - float(nearest)) <= 1e-8)
            )

        idx = np.where(mask.values)[0]
        selected = values[:, :, idx]
        flat = selected.reshape(-1)

    else:
        # Fallback multidimensional assumption: bg x ... x time
        if genotype not in bg_order:
            raise ValueError(f"Genotype {genotype} not found in model order {bg_order}")

        bg_idx = bg_order.index(genotype)
        time_len = sample_shape[-1]
        if time_len == len(time_values):
            available_times = time_values
        elif time_len == len(time_values) - 1:
            available_times = time_values[1:]
        else:
            available_times = list(range(time_len))

        t_idx = int(np.argmin(np.abs(np.asarray(available_times, dtype=float) - float(time_value))))

        slicer = [slice(None), slice(None)] + [slice(None)] * len(sample_shape)
        slicer[2] = bg_idx
        slicer[-1] = t_idx
        selected = values[tuple(slicer)]
        flat = selected.reshape(-1)

    flat = flat[np.isfinite(flat)]
    if len(flat) > max_samples:
        flat = rng.choice(flat, size=max_samples, replace=False)
    return flat


def observed_implied_counts(
    data: pd.DataFrame,
    genotype: str,
    time_value: float,
    n_trials: int,
) -> np.ndarray:
    d = data.copy()
    d["genotype_str"] = d["genotype"].astype(str).map(standardize_genotype_label)
    d["time_float"] = pd.to_numeric(d["time"], errors="coerce").astype(float)

    mask = (
        (d["genotype_str"] == standardize_genotype_label(genotype))
        & (np.abs(d["time_float"] - float(time_value)) <= 1e-8)
    )
    if not mask.any():
        return np.array([], dtype=int)

    sub = d.loc[mask].copy()
    if sub["count"].notna().any() and sub["trials"].notna().any():
        valid = sub.dropna(subset=["count", "trials"])
        if not valid.empty:
            return valid["count"].round().astype(int).values

    return np.rint(np.clip(sub["mutation_frequency"].values, 0, 1) * int(n_trials)).astype(int)


def sample_beta_binomial_counts(
    p: np.ndarray,
    n_trials: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample counts under binomial or beta-binomial.

    rho = 0 corresponds to binomial.
    For rho > 0:
        alpha = p * (1/rho - 1)
        beta  = (1-p) * (1/rho - 1)
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    n_trials = int(n_trials)

    if rho <= 0:
        return rng.binomial(n=n_trials, p=p).astype(int)

    concentration = (1.0 / float(rho)) - 1.0
    concentration = max(concentration, 1e-6)

    alpha = np.clip(p * concentration, 1e-9, None)
    beta = np.clip((1.0 - p) * concentration, 1e-9, None)

    q = rng.beta(alpha, beta)
    q = np.clip(q, 0.0, 1.0)
    return rng.binomial(n=n_trials, p=q).astype(int)


def count_layer_sensitivity_summary(
    idata,
    idata_path: Path,
    data: pd.DataFrame,
    selected_times: Sequence[float],
    default_trials: int,
    rho_values: Sequence[float],
    max_samples: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    pmf_rows = []

    layer_labels = {
        0.0: "Binomial",
    }

    for rho in rho_values:
        if float(rho) == 0.0:
            continue
        if rho <= 1e-5:
            layer_labels[float(rho)] = f"Beta-binomial low OD\nρ={rho:g}"
        else:
            layer_labels[float(rho)] = f"Beta-binomial moderate OD\nρ={rho:g}"

    genotypes = [g for g in GENOTYPE_ORDER if g in set(data["genotype"].astype(str))]
    if not genotypes:
        genotypes = ordered_genotypes(data["genotype"].astype(str))

    for genotype in genotypes:
        for time_value in selected_times:
            log10_samples = extract_ppc_log10_samples_for_panel(
                idata=idata,
                idata_path=idata_path,
                data=data,
                genotype=genotype,
                time_value=float(time_value),
                max_samples=max_samples,
                rng=rng,
            )

            p = np.clip(np.power(10.0, log10_samples), 0.0, 1.0)
            obs_counts = observed_implied_counts(data, genotype, float(time_value), default_trials)
            obs_mean = float(np.mean(obs_counts)) if len(obs_counts) else np.nan

            for rho in rho_values:
                rho = float(rho)
                counts = sample_beta_binomial_counts(
                    p=p,
                    n_trials=default_trials,
                    rho=rho,
                    rng=rng,
                )

                q05, q50, q95 = np.quantile(counts, [0.05, 0.50, 0.95])
                q025, q975 = np.quantile(counts, [0.025, 0.975])

                covered90 = bool((obs_mean >= q05) and (obs_mean <= q95)) if np.isfinite(obs_mean) else np.nan

                rows.append({
                    "genotype": genotype,
                    "time": float(time_value),
                    "layer": layer_labels.get(rho, f"rho={rho:g}"),
                    "rho": rho,
                    "n_trials": int(default_trials),
                    "obs_count_mean": obs_mean,
                    "median": float(q50),
                    "q05": float(q05),
                    "q95": float(q95),
                    "q025": float(q025),
                    "q975": float(q975),
                    "interval_width_90": float(q95 - q05),
                    "covered90": covered90,
                    "mean_count": float(np.mean(counts)),
                })

                # PMF for representative panel: priA late.
                if genotype == "priA" and float(time_value) == float(selected_times[-1]):
                    vals, freq = np.unique(counts, return_counts=True)
                    prob = freq / freq.sum()

                    # Keep plot range manageable.
                    lo = int(max(0, np.quantile(counts, 0.005)))
                    hi = int(np.quantile(counts, 0.995))
                    hi = max(hi, lo + 1)

                    if hi - lo > 600:
                        hi = lo + 600

                    keep = (vals >= lo) & (vals <= hi)
                    for k, pr in zip(vals[keep], prob[keep]):
                        pmf_rows.append({
                            "layer": layer_labels.get(rho, f"rho={rho:g}"),
                            "rho": rho,
                            "count": int(k),
                            "probability": float(pr),
                            "genotype": genotype,
                            "time": float(time_value),
                        })

    return pd.DataFrame(rows), pd.DataFrame(pmf_rows)


# =============================================================================
# Plotting panels
# =============================================================================

def plot_panel_a(ax, model_manifest: pd.DataFrame, rho_values: Sequence[float]) -> None:
    ax.axis("off")
    add_panel_label(ax, "A")

    ax.text(
        0.5, 0.95,
        "Sensitivity-design overview",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    boxes = [
        (0.06, 0.63, 0.26, 0.22, "Posterior source", "Hierarchical OU\ncore model"),
        (0.37, 0.63, 0.26, 0.22, "Prior sensitivity", "Additional fits\nvia manifest"),
        (0.68, 0.63, 0.26, 0.22, "Count layer", "Binomial vs\nbeta-binomial"),
        (0.21, 0.27, 0.26, 0.22, "Parameter context", "Equilibrium and\nOU parameters"),
        (0.53, 0.27, 0.26, 0.22, "Robustness readout", "Interval width and\nPMF sensitivity"),
    ]

    for x, y, w, h, head, body in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.0,
            edgecolor="0.25",
            facecolor="0.96",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h * 0.68,
            head,
            ha="center", va="center",
            fontsize=8.0,
            fontweight="bold",
        )
        ax.text(
            x + w / 2, y + h * 0.32,
            body,
            ha="center", va="center",
            fontsize=7.5,
        )

    n_fits = len(model_manifest)
    rho_text = ", ".join([f"{float(r):g}" for r in rho_values])

    ax.text(
        0.5,
        0.08,
        f"Posterior fits available: {n_fits}; count-layer overdispersion ρ values: {rho_text}",
        ha="center",
        va="center",
        fontsize=8,
        color="0.25",
    )

def plot_panel_b(ax, param_df: pd.DataFrame) -> None:
    add_panel_label(ax, "B")
    ax.set_title("Genotype-specific equilibrium estimates", fontsize=10, fontweight="bold")

    if param_df.empty:
        ax.text(0.5, 0.5, "No parameter summaries available", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    d = param_df[param_df["parameter"].str.startswith("mu_bg")].copy()
    if d.empty:
        ax.text(0.5, 0.5, "mu_bg unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    model_order = list(pd.Series(d["model_label"]).drop_duplicates())
    genotype_order = [g for g in GENOTYPE_ORDER if any(f"[{g}]" in p for p in d["parameter"])]

    if not genotype_order:
        genotype_order = ["priA", "recG", "WT"]

    x_base = np.arange(len(genotype_order))
    width = 0.16 if len(model_order) > 1 else 0.32

    for m_idx, model in enumerate(model_order):
        offset = (m_idx - (len(model_order) - 1) / 2) * width
        for g_idx, genotype in enumerate(genotype_order):
            row = d[(d["model_label"] == model) & (d["parameter"].str.contains(rf"\[{re.escape(genotype)}\]", regex=True))]
            if row.empty:
                continue
            row = row.iloc[0]
            x = x_base[g_idx] + offset
            ax.errorbar(
                x,
                row["median"],
                yerr=[[row["median"] - row["q025"]], [row["q975"] - row["median"]]],
                marker="o",
                markersize=4.5,
                capsize=2.5,
                linewidth=1.1,
                color=genotype_color(genotype),
                alpha=0.85,
                label=f"{genotype}" if m_idx == 0 else None,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(genotype_order, fontsize=8)
    ax.set_ylabel(r"Posterior $\mu_{\mathrm{bg}}$ on log$_{10}$ scale", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="best")
    style_axis(ax)

    if len(model_order) == 1:
        ax.text(
            0.02,
            0.04,
            "Primary fit shown; additional prior-sensitivity fits can be added via manifest.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="0.25",
        )


def plot_panel_c(ax, param_df: pd.DataFrame) -> None:
    add_panel_label(ax, "C")
    ax.set_title("Global OU/hierarchical parameters", fontsize=10, fontweight="bold")

    if param_df.empty:
        ax.text(0.5, 0.5, "No parameter summaries available", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    params = ["theta", "tau_mu", "mu_hyper", "sigma_obs"]
    d = param_df[param_df["parameter"].isin(params)].copy()

    if d.empty:
        ax.text(0.5, 0.5, "Global parameters unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    model_order = list(pd.Series(d["model_label"]).drop_duplicates())
    param_order = [p for p in params if p in set(d["parameter"])]

    x_base = np.arange(len(param_order))
    width = 0.16 if len(model_order) > 1 else 0.30

    for m_idx, model in enumerate(model_order):
        offset = (m_idx - (len(model_order) - 1) / 2) * width
        model_rows = d[d["model_label"] == model]
        for p_idx, param in enumerate(param_order):
            row = model_rows[model_rows["parameter"] == param]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.errorbar(
                x_base[p_idx] + offset,
                row["median"],
                yerr=[[row["median"] - row["q025"]], [row["q975"] - row["median"]]],
                marker="o",
                markersize=4.5,
                capsize=2.5,
                linewidth=1.1,
                label=model if p_idx == 0 else None,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels([parameter_display_name(p) for p in param_order], fontsize=8)
    ax.set_ylabel("Posterior estimate", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="best")
    style_axis(ax)


def plot_panel_d(ax, count_df: pd.DataFrame) -> None:
    add_panel_label(ax, "D")
    ax.set_title("Effective-count calibration by count layer", fontsize=10, fontweight="bold")

    if count_df.empty or "covered90" not in count_df.columns:
        ax.text(0.5, 0.5, "Count-layer calibration unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    d = (
        count_df.groupby("layer", sort=False)["covered90"]
        .mean()
        .reset_index(name="coverage")
    )

    x = np.arange(len(d))
    ax.bar(x, d["coverage"], width=0.58, alpha=0.78)
    ax.axhline(0.90, color="0.25", linestyle="--", linewidth=1.0, label="Nominal 90%")

    for i, row in d.iterrows():
        ax.text(
            i,
            min(float(row["coverage"]) + 0.035, 1.02),
            f"{row['coverage']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(d["layer"], fontsize=7, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction of genotype-time panels\nwith implied mean inside 90% interval", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style_axis(ax)

def plot_panel_e(ax, count_df: pd.DataFrame) -> None:
    add_panel_label(ax, "E")
    ax.set_title("Count-layer interval-width sensitivity", fontsize=10, fontweight="bold")

    if count_df.empty:
        ax.text(0.5, 0.5, "Count-layer sensitivity unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    d = (
        count_df.groupby("layer", sort=False)["interval_width_90"]
        .agg(["mean", "median"])
        .reset_index()
    )

    x = np.arange(len(d))
    ax.bar(x, d["mean"], width=0.58, alpha=0.78)

    for i, row in d.iterrows():
        ax.text(
            i,
            row["mean"],
            f"{row['mean']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(d["layer"], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Mean 90% count-interval width", fontsize=9)
    style_axis(ax)

def plot_panel_f(ax, pmf_df: pd.DataFrame, count_df: pd.DataFrame) -> None:
    add_panel_label(ax, "F")
    ax.set_title("Representative PMF sensitivity: priA late time", fontsize=10, fontweight="bold")

    if pmf_df.empty:
        ax.text(0.5, 0.5, "PMF sensitivity unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    layers = list(pd.Series(pmf_df["layer"]).drop_duplicates())

    for layer in layers:
        d = pmf_df[pmf_df["layer"] == layer].sort_values("count").copy()
        # Avoid exact zeros on semilog display.
        d = d[d["probability"] > 0]
        ax.plot(
            d["count"],
            d["probability"],
            linewidth=1.5,
            label=layer,
            alpha=0.9,
        )

    pri_late = count_df[(count_df["genotype"] == "priA") & (count_df["time"] == count_df["time"].max())]
    if not pri_late.empty:
        obs = float(pri_late.iloc[0]["obs_count_mean"])
        if np.isfinite(obs):
            ax.axvline(obs, color="black", linestyle="--", linewidth=1.1, label="Observed/implied mean")

    ax.set_yscale("log")
    ax.set_xlabel("Mutation count K", fontsize=9)
    ax.set_ylabel("Posterior predictive probability", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="best")
    style_axis(ax)

def make_figure(
    manifest: pd.DataFrame,
    param_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    count_df: pd.DataFrame,
    pmf_df: pd.DataFrame,
    rho_values: Sequence[float],
    output_dir: Path,
    dpi: int = 600,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12.8, 8.8),
        constrained_layout=False,
    )

    plt.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.10,
        wspace=0.30,
        hspace=0.43,
    )

    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]
    ax_e, ax_f = axes[2, 0], axes[2, 1]

    plot_panel_a(ax_a, manifest, rho_values)
    plot_panel_b(ax_b, param_df)
    plot_panel_c(ax_c, param_df)
    plot_panel_d(ax_d, count_df)
    plot_panel_e(ax_e, count_df)
    plot_panel_f(ax_f, pmf_df, count_df)

    fig.suptitle(
        "Supplementary Fig. S4. Sensitivity analysis for count-layer assumptions with posterior-parameter context.",
        fontsize=13,
        fontweight="bold",
        y=0.975,
    )

    fig.text(
        0.5,
        0.025,
        "Sensitivity is evaluated on the effective count scale because explicit count/trial columns were not detected in the input dataset.",
        ha="center",
        va="center",
        fontsize=8,
        color="0.25",
    )

    png_path = output_dir / f"{FIG_BASENAME}.png"
    pdf_path = output_dir / f"{FIG_BASENAME}.pdf"

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


# =============================================================================
# Main
# =============================================================================

def parse_rhos(text: str) -> List[float]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if 0.0 not in vals:
        vals = [0.0] + vals
    return vals


def parse_times(text: Optional[str], data: pd.DataFrame) -> List[float]:
    if text:
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    times = sorted([float(x) for x in data["time"].dropna().unique()])
    if len(times) >= 3:
        return [times[1], times[len(times) // 2], times[-1]]
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Fig. S4 sensitivity analysis."
    )
    parser.add_argument(
        "--idata",
        type=str,
        default=None,
        help="Primary trace_core.nc. If omitted, auto-detects /Figure_2/trace_core.nc.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Mutation-frequency CSV. If omitted, auto-detects Figure_2/mut_freq_data.csv.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional CSV manifest with model_label,idata_path,model_class.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )
    parser.add_argument(
        "--default-trials",
        type=int,
        default=1_000_000,
        help="Effective binomial trial size when K/N columns are absent.",
    )
    parser.add_argument(
        "--rho-values",
        type=str,
        default="0,1e-7,1e-6",
        help="Comma-separated beta-binomial overdispersion rho values. Include 0 for binomial.",
    )
    parser.add_argument(
        "--times",
        type=str,
        default="6,15,24",
        help="Comma-separated time points for count-layer sensitivity.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12000,
        help="Maximum posterior predictive samples per genotype/time panel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260613,
        help="Random seed for count-layer simulations.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution.",
    )
    return parser.parse_args()


def main() -> None:
    require_arviz()

    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    explicit_idata = Path(args.idata).expanduser().resolve() if args.idata else None
    explicit_data = Path(args.data).expanduser().resolve() if args.data else None
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None

    idata_path = detect_idata_file(explicit_idata)
    data_path = detect_data_file(explicit_data)

    raw = read_csv_lenient(data_path)
    data, colmap = prepare_dataset(raw)

    idata = az.from_netcdf(idata_path)

    manifest = load_manifest(manifest_path, idata_path)
    rho_values = parse_rhos(args.rho_values)
    times = parse_times(args.times, data)

    param_df = posterior_parameter_summary(manifest, data)
    coverage_df = posterior_predictive_coverage_summary(manifest, data)

    count_df, pmf_df = count_layer_sensitivity_summary(
        idata=idata,
        idata_path=idata_path,
        data=data,
        selected_times=times,
        default_trials=args.default_trials,
        rho_values=rho_values,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    param_path = output_dir / "supp_figure_s4_parameter_sensitivity.csv"
    coverage_path = output_dir / "supp_figure_s4_coverage_sensitivity.csv"
    count_path = output_dir / "supp_figure_s4_count_layer_sensitivity.csv"
    pmf_path = output_dir / "supp_figure_s4_representative_pmf_sensitivity.csv"
    manifest_path_out = output_dir / "supp_figure_s4_model_manifest_used.csv"

    param_df.to_csv(param_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    count_df.to_csv(count_path, index=False)
    pmf_df.to_csv(pmf_path, index=False)
    manifest.to_csv(manifest_path_out, index=False)

    png_path, pdf_path = make_figure(
        manifest=manifest,
        param_df=param_df,
        coverage_df=coverage_df,
        count_df=count_df,
        pmf_df=pmf_df,
        rho_values=rho_values,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    print("\n========== Supplementary Fig. S4 sensitivity summary ==========")
    print(f"Primary InferenceData : {idata_path}")
    print(f"Mutation data         : {data_path}")
    print(f"Models in manifest    : {len(manifest)}")
    print(f"Count-layer rho values: {rho_values}")
    print(f"Count-layer times     : {times}")
    print("Detected data columns:")
    for k, v in colmap.items():
        print(f"  {k:20s}: {v}")
    if not coverage_df.empty:
        print("\nCoverage summary:")
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print(coverage_df[["model_label", "coverage", "mean_interval_width", "n_observations", "ppc_var"]].to_string(index=False))
    if not count_df.empty:
        print("\nCount-layer interval-width summary:")
        tmp = count_df.groupby("layer", sort=False)["interval_width_90"].agg(["mean", "median"]).reset_index()
        with pd.option_context("display.max_columns", None, "display.width", 140):
            print(tmp.to_string(index=False))
    print("===============================================================\n")

    print("[DONE] Saved Supplementary Fig. S4 outputs:")
    print(f"  PNG             : {png_path}")
    print(f"  PDF             : {pdf_path}")
    print(f"  Parameter CSV   : {param_path}")
    print(f"  Coverage CSV    : {coverage_path}")
    print(f"  Count-layer CSV : {count_path}")
    print(f"  PMF CSV         : {pmf_path}")
    print(f"  Manifest used   : {manifest_path_out}")


if __name__ == "__main__":
    main()
