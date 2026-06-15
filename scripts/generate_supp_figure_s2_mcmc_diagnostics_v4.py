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
from matplotlib import gridspec


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

PROJECT_ROOT = Path("/Fig_S2")

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "Fig_S2"
)

DEFAULT_TRACE_CORE = (
    PROJECT_ROOT
    / "Figure_2"
    / "trace_core.nc"
)

FIG_BASENAME = "supp_figure_s2_mcmc_diagnostics"

CORE_VAR_NAMES = [
    "mu_bg",
    "mu_hyper",
    "tau_mu",
    "theta",
    "sigma_bg",
    "sigma_obs",
]

GLOBAL_VARS = [
    "mu_hyper",
    "tau_mu",
    "theta",
    "sigma_obs",
]

GENOTYPE_VARS = [
    "mu_bg",
    "sigma_bg",
]

CHAIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]


# =============================================================================
# Helpers
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


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.10,
        1.10,
        label,
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
    """Clean display names while avoiding regex backslash issues."""
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


def load_bg_categories(idata_path: Path) -> Optional[List[str]]:
    """
    Load genotype/background categories if available.

    The core script usually saves bg_categories.npy in the same folder as trace_core.nc.
    """
    candidates = [
        idata_path.parent / "bg_categories.npy",
        DEFAULT_TRACE_CORE.parent / "bg_categories.npy",
        PROJECT_ROOT / "Methodology" / "bg_categories.npy",
    ]

    for p in candidates:
        if p.exists():
            try:
                arr = np.load(p, allow_pickle=True)
                labels = [str(x) for x in arr.tolist()]
                # Standardize WT label.
                labels = ["WT" if x.lower() == "wt" else x for x in labels]
                return labels
            except Exception:
                pass

    return None


def detect_idata_file(explicit_idata: Optional[Path]) -> Path:
    """
    Prefer the core hierarchical OU trace rather than benchmark baseline posteriors.
    """
    if explicit_idata is not None:
        if not explicit_idata.exists():
            raise FileNotFoundError(f"Explicit InferenceData file does not exist: {explicit_idata}")
        return explicit_idata

    preferred = [
        DEFAULT_TRACE_CORE,
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
        candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
        print(f"[INFO] Auto-detected core InferenceData file: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "Could not find trace_core.nc automatically.\n"
        "Please run with:\n"
        "    python generate_supp_figure_s2_mcmc_diagnostics.py "
        "--idata /full/path/to/trace_core.nc"
    )


def load_idata(path: Path):
    require_arviz()
    return az.from_netcdf(path)


def available_core_vars(idata) -> List[str]:
    names = list(idata.posterior.data_vars)
    return [v for v in CORE_VAR_NAMES if v in names]


def sample_sizes(idata) -> Tuple[int, int]:
    posterior = idata.posterior
    n_chains = int(posterior.sizes.get("chain", 0))
    n_draws = int(posterior.sizes.get("draw", 0))
    return n_chains, n_draws


def scalar_components(
    idata,
    var_names: Sequence[str],
    bg_labels: Optional[List[str]] = None,
    max_components: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract scalar chain x draw arrays from selected variables.
    """
    out: Dict[str, np.ndarray] = {}

    for var in var_names:
        if var not in idata.posterior:
            continue

        arr = idata.posterior[var].transpose("chain", "draw", ...)
        values = np.asarray(arr.values)

        if values.ndim == 2:
            out[var] = values
        else:
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
                if var in {"mu_bg", "sigma_bg"} and bg_labels is not None and idx < len(bg_labels):
                    label = f"{var}[{bg_labels[idx]}]"
                elif idx < len(coord_labels):
                    label = f"{var}[{coord_labels[idx]}]"
                else:
                    label = f"{var}[{idx}]"
                out[label] = flat[:, :, idx]

        if max_components is not None and len(out) >= max_components:
            return dict(list(out.items())[:max_components])

    if max_components is not None:
        return dict(list(out.items())[:max_components])
    return out


def compute_summary(idata, var_names: Sequence[str]) -> pd.DataFrame:
    require_arviz()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = az.summary(
            idata,
            var_names=list(var_names),
            round_to=None,
            kind="all",
        )

    rename_map = {}
    for col in summary.columns:
        norm = normalize_name(col)
        if norm in {"r_hat", "rhat"}:
            rename_map[col] = "r_hat"
        elif norm in {"ess_bulk", "bulk_ess"}:
            rename_map[col] = "ess_bulk"
        elif norm in {"ess_tail", "tail_ess"}:
            rename_map[col] = "ess_tail"

    return summary.rename(columns=rename_map)


def divergence_count(idata) -> Optional[int]:
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        return int(np.asarray(idata.sample_stats["diverging"].values).sum())
    return None


# =============================================================================
# Plotting
# =============================================================================

def plot_trace_panel(ax, components: Dict[str, np.ndarray], title: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")

    if not components:
        ax.text(0.5, 0.5, "No matching parameters", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    n = len(components)
    offsets = np.linspace(0, n - 1, n)[::-1] * 3.2

    for idx, (label, values) in enumerate(components.items()):
        flat = values.reshape(-1)
        mu = np.nanmean(flat)
        sd = np.nanstd(flat)
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0

        offset = offsets[idx]

        for chain in range(values.shape[0]):
            color = CHAIN_COLORS[chain % len(CHAIN_COLORS)]
            y = (values[chain, :] - mu) / sd + offset
            ax.plot(np.arange(values.shape[1]), y, linewidth=0.55, alpha=0.85, color=color)

        ax.text(
            -0.02,
            offset,
            parameter_display_name(label),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.2,
        )

    ax.set_xlabel("Draw", fontsize=9)
    ax.set_ylabel("")
    ax.set_yticks([])
    style_axis(ax)

    n_chains = next(iter(components.values())).shape[0]
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
            lw=1.2,
            label=f"Chain {c + 1}",
        )
        for c in range(n_chains)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper right", ncol=min(n_chains, 4))


def plot_rhat_panel(ax, summary: pd.DataFrame, div_count: Optional[int]) -> None:
    ax.set_title("R-hat across monitored parameters", fontsize=10, fontweight="bold")

    vals = pd.to_numeric(summary.get("r_hat", pd.Series(dtype=float)), errors="coerce")
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

    if vals.empty:
        ax.text(0.5, 0.5, "R-hat unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    # Tight bins around well-converged values.
    x_min = min(0.995, float(vals.min()) - 0.002)
    x_max = max(1.010, float(vals.max()) + 0.002)
    bins = np.linspace(x_min, x_max, 18)

    ax.hist(vals, bins=bins, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(1.01, color="0.25", linestyle="--", linewidth=1.0)
    ax.axvline(1.05, color="0.45", linestyle=":", linewidth=1.0)

    div_text = "NA" if div_count is None else str(div_count)
    ax.text(
        0.98,
        0.92,
        f"median = {np.nanmedian(vals):.3f}\nmax = {np.nanmax(vals):.3f}\nN = {len(vals)}\ndivergences = {div_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.78", alpha=0.9),
    )

    ax.set_xlabel("R-hat", fontsize=9)
    ax.set_ylabel("Number of scalar parameters", fontsize=9)
    style_axis(ax)


def plot_ess_panel(ax, summary: pd.DataFrame) -> None:
    ax.set_title("Effective sample size", fontsize=10, fontweight="bold")

    ess_data = []
    labels = []

    for col, label in [("ess_bulk", "Bulk ESS"), ("ess_tail", "Tail ESS")]:
        if col in summary.columns:
            vals = pd.to_numeric(summary[col], errors="coerce")
            vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                ess_data.append(vals.values)
                labels.append(label)

    if not ess_data:
        ax.text(0.5, 0.5, "ESS unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    parts = ax.violinplot(
        ess_data,
        positions=np.arange(1, len(ess_data) + 1),
        widths=0.62,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor("0.72")
        body.set_edgecolor("0.25")
        body.set_alpha(0.45)
        body.set_linewidth(0.8)

    ax.boxplot(
        ess_data,
        positions=np.arange(1, len(ess_data) + 1),
        widths=0.25,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 1.2, "color": "black"},
        boxprops={"linewidth": 0.9, "facecolor": "white", "alpha": 0.85},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )

    for i, vals in enumerate(ess_data, start=1):
        ax.text(
            i,
            np.nanmax(vals),
            f"median {np.nanmedian(vals):.0f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ESS", fontsize=9)
    style_axis(ax)


def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        return np.zeros_like(grid)

    sd = np.std(values, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0

    n = len(values)
    bandwidth = 1.06 * sd * n ** (-1 / 5)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = sd / 2 if sd > 0 else 1.0

    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    return density


def plot_density_panel(ax, components: Dict[str, np.ndarray]) -> None:
    ax.set_title("Posterior density overlays by chain", fontsize=10, fontweight="bold")

    if not components:
        ax.text(0.5, 0.5, "No matching parameters", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    n = len(components)
    offsets = np.linspace(0, n - 1, n)[::-1] * 1.15

    for idx, (label, values) in enumerate(components.items()):
        flat = values.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if len(flat) < 3:
            continue

        lo, hi = np.nanpercentile(flat, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = np.nanmin(flat), np.nanmax(flat)
        if lo == hi:
            lo -= 1
            hi += 1

        grid = np.linspace(lo, hi, 200)
        offset = offsets[idx]

        densities = []
        max_density = 0.0
        for chain in range(values.shape[0]):
            dens = kde_curve(values[chain, :], grid)
            densities.append(dens)
            max_density = max(max_density, float(np.nanmax(dens)) if len(dens) else 0.0)

        if max_density <= 0 or not np.isfinite(max_density):
            max_density = 1.0

        for chain, dens in enumerate(densities):
            color = CHAIN_COLORS[chain % len(CHAIN_COLORS)]
            y = dens / max_density * 0.85 + offset
            ax.plot(grid, y, linewidth=1.0, alpha=0.9, color=color)

        ax.text(
            0.01,
            offset + 0.20,
            parameter_display_name(label),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7.2,
        )

    ax.set_xlabel("Parameter value", fontsize=9)
    ax.set_yticks([])
    style_axis(ax)


def plot_rank_panel(ax, components: Dict[str, np.ndarray]) -> None:
    ax.set_title("Rank-normalized chain diagnostics", fontsize=10, fontweight="bold")

    if not components:
        ax.text(0.5, 0.5, "No matching parameters", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    n_bins = 20
    rows = []
    row_labels = []

    for label, values in components.items():
        chains, draws = values.shape
        flat = values.reshape(-1)
        if np.nanstd(flat) == 0 or not np.isfinite(flat).all():
            continue

        order = np.argsort(np.argsort(flat))
        ranks = order.reshape(chains, draws) / max(len(flat) - 1, 1)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        expected = draws / n_bins

        for chain in range(chains):
            counts, _ = np.histogram(ranks[chain, :], bins=bin_edges)
            rel_dev = (counts - expected) / max(expected, 1)
            rows.append(rel_dev)
            row_labels.append(f"{parameter_display_name(label)} | C{chain + 1}")

    if not rows:
        ax.text(0.5, 0.5, "Rank diagnostics unavailable", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    mat = np.asarray(rows)
    vmax = np.nanpercentile(np.abs(mat), 95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.5
    vmax = max(vmax, 0.25)

    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[0, 1, len(rows), 0],
    )

    ax.set_xlabel("Rank quantile", fontsize=9)
    ax.set_ylabel("Parameter | chain", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))

    if len(row_labels) <= 16:
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=6.5)
    else:
        ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Relative deviation", fontsize=8)

    style_axis(ax)


# =============================================================================
# Main figure
# =============================================================================

def make_figure(
    idata,
    summary: pd.DataFrame,
    output_dir: Path,
    idata_path: Path,
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

    bg_labels = load_bg_categories(idata_path)

    global_components = scalar_components(
        idata,
        [v for v in GLOBAL_VARS if v in idata.posterior],
        bg_labels=bg_labels,
        max_components=4,
    )

    genotype_components = scalar_components(
        idata,
        [v for v in GENOTYPE_VARS if v in idata.posterior],
        bg_labels=bg_labels,
        max_components=6,
    )

    density_components = scalar_components(
        idata,
        ["theta", "mu_bg", "sigma_bg", "sigma_obs"],
        bg_labels=bg_labels,
        max_components=5,
    )

    rank_components = scalar_components(
        idata,
        ["theta", "mu_bg", "sigma_bg", "sigma_obs"],
        bg_labels=bg_labels,
        max_components=4,
    )

    fig = plt.figure(figsize=(12.8, 8.8), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.30,
        hspace=0.44,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    add_panel_label(ax_a, "A")
    plot_trace_panel(ax_a, global_components, "Trace plots: global OU parameters")

    add_panel_label(ax_b, "B")
    plot_trace_panel(ax_b, genotype_components, "Trace plots: genotype-specific parameters")

    add_panel_label(ax_c, "C")
    plot_rhat_panel(ax_c, summary, divergence_count(idata))

    add_panel_label(ax_d, "D")
    plot_ess_panel(ax_d, summary)

    add_panel_label(ax_e, "E")
    plot_density_panel(ax_e, density_components)

    add_panel_label(ax_f, "F")
    plot_rank_panel(ax_f, rank_components)

    fig.suptitle(
        "Supplementary Fig. S2. MCMC diagnostics for the hierarchical OU model",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    png_path = output_dir / f"{FIG_BASENAME}.png"
    pdf_path = output_dir / f"{FIG_BASENAME}.pdf"

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def print_summary(idata, summary: pd.DataFrame, idata_path: Path) -> None:
    n_chains, n_draws = sample_sizes(idata)
    div_count = divergence_count(idata)

    print("\n========== Supplementary Fig. S2 MCMC summary ==========")
    print(f"InferenceData file: {idata_path}")
    print(f"Chains           : {n_chains}")
    print(f"Draws per chain  : {n_draws}")
    print(f"Posterior vars   : {list(idata.posterior.data_vars)}")
    print(f"Monitored vars   : {available_core_vars(idata)}")
    print(f"Summary scalars  : {len(summary)}")
    print(f"Divergences      : {div_count if div_count is not None else 'NA'}")

    if "r_hat" in summary.columns:
        vals = pd.to_numeric(summary["r_hat"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            print(f"R-hat median/max : {np.nanmedian(vals):.4f} / {np.nanmax(vals):.4f}")

    if "ess_bulk" in summary.columns:
        vals = pd.to_numeric(summary["ess_bulk"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            print(f"Bulk ESS median  : {np.nanmedian(vals):.1f}")

    if "ess_tail" in summary.columns:
        vals = pd.to_numeric(summary["ess_tail"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not vals.empty:
            print(f"Tail ESS median  : {np.nanmedian(vals):.1f}")

    print("========================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Supplementary Fig. S2 MCMC diagnostics.")
    parser.add_argument(
        "--idata",
        type=str,
        default=None,
        help="Path to trace_core.nc. If omitted, the script searches for the core trace.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for PNG/PDF/CSV.",
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

    idata_path = detect_idata_file(explicit_idata)
    idata = load_idata(idata_path)

    monitored_vars = available_core_vars(idata)
    if not monitored_vars:
        raise ValueError(
            "None of the expected core variables were found.\n"
            f"Expected any of: {CORE_VAR_NAMES}\n"
            f"Available: {list(idata.posterior.data_vars)}"
        )

    summary = compute_summary(idata, monitored_vars)

    print_summary(idata, summary, idata_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "supp_figure_s2_arviz_summary.csv"
    summary.to_csv(summary_path)

    png_path, pdf_path = make_figure(
        idata=idata,
        summary=summary,
        output_dir=output_dir,
        idata_path=idata_path,
        dpi=args.dpi,
    )

    print("[DONE] Saved Supplementary Fig. S2 outputs:")
    print(f"  PNG          : {png_path}")
    print(f"  PDF          : {pdf_path}")
    print(f"  ArviZ summary: {summary_path}")


if __name__ == "__main__":
    main()
