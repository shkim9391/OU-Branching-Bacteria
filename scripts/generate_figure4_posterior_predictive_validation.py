from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

DEFAULT_BASE_DIR = Path(
    "/Figure_4"
)
BASE_DIR = Path(os.environ.get("OU_FIG4_DIR", DEFAULT_BASE_DIR))

DATA_PATH = BASE_DIR / "mut_freq_data.csv"
TRACE_PATH = BASE_DIR / "trace_core.nc"
BG_PATH = BASE_DIR / "bg_categories.npy"
OUT_PREFIX = BASE_DIR / "Figure4_posterior_predictive_validation_real"

RANDOM_SEED = 123
N_DENSITY_DRAWS = 150

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

DISPLAY_NAME = {"priA": r"$priA$", "recG": r"$recG$", "wt": "WT", "WT": "WT"}
COLOR_MAP = {"priA": "#d62728", "recG": "#7f7f7f", "wt": "#ff7f0e", "WT": "#ff7f0e"}


def normalize_bg(x):
    s = str(x).strip()
    low = s.lower()
    if low == "wt":
        return "wt"
    if low == "pria":
        return "priA"
    if low == "recg":
        return "recG"
    return s


def display_bg(x):
    return DISPLAY_NAME.get(str(x), str(x))


def add_panel_label(ax, label):
    ax.text(-0.13, 1.08, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def simple_kde_on_grid(values, grid, bw=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.zeros_like(grid)
    sd = np.std(values, ddof=1)
    if sd <= 0:
        sd = 1e-6
    if bw is None:
        bw = 1.06 * sd * len(values) ** (-1 / 5)
    bw = max(bw, 1e-6)
    z = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))


def density_grid(*arrays, n=450, pad_frac=0.08):
    vals = np.concatenate([np.asarray(a).reshape(-1) for a in arrays])
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    pad = pad_frac * (hi - lo + 1e-9)
    return np.linspace(lo - pad, hi + pad, n)


def load_data_and_ppc():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Missing trace file: {TRACE_PATH}")
    if not BG_PATH.exists():
        raise FileNotFoundError(f"Missing background metadata file: {BG_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df[["background", "replicate", "t", "x"]].copy()
    df["background"] = df["background"].map(normalize_bg)
    df["replicate"] = df["replicate"].astype(int)
    df["t"] = df["t"].astype(float)
    df["x"] = df["x"].astype(float)
    df["Y_obs"] = np.log10(df["x"] + 1e-9)

    bg_order = np.load(BG_PATH, allow_pickle=True).astype(str).tolist()
    bg_order = [normalize_bg(b) for b in bg_order]

    bg_to_idx = {bg: i for i, bg in enumerate(bg_order)}
    df["bg_idx"] = df["background"].map(bg_to_idx)
    df = df.sort_values(["bg_idx", "replicate", "t"]).reset_index(drop=True)

    y_table = (
        df.pivot_table(index=["bg_idx", "replicate"], columns="t", values="Y_obs")
        .sort_index(axis=1)
    )
    original_times = y_table.columns.values.astype(float)
    n_bg = len(bg_order)
    reps_per_bg = df.groupby("bg_idx")["replicate"].nunique().iloc[0]
    n_time = len(original_times)
    y_obs = y_table.to_numpy().reshape(n_bg, reps_per_bg, n_time)
    y_target = y_obs[:, :, 1:]

    idata = az.from_netcdf(TRACE_PATH)
    if "posterior_predictive" not in idata.groups():
        raise ValueError("trace_core.nc does not contain posterior_predictive group.")
    if "Y_transition" not in idata.posterior_predictive:
        raise ValueError("posterior_predictive does not contain Y_transition.")

    y_ppc = idata.posterior_predictive["Y_transition"].values
    y_ppc = y_ppc.reshape(-1, *y_ppc.shape[-3:])

    return df, bg_order, original_times[1:], y_target, y_ppc, idata


def flatten_by_bg(y_array, bg_idx):
    if y_array.ndim == 3:
        return y_array[bg_idx, :, :].reshape(-1)
    if y_array.ndim == 4:
        return y_array[:, bg_idx, :, :].reshape(y_array.shape[0], -1)
    raise ValueError("Unsupported array dimension.")


def panel_a_overall_ppc(ax, y_target, y_ppc):
    obs = y_target.reshape(-1)
    pred = y_ppc.reshape(y_ppc.shape[0], -1)
    grid = density_grid(obs, pred)
    obs_dens = simple_kde_on_grid(obs, grid)

    rng = np.random.default_rng(RANDOM_SEED)
    use = rng.choice(np.arange(pred.shape[0]), size=min(N_DENSITY_DRAWS, pred.shape[0]), replace=False)

    densities = []
    for idx in use:
        dens = simple_kde_on_grid(pred[idx], grid)
        densities.append(dens)
        ax.plot(grid, dens, color="#1f77b4", alpha=0.05, lw=0.8)

    mean_dens = np.asarray(densities).mean(axis=0)
    ax.plot(grid, obs_dens, color="black", lw=2.0, label="Observed")
    ax.plot(grid, mean_dens, color="#1f77b4", lw=2.0, label="PPC mean")

    ax.set_title("Overall posterior predictive distribution", fontweight="bold")
    ax.set_xlabel(r"$\log_{10}$ mutation frequency")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")


def panel_b_genotype_stratified(ax, y_target, y_ppc, bg_order):
    for b_idx, bg in enumerate(bg_order):
        color = COLOR_MAP[bg]
        obs = flatten_by_bg(y_target, b_idx)
        pred = flatten_by_bg(y_ppc, b_idx)

        grid = density_grid(obs, pred)
        obs_dens = simple_kde_on_grid(obs, grid)
        pred_dens = simple_kde_on_grid(pred.reshape(-1), grid)

        ax.plot(grid, obs_dens, color=color, lw=1.8, label=f"{display_bg(bg)} observed")
        ax.plot(grid, pred_dens, color=color, lw=1.8, ls="--", label=f"{display_bg(bg)} PPC")

    ax.set_title("Genotype-stratified predictive checks", fontweight="bold")
    ax.set_xlabel(r"$\log_{10}$ mutation frequency")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.7, ncol=2, loc="upper right")


def panel_c_uncertainty_band(ax, y_target, y_ppc):
    obs = y_target.reshape(-1)
    pred = y_ppc.reshape(y_ppc.shape[0], -1)
    grid = density_grid(obs, pred)
    obs_dens = simple_kde_on_grid(obs, grid)

    rng = np.random.default_rng(RANDOM_SEED)
    use = rng.choice(np.arange(pred.shape[0]), size=min(500, pred.shape[0]), replace=False)
    densities = np.asarray([simple_kde_on_grid(pred[idx], grid) for idx in use])

    mean_dens = densities.mean(axis=0)
    lo, hi = np.percentile(densities, [2.5, 97.5], axis=0)
    lo50, hi50 = np.percentile(densities, [25, 75], axis=0)

    ax.fill_between(grid, lo, hi, color="#1f77b4", alpha=0.16, label="95% PPC band")
    ax.fill_between(grid, lo50, hi50, color="#1f77b4", alpha=0.26, label="50% PPC band")
    ax.plot(grid, mean_dens, color="#1f77b4", lw=2.0, label="PPC mean")
    ax.plot(grid, obs_dens, color="black", lw=2.0, label="Observed")

    ax.set_title("Posterior predictive density uncertainty", fontweight="bold")
    ax.set_xlabel(r"$\log_{10}$ mutation frequency")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.0, loc="best")


def panel_d_quantile_agreement(ax, y_target, y_ppc):
    obs = y_target.reshape(-1)
    pred = y_ppc.reshape(y_ppc.shape[0], -1)

    probs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    # Observed quantiles: shape = (n_quantiles,)
    obs_q = np.quantile(obs, probs)

    # Posterior predictive quantiles.
    # np.quantile(..., axis=1) returns shape = (n_quantiles, n_draws),
    # so transpose to obtain shape = (n_draws, n_quantiles).
    pred_q = np.quantile(pred, probs, axis=1).T

    pred_q_mean = pred_q.mean(axis=0)
    pred_q_lo, pred_q_hi = np.percentile(pred_q, [2.5, 97.5], axis=0)

    min_val = min(obs_q.min(), pred_q_lo.min())
    max_val = max(obs_q.max(), pred_q_hi.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        lw=1.0,
        ls="--",
        alpha=0.65,
    )

    ax.errorbar(
        obs_q,
        pred_q_mean,
        yerr=[pred_q_mean - pred_q_lo, pred_q_hi - pred_q_mean],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.2,
        capsize=3,
        markersize=5,
    )

    for x, y, p in zip(obs_q, pred_q_mean, probs):
        ax.text(
            x,
            y,
            f"{int(p * 100)}",
            fontsize=6.5,
            ha="left",
            va="bottom",
        )

    ax.set_title("Observed versus predictive quantiles", fontweight="bold")
    ax.set_xlabel("Observed quantile")
    ax.set_ylabel("Posterior predictive quantile")
    ax.set_xlim(min_val - 0.2, max_val + 0.2)
    ax.set_ylim(min_val - 0.2, max_val + 0.2)
    ax.grid(alpha=0.25)

    ax.text(
        0.04,
        0.94,
        "points: posterior predictive mean quantiles\nbars: 95% posterior predictive interval",
        transform=ax.transAxes,
        fontsize=6.8,
        ha="left",
        va="top",
    )


def stat_dict(vals):
    return {
        "mean": np.mean(vals),
        "sd": np.std(vals, ddof=1),
        "q05": np.quantile(vals, 0.05),
        "q50": np.quantile(vals, 0.50),
        "q95": np.quantile(vals, 0.95),
    }


def summarize_ppc(y_target, y_ppc, bg_order):
    rows = []
    obs = y_target.reshape(-1)
    pred_flat = y_ppc.reshape(y_ppc.shape[0], -1)
    groups = [("overall", obs, pred_flat)]

    for b_idx, bg in enumerate(bg_order):
        groups.append((bg, flatten_by_bg(y_target, b_idx), flatten_by_bg(y_ppc, b_idx)))

    for group, obs_vals, pred_vals in groups:
        obs_stats = stat_dict(obs_vals)
        pred_stats = pd.DataFrame([stat_dict(pred_vals[i]) for i in range(pred_vals.shape[0])])
        for stat in ["mean", "sd", "q05", "q50", "q95"]:
            rows.append({
                "group": group,
                "statistic": stat,
                "observed": obs_stats[stat],
                "ppc_mean": pred_stats[stat].mean(),
                "ppc_q2.5": pred_stats[stat].quantile(0.025),
                "ppc_q97.5": pred_stats[stat].quantile(0.975),
            })
    return pd.DataFrame(rows)


def main():
    print(f"Using BASE_DIR: {BASE_DIR}")
    print(f"Reading posterior predictive from: {TRACE_PATH}")

    df, bg_order, target_times, y_target, y_ppc, idata = load_data_and_ppc()

    print("Background order:", bg_order)
    print("Posterior predictive shape:", y_ppc.shape)
    print("Observed transition target shape:", y_target.shape)
    print("Target times:", target_times)

    summary = summarize_ppc(y_target, y_ppc, bg_order)
    summary_path = BASE_DIR / "figure4_ppc_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nPPC summary:")
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5))
    axA, axB, axC, axD = axes.ravel()

    panel_a_overall_ppc(axA, y_target, y_ppc)
    panel_b_genotype_stratified(axB, y_target, y_ppc, bg_order)
    panel_c_uncertainty_band(axC, y_target, y_ppc)
    panel_d_quantile_agreement(axD, y_target, y_ppc)

    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Posterior predictive validation of mutation-frequency distributions",
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
    print(summary_path)


if __name__ == "__main__":
    main()
