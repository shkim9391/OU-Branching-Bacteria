from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az


# =============================================================================
# 0. Paths and user settings
# =============================================================================

DEFAULT_BASE_DIR = Path(
    "/Figure_3"
)

# If Figure 3 is stored in the same folder as Figure 2, use:
# export OU_FIG3_DIR="/Figure_2"
BASE_DIR = Path(os.environ.get("OU_FIG3_DIR", DEFAULT_BASE_DIR))

DATA_PATH = BASE_DIR / "mut_freq_data.csv"
TRACE_PATH = BASE_DIR / "trace_core.nc"
TIMES_PATH = BASE_DIR / "times.npy"
BG_PATH = BASE_DIR / "bg_categories.npy"

OUT_PREFIX = BASE_DIR / "Figure3_OU_branching_count_layer_real"

# Current Figure 2 trace is on log10 mutation-frequency scale.
# Use "power10" for p(t)=10^Z_t. Use "inverse_logit" only if the OU model was fit on logit scale.
PROBABILITY_LINK = "power10"

# Effective number of mutation opportunities/trials for the count layer.
# For visualization, 1e6 makes rare mutation-frequency probabilities interpretable as expected counts.
# Change this to the true assay denominator if known.
N_EFFECTIVE = 1_000_000

# Number of posterior paths used for visualization.
N_POSTERIOR_PATHS = 1500

RANDOM_SEED = 123


# =============================================================================
# 1. Style
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

DISPLAY_NAME = {"priA": r"$priA$", "recG": r"$recG$", "wt": "WT", "WT": "WT"}

COLOR_MAP = {
    "priA": "#d62728",  # red
    "recG": "#7f7f7f",  # gray
    "wt":   "#ff7f0e",  # orange
    "WT":   "#ff7f0e",
}


# =============================================================================
# 2. Helper functions
# =============================================================================

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
    ax.text(
        -0.13, 1.08, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def simple_kde(values, gridsize=350):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.array([]), np.array([])
    sd = np.std(values, ddof=1)
    if sd <= 0:
        sd = 1e-6
    bw = max(1.06 * sd * len(values) ** (-1 / 5), 1e-9)
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = 0.20 * (hi - lo + 1e-12)
    xs = np.linspace(lo - pad, hi + pad, gridsize)
    z = (xs[:, None] - values[None, :]) / bw
    ys = np.exp(-0.5 * z ** 2).sum(axis=1) / (
        len(values) * bw * np.sqrt(2 * np.pi)
    )
    return xs, ys


def draw_kde(ax, values, label, color, lw=1.8):
    xs, ys = simple_kde(values)
    if len(xs) == 0:
        return
    ax.plot(xs, ys, lw=lw, label=label, color=color)
    ax.fill_between(xs, 0, ys, alpha=0.16, color=color)


def probability_transform(z):
    """Transform latent OU state to mutation-propagation probability."""
    z = np.asarray(z, dtype=float)

    if PROBABILITY_LINK == "power10":
        p = 10.0 ** z
    elif PROBABILITY_LINK == "inverse_logit":
        p = 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError("PROBABILITY_LINK must be 'power10' or 'inverse_logit'.")

    return np.clip(p, 1e-15, 1.0 - 1e-12)


def load_observed_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df[["background", "replicate", "t", "x"]].copy()
    df["background"] = df["background"].map(normalize_bg)
    df["replicate"] = df["replicate"].astype(int)
    df["t"] = df["t"].astype(float)
    df["x"] = df["x"].astype(float)
    df["Y_obs"] = np.log10(df["x"] + 1e-9)
    return df


def load_posterior_and_metadata():
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Missing posterior file: {TRACE_PATH}")
    if not TIMES_PATH.exists():
        raise FileNotFoundError(f"Missing times metadata file: {TIMES_PATH}")
    if not BG_PATH.exists():
        raise FileNotFoundError(f"Missing background metadata file: {BG_PATH}")

    idata = az.from_netcdf(TRACE_PATH)

    times = np.load(TIMES_PATH, allow_pickle=True).astype(float)
    bg_order = np.load(BG_PATH, allow_pickle=True).astype(str).tolist()
    bg_order = [normalize_bg(b) for b in bg_order]

    posterior = idata.posterior

    mu = posterior["mu_bg"].values.reshape(-1, len(bg_order))       # sample, bg
    sigma = posterior["sigma_bg"].values.reshape(-1, len(bg_order)) # sample, bg
    theta = posterior["theta"].values.reshape(-1)                   # sample

    return idata, times, bg_order, mu, sigma, theta


def simulate_ou_paths(df, times, bg_order, mu, sigma, theta):
    """
    Simulate posterior OU latent paths under the fitted transition kernel.

    The marginal Figure 2 trace does not sample Z_latent explicitly. For Figure 3,
    we generate posterior representative latent trajectories by initializing each
    genotype at its empirical mean at the first time point and propagating forward
    under posterior μ, θ, and σ.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    n_total = mu.shape[0]
    n_use = min(N_POSTERIOR_PATHS, n_total)
    draw_idx = rng.choice(np.arange(n_total), size=n_use, replace=False)

    mu_use = mu[draw_idx, :]
    sigma_use = sigma[draw_idx, :]
    theta_use = theta[draw_idx]

    n_t = len(times)
    rows = []
    path_dict = {}

    for b_idx, bg in enumerate(bg_order):
        g = df[df["background"] == bg].copy()
        y0 = g[g["t"] == np.min(times)]["Y_obs"].mean()

        z_paths = np.zeros((n_use, n_t), dtype=float)
        z_paths[:, 0] = y0

        for j in range(1, n_t):
            dt = 1.0  # rescaled model time
            exp_term = np.exp(-theta_use * dt)
            mean_j = mu_use[:, b_idx] + (z_paths[:, j - 1] - mu_use[:, b_idx]) * exp_term
            var_j = (sigma_use[:, b_idx] ** 2 / (2.0 * theta_use)) * (
                1.0 - np.exp(-2.0 * theta_use * dt)
            )
            sd_j = np.sqrt(np.maximum(var_j, 1e-12))
            z_paths[:, j] = rng.normal(mean_j, sd_j)

        p_paths = probability_transform(z_paths)
        count_paths = rng.binomial(N_EFFECTIVE, p_paths)

        path_dict[bg] = {
            "z": z_paths,
            "p": p_paths,
            "counts": count_paths,
        }

        for s in range(n_use):
            for j, t in enumerate(times):
                rows.append({
                    "background": bg,
                    "posterior_path": s,
                    "time": t,
                    "Z_latent": z_paths[s, j],
                    "p_mutation": p_paths[s, j],
                    "count": count_paths[s, j],
                    "N_effective": N_EFFECTIVE,
                    "probability_link": PROBABILITY_LINK,
                })

    sim_df = pd.DataFrame(rows)
    return sim_df, path_dict


def summarize_paths(arr):
    """Return mean, 2.5%, 97.5% across posterior paths for each time."""
    return (
        np.mean(arr, axis=0),
        np.percentile(arr, 2.5, axis=0),
        np.percentile(arr, 97.5, axis=0),
    )


# =============================================================================
# 3. Plot panels
# =============================================================================

def panel_a_latent(ax, times, bg_order, path_dict):
    for bg in bg_order:
        color = COLOR_MAP[bg]
        mean, lo, hi = summarize_paths(path_dict[bg]["z"])

        ax.plot(times, mean, lw=2.2, color=color, label=display_bg(bg))
        ax.fill_between(times, lo, hi, color=color, alpha=0.16, linewidth=0)

    ax.set_title("Posterior OU latent trajectories", fontweight="bold")
    ax.set_xlabel("Time point")
    ax.set_ylabel(r"Latent state $Z_t$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")


def panel_b_probability(ax, times, bg_order, path_dict):
    for bg in bg_order:
        color = COLOR_MAP[bg]
        mean, lo, hi = summarize_paths(path_dict[bg]["p"])

        ax.plot(times, mean, lw=2.2, color=color, label=display_bg(bg))
        ax.fill_between(times, lo, hi, color=color, alpha=0.16, linewidth=0)

    ax.set_title("OU states mapped to propagation probabilities", fontweight="bold")
    ax.set_xlabel("Time point")
    ax.set_ylabel(r"Mutation probability $p(t)$")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="best")

    if PROBABILITY_LINK == "power10":
        transform_text = r"$p(t)=10^{Z_t}$"
    else:
        transform_text = r"$p(t)=\mathrm{logit}^{-1}(Z_t)$"

    ax.text(
        0.03, 0.06,
        transform_text,
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="bottom",
    )


def panel_c_counts(ax, times, bg_order, path_dict):
    for bg in bg_order:
        color = COLOR_MAP[bg]
        mean, lo, hi = summarize_paths(path_dict[bg]["counts"])

        ax.plot(times, mean, lw=2.2, color=color, label=display_bg(bg))
        ax.fill_between(times, lo, hi, color=color, alpha=0.16, linewidth=0)

    ax.set_title("Posterior predictive count layer", fontweight="bold")
    ax.set_xlabel("Time point")
    ax.set_ylabel(r"Posterior predictive counts per $10^6$ trials")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")


def panel_d_probability_distributions(ax, bg_order, path_dict):
    for bg in bg_order:
        color = COLOR_MAP[bg]
        # use all posterior paths/time points
        vals = path_dict[bg]["p"].reshape(-1)
        # Plot on log10 probability scale for clarity
        log_vals = np.log10(np.clip(vals, 1e-15, 1.0))
        draw_kde(ax, log_vals, label=display_bg(bg), color=color)

    ax.set_title("Mutation-propagation probability regimes", fontweight="bold")
    ax.set_xlabel(r"$\log_{10}\,p(t)$")
    ax.set_ylabel("Posterior density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")


# =============================================================================
# 4. Main
# =============================================================================

def main():
    print(f"Using BASE_DIR: {BASE_DIR}")
    print(f"Probability link: {PROBABILITY_LINK}")
    print(f"N_EFFECTIVE: {N_EFFECTIVE:,}")

    df = load_observed_data()
    idata, times, bg_order, mu, sigma, theta = load_posterior_and_metadata()

    print("Background order:", bg_order)
    print("Times:", times)

    sim_df, path_dict = simulate_ou_paths(df, times, bg_order, mu, sigma, theta)
    sim_out = BASE_DIR / "figure3_simulated_latent_probability_counts.csv"
    sim_df.to_csv(sim_out, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5))
    axA, axB, axC, axD = axes.ravel()

    panel_a_latent(axA, times, bg_order, path_dict)
    panel_b_probability(axB, times, bg_order, path_dict)
    panel_c_counts(axC, times, bg_order, path_dict)
    panel_d_probability_distributions(axD, bg_order, path_dict)

    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Coupling continuous OU latent dynamics to discrete mutation/count processes",
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
    print(sim_out)


if __name__ == "__main__":
    main()
