from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import arviz as az

DEFAULT_BASE_DIR = Path(
    "/Figure_2"
)
BASE_DIR = Path(os.environ.get("OU_FIG2_DIR", DEFAULT_BASE_DIR))

DATA_PATH = BASE_DIR / "mut_freq_data.csv"
TRACE_PATH = BASE_DIR / "trace_core.nc"
BG_PATH = BASE_DIR / "bg_categories.npy"
OUT_PREFIX = BASE_DIR / "Figure2_bacterial_validation_OU_parameters_real"

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
}


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
    bw = max(1.06 * sd * len(values) ** (-1 / 5), 1e-6)
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = 0.20 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, gridsize)
    z = (xs[:, None] - values[None, :]) / bw
    ys = np.exp(-0.5 * z ** 2).sum(axis=1) / (
        len(values) * bw * np.sqrt(2 * np.pi)
    )
    return xs, ys


def draw_kde(ax, values, label, lw=1.8):
    xs, ys = simple_kde(values)
    if len(xs) == 0:
        return
    ax.plot(xs, ys, lw=lw, label=label)
    ax.fill_between(xs, 0, ys, alpha=0.16)


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


def load_posterior_long():
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Missing posterior file: {TRACE_PATH}")
    if not BG_PATH.exists():
        raise FileNotFoundError(f"Missing background metadata file: {BG_PATH}")

    idata = az.from_netcdf(TRACE_PATH)
    bg_categories = np.load(BG_PATH, allow_pickle=True).astype(str).tolist()
    bg_categories = [normalize_bg(b) for b in bg_categories]

    mu = idata.posterior["mu_bg"].values
    sigma = idata.posterior["sigma_bg"].values

    rows = []
    for b_idx, bg in enumerate(bg_categories):
        mu_vals = mu[:, :, b_idx].reshape(-1)
        sigma_vals = sigma[:, :, b_idx].reshape(-1)
        rows.extend(
            {"background": bg, "mu": float(m), "sigma": float(s)}
            for m, s in zip(mu_vals, sigma_vals)
        )

    posterior_long = pd.DataFrame(rows)
    return posterior_long, bg_categories, idata


def summarize_parameters(posterior_long, idata):
    rows = []
    for bg, g in posterior_long.groupby("background", sort=False):
        for var in ["mu", "sigma"]:
            vals = g[var].values
            rows.append({
                "parameter": f"{var}_{bg}",
                "background": bg,
                "mean": np.mean(vals),
                "sd": np.std(vals, ddof=1),
                "q2.5": np.percentile(vals, 2.5),
                "q50": np.percentile(vals, 50),
                "q97.5": np.percentile(vals, 97.5),
            })

    for var in ["theta", "mu_hyper", "tau_mu", "sigma_obs"]:
        if var in idata.posterior:
            vals = idata.posterior[var].values.reshape(-1)
            rows.append({
                "parameter": var,
                "background": "",
                "mean": np.mean(vals),
                "sd": np.std(vals, ddof=1),
                "q2.5": np.percentile(vals, 2.5),
                "q50": np.percentile(vals, 50),
                "q97.5": np.percentile(vals, 97.5),
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(BASE_DIR / "figure2_parameter_summary.csv", index=False)
    return summary_df


def panel_a_observed(ax, df, bg_order):
    for bg in bg_order:
        g = df[df["background"] == bg].copy()
        if g.empty:
            continue
        for _, rg in g.groupby("replicate"):
            rg = rg.sort_values("t")
            ax.plot(rg["t"], rg["Y_obs"], lw=0.9, alpha=0.42)
            ax.scatter(rg["t"], rg["Y_obs"], s=12, alpha=0.65)

        mean_g = g.groupby("t", as_index=False)["Y_obs"].mean().sort_values("t")
        ax.plot(mean_g["t"], mean_g["Y_obs"], lw=2.2, label=display_bg(bg))

    ax.set_title("Observed longitudinal trajectories", fontweight="bold")
    ax.set_xlabel("Time point")
    ax.set_ylabel(r"$\log_{10}$ mutation frequency")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    
    color = COLOR_MAP[bg]
    
    ax.plot(
        rg["t"], rg["Y_obs"],
        lw=0.9,
        alpha=0.35,
        color=color,
    )
    
    ax.scatter(
        rg["t"], rg["Y_obs"],
        s=12,
        alpha=0.55,
        color=color,
    )
    
    ax.plot(
        mean_g["t"],
        mean_g["Y_obs"],
        lw=2.4,
        color=color,
        label=display_bg(bg),
    )

def draw_kde(ax, values, label, color, lw=1.8):
    xs, ys = simple_kde(values)
    if len(xs) == 0:
        return
    ax.plot(xs, ys, lw=lw, label=label, color=color)
    ax.fill_between(xs, 0, ys, alpha=0.16, color=color)

def panel_b_mu(ax, posterior_long, bg_order):
    for bg in bg_order:
        vals = posterior_long.loc[posterior_long["background"] == bg, "mu"].values
        draw_kde(ax, vals, label=display_bg(bg), color=COLOR_MAP[bg])
    ax.set_title("Genotype-specific OU equilibrium", fontweight="bold")
    ax.set_xlabel(r"OU equilibrium mean $\mu_i$")
    ax.set_ylabel("Posterior density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")


def panel_c_sigma(ax, posterior_long, bg_order):
    for bg in bg_order:
        vals = posterior_long.loc[posterior_long["background"] == bg, "sigma"].values
        draw_kde(ax, vals, label=display_bg(bg), color=COLOR_MAP[bg])
    ax.set_title("Genotype-specific stochastic variability", fontweight="bold")
    ax.set_xlabel(r"Diffusion scale $\sigma_i$")
    ax.set_ylabel("Posterior density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")


def panel_d_shrinkage(ax, df, posterior_long, bg_order):
    y_positions = {bg: len(bg_order) - 1 - i for i, bg in enumerate(bg_order)}

    posterior_means = posterior_long.groupby("background")["mu"].mean().reindex(bg_order).dropna()
    if len(posterior_means) > 0:
        pooled_ref = posterior_means.mean()
        ax.axvline(pooled_ref, ls="--", lw=1.0, color="black", alpha=0.65)
        ax.text(
            pooled_ref,
            len(bg_order) - 0.45,
            "pooled\nreference",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )

    for bg in bg_order:
        y = y_positions[bg]
        obs_vals = df.loc[df["background"] == bg, "Y_obs"].values
        post_vals = posterior_long.loc[posterior_long["background"] == bg, "mu"].values

        raw_mean = np.mean(obs_vals)
        post_mean = np.mean(post_vals)
        ci_low, ci_high = np.percentile(post_vals, [2.5, 97.5])

        color = COLOR_MAP[bg]
        
        ax.hlines(y, ci_low, ci_high, lw=2.4, color=color)
        ax.scatter(post_mean, y, s=36, color=color, zorder=5)
        ax.scatter(
            raw_mean,
            y + 0.15,
            s=34,
            facecolors="white",
            edgecolors=color,
            linewidths=1.2,
            zorder=6,
        )

        arrow = FancyArrowPatch(
            (raw_mean, y + 0.12),
            (post_mean, y + 0.02),
            arrowstyle="-|>",
            mutation_scale=9,
            lw=0.9,
            color="black",
            alpha=0.75,
        )
        ax.add_patch(arrow)

    ax.set_yticks([y_positions[bg] for bg in bg_order])
    ax.set_yticklabels([display_bg(bg) for bg in bg_order])
    ax.set_xlabel(r"$\log_{10}$ mutation-frequency scale")
    ax.set_title("Hierarchical shrinkage of equilibrium means", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    ax.text(
        0.08,
        0.12,
        "open circle: empirical mean\nfilled circle/line: posterior mean and 95% CrI",
        transform=ax.transAxes,
        fontsize=6.7,
        ha="left",
        va="bottom",
    )


def main():
    print(f"Using BASE_DIR: {BASE_DIR}")
    df = load_observed_data()
    posterior_long, bg_order, idata = load_posterior_long()

    posterior_long.to_csv(BASE_DIR / "posterior_mu_sigma_long.csv", index=False)
    summary_df = summarize_parameters(posterior_long, idata)

    print("\nBackground order from bg_categories.npy:")
    print(bg_order)
    print("\nFigure 2 parameter summary:")
    print(summary_df.to_string(index=False))

    if "diverging" in idata.sample_stats:
        print(f"\nDivergences: {int(idata.sample_stats['diverging'].sum())}")

    var_names = [v for v in ["mu_bg", "theta", "sigma_bg", "sigma_obs"] if v in idata.posterior]
    print("\nArviZ summary:")
    print(az.summary(idata, var_names=var_names, round_to=3))

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5))
    axA, axB, axC, axD = axes.ravel()

    panel_a_observed(axA, df, bg_order)
    panel_b_mu(axB, posterior_long, bg_order)
    panel_c_sigma(axC, posterior_long, bg_order)
    panel_d_shrinkage(axD, df, posterior_long, bg_order)

    for ax, label in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        add_panel_label(ax, label)

    fig.suptitle(
        "Controlled bacterial validation platform and hierarchical OU parameter inference",
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
    print(BASE_DIR / "posterior_mu_sigma_long.csv")
    print(BASE_DIR / "figure2_parameter_summary.csv")


if __name__ == "__main__":
    main()
