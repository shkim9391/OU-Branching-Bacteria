"""
Multi-panel OU figure 2 for WT / priA / recG E. coli

Panels:
A. μ (log10 mutation frequency) posterior distributions
B. σ (diffusion) posterior distributions
C. Posterior predictive vs observed density
D. Hierarchical shrinkage of μ across backgrounds
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 0. Load fitted model and metadata
# -------------------------------------------------------
trace_core = az.from_netcdf(
    "/trace_core.nc"
)
times = np.load(
    "/times.npy"
)

bg_names = ["priA", "recG", "wt"]
colors = {"priA": "#E66100", "recG": "#5D9C59", "wt": "#0072B2"}

# Stack chains/draws for convenience
post_mu = trace_core.posterior["mu_bg"].stack(sample=("chain", "draw")).values
post_sigma = trace_core.posterior["sigma_bg"].stack(sample=("chain", "draw")).values
post_theta = trace_core.posterior["theta"].stack(sample=("chain", "draw")).values.flatten()

# Posterior predictive + observed
y_pp = trace_core.posterior_predictive["Y_lik"].stack(sample=("chain", "draw")).values.flatten()
y_obs = trace_core.observed_data["Y_lik"].values.flatten()

# -------------------------------------------------------
# 1. Create 2×2 figure
# -------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.subplots_adjust(hspace=0.45)   # good amount of space
axA, axB, axC, axD = axes.flatten()

# ---------------- Panel A: μ ridgeplot ----------------
for i, bg in enumerate(bg_names):
    sns.kdeplot(
        post_mu[i],
        fill=True,
        alpha=0.5,
        linewidth=1.2,
        color=colors[bg],
        label=bg,
        ax=axA,
    )
axA.set_xlabel("μ (log10 mutation frequency)")
axA.set_ylabel("Density")
axA.set_title("A  μ posteriors by background")
axA.legend(frameon=False)

# ---------------- Panel B: σ ridgeplot ----------------
for i, bg in enumerate(bg_names):
    sns.kdeplot(
        post_sigma[i],
        fill=True,
        alpha=0.5,
        linewidth=1.2,
        color=colors[bg],
        label=bg,
        ax=axB,
    )
axB.set_xlabel("σ (diffusion)")
axB.set_ylabel("Density")
axB.set_title("B  σ diffusion by background")
axB.legend(frameon=False)

# ---------------- Panel C: PPC density ----------------
sns.kdeplot(
    y_pp,
    fill=True,
    alpha=0.35,
    color="#56B4E9",
    label="Posterior predictive",
    ax=axC,
)
sns.kdeplot(
    y_obs,
    color="black",
    linewidth=2,
    label="Observed",
    ax=axC,
)
axC.set_xlabel("log10 mutation frequency")
axC.set_ylabel("Density")
axC.set_title("C  Posterior predictive vs observed")
axC.legend(frameon=False)

# ---------------- Panel D: Shrinkage plot -------------
# Panel D — Horizontal shrinkage plot
for i, bg in enumerate(bg_names):
    sns.kdeplot(
        post_mu[i],
        fill=True,
        alpha=0.4,
        color=colors[bg],
        ax=axD,
    )
    # Mean indicator
    axD.scatter(post_mu[i].mean(), i, color=colors[bg], s=60)

axD.set_yticks(range(len(bg_names)))
axD.set_yticklabels(bg_names)
axD.set_xlabel("μ (log10 mutation frequency)")
axD.set_ylabel("Background", labelpad=0.5)
axD.yaxis.set_label_coords(-0.04, 0.5)
axD.set_title("D  Hierarchical shrinkage of μ")

axD.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout(pad=1.5)
plt.show()

# Save high-res version
fig.savefig(
    "/ou_multifigure_ABCD.png",
    dpi=300,
    bbox_inches="tight",
)
