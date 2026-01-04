#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#figure3_hybrid_ou_branching.py

"""
Figure 3 — Hybrid OU–Branching: latent OU dynamics + branching (count) layer.
Uses the Z_latent samples already in trace_core.nc.
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# Load OU posterior + time grid
# -------------------------------------------------------
trace_core = az.from_netcdf(
    "/Users/seung-hwan.kim/Desktop/Hierarchical_OU_Branching_Model/Methodology/trace_core.nc"
)
times = np.load(
    "/Users/seung-hwan.kim/Desktop/Hierarchical_OU_Branching_Model/Methodology/times.npy"
)

bg_names = ["priA", "recG", "wt"]
colors = {"priA": "#E66100", "recG": "#5D9C59", "wt": "#0072B2"}

# Extract latent OU samples
Z = trace_core.posterior["Z_latent"].values  
n_chain, n_draw, n_bg, n_rep, n_time = Z.shape
Z_flat = Z.reshape(-1, n_bg, n_rep, n_time)

# -------------------------------------------------------
# Panel A — OU trajectories (posterior mean + CI)
# -------------------------------------------------------
def panel_A(ax):
    for k, bg in enumerate(bg_names):
        trajs = Z_flat[:, k, :, :]
        traj_mean = trajs.mean(axis=1)

        mean_curve = traj_mean.mean(axis=0)
        low = np.percentile(traj_mean, 2.5, axis=0)
        high = np.percentile(traj_mean, 97.5, axis=0)

        ax.plot(times, mean_curve, color=colors[bg], lw=2, label=bg)
        ax.fill_between(times, low, high, color=colors[bg], alpha=0.25)

    ax.set_title("A  Latent OU trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Latent state $X_t$ (logit probability)")
    ax.legend(frameon=False)


# -------------------------------------------------------
# Panel B — Branching model: Binomial count layer
# -------------------------------------------------------
# Simulate counts: Y_t ~ Binomial(N=1000, p = logistic(X_t))
def logistic(x):
    return 1 / (1 + np.exp(-x))

N = 1000  # assumed population size for demonstration

def panel_B(ax):
    for k, bg in enumerate(bg_names):
        trajs = Z_flat[:, k, :, :]
        traj_mean = trajs.mean(axis=1)
        latent = traj_mean.mean(axis=0)
        p = logistic(latent)
        y = np.random.binomial(N, p)

        ax.plot(times, y, "-o", color=colors[bg], label=bg, alpha=0.8)

    ax.set_title("B  Branching (Binomial) count layer")
    ax.set_xlabel("Time")
    ax.set_ylabel("Counts $Y_t$")
    ax.legend(frameon=False, loc="upper left")


# -------------------------------------------------------
# Panel C — Posterior predictive distribution for counts
# -------------------------------------------------------
def panel_C(ax):
    for k, bg in enumerate(bg_names):
        trajs = Z_flat[:, k, :, :]
        traj_mean = trajs.mean(axis=1)
        latent = traj_mean.mean(axis=0)
        p = logistic(latent)

        y_pp = np.random.binomial(N, p, size=(500, len(p))).flatten()

        sns.kdeplot(
            y_pp,
            color=colors[bg],
            fill=True,
            alpha=0.3,
            label=bg,
            ax=ax,
            common_norm=False,   # important when overlaying groups
        )

    ax.set_title("C  Posterior predictive (count layer)")
    ax.set_xlabel("Counts $Y_t$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)


# -------------------------------------------------------
# Panel D — Distribution of branching probabilities p(t)
# -------------------------------------------------------
def panel_D(ax):
    for k, bg in enumerate(bg_names):
        trajs = Z_flat[:, k, :, :]
        traj_mean = trajs.mean(axis=1)
        latent = traj_mean.mean(axis=0)
        p = logistic(latent)

        sns.kdeplot(p, color=colors[bg], fill=True, alpha=0.3, label=bg, ax=ax)

    ax.set_title("D  Branching probabilities $p(t)$")
    ax.set_xlabel("$p(t)$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)


# -------------------------------------------------------
# Build multi-panel Figure 3
# -------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axA, axB, axC, axD = axes.flatten()

panel_A(axA)
panel_B(axB)
panel_C(axC)
panel_D(axD)

plt.tight_layout()
fig.savefig(
    "/Users/seung-hwan.kim/Desktop/Hierarchical_OU_Branching_Model/Methodology/Figure_3/Figure3_Hybrid_OU_Branching.png",
    dpi=300, bbox_inches="tight")

plt.show()
