#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 05:45:31 2025

@author: seung-hwan.kim
"""
#publication_ready_ou_visualization_suite.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

# Load trace and time grid
trace_core = az.from_netcdf("/Users/seung-hwan.kim/Desktop/Hierarchical_OU_Branching_Model/Methodology/trace_core.nc")
times = np.load("/Users/seung-hwan.kim/Desktop/Hierarchical_OU_Branching_Model/Methodology/times.npy")

bg_names = ["priA", "recG", "wt"]
colors = {"priA": "#E66100", "recG": "#5D9C59", "wt": "#0072B2"}

# Extract Z_latent
Z = trace_core.posterior["Z_latent"].values  # (chain, draw, bg, rep, time)
Z_flat = Z.reshape(-1, len(bg_names), Z.shape[3], len(times))  # flatten chains & draws

plt.figure(figsize=(10,6))

for k, bg in enumerate(bg_names):
    trajs = Z_flat[:, k, :, :]            # (samples, rep, time)
    traj_mean = trajs.mean(axis=1)        # posterior mean per sample
    mean_curve = traj_mean.mean(axis=0)   # mean over samples
    low = np.percentile(traj_mean, 2.5, axis=0)
    high = np.percentile(traj_mean, 97.5, axis=0)

    plt.plot(times, mean_curve, color=colors[bg], lw=2, label=f"{bg}")
    plt.fill_between(times, low, high, color=colors[bg], alpha=0.25)

plt.xlabel("Time")
plt.ylabel("Latent OU state (log10 mutation frequency)")
plt.title("Posterior latent OU trajectories (mean ± 95% CI)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 0. Load fitted model results (must run BEFORE plotting)
# -------------------------------------------------------

trace_core = az.from_netcdf(
    "/Users/seung-hwan.kim/Desktop/Hierarchical_OU_Branching_Model/Methodology/trace_core.nc"
)
times = np.load(
    "/Users/seung-hwan.kim/Desktop/Hierarchical_OU_Branching_Model/Methodology/times.npy"
)

bg_names = ["priA", "recG", "wt"]
colors = {"priA": "#E66100", "recG": "#5D9C59", "wt": "#0072B2"}
n_time = len(times)

# -------------------------------------------------------
# 1. Ridgeplots
# -------------------------------------------------------

def ridgeplot_param(param, label):
    post = trace_core.posterior[param].stack(sample=("chain","draw")).values

    if param == "theta":
        # scalar parameter
        post_vals = post.values.flatten() if hasattr(post, "values") else post.flatten()
        plt.figure(figsize=(6,4))
        sns.kdeplot(post_vals, fill=True, alpha=0.5, color="steelblue")
        plt.xlabel(label)
        plt.title(label)
        plt.tight_layout()
        plt.show()
        return

    # multi-background case
    plt.figure(figsize=(7,4))
    for i, bg in enumerate(bg_names):
        sns.kdeplot(
            post[i],
            fill=True,
            alpha=0.4,
            color=colors[bg],
            label=bg
        )
    plt.xlabel(label)
    plt.title(label)
    plt.legend()
    plt.tight_layout()
    plt.show()

ridgeplot_param("mu_bg", "μ (log10 mutation frequency)")
ridgeplot_param("sigma_bg", "σ diffusion")
ridgeplot_param("theta", "θ mean–reversion rate")

# -------------------------------------------------------
# 2. Shrinkage plot for μ_bg
# -------------------------------------------------------

def shrinkage_plot_mu():
    post_mu = trace_core.posterior["mu_bg"].stack(sample=("chain","draw")).values
    group_means = post_mu.mean(axis=1)

    plt.figure(figsize=(6,4))
    for i, bg in enumerate(bg_names):
        sns.kdeplot(post_mu[i], vertical=True, fill=True, alpha=0.5, color=colors[bg])
        plt.scatter(i, group_means[i], color=colors[bg], s=50)

    plt.xticks(range(3), bg_names)
    plt.ylabel("μ (log10 frequency)")
    plt.title("Hierarchical shrinkage of μ across backgrounds")
    plt.tight_layout()
    plt.show()

shrinkage_plot_mu()

# -------------------------------------------------------
# 3. OU trajectories
# -------------------------------------------------------

# assume trace_core, times, bg_names, colors already defined

def ou_trajectory_plot_with_ci():
    """
    Posterior latent OU trajectories with mean and 95% credible interval bands
    for priA, recG, wt.

    Uses the Z_latent deterministic saved in trace_core.
    """
    # Z_latent has shape: (chain, draw, bg, rep, time)
    Z = trace_core.posterior["Z_latent"].values
    n_chain, n_draw, n_bg, n_rep, n_time = Z.shape

    # Flatten chains and draws into one "sample" axis
    Z_flat = Z.reshape(-1, n_bg, n_rep, n_time)  # (samples, bg, rep, time)

    plt.figure(figsize=(8, 5))

    for k, bg in enumerate(bg_names):
        # All samples & replicates for this background
        trajs = Z_flat[:, k, :, :]          # (samples, rep, time)

        # Average over replicates for each posterior sample
        traj_mean_per_sample = trajs.mean(axis=1)  # (samples, time)

        # Posterior mean across samples
        mean_curve = traj_mean_per_sample.mean(axis=0)
        # 95% credible interval across samples
        low = np.percentile(traj_mean_per_sample, 2.5, axis=0)
        high = np.percentile(traj_mean_per_sample, 97.5, axis=0)

        # Plot mean and ribbon
        plt.plot(times, mean_curve, color=colors[bg], lw=2, label=bg)
        plt.fill_between(times, low, high, color=colors[bg], alpha=0.25)

    plt.xlabel("Time")
    plt.ylabel("Latent OU state (log10 mutation frequency)")
    plt.title("Posterior latent OU trajectories (mean ± 95% CI)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------
# 4. PPC
# -------------------------------------------------------

def clean_ppc_plot():
    y_pp = trace_core.posterior_predictive["Y_lik"].stack(sample=("chain","draw")).values.flatten()
    y_obs = trace_core.observed_data["Y_lik"].values.flatten()

    plt.figure(figsize=(7,4))
    sns.kdeplot(y_pp, fill=True, alpha=0.4, color="#56B4E9", label="Posterior predictive")
    sns.kdeplot(y_obs, color="black", linewidth=2, label="Observed")
    plt.xlabel("log10 mutation frequency")
    plt.title("Posterior predictive vs observed")
    plt.legend()
    plt.tight_layout()
    plt.show()

clean_ppc_plot()

# -------------------------------------------------------
# 5. Diffusion comparison
# -------------------------------------------------------

def diffusion_comparison():
    post_sigma = trace_core.posterior["sigma_bg"].stack(sample=("chain","draw")).values

    plt.figure(figsize=(6,4))
    for i, bg in enumerate(bg_names):
        sns.kdeplot(post_sigma[i], fill=True, alpha=0.4, color=colors[bg], label=bg)
    plt.xlabel("σ (diffusion)")
    plt.title("Genotype-specific diffusion σ_bg")
    plt.legend()
    plt.tight_layout()
    plt.show()

diffusion_comparison()