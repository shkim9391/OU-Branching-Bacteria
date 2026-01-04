#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#s4_figure.py

Generate Supplementary Figure S4:
OU latent trajectories + branching counts + posterior predictive count PMFs +
branching probabilities p(t) for priA, recG, and WT.
"""

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


def main():
    # ---------------------------------------------------------
    # 1. Load posterior and time grid
    # ---------------------------------------------------------
    trace = az.from_netcdf("trace_core.nc")
    times = np.load("times.npy")

    # Latent OU states: chain x draw x bg x rep x time
    Z = trace.posterior["Z_latent"].values
    n_chain, n_draw, n_bg, n_rep, n_time = Z.shape

    # Flatten chains and draws into a "samples" axis
    # Z_flat: samples × bg × rep × time
    Z_flat = Z.reshape(-1, n_bg, n_rep, n_time)

    # ---------------------------------------------------------
    # 2. Posterior summaries of latent OU trajectories
    # ---------------------------------------------------------
    # Average over replicates, then summarize over samples
    Z_mean_reps = Z_flat.mean(axis=2)      # samples × bg × time
    Z_mean      = Z_mean_reps.mean(axis=0) # bg × time
    Z_low       = np.percentile(Z_mean_reps, 2.5, axis=0)
    Z_high      = np.percentile(Z_mean_reps, 97.5, axis=0)

    # ---------------------------------------------------------
    # 3. Branching probabilities p(t) = logistic(X_t)
    # ---------------------------------------------------------
    p_samples = expit(Z_mean_reps)         # samples × bg × time
    p_mean    = p_samples.mean(axis=0)     # bg × time
    p_low     = np.percentile(p_samples, 2.5, axis=0)
    p_high    = np.percentile(p_samples, 97.5, axis=0)

    # ---------------------------------------------------------
    # 4. Example Binomial counts from p(t)
    # ---------------------------------------------------------
    N = 20  # Binomial sample size
    rng = np.random.default_rng(42)
    example_counts = rng.binomial(N, p_mean)  # bg × time

    # ---------------------------------------------------------
    # 5. Posterior predictive counts from the Binomial layer
    #    (simulate using p_samples)
    # ---------------------------------------------------------
    # p_samples: samples × bg × time
    counts_pp = rng.binomial(N, p_samples)  # samples × bg × time

    # ---------------------------------------------------------
    # 6. Build S4 figure
    # ---------------------------------------------------------
    bg_names = ["priA", "recG", "WT"]
    colors   = ["#E66100", "#5D9C59", "#0072B2"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axA, axB, axC, axD = axes.flatten()

    # Panel A: Latent OU trajectories
    for i, bg in enumerate(bg_names):
        axA.plot(times, Z_mean[i], label=bg, color=colors[i])
        axA.fill_between(times, Z_low[i], Z_high[i],
                         color=colors[i], alpha=0.25)
    axA.set_title("A  Latent OU trajectories")
    axA.set_xlabel("Time")
    axA.set_ylabel("Latent state $X_t$")
    axA.legend()

    # Panel B: Example Binomial branching counts
    for i, bg in enumerate(bg_names):
        axB.plot(times, example_counts[i],
                 marker="o", color=colors[i], label=bg)
    axB.set_title("B  Branching (Binomial) counts")
    axB.set_xlabel("Time")
    axB.set_ylabel("Counts $Y_t$")
    axB.legend()

    # Panel C: Posterior predictive count PMFs
    k_vals = np.arange(0, N + 1)
    for i, bg in enumerate(bg_names):
        # counts_pp: samples × bg × time
        # Flatten over samples and time for genotype i
        geno_counts = counts_pp[:, i, :].reshape(-1)
        pmf = np.array([(geno_counts == k).mean() for k in k_vals])
        axC.plot(k_vals, pmf, marker='o', color=colors[i], label=bg)
    axC.set_title("C  Posterior predictive PMF (count layer)")
    axC.set_xlabel("Counts $Y_t$")
    axC.set_ylabel("Posterior predictive $P(Y_t = k)$")
    axC.set_xlim(-0.5, N + 0.5)
    axC.legend()

    # Panel D: Branching probabilities p(t)
    for i, bg in enumerate(bg_names):
        axD.plot(times, p_mean[i], color=colors[i], label=bg)
        axD.fill_between(times, p_low[i], p_high[i],
                         color=colors[i], alpha=0.25)
    axD.set_title("D  Branching probabilities $p(t)$")
    axD.set_xlabel("Time")
    axD.set_ylabel("$p(t)$")
    axD.legend()

    fig.tight_layout()

    # ---------------------------------------------------------
    # 7. Save figure
    # ---------------------------------------------------------
    fig.savefig("S4_OU_branching.png", dpi=300, bbox_inches="tight")
    fig.savefig("S4_OU_branching.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
