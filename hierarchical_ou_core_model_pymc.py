"""
Stabilized core hierarchical OU model for mutation frequencies
in WT, priA, recG E. coli.

- Log10-transformed frequencies
- Hierarchical μ across backgrounds (WT, priA, recG)
- Shared mean-reversion rate θ across backgrounds
- Background-specific diffusion σ_bg
- Small, fixed observation noise (sigma_obs)
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as at
import arviz as az
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 0. PyTensor config (no C++ compiler, optimized mode)
# --------------------------------------------------------------------
pytensor.config.cxx = ""
pytensor.config.mode = "FAST_RUN"

# --------------------------------------------------------------------
# 1. Load & preprocess data
# --------------------------------------------------------------------
data_path = (
    "/mut_freq_data.csv"
)

df = pd.read_csv(data_path)
df = df[["background", "replicate", "t", "x"]].copy()

df["replicate"] = df["replicate"].astype(int)
df["t"] = df["t"].astype(float)
df["x"] = df["x"].astype(float)

# Encode backgrounds as integers in sorted order: ['priA','recG','wt']
bg_categories = sorted(df["background"].unique())
bg_to_idx = {bg: i for i, bg in enumerate(bg_categories)}
df["bg_idx"] = df["background"].map(bg_to_idx)

df = df.sort_values(["bg_idx", "replicate", "t"]).reset_index(drop=True)

# Log10-transform frequencies
eps = 1e-9
df["Y_obs"] = np.log10(df["x"] + eps)

# Shapes
n_bg = df["bg_idx"].nunique()
reps_per_bg = df.groupby("bg_idx")["replicate"].nunique().iloc[0]
n_time = df.groupby(["bg_idx", "replicate"])["t"].size().iloc[0]

print(f"Genotypes (backgrounds): {bg_categories}")
print(f"n_bg = {n_bg}, reps_per_bg = {reps_per_bg}, n_time = {n_time}")

# Pivot to 3D array: (bg, rep, time)
Y_obs_table = (
    df.pivot_table(
        index=["bg_idx", "replicate"],
        columns="t",
        values="Y_obs",
    )
    .sort_index(axis=1)
)

times = Y_obs_table.columns.values
Y_obs = Y_obs_table.to_numpy().reshape(n_bg, reps_per_bg, n_time)

# Check regular time grid
dt = np.diff(times)
if not np.allclose(dt, dt[0]):
    raise ValueError("Time grid is not regular; OU kernel assumes constant Δt.")
delta_t = float(dt[0])
print(f"Δt = {delta_t}")

# --------------------------------------------------------------------
# 2. Stabilized hierarchical OU model
# --------------------------------------------------------------------
with pm.Model() as ou_core_model:

    # ------- Hyperpriors for OU mean μ (log10 scale) -------
    global_mean = df["Y_obs"].mean()
    mu_hyper = pm.Normal("mu_hyper", mu=global_mean, sigma=1.0)
    tau_mu = pm.HalfNormal("tau_mu", sigma=0.5)

    # Non-centered genotype-specific μ
    z_mu = pm.Normal("z_mu", mu=0.0, sigma=1.0, shape=n_bg)
    mu_bg = pm.Deterministic("mu_bg", mu_hyper + tau_mu * z_mu)  # (n_bg,)

    # ------- Shared mean-reversion rate θ -------
    log_theta = pm.Normal("log_theta", mu=np.log(0.3), sigma=0.15)
    theta = pm.Deterministic("theta", at.exp(log_theta))  # > 0
    theta_bg = at.ones(n_bg) * theta

    # ------- Background-specific diffusion σ_bg -------
    sigma_bg = pm.HalfNormal("sigma_bg", sigma=0.2, shape=n_bg)

    # ------- Initial latent variance & observation noise -------
    sigma_Z0 = pm.HalfNormal("sigma_Z0", sigma=0.3)
    sigma_obs_value = 0.03  # fixed small observation noise

    # ------- Latent OU states: Z[bg, rep, time] -------
    Z0 = pm.Normal(
        "Z0",
        mu=mu_bg[:, None],
        sigma=sigma_Z0,
        shape=(n_bg, reps_per_bg),
    )

    Z_latent = at.zeros((n_bg, reps_per_bg, n_time))
    Z_latent = at.set_subtensor(Z_latent[:, :, 0], Z0)

    eps = pm.Normal(
        "eps",
        mu=0.0,
        sigma=1.0,
        shape=(n_bg, reps_per_bg, n_time - 1),
    )

    for j in range(1, n_time):
        prev = Z_latent[:, :, j - 1]

        exp_term = at.exp(-theta_bg * delta_t)[:, None]
        mean_j = mu_bg[:, None] + (prev - mu_bg[:, None]) * exp_term

        var_j = (sigma_bg**2 / (2.0 * theta_bg)) * (
            1.0 - at.exp(-2.0 * theta_bg * delta_t)
        )
        std_j = at.sqrt(var_j)[:, None]

        eps_j = eps[:, :, j - 1]
        Z_latent = at.set_subtensor(
            Z_latent[:, :, j],
            mean_j + std_j * eps_j,
        )

    # Make latent states available in posterior
    Z_latent_det = pm.Deterministic("Z_latent", Z_latent)

    # ------- Observation model -------
    pm.Normal(
        "Y_lik",
        mu=Z_latent_det,
        sigma=sigma_obs_value,
        observed=Y_obs,
    )

    # ------- Sampling -------
    trace_core = pm.sample(
        draws=500,
        tune=500,
        target_accept=0.97,
        max_treedepth=11,
        chains=2,
        cores=2,
        random_seed=123,
        return_inferencedata=True,
    )

    ppc_core = pm.sample_posterior_predictive(trace_core)

# ➜ Attach posterior_predictive group to trace_core
trace_core.extend(ppc_core)

# ➜ THEN save
trace_core.to_netcdf(
    "/trace_core.nc"
)
np.save(
    "/times.npy",
    times,
)

# Optional: quick summary
print(
    az.summary(trace_core, var_names=["mu_bg", "theta", "sigma_bg"], round_to=3)
)
