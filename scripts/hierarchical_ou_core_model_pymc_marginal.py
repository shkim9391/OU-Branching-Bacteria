import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as at
import arviz as az
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 0. PyTensor config
# --------------------------------------------------------------------
# Do not modify pytensor.config.mode inside the script.
# Use PYTENSOR_FLAGS from the shell before Python starts.

# --------------------------------------------------------------------
# 1. Load & preprocess data
# --------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(
    "/Figure_2"
)

data_path = BASE_DIR / "mut_freq_data.csv"
trace_path = BASE_DIR / "trace_core.nc"
times_path = BASE_DIR / "times.npy"
bg_path = BASE_DIR / "bg_categories.npy"

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

original_times = Y_obs_table.columns.values
Y_obs = Y_obs_table.to_numpy().reshape(n_bg, reps_per_bg, n_time)

# Check regular raw time grid
dt_raw = np.diff(original_times)
if not np.allclose(dt_raw, dt_raw[0]):
    raise ValueError("Time grid is not regular; OU kernel assumes constant spacing.")

# Use rescaled model time so that Δt = 1, as described in the manuscript
model_times = np.arange(n_time)
delta_t = 1.0

print(f"Original time points = {original_times}")
print(f"Model time grid = {model_times}")
print(f"Model Δt = {delta_t}")

global_mean = df["Y_obs"].mean()
print(f"Global mean log10 mutation frequency = {global_mean:.3f}")

# --------------------------------------------------------------------
# 2. Stabilized hierarchical OU model
# --------------------------------------------------------------------
with pm.Model() as ou_core_model:

    # ------- Hyperpriors for OU mean μ (log10 scale) -------
    mu_hyper = pm.Normal("mu_hyper", mu=global_mean, sigma=1.5)
    tau_mu = pm.HalfNormal("tau_mu", sigma=1.5)
    
    z_mu = pm.Normal("z_mu", 0.0, 1.0, shape=n_bg)
    mu_bg = pm.Deterministic("mu_bg", mu_hyper + tau_mu * z_mu)
    
    log_theta = pm.Normal("log_theta", mu=np.log(0.3), sigma=0.5)
    theta = pm.Deterministic("theta", at.exp(log_theta))
    
    sigma_bg = pm.HalfNormal("sigma_bg", sigma=1.0, shape=n_bg)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.15)

    # ------- Marginal OU transition likelihood -------
    Y_prev = Y_obs[:, :, :-1]
    Y_next = Y_obs[:, :, 1:]
    
    exp_term = at.exp(-theta * delta_t)
    
    mean_next = mu_bg[:, None, None] + (
        Y_prev - mu_bg[:, None, None]
    ) * exp_term
    
    var_next = (sigma_bg[:, None, None] ** 2 / (2.0 * theta)) * (
        1.0 - at.exp(-2.0 * theta * delta_t)
    )
    
    sigma_trans = at.sqrt(var_next + sigma_obs**2)
    
    pm.Normal(
        "Y_transition",
        mu=mean_next,
        sigma=sigma_trans,
        observed=Y_next,
    )

    # ------- Sampling -------
    trace_core = pm.sample(
        draws=1000,
        tune=1000,
        target_accept=0.95,
        max_treedepth=12,
        chains=4,
        cores=2,
        random_seed=123,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )

    ppc_core = pm.sample_posterior_predictive(trace_core)

# ➜ Attach posterior_predictive group to trace_core
trace_core.extend(ppc_core)

# ➜ THEN save
trace_core.to_netcdf(trace_path)
np.save(times_path, original_times)
np.save(bg_path, np.array(bg_categories))

# Optional: quick summary
print(
    az.summary(trace_core, var_names=["mu_bg", "theta", "sigma_bg"], round_to=3)
)
