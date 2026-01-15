"""
Common utilities + model builders for Fig2 posterior fitting.

Shared by three thin wrappers:
  - fit_fig5_ou_nb.py        : OU latent x + NegBin count layer
  - fit_fig5_rw_nb.py        : Random-walk latent x + NegBin count layer
  - fit_fig5_ou_branch_nb.py : OU latent x with count-coupled mean (branch-like) + NegBin count layer

All scripts write to --out_dir:
  - posterior.nc
  - posterior_summary.csv
  - group_order_used.csv
  - loo_waic.csv  (LOO/WAIC computed on combined loglik_total)

NegBin count layer is INCLUDED in ALL models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

SEED_DEFAULT = 13


# -----------------------------
# I/O + preprocessing
# -----------------------------
def read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    need = {"Patient_ID", "series", "t", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[series_csv] Missing columns: {missing}. Found: {df.columns.tolist()}")

    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df["series"] = df["series"].astype("string").str.strip().str.lower()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.loc[df["series"].isin(["x", "n"])].dropna(subset=["Patient_ID", "series", "t", "value"]).copy()
    df = df.sort_values(["Patient_ID", "series", "t"], kind="mergesort").reset_index(drop=True)
    return df


def read_patient_groups(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # allow common variants
    if "patient_id" in df.columns and "Patient_ID" not in df.columns:
        df = df.rename(columns={"patient_id": "Patient_ID"})
    if "patient" in df.columns and "Patient_ID" not in df.columns:
        df = df.rename(columns={"patient": "Patient_ID"})

    need = {"Patient_ID", "Group"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"[patient_group_csv] Missing columns: {missing}. Found: {df.columns.tolist()}")

    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df["Group"] = df["Group"].astype("string").str.strip()

    df = df.dropna(subset=["Patient_ID", "Group"]).drop_duplicates(subset=["Patient_ID"]).reset_index(drop=True)
    return df


def _sanitize_pid(pid: str) -> str:
    pid = str(pid).strip()
    pid = re.sub(r"[^0-9a-zA-Z_]+", "_", pid)
    pid = re.sub(r"_+", "_", pid).strip("_")
    return pid or "pid"


def split_patient_series(series_df: pd.DataFrame, pid: str):
    dfx = series_df.query("Patient_ID == @pid and series == 'x'").copy()
    dfn = series_df.query("Patient_ID == @pid and series == 'n'").copy()
    tx = dfx["t"].to_numpy(dtype=float)
    xv = dfx["value"].to_numpy(dtype=float)
    tn = dfn["t"].to_numpy(dtype=float)
    nv = dfn["value"].to_numpy(dtype=float)
    return tx, xv, tn, nv


def build_count_covariate_for_x_times(tx: np.ndarray, tn: np.ndarray, nv: np.ndarray) -> np.ndarray:
    """
    Deterministic count-derived covariate aligned to x timepoints.
    Uses log(n+1) and linear interpolation in time; ends are held constant.
    """
    if tx.size == 0:
        return np.array([], dtype=float)
    if tn.size == 0 or nv.size == 0:
        return np.zeros_like(tx, dtype=float)

    y = np.log(nv.astype(float) + 1.0)
    order = np.argsort(tn)
    tn_s = tn[order]
    y_s = y[order]
    return np.interp(tx, tn_s, y_s, left=float(y_s[0]), right=float(y_s[-1]))


@dataclass
class PatientPack:
    pid: str
    pid_safe: str
    group_idx: int
    tx: np.ndarray
    xv: np.ndarray
    tn: np.ndarray
    nv: np.ndarray
    cx: np.ndarray  # covariate aligned to x times


def pack_patients(series_df: pd.DataFrame, pg_df: pd.DataFrame) -> Tuple[List[PatientPack], List[str]]:
    patients = series_df["Patient_ID"].dropna().unique().tolist()

    pg = pg_df.set_index("Patient_ID").reindex(patients)
    if pg["Group"].isna().any():
        missing = pg.index[pg["Group"].isna()].tolist()
        raise ValueError(
            f"Missing Group labels for {len(missing)} patients in patient_group_csv. "
            f"Example: {missing[:10]}"
        )

    group_names = sorted(pg["Group"].unique().tolist())
    group_to_idx = {g: i for i, g in enumerate(group_names)}

    packs: List[PatientPack] = []
    for pid in patients:
        tx, xv, tn, nv = split_patient_series(series_df, pid)
        cx = build_count_covariate_for_x_times(tx, tn, nv)
        packs.append(
            PatientPack(
                pid=str(pid),
                pid_safe=_sanitize_pid(pid),
                group_idx=group_to_idx[str(pg.loc[pid, "Group"])],
                tx=tx,
                xv=xv,
                tn=tn,
                nv=nv,
                cx=cx,
            )
        )
    return packs, group_names


# -----------------------------
# Log-likelihood helper
# -----------------------------
def add_combined_loglik(idata, name: str = "loglik_total"):
    """
    Sum pointwise log-likelihood over ALL observed RVs and observation dimensions,
    leaving only (chain, draw). Stores as idata.log_likelihood[name].
    """
    import xarray as xr

    ll_ds = idata.log_likelihood
    per_var = []
    for v in ll_ds.data_vars:
        x = ll_ds[v]
        dims_to_sum = [d for d in x.dims if d not in ("chain", "draw")]
        per_var.append(x.sum(dim=dims_to_sum))

    ll_total = xr.concat(per_var, dim="__obsvar__").sum(dim="__obsvar__")
    idata.log_likelihood[name] = ll_total
    return idata


# -----------------------------
# Model builder
# -----------------------------
def fit_model_and_write(
    model_kind: str,  # {"ou","rw","ou_branch"}
    series_csv: Path,
    patient_group_csv: Path,
    out_dir: Path,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = SEED_DEFAULT,
    prefer_numpyro: bool = True,
):
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    out_dir.mkdir(parents=True, exist_ok=True)

    series_df = read_series(series_csv)
    pg_df = read_patient_groups(patient_group_csv)

    packs, group_names = pack_patients(series_df, pg_df)
    G = len(group_names)

    # Prefer numpyro sampler if available
    use_numpyro = False
    if prefer_numpyro:
        try:
            import jax  # noqa
            import numpyro  # noqa
            use_numpyro = True
        except Exception:
            use_numpyro = False

    eps = 1e-8

    with pm.Model() as model:
        # --- Group parameters
        if model_kind in {"ou", "ou_branch"}:
            theta_raw = pm.Normal("theta_raw", 0.0, 1.0, shape=(G,))
            theta = pm.Deterministic("theta", 0.05 + pm.math.log1pexp(theta_raw))

        sigma_raw = pm.Normal("sigma_raw", 0.0, 1.0, shape=(G,))
        sigma = pm.Deterministic("sigma", 0.02 + 0.50 * pm.math.log1pexp(sigma_raw))

        r_raw = pm.Normal("r_raw", 0.0, 1.0, shape=(G,))
        r = pm.Deterministic("r", 0.50 * r_raw)

        # Count-coupling for OU mean (only for "ou_branch")
        if model_kind == "ou_branch":
            beta_raw = pm.Normal("beta_raw", 0.0, 1.0, shape=(G,))
            beta = pm.Deterministic("beta", 0.50 * beta_raw)

        # --- Global noise / dispersion (shared across all models)
        tau_x = pm.HalfNormal("tau_x", 0.30)   # x obs noise
        phi = pm.HalfNormal("phi", 2.00)       # NegBin alpha

        # --- Patient random effects
        mu_mu = pm.Normal("mu_mu", 0.0, 1.0)
        mu_sigma = pm.HalfNormal("mu_sigma", 1.0)

        logn_mu = pm.Normal("logn_mu", 0.0, 2.0)
        logn_sigma = pm.HalfNormal("logn_sigma", 1.0)

        for pack in packs:
            gix = pack.group_idx
            pid = pack.pid_safe
            tx, xv, tn, nv, cx = pack.tx, pack.xv, pack.tn, pack.nv, pack.cx

            # patient set-point for x
            z_mu = pm.Normal(f"z_mu_{pid}", 0.0, 1.0)
            mu_i = pm.Deterministic(f"mu_{pid}", mu_mu + mu_sigma * z_mu)

            # ---------- x-layer (OU or RW) ----------
            if len(tx) >= 2:
                if model_kind in {"ou", "ou_branch"}:
                    mu_t0 = mu_i + beta[gix] * float(cx[0]) if model_kind == "ou_branch" else mu_i

                    x0 = pm.Normal(
                        f"x0_{pid}",
                        mu=mu_t0,
                        sigma=pt.sqrt((sigma[gix] ** 2) / (2.0 * theta[gix]) + eps),
                    )
                    x_lat = [x0]

                    for k in range(1, len(tx)):
                        dt = float(tx[k] - tx[k - 1])
                        mu_k = mu_i + beta[gix] * float(cx[k]) if model_kind == "ou_branch" else mu_i

                        m = mu_k + (x_lat[-1] - mu_k) * pt.exp(-theta[gix] * dt)
                        v = (sigma[gix] ** 2) / (2.0 * theta[gix]) * (1.0 - pt.exp(-2.0 * theta[gix] * dt)) + eps

                        xk = pm.Normal(f"x_{pid}_{k}", mu=m, sigma=pt.sqrt(v))
                        x_lat.append(xk)

                    pm.Normal(f"x_obs_{pid}", mu=pt.stack(x_lat), sigma=tau_x, observed=xv)

                elif model_kind == "rw":
                    x0 = pm.Normal(f"x0_{pid}", mu=mu_i, sigma=1.0)  # broad initial prior
                    x_lat = [x0]
                    for k in range(1, len(tx)):
                        dt = float(tx[k] - tx[k - 1])
                        v = (sigma[gix] ** 2) * dt + eps
                        xk = pm.Normal(f"x_{pid}_{k}", mu=x_lat[-1], sigma=pt.sqrt(v))
                        x_lat.append(xk)
                    pm.Normal(f"x_obs_{pid}", mu=pt.stack(x_lat), sigma=tau_x, observed=xv)
                else:
                    raise ValueError(f"Unknown model_kind={model_kind}")

            elif len(tx) == 1:
                pm.Normal(f"x_obs_{pid}", mu=mu_i, sigma=tau_x, observed=xv)

            # ---------- n-layer (NegBin) — kept for ALL models ----------
            z_ln = pm.Normal(f"z_logn_{pid}", 0.0, 1.0)
            logn0_i = pm.Deterministic(f"logn0_{pid}", logn_mu + logn_sigma * z_ln)

            if len(tn) >= 1:
                mu_n = pt.exp(logn0_i + r[gix] * pt.as_tensor_variable(tn.astype(float)))
                mu_n = pt.clip(mu_n, 1e-6, 1e9)
                pm.NegativeBinomial(f"n_obs_{pid}", mu=mu_n, alpha=phi, observed=nv.astype(int))

        # -------- Sampling (LOO/WAIC-ready)
        sample_kwargs = dict(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            random_seed=seed,
            target_accept=target_accept,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=True,
        )

        if use_numpyro:
            idata = pm.sample(**sample_kwargs, nuts_sampler="numpyro")
        else:
            idata = pm.sample(**sample_kwargs, max_treedepth=15, init="jitter+adapt_diag")

    # Combine LL + compute LOO/WAIC on combined loglik
    idata = add_combined_loglik(idata, name="loglik_total")
    loo = az.loo(idata, var_name="loglik_total")
    waic = az.waic(idata, var_name="loglik_total")

    # Write outputs
    az.to_netcdf(idata, out_dir / "posterior.nc")

    var_names = ["theta_raw", "theta", "sigma_raw", "sigma", "r_raw", "r", "tau_x", "phi"]
    if model_kind == "rw":
        var_names = ["sigma_raw", "sigma", "r_raw", "r", "tau_x", "phi"]
    if model_kind == "ou_branch":
        var_names = var_names + ["beta_raw", "beta"]

    summ = az.summary(idata, var_names=var_names)
    summ.to_csv(out_dir / "posterior_summary.csv")

    pd.DataFrame({"Group": group_names}).to_csv(out_dir / "group_order_used.csv", index=False)

    pd.DataFrame(
        {
            "model_kind": [model_kind],
            "loo_elpd": [float(loo.elpd_loo)],
            "loo_se": [float(loo.se)],
            "p_loo": [float(loo.p_loo)],
            "waic_elpd": [float(waic.elpd_waic)],
            "waic_se": [float(waic.se)],
            "p_waic": [float(waic.p_waic)],
        }
    ).to_csv(out_dir / "loo_waic.csv", index=False)

    print(f"[OK] model_kind={model_kind}")
    print(f"[OK] wrote {out_dir / 'posterior.nc'}")
    print(f"[OK] wrote {out_dir / 'posterior_summary.csv'}")
    print(f"[OK] wrote {out_dir / 'group_order_used.csv'}")
    print(f"[OK] wrote {out_dir / 'loo_waic.csv'}")
    print(f"[OK] groups (order): {group_names}")

    return idata, loo, waic
