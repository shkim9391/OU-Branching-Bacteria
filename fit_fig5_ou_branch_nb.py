"""
OU latent x with count-coupled mean (branch-like) + NegBin count layer
Coupling uses deterministic covariate c(t)=log(n(t)+1) interpolated onto x timepoints.
"""

import argparse
from pathlib import Path
from fig5_common import fit_model_and_write

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_csv", required=True)
    ap.add_argument("--patient_group_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--prefer_numpyro", action="store_true")
    args = ap.parse_args()

    fit_model_and_write(
        model_kind="ou_branch",
        series_csv=Path(args.series_csv),
        patient_group_csv=Path(args.patient_group_csv),
        out_dir=Path(args.out_dir),
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
        prefer_numpyro=args.prefer_numpyro,
    )

if __name__ == "__main__":
    main()
