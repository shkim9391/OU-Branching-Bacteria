"""
Compare three fitted models using ArviZ compare on "loglik_total".
"""

import argparse
from pathlib import Path
import arviz as az

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rw", required=True)
    ap.add_argument("--ou", required=True)
    ap.add_argument("--ou_branch", required=True)
    ap.add_argument("--ic", default="loo", choices=["loo", "waic"])
    args = ap.parse_args()

    id_rw = az.from_netcdf(Path(args.rw))
    id_ou = az.from_netcdf(Path(args.ou))
    id_ob = az.from_netcdf(Path(args.ou_branch))

    cmp = az.compare({"RW": id_rw, "OU": id_ou, "OU-Branch": id_ob}, ic=args.ic, var_name="loglik_total")
    print(cmp)
    out_path = Path(args.rw).parent / f"model_compare_{args.ic}.csv"
    cmp.to_csv(out_path)
    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
