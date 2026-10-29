"""
Null distribution of the random-retrieval baseline.

The manuscript's random baseline is a single draw.  This script rebuilds the
38,804-structure retrieval pool exactly as described in SI Section S3 and
repeats random retrieval many times, so the random row of the retrieval table
can be reported as a mean with an interval, and the observed band-gap
Spearman of 0.168 can be placed in its null distribution.

Per draw and per property, every query receives k structures drawn uniformly
without replacement from the pool; the neighbor estimate is the mean over the
drawn structures that carry the property, and a query counts only when at least
one of them does (the rule used for K_VRH in the paper).  Reported per property:
per-query MAE and Spearman (mean, std, 2.5/97.5 percentiles over draws), the
usable-query count, and the pool-mean and pool-median constant baselines.
With --k-physical, queries and pool entries whose K_VRH is outside
[--kmin-gpa, --kmax-gpa] are excluded from the K_VRH column.

The pool comes either from data/labels/retrieval_pool.csv, which this
repository ships (--pool-csv), or is rebuilt from the alex_mp20 release
(--alex-zip, which also needs --mp20-csv).
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

COLS = {"eform": "formation_energy_per_atom", "bandgap": "dft_band_gap", "gvrh": "log10_bulk_modulus"}
QCOLS = {"eform": "formation_energy_per_atom", "bandgap": "band_gap", "gvrh": "bulk_modulus"}


def build_pool(alex_zip, mp20_csv, exclude_ids):
    with zipfile.ZipFile(alex_zip) as z:
        use = ["material_id", "dft_band_gap", "dft_bulk_modulus"]
        parts = [pd.read_csv(z.open(f"alex_mp_20/{s}.csv"), usecols=use) for s in ("train", "val")]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("material_id")
    n_release = len(df)
    df = df[df["dft_band_gap"].notna() | df["dft_bulk_modulus"].notna()]
    n_prop = len(df)
    df = df[~df["material_id"].isin(exclude_ids)].reset_index(drop=True)
    k = df["dft_bulk_modulus"]
    df["log10_bulk_modulus"] = np.where(k.notna() & (k > 0), np.log10(k.clip(lower=1e-6)), np.nan)
    ef = pd.read_csv(mp20_csv, usecols=["material_id", "formation_energy_per_atom"]).drop_duplicates("material_id")
    df = df.merge(ef, on="material_id", how="left")
    audit = {
        "release_rows": int(n_release), "with_bandgap_or_bulk": int(n_prop), "pool": int(len(df)),
        "with_eform": int(df["formation_energy_per_atom"].notna().sum()),
        "with_bandgap": int(df["dft_band_gap"].notna().sum()),
        "with_bulk": int(df["log10_bulk_modulus"].notna().sum()),
        "mp_ids": int(df["material_id"].str.startswith("mp-").sum()),
    }
    return df, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool-csv", type=Path,
                     help="pre-built retrieval pool (data/labels/retrieval_pool.csv); "
                          "the mp20_5k ids are already excluded")
    src.add_argument("--alex-zip", type=Path,
                     help="alex_mp_20.zip, to rebuild the pool from the release instead")
    ap.add_argument("--mp20-csv", type=Path, default=None,
                    help="45,229-row MP20 table for the formation-energy join (only with --alex-zip)")
    ap.add_argument("--labels-dir", type=Path, required=True, help="prepared/mp20_5k with {train,val,test}/labels.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-physical", action="store_true")
    ap.add_argument("--kmin-gpa", type=float, default=1.0)
    ap.add_argument("--kmax-gpa", type=float, default=600.0)
    ap.add_argument("--observed-json", type=Path, default=None,
                    help="optional {prop: {mae:..., spearman:...}} of the single published draw")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    splits = {s: pd.read_csv(a.labels_dir / s / "labels.csv") for s in ("train", "val", "test")}
    exclude = set(pd.concat(splits.values())["material_id"])
    if a.pool_csv:
        pool = pd.read_csv(a.pool_csv)
        audit = {"source": str(a.pool_csv), "pool": int(len(pool)),
                 "with_eform": int(pool["formation_energy_per_atom"].notna().sum()),
                 "with_bandgap": int(pool["dft_band_gap"].notna().sum()),
                 "with_bulk": int(pool["log10_bulk_modulus"].notna().sum())}
    else:
        if a.mp20_csv is None:
            ap.error("--mp20-csv is required with --alex-zip")
        pool, audit = build_pool(a.alex_zip, a.mp20_csv, exclude)
    q = splits["test"]
    print(json.dumps(audit))

    rng = np.random.default_rng(a.seed)
    n_q, n_pool = len(q), len(pool)
    lo, hi = np.log10(a.kmin_gpa), np.log10(a.kmax_gpa)
    result = {"pool_audit": audit, "k": a.k, "draws": a.draws, "seed": a.seed,
              "n_queries": n_q, "k_physical": a.k_physical, "properties": {}}

    vals = {p: pool[COLS[p]].to_numpy(float) for p in COLS}
    yq = {p: q[QCOLS[p]].to_numpy(float) for p in QCOLS}
    if a.k_physical:
        g = vals["gvrh"]
        vals["gvrh"] = np.where((g >= lo) & (g <= hi), g, np.nan)
        yq["gvrh"] = np.where((yq["gvrh"] >= lo) & (yq["gvrh"] <= hi), yq["gvrh"], np.nan)

    mae = {p: np.empty(a.draws) for p in COLS}
    rho = {p: np.empty(a.draws) for p in COLS}
    usable = {p: np.empty(a.draws) for p in COLS}
    for d in range(a.draws):
        idx = np.stack([rng.choice(n_pool, a.k, replace=False) for _ in range(n_q)])  # (n_q, k)
        for p in COLS:
            nb = vals[p][idx]
            cnt = np.isfinite(nb).sum(1)
            est = np.where(cnt > 0, np.nansum(nb, 1) / np.maximum(cnt, 1), np.nan)
            ok = np.isfinite(est) & np.isfinite(yq[p])
            usable[p][d] = ok.sum()
            mae[p][d] = np.abs(yq[p][ok] - est[ok]).mean()
            rho[p][d] = spearmanr(yq[p][ok], est[ok]).correlation

    obs = json.loads(a.observed_json.read_text()) if a.observed_json else {}
    for p in COLS:
        v = vals[p][np.isfinite(vals[p])]
        y = yq[p][np.isfinite(yq[p])]
        r = {
            "mae_mean": float(mae[p].mean()), "mae_std": float(mae[p].std()),
            "mae_ci95": [float(x) for x in np.percentile(mae[p], [2.5, 97.5])],
            "spearman_mean": float(np.nanmean(rho[p])), "spearman_std": float(np.nanstd(rho[p])),
            "spearman_ci95": [float(x) for x in np.nanpercentile(rho[p], [2.5, 97.5])],
            "usable_queries_mean": float(usable[p].mean()),
            "usable_queries_range": [int(usable[p].min()), int(usable[p].max())],
            "pool_mean_mae": float(np.abs(y - v.mean()).mean()),
            "pool_median_mae": float(np.abs(y - np.median(v)).mean()),
        }
        if p in obs:
            if "spearman" in obs[p]:
                r["frac_draws_spearman_ge_observed"] = float(np.mean(rho[p] >= obs[p]["spearman"]))
            if "mae" in obs[p]:
                r["frac_draws_mae_le_observed"] = float(np.mean(mae[p] <= obs[p]["mae"]))
        result["properties"][p] = r
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["properties"], indent=2))


if __name__ == "__main__":
    main()
