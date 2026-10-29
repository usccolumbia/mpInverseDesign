"""
Pareto-front recovery on the held-out library with non-physical K_VRH entries
removed from the library before any front is computed.

The manuscript's Tables 4-5 use the library as released, which contains bulk
moduli far outside the physical range (up to ~1e10 GPa, and values below 1 GPa).
Those entries can sit on the true front -- in particular under every objective
that maximises K_VRH -- and they stretch the min-max rescaling used by IGD+ and
HV.  This script drops library structures whose *measured* K_VRH lies outside
[--kmin-gpa, --kmax-gpa] and recomputes, per arm and seed, front size, recall,
precision, enrichment, IGD+ and HV ratio under all four objective conventions.
The unfiltered numbers are recomputed alongside as a consistency check against
the manuscript.

Domination, metrics and normalisation are imported unchanged from the
repository's compare_pareto_erm_vs_mex.py (HV reference point 1.1 in min-max
scaled cost space).

Arms are given as  NAME=KIND:PATTERN  where KIND is
  triple : three files, one per property; PATTERN contains {prop} and {seed}
  single : one file holding y_eform, y_bandgap, y_gvrh; PATTERN contains {seed}
Each file may store its prediction as 'y_<prop>' (multi trainer) or 'y'
(single-property trainers), with matching '_gts'.

Example:
  python pareto_filtered.py --repo code \
    --arm "MEX-Single=triple:runs/mex_single_{prop}_seed{seed}/results/heldout/*predictions*.npz" \
    --arm "MEX-Multi=single:/path/mex_multi_seed{seed}/results/heldout*/ocp_predictions.npz" \
    --out-dir reports/pareto_filtered
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROPS = ["eform", "bandgap", "gvrh"]
DIRECTIONS = {
    "min_min_min": ["min", "min", "min"],
    "min_1.4_max": ["min", "1.4", "max"],
    "min_max_max": ["min", "max", "max"],
    "min_min_max": ["min", "min", "max"],
}


def _read(path, prop):
    d = np.load(path, allow_pickle=True)
    key = f"y_{prop}" if f"y_{prop}" in d else "y"
    ids = np.asarray(d["ids"]).reshape(-1).astype(str)
    return ids, np.asarray(d[key]).reshape(-1), np.asarray(d[f"{key}_gts"]).reshape(-1)


def _one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True), key=lambda p: Path(p).stat().st_mtime)
    return hits[-1] if hits else None


def load_arm(kind, pattern, seed):
    """(ids, preds (N,3), gts (N,3)) or None if any file is missing."""
    if kind == "single":
        f = _one(pattern.format(seed=seed))
        if f is None:
            return None
        cols = [_read(f, p) for p in PROPS]
        files = [f]
    else:
        files = [_one(pattern.format(prop=p, seed=seed)) for p in PROPS]
        if any(f is None for f in files):
            return None
        cols = [_read(f, p) for f, p in zip(files, PROPS)]
    common = set(cols[0][0])
    for c in cols[1:]:
        common &= set(c[0])
    order = [i for i in cols[0][0] if i in common]
    preds = np.empty((len(order), 3))
    gts = np.empty((len(order), 3))
    for k, (ids, pr, gt) in enumerate(cols):
        pos = {s: j for j, s in enumerate(ids)}
        idx = np.array([pos[s] for s in order])
        preds[:, k], gts[:, k] = pr[idx], gt[idx]
    return np.array(order), preds, gts, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="mex_multi_property checkout (for compare_pareto_erm_vs_mex)")
    ap.add_argument("--arm", action="append", required=True, help="NAME=KIND:PATTERN")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--kmin-gpa", type=float, default=1.0)
    ap.add_argument("--kmax-gpa", type=float, default=600.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    from compare_pareto_erm_vs_mex import front_metrics  # noqa: E402

    lo, hi = np.log10(a.kmin_gpa), np.log10(a.kmax_gpa)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows, audit = [], {"kmin_gpa": a.kmin_gpa, "kmax_gpa": a.kmax_gpa, "arms": {}}

    for spec in a.arm:
        name, rest = spec.split("=", 1)
        kind, pattern = rest.split(":", 1)
        for seed in a.seeds:
            got = load_arm(kind, pattern, seed)
            if got is None:
                print(f"[skip] {name} seed {seed}: files not found for {pattern}")
                continue
            ids, preds, gts, files = got
            keep = (gts[:, 2] >= lo) & (gts[:, 2] <= hi) & np.all(np.isfinite(gts), axis=1)
            audit["arms"].setdefault(name, {})[seed] = {
                "files": [str(f) for f in files], "n_library": int(len(ids)),
                "n_kept": int(keep.sum()), "n_removed": int((~keep).sum()),
                "removed_ids": ids[~keep].tolist(),
            }
            for label, mask in (("unfiltered", np.ones(len(ids), bool)), ("filtered", keep)):
                for dname, dirs in DIRECTIONS.items():
                    m = front_metrics(preds[mask], gts[mask], dirs)
                    rows.append({
                        "library": label, "directions": dname, "arm": name, "seed": seed,
                        "n": m["n"], "front": m["n_model"], "true_front": m["n_true"],
                        "recovered": m["hit"], "recall_pct": 100 * m["recall"],
                        "precision_pct": 100 * m["precision"], "enrichment": m["enrichment"],
                        "igd_plus": m["igd_plus"], "hv_ratio": m["hv_ratio"],
                    })

    if not rows:
        raise SystemExit("no arm/seed could be loaded")
    with open(a.out_dir / "pareto_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.out_dir / "library_filter_audit.json").write_text(json.dumps(audit, indent=1))

    # Summary: per-seed counts plus mean +/- population std, as in the manuscript.
    lines = ["# Pareto recovery, held-out library",
             f"K_VRH kept in [{a.kmin_gpa:g}, {a.kmax_gpa:g}] GPa for the filtered rows.", ""]
    keys = sorted({(r["library"], r["directions"]) for r in rows}, key=lambda k: (k[0] != "unfiltered", k[1]))
    for lib, dname in keys:
        sub = [r for r in rows if r["library"] == lib and r["directions"] == dname]
        tf = sorted({r["true_front"] for r in sub})
        lines += [f"## {lib}, {dname}  (n={sub[0]['n']}, |F*|={tf})", "",
                  "| arm | front | rec. per seed | recall % | precision % | enrich. | IGD+ | HV ratio |",
                  "|---|---|---|---|---|---|---|---|"]
        for arm in dict.fromkeys(r["arm"] for r in sub):
            s = [r for r in sub if r["arm"] == arm]
            def ms(k, fmt):
                v = np.array([r[k] for r in s], float)
                return f"{v.mean():{fmt}} ± {v.std():{fmt}}"
            fr = np.array([r["front"] for r in s], float)
            hit = np.array([r["recovered"] for r in s], float)
            prec = 100 * hit.mean() / fr.mean() if fr.mean() else float("nan")
            lines.append(f"| {arm} | {ms('front', '.1f')} | {', '.join(str(r['recovered']) for r in s)} | "
                         f"{ms('recall_pct', '.1f')} | {prec:.1f} | {prec / (100 * s[0]['true_front'] / s[0]['n']):.1f}× | "
                         f"{ms('igd_plus', '.4f')} | {ms('hv_ratio', '.4f')} |")
        lines.append("")
    (a.out_dir / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
