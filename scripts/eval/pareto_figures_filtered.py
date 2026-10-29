"""
Pareto-front figures (Figure 3 / Figure S1 style) on the held-out library after
removing structures with non-physical measured K_VRH, using the repository's
own plotting code (plot_pareto_front_arms.make_figure) and front logic
(compare_pareto_erm_vs_mex.front_metrics).  Model fronts are selected on
predictions and plotted at measured DFT values.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto_filtered import load_arm  # noqa: E402

SETS = {"mex": ("MEX-Single", "MEX-Multi"), "erm": ("ERM-Single", "ERM-Multi")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=KIND:PATTERN (as pareto_filtered.py)")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--directions", nargs=3, default=["min", "min", "min"])
    ap.add_argument("--kmin-gpa", type=float, default=1.0)
    ap.add_argument("--kmax-gpa", type=float, default=600.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    from compare_pareto_erm_vs_mex import front_metrics  # noqa: E402
    from plot_pareto_front_arms import make_figure  # noqa: E402

    specs = {}
    for s in a.arm:
        name, rest = s.split("=", 1)
        specs[name] = rest.split(":", 1)
    lo, hi = np.log10(a.kmin_gpa), np.log10(a.kmax_gpa)

    for set_name, arms in SETS.items():
        per_seed = []
        for seed in a.seeds:
            arm_pts, true_pts = {}, None
            for arm in arms:
                ids, preds, gts, _ = load_arm(*specs[arm], seed)
                keep = (gts[:, 2] >= lo) & (gts[:, 2] <= hi)
                preds, gts = preds[keep], gts[keep]
                m = front_metrics(preds, gts, a.directions)
                arm_pts[arm] = gts[m["model_front"]]
                t = gts[m["true_front"]]
                if true_pts is not None:
                    assert len(t) == len(true_pts), "true fronts differ between arms"
                true_pts = t
                print(f"{set_name} seed {seed} {arm}: front {m['n_model']}, recovered {m['hit']}/{m['n_true']}")
            per_seed.append((f"seed {seed}", true_pts, arm_pts))
        tag = "_".join(a.directions)
        make_figure(set_name, per_seed, "dft", a.out_dir / f"pareto_arms_{set_name}_dft_heldout_kphys_{tag}.png")


if __name__ == "__main__":
    main()
