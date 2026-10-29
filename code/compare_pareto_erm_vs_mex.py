"""
compare_pareto_erm_vs_mex.py
============================
Does the Pareto-front result need the energy heads?

sec:pareto_viz runs one DFO pass per property, collapses each energy head to a
point estimate, and applies NSGA-II non-dominated sorting across the test
library.  That procedure needs *a predictor*, not specifically an energy-based
one.  A plain regression (ERM) model produces point estimates too -- and on
mp20_5k it produces better ones (MAE 0.061/0.287/0.083 vs MEX-Multi's
0.100/0.328/0.088).

So the obvious reviewer question is: run the same non-dominated sorting on the
ERM predictions and see whether the front is any worse.  This script answers
it.  If MEX-Multi recovers more of the true front, the Pareto section has a
defensible claim.  If ERM matches or beats it, the section cannot be framed as
requiring MEX, and that is better learned here than in review.

WHAT IT COMPARES
----------------
For each seed, and for each of {MEX-Multi, ERM}:
  * model front  : non-dominated subset of the 250 predicted property vectors
  * true front   : non-dominated subset of the 250 DFT-measured vectors
                   (identical for both models by construction -- it depends
                   only on ground truth, so it doubles as a sanity check)
  * recall       : |model front INTERSECT true front| / |true front|
  * precision    : |model front INTERSECT true front| / |model front|
  * enrichment   : precision / (|true front| / N)   i.e. lift over a random
                   shortlist of the same size

MERGING ERM PREDICTIONS
-----------------------
ERM trains one model per property, so its predictions arrive as three separate
npz files each holding 'y' and 'y_gts'.  MEXMultiTrainer writes one file with
'y_eform', 'y_bandgap', 'y_gvrh' (+ '_gts').  This script merges the three ERM
files into the same (N, 3) layout, aligning rows by the 'ids' field rather
than assuming the files are in the same order.  (They are -- all four mp20_5k
test LMDBs hold the same 250 structures in the same order -- but aligning by
id makes that a checked fact rather than an assumption.)

DOMINATION LOGIC
----------------
Imported from plot_pareto_front_structurelevel.py rather than reimplemented,
so both this script and the figure use byte-identical definitions of
"dominates".  The convention there is minimise-all-three, inherited from
MEXMultiTrainer.  Use --directions to test other conventions.

USAGE
-----
Auto-discovery (expects the layout the training scripts create):

    python compare_pareto_erm_vs_mex.py \\
        --erm-root outputs --mex-root outputs --seeds 0 1 2

Explicit paths for one seed:

    python compare_pareto_erm_vs_mex.py \\
        --erm  <eform.npz> <bandgap.npz> <gvrh.npz> \\
        --mex  <multi.npz> --label seed0

Sensitivity to the objective directions (sec:pareto_viz reports four):

    python compare_pareto_erm_vs_mex.py --erm-root outputs --mex-root outputs \\
        --seeds 0 1 2 --directions min min max
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse the exact domination logic the figure uses.  Importing rather than
# copying guarantees this script and plot_pareto_front_structurelevel.py can
# never drift apart on what "non-dominated" means.
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))
from plot_pareto_front_structurelevel import (  # noqa: E402
    non_dominated_front,
    PROP_NAMES,
)

PROP_LABELS = {"eform": "formation energy", "bandgap": "band gap", "gvrh": "bulk modulus"}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_multi_npz(path):
    """MEXMultiTrainer.predict() output -> (preds, gts, ids), each (N, 3)."""
    d = np.load(path, allow_pickle=True)
    missing = [f"y_{p}" for p in PROP_NAMES if f"y_{p}" not in d]
    if missing:
        raise ValueError(f"{path}: missing {missing}. Expected a multi-property predict output.")
    preds = np.stack([np.asarray(d[f"y_{p}"]).reshape(-1) for p in PROP_NAMES], axis=1)
    gts = np.stack([np.asarray(d[f"y_{p}_gts"]).reshape(-1) for p in PROP_NAMES], axis=1)
    ids = (np.asarray(d["ids"]).reshape(-1).astype(str)
           if "ids" in d else np.arange(len(preds)).astype(str))
    return preds, gts, ids


def load_erm_npzs(paths):
    """
    Three single-property OCPTrainer.predict() outputs -> (preds, gts, ids).

    paths must be in PROP_NAMES order (eform, bandgap, gvrh).  Rows are
    aligned by 'ids'; if the three files disagree on which structures they
    cover, we keep the intersection and say so.
    """
    if len(paths) != len(PROP_NAMES):
        raise ValueError(f"--erm needs {len(PROP_NAMES)} files in order {PROP_NAMES}, got {len(paths)}")

    per_prop = []
    for prop, path in zip(PROP_NAMES, paths):
        d = np.load(path, allow_pickle=True)
        if "y" not in d or "y_gts" not in d:
            raise ValueError(f"{path}: missing 'y' or 'y_gts'. Expected a single-property predict output.")
        ids = (np.asarray(d["ids"]).reshape(-1).astype(str)
               if "ids" in d else np.arange(len(d["y"])).astype(str))
        per_prop.append({
            "ids": ids,
            "pred": np.asarray(d["y"]).reshape(-1),
            "gt": np.asarray(d["y_gts"]).reshape(-1),
        })

    common = set(per_prop[0]["ids"])
    for pp in per_prop[1:]:
        common &= set(pp["ids"])
    if not common:
        raise ValueError("The three ERM files share no structure ids.")

    # Order by the first file so output is deterministic.
    order = [i for i in per_prop[0]["ids"] if i in common]
    n_dropped = len(per_prop[0]["ids"]) - len(order)
    if n_dropped:
        print(f"    note: {n_dropped} structures not present in all three ERM files; using the "
              f"{len(order)} shared ones")

    preds = np.empty((len(order), len(PROP_NAMES)))
    gts = np.empty((len(order), len(PROP_NAMES)))
    for k, pp in enumerate(per_prop):
        pos = {sid: j for j, sid in enumerate(pp["ids"])}
        idx = np.array([pos[i] for i in order])
        preds[:, k] = pp["pred"][idx]
        gts[:, k] = pp["gt"][idx]
    return preds, gts, np.array(order)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def apply_directions(mat, directions):
    """
    Turn raw property values into costs that non_dominated_front can minimise.

    'min'   -> as-is                (lower is better)
    'max'   -> negate               (higher is better)
    '<val>' -> |value - target|     (closer to <val> is better, e.g. 1.4 for a
                                     photovoltaic band gap)
    """
    out = mat.astype(float).copy()
    for k, d in enumerate(directions):
        if d == "min":
            continue
        elif d == "max":
            out[:, k] = -out[:, k]
        else:
            try:
                target = float(d)
            except ValueError:
                raise ValueError(f"bad direction {d!r}: expected 'min', 'max', or a number")
            out[:, k] = np.abs(out[:, k] - target)
    return out


def fast_non_dominated_front(costs):
    """
    Same result as non_dominated_front, without its O(n^2) Python loop.

    The reference implementation is fine at n=250 but takes minutes at
    n=4,803 (the held-out library), and the comparison calls it ~100 times.
    This one is the standard iterative-filter method, with exact duplicate rows
    collapsed first so that identical points -- which do NOT dominate each
    other under `all(a<=b) and any(a<b)` -- are all kept or all dropped
    together.  Verified byte-identical to non_dominated_front on real data
    across all four objective conventions and on tie-heavy random matrices;
    check_front_agreement() below re-verifies at runtime on small inputs.
    """
    costs = np.asarray(costs, dtype=float)
    if len(costs) == 0:
        return np.zeros(0, dtype=bool)
    uniq, inv = np.unique(costs, axis=0, return_inverse=True)
    order = np.argsort(uniq[:, 0], kind="stable")
    c = uniq[order]
    idx = np.arange(len(c))
    i = 0
    while i < len(c):
        keep = np.any(c < c[i], axis=1)
        keep[i] = True
        idx = idx[keep]
        c = c[keep]
        i = int(np.sum(keep[:i])) + 1
    eff = np.zeros(len(uniq), dtype=bool)
    eff[order[idx]] = True
    return eff[inv]


def check_front_agreement(costs, limit=400):
    """Cross-check the fast front against the reference; cheap at small n."""
    if len(costs) <= limit:
        assert np.array_equal(non_dominated_front(costs), fast_non_dominated_front(costs)), (
            "fast_non_dominated_front disagrees with non_dominated_front"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Continuous front-quality metrics
#
# Recall counts exact hits and therefore has a ceiling: at |F*|=18, an arm
# sitting at 16 has two structures of headroom and every comparison collapses
# to "indistinguishable".  IGD and hypervolume have no ceiling -- they give
# partial credit for near-misses, so an arm that selects structures just off
# the front scores better than one that selects structures far from it, and
# two arms that both recover 16/18 can still be separated.
#
# WHICH COORDINATES
# -----------------
# The model's front is SELECTED in predicted space and SCORED at DFT
# coordinates: take the structures the model puts on its front, look up their
# measured properties, and ask how good that set actually is.  This is the
# design-relevant question -- you propose these structures, you synthesise
# them, how good is what you got -- and unlike scoring in predicted space it
# cannot be gamed by a model that simply predicts optimistic values.
#
# NORMALISATION
# -------------
# Objectives are min-max scaled to [0,1] using the ground truth of the whole
# library, after direction mapping.  The scaling is therefore a property of the
# data, identical for every arm and every seed, and the three properties
# (eV/atom, eV, log10 GPa) contribute comparably instead of whichever has the
# largest raw range dominating.  Every scored point is a library point, so all
# coordinates land in [0,1] and the reference point 1.1 is dominated by all of
# them.
# ─────────────────────────────────────────────────────────────────────────────

HV_REF = 1.1


def _hv_2d(points, ref):
    """Exact 2D hypervolume (minimisation) of the union of [p, ref] boxes."""
    if len(points) == 0:
        return 0.0
    # Sort by x ascending. Sweeping left to right, the strip [x_k, x_{k+1}) is
    # covered in y down to the best (lowest) y seen at or before x_k, because
    # every such point's box extends rightwards to ref.
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    xs, ys = pts[:, 0], pts[:, 1]
    area = 0.0
    y_best = np.inf
    for k in range(len(pts)):
        y_best = min(y_best, ys[k])
        x_next = xs[k + 1] if k + 1 < len(pts) else ref[0]
        area += max(x_next - xs[k], 0.0) * max(ref[1] - y_best, 0.0)
    return float(area)


def hypervolume(points, ref=None):
    """
    Exact hypervolume for 2 or 3 minimised objectives.

    3D is computed by sweeping the third objective: between consecutive z
    values the cross-section is the 2D hypervolume of everything seen so far.
    Points not strictly dominating `ref` are dropped, as they contribute
    nothing.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) == 0:
        return 0.0
    d = points.shape[1]
    ref_v = np.full(d, HV_REF, dtype=float) if ref is None else np.asarray(ref, float)

    points = points[np.all(points < ref_v, axis=1)]
    if len(points) == 0:
        return 0.0
    points = points[fast_non_dominated_front(points)]   # HV(S) == HV(front(S))

    if d == 2:
        return _hv_2d(points, ref_v)
    if d != 3:
        raise NotImplementedError(f"hypervolume supports 2 or 3 objectives, got {d}")

    pts = points[np.argsort(points[:, 2], kind="stable")]
    vol = 0.0
    for k in range(len(pts)):
        z = pts[k, 2]
        z_next = pts[k + 1, 2] if k + 1 < len(pts) else ref_v[2]
        if z_next <= z:
            continue                       # zero-thickness slab (tied z)
        area = _hv_2d(pts[: k + 1, :2], ref_v[:2])
        vol += area * (z_next - z)
    return float(vol)


def igd(reference_set, approx, plus=False):
    """
    Inverted generational distance from `approx` to `reference_set`.

    Mean over true-front points of the distance to the nearest selected point.
    Lower is better; 0 means every true-front point is matched exactly.

    plus=True gives IGD+ (Ishibuchi et al.), which measures only the component
    by which the selected point is WORSE than the reference point.  IGD+ is
    Pareto-compliant -- a front that dominates another can never score worse --
    whereas plain IGD can prefer a dominated front that happens to sit closer
    in Euclidean terms.  Both are reported; IGD+ is the one to trust when they
    disagree.
    """
    R = np.asarray(reference_set, float)
    A = np.asarray(approx, float)
    if len(R) == 0:
        return float("nan")
    if len(A) == 0:
        return float("inf")
    diff = A[None, :, :] - R[:, None, :]          # (|R|, |A|, d)
    if plus:
        diff = np.maximum(diff, 0.0)
    return float(np.sqrt((diff ** 2).sum(axis=2)).min(axis=1).mean())


def front_metrics(preds, gts, directions):
    """
    Model front, true front, and both count-based and continuous statistics.

    Count-based (recall / precision / enrichment) are exact-hit measures and
    saturate on a small front.  Continuous (igd / igd_plus / hv_ratio) do not,
    and are computed on the model's selected structures scored at their DFT
    coordinates -- see the section comment above.
    """
    n = len(preds)
    pred_costs = apply_directions(preds, directions)
    gt_costs = apply_directions(gts, directions)

    check_front_agreement(pred_costs)
    model_front = fast_non_dominated_front(pred_costs)
    true_front = fast_non_dominated_front(gt_costs)

    hit = int((model_front & true_front).sum())
    n_model = int(model_front.sum())
    n_true = int(true_front.sum())

    recall = hit / n_true if n_true else float("nan")
    precision = hit / n_model if n_model else float("nan")
    base_rate = n_true / n if n else float("nan")
    enrichment = precision / base_rate if base_rate else float("nan")

    # Scale by the library's ground truth so the normalisation is identical for
    # every arm and seed. A degenerate (constant) objective gets span 1 so it
    # contributes nothing rather than producing a divide-by-zero.
    lo = gt_costs.min(axis=0)
    span = gt_costs.max(axis=0) - lo
    span[span <= 0] = 1.0
    gt_norm = (gt_costs - lo) / span

    selected = gt_norm[model_front]     # selected in predicted space, scored at DFT
    reference = gt_norm[true_front]

    hv_model = hypervolume(selected)
    hv_true = hypervolume(reference)

    return {
        "n": n, "n_model": n_model, "n_true": n_true, "hit": hit,
        "recall": recall, "precision": precision,
        "base_rate": base_rate, "enrichment": enrichment,
        "igd": igd(reference, selected, plus=False),
        "igd_plus": igd(reference, selected, plus=True),
        "hv_model": hv_model, "hv_true": hv_true,
        "hv_ratio": hv_model / hv_true if hv_true > 0 else float("nan"),
        "model_front": model_front, "true_front": true_front,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def newest(paths):
    paths = list(paths)
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def find_erm(root, seed, pattern=None):
    """
    Locate the three single-property ERM prediction files for one seed.

    pattern may contain {prop} and {seed}.  Default matches the layout created
    by run_scripts/train_mp20_5k_erm_slurm.sh, under either the flat or the
    results/<timestamp_id>/ arrangement.
    """
    pattern = pattern or "mp20_5k_erm_{prop}_seed{seed}/**/ocp_predictions.npz"
    out = []
    for prop in PROP_NAMES:
        pat = pattern.format(prop=prop, seed=seed)
        p = newest(Path(root).glob(pat))
        if p is None:
            return None
        out.append(p)
    return out


def find_mex(root, seed, pattern=None):
    """
    Locate the multi-property prediction file for one seed.

    A MEX-Multi run directory can hold more than one ocp_predictions.npz: one
    written during training at every best-checkpoint save, and one per
    explicit --mode predict / pareto pass.  They come from the same checkpoint,
    but pass --mex-pattern to pin a specific one rather than relying on mtime.
    """
    if pattern:
        return newest(Path(root).glob(pattern.format(seed=seed)))
    pats = [
        f"*mex_multi_seed{seed}*/results/**/ocp_predictions.npz",
        f"*mex_multi_seed{seed}*/ocp_predictions.npz",
        f"*multi_seed{seed}*/results/**/ocp_predictions.npz",
        f"*multi_seed{seed}*/ocp_predictions.npz",
    ]
    for pat in pats:
        p = newest(Path(root).glob(pat))
        if p is not None:
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Arm definitions
#
# Four arms: {single, multi} x {MEX head, regression head}.  "single" arms keep
# one model per property, so their predictions arrive as three npz files that
# get merged by load_erm_npzs; "multi" arms write one npz with all three
# targets and are read by load_multi_npz.
#
# A NOTE ON PROCEDURE, which matters when reading the output below.  The two
# MEX arms reach their point estimates by derivative-free search: S=2000
# candidate property values, T=5 refinement iterations, noise 0.5 shrinking
# x0.5, inside bounds anchored to the training-set extremes
# (mex_multi_trainer.py:280-325).  The two regression arms have no such stage
# at all -- one forward pass, read the heads.  These are therefore different
# inference procedures, not just different heads, and "identical DFO settings
# across all four arms" is not a thing that can be arranged.  What IS held
# fixed, and what makes the comparison meaningful, is everything downstream:
# the same 250 test structures, one point-estimate vector per structure, the
# same non-dominated sorting code, the same objective directions, and the same
# DFT-derived true front.
# ─────────────────────────────────────────────────────────────────────────────

ARMS = [
    # (display name, kind, root attr, pattern attr, default pattern)
    ("MEX-single",  "triple", "mex_single_root", "mex_single_pattern", None),
    ("MEX-multi",   "single", "mex_root",        "mex_pattern",        None),
    ("ERM-single",  "triple", "erm_root",        "erm_pattern",        None),
    ("ERM-multi",   "single", "erm_multi_root",  "erm_multi_pattern",
     "mp20_5k_erm_multi_seed{seed}/**/ocp_predictions.npz"),
]


def compare_pair(name_a, ms_a, name_b, ms_b, n_true):
    """
    Plain-language verdict on whether two arms' fronts differ.

    With an 18-structure true front, one recovered structure is 5.6 points of
    recall, and three seeds give almost no power to resolve anything smaller
    than that.  So rather than quoting a p-value that three points cannot
    support, the rule here is deliberately blunt and stated in the output:

      * if the per-seed recovered counts of the two arms overlap in range, or
      * if the gap in mean recovered count is under one structure, or
      * if the gap is smaller than the larger of the two cross-seed spreads,

    the arms are called indistinguishable.  Only a gap that clears all three
    is reported as a difference, and even then the per-seed counts are printed
    so the reader can judge it themselves.
    """
    a = np.array([m["hit"] for m in ms_a], dtype=float)
    b = np.array([m["hit"] for m in ms_b], dtype=float)
    delta = a.mean() - b.mean()
    spread = max(a.std(), b.std())
    overlap = (a.min() <= b.max()) and (b.min() <= a.max())

    per_seed = (f"{name_a} {[int(x) for x in a]} vs "
                f"{name_b} {[int(x) for x in b]} out of {n_true}")

    if overlap or abs(delta) < 1.0 or abs(delta) <= spread:
        reason = ("ranges overlap" if overlap
                  else "gap under one structure" if abs(delta) < 1.0
                  else "gap within cross-seed spread")
        return f"    {name_a} vs {name_b}: INDISTINGUISHABLE ({reason}). {per_seed}"

    ahead, behind = (name_a, name_b) if delta > 0 else (name_b, name_a)
    return (f"    {name_a} vs {name_b}: {ahead} ahead of {behind} by "
            f"{abs(delta):.1f} structures on average. {per_seed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--erm", nargs=3, type=Path,
                    help=f"three single-property npz in order {PROP_NAMES}")
    ap.add_argument("--mex", type=Path, help="one multi-property npz")
    ap.add_argument("--label", default="run", help="label when using explicit paths")

    ap.add_argument("--erm-root", default=None,
                    help="auto-discover single-property ERM predictions under here")
    ap.add_argument("--mex-root", default=None,
                    help="auto-discover MEX-Multi predictions under here")
    ap.add_argument("--erm-multi-root", default=None,
                    help="auto-discover multi-property ERM predictions under here "
                         "(the equiformer_v2-direct-multi arm)")
    ap.add_argument("--mex-single-root", default=None,
                    help="auto-discover single-property MEX predictions under here")

    ap.add_argument("--erm-pattern", default=None,
                    help="glob under --erm-root; may use {prop} and {seed}. "
                         "Default: mp20_5k_erm_{prop}_seed{seed}/**/ocp_predictions.npz")
    ap.add_argument("--mex-pattern", default=None,
                    help="glob under --mex-root; may use {seed}. Pin this when a run dir holds "
                         "several ocp_predictions.npz, e.g. "
                         "'*_seed{seed}/results/*structurelevel_pareto*/ocp_predictions.npz'")
    ap.add_argument("--erm-multi-pattern", default=None,
                    help="glob under --erm-multi-root; may use {seed}")
    ap.add_argument("--mex-single-pattern", default=None,
                    help="glob under --mex-single-root; may use {prop} and {seed}")

    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--directions", nargs=3, default=["min", "min", "min"],
                    metavar=("EFORM", "BANDGAP", "GVRH"),
                    help="'min', 'max', or a target value per property (default: min min min, "
                         "the convention used in sec:pareto_viz)")
    args = ap.parse_args()

    # jobs: label -> {arm name: source path(s)}
    jobs = []
    if args.erm or args.mex:
        srcs = {}
        if args.mex:
            srcs["MEX-multi"] = args.mex
        if args.erm:
            srcs["ERM-single"] = args.erm
        jobs.append((args.label, srcs))
    else:
        active = [a for a in ARMS if getattr(args, a[2]) is not None]
        if not active:
            ap.error("give --erm/--mex, or at least one of --erm-root/--mex-root/"
                     "--erm-multi-root/--mex-single-root")
        for s in args.seeds:
            srcs = {}
            for name, kind, root_attr, pat_attr, default_pat in active:
                root = getattr(args, root_attr)
                pattern = getattr(args, pat_attr) or default_pat
                found = (find_erm(root, s, pattern) if kind == "triple"
                         else find_mex(root, s, pattern))
                if found is not None:
                    srcs[name] = found
                else:
                    print(f"seed {s}: {name} not found under {root}")
            if not srcs:
                print(f"seed {s}: nothing found, skipping")
                continue
            jobs.append((f"seed{s}", srcs))

    if not jobs:
        print("No prediction files found. Check the --*-root arguments.")
        return 1

    kinds = {name: kind for name, kind, *_ in ARMS}

    print(f"\nobjective directions: " +
          ", ".join(f"{p}={d}" for p, d in zip(PROP_NAMES, args.directions)))
    print("(non-domination convention imported from plot_pareto_front_structurelevel.py)")
    print("NOTE: MEX arms reach point estimates by derivative-free search; regression\n"
          "      arms by a single forward pass. Different procedures, same downstream\n"
          "      sorting. See the ARMS comment in this file.\n")

    rows = []
    for label, srcs in jobs:
        print(f"═══ {label}")
        for name, _kind, *_ in ARMS:
            src = srcs.get(name)
            if src is None:
                continue
            loader = load_erm_npzs if kinds[name] == "triple" else load_multi_npz
            try:
                preds, gts, ids = loader(src)
            except Exception as exc:
                print(f"  {name:12s} FAILED: {exc}")
                continue
            m = front_metrics(preds, gts, args.directions)
            rows.append((label, name, m))
            print(f"  {name:12s} n={m['n']:4d}  model front={m['n_model']:3d}  "
                  f"true front={m['n_true']:3d}  recovered={m['hit']:3d}  "
                  f"recall={m['recall']*100:5.1f}%  precision={m['precision']*100:5.1f}%  "
                  f"enrichment={m['enrichment']:4.1f}x")
            print(f"               IGD={m['igd']:.4f}  IGD+={m['igd_plus']:.4f} (lower better)  "
                  f"HV ratio={m['hv_ratio']:.4f} (higher better)")
            # A single-property arm is THREE files merged by structure id, one
            # per property. Print all of them: showing only the first made it
            # look as though the whole arm came from the eform run.
            if kinds[name] == "triple":
                for prop, path in zip(PROP_NAMES, src):
                    print(f"               [{prop:7s}] {path}")
            else:
                print(f"               {src}")
        print()

    by_model = {}
    for _, model, m in rows:
        by_model.setdefault(model, []).append(m)

    if len(by_model) < 2:
        return 0

    # ── per-seed recovered counts, printed before any mean ────────────────────
    # The means below hide the thing that matters most at this front size: how
    # few structures separate the arms.  Print the raw counts first.
    true_sizes = {m["n_true"] for _, _, m in rows}
    n_true = true_sizes.copy().pop() if len(true_sizes) == 1 else None
    seed_labels = [lab for lab, _ in jobs]

    print("═══ per-seed recovered counts (out of the true front)")
    header = "  {:12s}".format("model") + "".join(f"{s:>10s}" for s in seed_labels)
    print(header)
    for name in by_model:
        cells = []
        for lab in seed_labels:
            hit = next((m["hit"] for l, mo, m in rows if l == lab and mo == name), None)
            nt = next((m["n_true"] for l, mo, m in rows if l == lab and mo == name), None)
            cells.append(f"{hit}/{nt}" if hit is not None else "-")
        print("  {:12s}".format(name) + "".join(f"{c:>10s}" for c in cells))
    if n_true:
        print(f"\n  Scale: the true front has {n_true} structures, so one recovered "
              f"structure is {100.0/n_true:.1f} points of recall.")
        print(f"  With 3 seeds, differences of one or two structures are within noise.")

    # ── means ─────────────────────────────────────────────────────────────────
    print("\n═══ summary (mean ± std over seeds)")
    print(f"  {'model':12s} {'front size':>14s} {'recovered':>12s} {'recall':>14s} {'enrichment':>13s}")
    for model, ms in by_model.items():
        fs = np.array([m["n_model"] for m in ms], dtype=float)
        hit = np.array([m["hit"] for m in ms], dtype=float)
        rc = np.array([m["recall"] for m in ms]) * 100
        en = np.array([m["enrichment"] for m in ms])
        print(f"  {model:12s} {fs.mean():7.1f}±{fs.std():4.1f} {hit.mean():7.1f}±{hit.std():3.1f} "
              f"{rc.mean():8.1f}%±{rc.std():4.1f} {en.mean():8.1f}x±{en.std():3.1f}")

    # Continuous metrics, which do not saturate the way recall does. When recall
    # ties, these are where a difference -- if there is one -- shows up.
    print(f"\n  {'model':12s} {'IGD':>16s} {'IGD+':>16s} {'HV ratio':>16s}")
    print(f"  {'':12s} {'(lower better)':>16s} {'(lower better)':>16s} {'(higher better)':>16s}")
    for model, ms in by_model.items():
        ig = np.array([m["igd"] for m in ms])
        ip = np.array([m["igd_plus"] for m in ms])
        hv = np.array([m["hv_ratio"] for m in ms])
        print(f"  {model:12s} {ig.mean():8.4f}±{ig.std():.4f} {ip.mean():8.4f}±{ip.std():.4f} "
              f"{hv.mean():8.4f}±{hv.std():.4f}")

    if n_true:
        print(f"\n  sanity check: true front is {n_true} structures in every run, as it "
              f"must be (it depends only on ground truth).")
    else:
        print(f"\n  WARNING: true front size differs across runs {sorted(true_sizes)} — the models "
              f"are not being evaluated on the same structures.")

    # ── pairwise verdicts ─────────────────────────────────────────────────────
    print("\n═══ pairwise verdicts")
    names = [n for n, *_ in ARMS if n in by_model]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(compare_pair(a, by_model[a], b, by_model[b], n_true or "?"))

    print("\n  These verdicts are a blunt rule, not a hypothesis test: three seeds "
          "cannot\n  support one at this front size. Read the per-seed counts above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
