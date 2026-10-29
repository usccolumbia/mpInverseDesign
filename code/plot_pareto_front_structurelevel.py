"""
plot_pareto_front_structurelevel.py
====================================
Corrected structure-level Pareto front, replacing the per-structure DFO
resample front produced by MEXMultiTrainer.pareto_predict() /
plot_pareto_front.py.

Why the old front was wrong
----------------------------
pareto_predict() draws N=200 *independent* DFO samples per property for a
single structure (one stochastic optimiser run per sample, per property),
then pairs sample i of eform with sample i of band gap with sample i of
K_VRH purely by array index and computes non-domination over those 200
triples. The three per-property DFO runs share no coupling, so the pairing
is arbitrary: the resulting "front" mixes DFO sampling noise across three
unrelated random sequences, not a real trade-off. A structure has one true
formation energy, not 200 candidate ones.

What this script does instead
-------------------------------
Takes the standard point-estimate `*_predictions.npz` written by
MEXMultiTrainer.predict() (the same file the MAE/Spearman/EGM numbers in
tab:main_results come from) -- one predicted (eform, bandgap, gvrh) vector
per test structure, plus the matching ground-truth vector -- and computes
non-domination *across the structure library*: candidate i can only be
beaten by a DIFFERENT structure's prediction, never by resampling noise on
itself. This is the front described in \\refsec{sec:intro} ("the set of
property vectors that are jointly attainable... together with the
structures that realise it") and in the \\PLACEHOLDER note in
\\refsec{sec:method}: "the design-relevant front is over structures: one
predicted property vector per candidate, non-domination computed across the
library."

Input
-----
A `*_predictions.npz` from `--mode predict` (NOT `--mode pareto`), with keys:
  y_<prop>       -- model point-estimate prediction, one value per structure
  y_<prop>_gts   -- ground-truth value, one value per structure
  ids            -- structure ids (optional)
for prop in {eform, bandgap, gvrh}. If you don't already have this file
saved from generating tab:main_results, `--mode predict` is a single forward
pass per structure (one DFO run per property, not the N=200 resampling used
for the old pareto mode), so it is cheap to regenerate.

Usage
-----
    python plot_pareto_front_structurelevel.py \\
        --pred-file outputs/mp20_5k_multi_seed0/results/mp20_5k_multi_seed0_predictions.npz \\
        --out-png figures/pareto_front_structurelevel_seed0.png

    # all three seeds combined into one 3x1 grid, matching the existing figure:
    python plot_pareto_front_structurelevel.py \\
        --pred-file outputs/mp20_5k_multi_seed0/results/..._predictions.npz \\
                    outputs/mp20_5k_multi_seed1/results/..._predictions.npz \\
                    outputs/mp20_5k_multi_seed2/results/..._predictions.npz \\
        --out-png figures/pareto_front_structurelevel_all_seeds.png
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Keys as stored in the .npz files.  "gvrh" is a legacy field name inherited
# from the upstream MEX repo; the values it holds are the Voigt-Reuss-Hill
# BULK modulus (MP elasticity `bulk_modulus.vrh`), not the shear modulus.
# See files/download_bulk_modulus.py.  Do not rename these -- they must match
# the npz keys.
PROP_NAMES = ["eform", "bandgap", "gvrh"]

# Display names used for axis labels, so no figure ever shows "gvrh".
PROP_LABELS = [
    r"$E_\mathrm{form}$ (eV/atom)",
    r"$E_g$ (eV)",
    r"$K_\mathrm{VRH}$ (log$_{10}$ GPa)",
]

_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRIDLINE = "#e1e0d9"
_AXIS_LINE = "#c3c2b7"
_SURFACE = "#ffffff"
_COLOR_CANDIDATES = "#2a78d6"
_COLOR_GROUND_TRUTH = "#eb6834"


def _apply_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "text.color": _INK_PRIMARY,
        "axes.edgecolor": _INK_PRIMARY,
        "axes.labelcolor": _INK_PRIMARY,
        "xtick.color": _INK_PRIMARY,
        "ytick.color": _INK_PRIMARY,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "figure.facecolor": _SURFACE,
        "axes.facecolor": _SURFACE,
        "savefig.facecolor": _SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_front(costs: np.ndarray) -> np.ndarray:
    """Boolean mask of the rank-0 (non-dominated) rows of `costs` (N, K), minimisation."""
    n = len(costs)
    is_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_front[i]:
            continue
        for j in range(n):
            if i != j and is_front[j] and _dominates(costs[j], costs[i]):
                is_front[i] = False
                break
    return is_front


def load_predict_npz(path: Path):
    """Load a standard MEXMultiTrainer.predict() output file.

    Returns (pred_matrix, gt_matrix, ids) where each matrix is (N, 3) in
    PROP_NAMES order, restricted to rows where both prediction and
    ground truth are finite for all three properties.
    """
    data = np.load(path, allow_pickle=True)
    for p in PROP_NAMES:
        if f"y_{p}" not in data or f"y_{p}_gts" not in data:
            raise ValueError(
                f"{path} is missing 'y_{p}' or 'y_{p}_gts'. This script needs a "
                f"--mode predict output, not a --mode pareto one."
            )
    preds = np.stack([np.asarray(data[f"y_{p}"]).reshape(-1) for p in PROP_NAMES], axis=1)
    gts = np.stack([np.asarray(data[f"y_{p}_gts"]).reshape(-1) for p in PROP_NAMES], axis=1)
    ids = np.asarray(data["ids"]).reshape(-1) if "ids" in data else np.arange(len(preds)).astype(str)

    valid = np.all(np.isfinite(preds), axis=1) & np.all(np.isfinite(gts), axis=1)
    return preds[valid], gts[valid], ids[valid]


def _scatter_2d(ax, x, y, xlabel, ylabel, gt_x, gt_y):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(labelsize=14, width=1.4, length=5, color=_INK_PRIMARY)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.scatter(x, y, c=_COLOR_CANDIDATES, s=16, alpha=0.7, linewidths=0,
               zorder=2, label=f"Model structure-level front (n={len(x)})")
    ax.scatter(gt_x, gt_y, c=_COLOR_GROUND_TRUTH, s=46, alpha=0.9,
               edgecolors=_SURFACE, linewidths=1.0, zorder=3,
               label=f"True front (n={len(gt_x)})")
    ax.set_xlabel(xlabel, fontsize=15, fontweight="bold", color=_INK_PRIMARY)
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold", color=_INK_PRIMARY)


def plot_one_seed(preds: np.ndarray, gts: np.ndarray, axrow, seed_label: str):
    pred_front = non_dominated_front(preds)
    gt_front = non_dominated_front(gts)
    pf, gf = preds[pred_front], gts[gt_front]

    pairs = list(combinations(range(3), 2))
    for (i, j), ax in zip(pairs, axrow):
        pi, pj = PROP_LABELS[i], PROP_LABELS[j]
        _scatter_2d(ax, pf[:, j], pf[:, i], xlabel=pj, ylabel=pi,
                    gt_x=gf[:, j], gt_y=gf[:, i])

    print(f"[{seed_label}] structures={len(preds)}  "
          f"model front size={pred_front.sum()}  true front size={gt_front.sum()}")
    return pred_front.sum(), gt_front.sum(), len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-file", type=Path, nargs="+", required=True,
                     help="One or more --mode predict npz files (one per seed).")
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()

    _apply_pub_style()
    n_seeds = len(args.pred_file)
    fig, axes = plt.subplots(n_seeds, 3, figsize=(15, 5.2 * n_seeds), squeeze=False)

    counts = []
    for row, path in enumerate(args.pred_file):
        preds, gts, ids = load_predict_npz(path)
        seed_label = path.stem
        n_model, n_true, n_struct = plot_one_seed(preds, gts, axes[row], seed_label)
        counts.append((n_model, n_true, n_struct))

    # Figure-level legend carries no per-seed counts: each row has its own
    # model-front size, so a single "n=..." here would report only row 0 and
    # misstate the other rows. The counts are printed to stdout above and
    # belong in the caption.
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=8,
               markerfacecolor=_COLOR_CANDIDATES, markeredgecolor="none",
               alpha=0.7, label="Model structure-level front"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=_COLOR_GROUND_TRUTH, markeredgecolor=_SURFACE,
               markeredgewidth=1.0, alpha=0.9, label="True front"),
    ]
    legend = fig.legend(handles=handles, loc="upper center",
                        bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
                        fontsize=15, handletextpad=0.7, columnspacing=2.4)
    for text in legend.get_texts():
        text.set_color(_INK_PRIMARY)
        text.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(str(args.out_png), dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    print(f"Saved plot to: {args.out_png}")
    if args.out_png.suffix.lower() == ".png":
        pdf_path = args.out_png.with_suffix(".pdf")
        fig.savefig(str(pdf_path), bbox_inches="tight", facecolor=_SURFACE)
        print(f"Saved vector copy to: {pdf_path}")


if __name__ == "__main__":
    main()
