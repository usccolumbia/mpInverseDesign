"""
plot_pareto_front.py
--------------------
Plot the Pareto front from a pareto_predictions.npz file produced by
MEXMultiTrainer.pareto_predict(), optionally overlaid with the TRUE Pareto
front computed from the actual DFT-labelled test-set structures.

The npz file is expected to contain:
  - 'y_<prop>'   : denormalised predictions for each property
  - 'front_rank' : Pareto front rank per point (0 = Pareto front)
  - 'ids'        : structure ids (optional, used for hover labels)

For 2 properties  -> a single 2-D scatter
For 3 properties  -> a 3x3 pair-plot grid (all 2-D projections)
For N properties  -> NxN pair-plot grid

Ground-truth overlay
---------------------
Passing --test-lmdb computes the non-dominated front over the REAL test-set
structures' DFT-computed properties (not model candidates) and draws it as a
second series. This answers a question the model-only plot cannot: does
\\model's proposed candidate front actually reach as far as (or further than)
what's achievable among materials that were really synthesised/computed, or
does it fall short of / never explore regions the real data already covers?

Usage:
    python plot_pareto_front.py --pred-file <path/to/pareto_predictions.npz>
    python plot_pareto_front.py --pred-file <path>.npz --test-lmdb data/mp20_5k/test/data.lmdb \\
        --out-png pareto.png
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "inverse_design"))

# ─────────────────────────────────────────────────────────────────────────────
# Palette / chart chrome (validated categorical set, CVD-safe adjacent pair)
# ─────────────────────────────────────────────────────────────────────────────

_INK_PRIMARY   = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRIDLINE      = "#e1e0d9"
_AXIS_LINE     = "#c3c2b7"
_SURFACE       = "#ffffff"

_COLOR_CANDIDATES = "#2a78d6"   # categorical slot 1 (blue)
_COLOR_GROUND_TRUTH = "#eb6834"  # categorical slot 2 (orange)


def _apply_pub_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         11,
        "text.color":        _INK_PRIMARY,
        "axes.edgecolor":    _AXIS_LINE,
        "axes.labelcolor":   _INK_PRIMARY,
        "xtick.color":       _INK_SECONDARY,
        "ytick.color":       _INK_SECONDARY,
        "axes.linewidth":    0.8,
        "figure.facecolor":  _SURFACE,
        "axes.facecolor":    _SURFACE,
        "savefig.facecolor": _SURFACE,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Pareto utilities (mirrors ocpmodels/trainers/mex_multi_trainer.py so the
# ground-truth reference front is computed with the exact same definition of
# dominance as the model's own candidate front)
# ─────────────────────────────────────────────────────────────────────────────

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_front(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of the rank-0 (non-dominated) points in `costs` (N, K), minimisation."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pareto(npz_path: Path):
    """Return (property_names, values_dict, front_rank)."""
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)

    prop_names = sorted(
        k[len("y_"):] for k in keys if k.startswith("y_") and not k.endswith("_gts")
    )
    if not prop_names:
        raise ValueError(f"No 'y_<prop>' keys found in {npz_path.name}. Keys: {keys}")

    values = {p: np.array(data[f"y_{p}"]).reshape(-1) for p in prop_names}
    front_rank = np.array(data["front_rank"]).reshape(-1) if "front_rank" in data else np.zeros(len(values[prop_names[0]]), dtype=int)
    return prop_names, values, front_rank


def load_ground_truth_front(test_lmdb: Path, prop_names):
    """Compute the non-dominated front over the REAL test-set structures'
    DFT-computed properties. Sign convention matches pareto_predict(): lower
    is assumed better for every property fed to non_dominated_front (the
    model's own candidates already are, since pareto_predict never negates
    formation energy — we mirror that convention here unmodified)."""
    from random_retrieval_baseline import load_test_properties

    test_data = load_test_properties(str(test_lmdb))
    values = {}
    valid = np.ones(len(test_data["material_id"]), dtype=bool)
    for p in prop_names:
        key = f"y_{p}"
        if key not in test_data:
            raise ValueError(f"Ground-truth test set has no '{key}' — cannot overlay for property '{p}'.")
        v = test_data[key]
        values[p] = v
        valid &= ~np.isnan(v)

    for p in prop_names:
        values[p] = values[p][valid]

    costs = np.stack([values[p] for p in prop_names], axis=1)
    front_mask = non_dominated_front(costs)
    return {p: values[p][front_mask] for p in prop_names}, int(valid.sum()), int(front_mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_2d(ax, x, y, xlabel, ylabel, gt_x=None, gt_y=None):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, color=_AXIS_LINE)

    ax.scatter(x, y, c=_COLOR_CANDIDATES, s=16, alpha=0.35, linewidths=0,
               zorder=2, label=f"\\model candidates (n={len(x)})")

    if gt_x is not None:
        ax.scatter(gt_x, gt_y, c=_COLOR_GROUND_TRUTH, s=46, alpha=0.9,
                   edgecolors=_SURFACE, linewidths=1.0, zorder=3,
                   label=f"Test-set materials, true front (n={len(gt_x)})")

    ax.set_xlabel(xlabel, fontsize=10, color=_INK_PRIMARY)
    ax.set_ylabel(ylabel, fontsize=10, color=_INK_PRIMARY)


# ── main plotting function ────────────────────────────────────────────────────

def plot_pareto(npz_path: Path, out_png: Path, test_lmdb: Path = None):
    _apply_pub_style()
    prop_names, values, front_rank = load_pareto(npz_path)
    K = len(prop_names)

    print(f"Properties  : {prop_names}")
    print(f"Total points: {len(front_rank)}")
    print(f"Front points: {(front_rank == 0).sum()}  (rank 0)")

    if K == 1:
        raise ValueError("Need at least 2 properties to plot a Pareto front.")

    gt_values, gt_n_valid, gt_n_front = (None, None, None)
    if test_lmdb is not None:
        gt_values, gt_n_valid, gt_n_front = load_ground_truth_front(test_lmdb, prop_names)
        print(f"Ground-truth test structures with all properties: {gt_n_valid}")
        print(f"Ground-truth non-dominated (true Pareto front)  : {gt_n_front}")

    pairs = list(combinations(range(K), 2))

    if K == 2:
        fig, axes = plt.subplots(1, 1, figsize=(5.5, 4.5))
        axes = np.array([[axes]])
        pair_axes = {pairs[0]: axes[0, 0]}
    elif K == 3:
        fig, axrow = plt.subplots(1, 3, figsize=(15, 5.2))
        pair_axes = {pairs[i]: axrow[i] for i in range(3)}
    else:
        fig, axes = plt.subplots(K, K, figsize=(3.2 * K, 3.2 * K))
        pair_axes = {}
        for i, j in pairs:
            pair_axes[(i, j)] = axes[i, j]
        for r in range(K):
            for c in range(K):
                if (r, c) not in pair_axes:
                    axes[r, c].set_visible(False)

    handles, labels = None, None
    for (i, j), ax in pair_axes.items():
        pi, pj = prop_names[i], prop_names[j]
        x = values[pj]
        y = values[pi]
        gt_x = gt_values[pj] if gt_values is not None else None
        gt_y = gt_values[pi] if gt_values is not None else None
        _scatter_2d(ax, x, y, xlabel=pj, ylabel=pi, gt_x=gt_x, gt_y=gt_y)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    title = f"Pareto-optimal candidates vs. true test-set front — {', '.join(prop_names)}"
    fig.suptitle(title, fontsize=13, color=_INK_PRIMARY, y=1.04)

    legend = fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, frameon=False, fontsize=10, handletextpad=0.6, columnspacing=2.0,
        markerscale=1.6,
    )
    for text in legend.get_texts():
        text.set_color(_INK_PRIMARY)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    print(f"Saved plot to: {out_png}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot Pareto front from MEX multi predictions")
    parser.add_argument("--pred-file", type=Path, required=True,
                        help="Path to the pareto_predictions.npz file")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG path (default: <pred-file dir>/pareto_front.png)")
    parser.add_argument("--test-lmdb", type=Path, default=None,
                        help="Multi-property test LMDB (e.g. data/mp20_5k/test/data.lmdb). "
                             "If given, overlays the TRUE non-dominated front computed from "
                             "the actual DFT-labelled test structures.")
    args = parser.parse_args()

    out_png = args.out_png or (args.pred_file.parent / "pareto_front.png")
    plot_pareto(args.pred_file, out_png, test_lmdb=args.test_lmdb)


if __name__ == "__main__":
    main()
