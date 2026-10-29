"""
replot_retrieval_histograms.py
==============================
Manuscript-quality replacement for the 3-panel retrieval histogram that
embed_and_match_alex_v3.py emits as a side effect (test_vs_matched_alex_
histograms.png).

This script does NO model loading and NO embedding. It reads only the
artifacts an existing run already wrote to its output directory:

    <seed_dir>/test_embeddings.npz        (test properties; embeddings unused)
    <seed_dir>/alex_embeddings_dedup.npz  (alex properties; embeddings unused)
    <seed_dir>/matches_v3.npy             (N_test, K) indices into the alex pool

so it is cheap to re-run repeatedly while tuning the figure.

Multiple --seed-dirs are pooled into one retrieved set (K * n_seeds matches
per query), matching how compare_kl_v3.py seed-ensembles a method and how
sec:retrieval describes the pooling. The alex pool ordering is verified to be
identical across seed dirs before pooling, since the match indices are only
meaningful against the pool they were computed from.

USAGE
-----
Single seed:
    python inverse_design/replot_retrieval_histograms.py \\
        --seed-dirs outputs/alex_match_5k_gvrh_nce_seed0 \\
        --out figures/retrieval_histograms.pdf

Pooled over 3 seeds (what the manuscript reports):
    python inverse_design/replot_retrieval_histograms.py \\
        --seed-dirs outputs/alex_match_5k_gvrh_nce_seed0 \\
                    outputs/alex_match_5k_gvrh_nce_seed1 \\
                    outputs/alex_match_5k_gvrh_nce_seed2 \\
        --out figures/retrieval_histograms.pdf

Writes both the requested file and a .png sibling (some viewers/reviewers
want a raster); pass --no-png to skip that.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Local copies of two helpers, rather than importing them.
#
# `from embed_and_match_alex import _load_cache` transitively imports
# MEXInferenceEngine -> equiformer_v2 -> e3nn -> torch, so a script that only
# reads .npz files and draws a histogram would require the full training
# environment. These are faithful copies of:
#     _load_cache   : inverse_design/embed_and_match_alex.py
#     kl_divergence : inverse_design/random_retrieval_baseline.py
# Keep them in sync if either original changes (the KL definition in
# particular must stay identical to what compare_kl_v3.py reports).
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache(path: Path) -> dict:
    raw = np.load(str(path), allow_pickle=True)
    out = {}
    for k in raw.files:
        v = raw[k]
        # numpy stores lists of strings wrapped in 0-d object array; unwrap
        if v.dtype == object and v.ndim == 0:
            out[k] = v.item().tolist()
        elif v.dtype == object:
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def kl_divergence(p_vals, q_vals, n_bins=50, xmin=None, xmax=None):
    """
    Estimate KL(P || Q) using histogram-based density estimation.
    p_vals: reference (test set), q_vals: approximation (retrieved).
    Returns float (nats). Returns nan if either distribution is empty.
    """
    if len(p_vals) == 0 or len(q_vals) == 0:
        return float("nan")
    lo = xmin if xmin is not None else min(p_vals.min(), q_vals.min())
    hi = xmax if xmax is not None else max(p_vals.max(), q_vals.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, density=True)
    # Smooth to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(entropy(p_hist, q_hist))


# ─────────────────────────────────────────────────────────────────────────────
# Chart chrome — matches plot_pareto_front_structurelevel.py / compare_kl_v3.py
# so every figure in the paper reads as one visual system.
# ─────────────────────────────────────────────────────────────────────────────

_INK_PRIMARY   = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRIDLINE      = "#e1e0d9"
_AXIS_LINE     = "#c3c2b7"
_SURFACE       = "#ffffff"

_COLOR_TEST      = "#2a78d6"   # categorical slot 1 (blue)  — reference
_COLOR_RETRIEVED = "#eb6834"   # categorical slot 2 (orange) — retrieved

# (test_key, alex_key, axis_label, xmin, xmax, nbins)
# Axis labels carry the physical quantity and unit only; which dataset column
# each came from is documentation, not chart content, so it lives in the
# docstring/caption rather than on the axis.
PANELS = [
    ("y_eform",   "formation_energy_per_atom",
     r"Formation energy (eV/atom)",        -5.0, 1.0, 50),
    ("y_bandgap", "dft_band_gap",
     r"Band gap (eV)",                      0.0, 10.0, 50),
    ("y_gvrh",    "log10_bulk_modulus",
     r"Bulk modulus ($\log_{10}$GPa)",     -1.0, 3.5, 50),
]


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
        "pdf.fonttype":      42,   # embed TrueType, not Type3 — required by
        "ps.fonttype":       42,   # most publishers
    })


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_seed_dirs(seed_dirs):
    """Pool matches across seed dirs. Returns (test_data, alex_data, matches)."""
    test_data = alex_data = None
    alex_ids_ref = None
    all_matches = []

    for d in seed_dirs:
        d = Path(d)
        mpath = d / "matches_v3.npy"
        tpath = d / "test_embeddings.npz"
        apath = d / "alex_embeddings_dedup.npz"

        missing = [p.name for p in (mpath, tpath, apath) if not p.exists()]
        if missing:
            logging.warning(f"[{d.name}] missing {missing} — skipping this seed")
            continue

        if test_data is None:
            test_data = _load_cache(tpath)
        if alex_data is None:
            alex_data = _load_cache(apath)
            alex_ids_ref = list(alex_data["material_id"])
        else:
            # Match indices only mean anything against the pool they were
            # computed from, so pooling across seeds requires identical pools.
            alex_ids_this = list(_load_cache(apath)["material_id"])
            if alex_ids_this != alex_ids_ref:
                raise SystemExit(
                    f"ERROR: alex pool in {d} differs from the first seed dir "
                    f"({len(alex_ids_this)} vs {len(alex_ids_ref)} entries, or "
                    f"different ordering). Pooling their match indices would "
                    f"silently mix up structures. Re-run those seeds against "
                    f"the same alex pool, or plot them separately."
                )

        m = np.load(str(mpath))
        all_matches.append(m)
        logging.info(f"[{d.name}] matches {m.shape}")

    if not all_matches:
        raise SystemExit("ERROR: no usable seed directories found.")

    n_q = all_matches[0].shape[0]
    for m in all_matches:
        if m.shape[0] != n_q:
            raise SystemExit(
                f"ERROR: seed dirs disagree on the number of test queries "
                f"({[m.shape[0] for m in all_matches]}). They are not the same "
                f"test split; refusing to pool."
            )

    matches = np.concatenate(all_matches, axis=1)   # (N_test, K * n_seeds)
    logging.info(f"Pooled matches: {matches.shape} "
                 f"({len(all_matches)} seed(s) x K={all_matches[0].shape[1]})")
    return test_data, alex_data, matches


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot(test_data, alex_data, matches, out_path, n_seeds, write_png=True):
    _apply_pub_style()

    flat_idx = matches.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    summary = {}

    for ax, (tkey, akey, xlabel, xmin, xmax, nbins) in zip(axes, PANELS):
        test_vals = np.asarray(test_data[tkey]).ravel()
        test_vals = test_vals[~np.isnan(test_vals)]

        alex_prop = np.asarray(alex_data[akey])
        matched_vals = alex_prop[flat_idx]
        matched_vals = matched_vals[~np.isnan(matched_vals)]

        bins = np.linspace(xmin, xmax, nbins + 1)

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, color=_AXIS_LINE)

        # Retrieved: filled, sits behind. Test: step outline on top. Two filled
        # alpha-blended series (the original approach) produce a third muddy
        # colour wherever they overlap, which is exactly the region the reader
        # needs to judge; an outline over a fill keeps both readable.
        if len(matched_vals) > 0:
            ax.hist(matched_vals, bins=bins, density=True,
                    color=_COLOR_RETRIEVED, alpha=0.55, linewidth=0, zorder=2)
        ax.hist(test_vals, bins=bins, density=True,
                histtype="step", color=_COLOR_TEST, linewidth=1.6, zorder=3)

        kl = kl_divergence(test_vals, matched_vals, n_bins=nbins,
                           xmin=xmin, xmax=xmax) if len(matched_vals) else float("nan")

        # KL annotated in-panel, top-right, where no histogram mass usually is.
        ax.text(0.97, 0.94, f"$D_\\mathrm{{KL}}$ = {kl:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color=_INK_PRIMARY)

        ax.set_xlabel(xlabel, fontsize=10.5, color=_INK_PRIMARY)
        ax.set_xlim(xmin, xmax)

        summary[tkey] = {
            "kl": kl,
            "n_test": len(test_vals),
            "n_matched": len(matched_vals),
            "test_mean": float(np.mean(test_vals)),
            "test_std": float(np.std(test_vals)),
            "matched_mean": float(np.mean(matched_vals)) if len(matched_vals) else float("nan"),
            "matched_std": float(np.std(matched_vals)) if len(matched_vals) else float("nan"),
        }

    axes[0].set_ylabel("Density", fontsize=10.5, color=_INK_PRIMARY)

    n_test = summary["y_eform"]["n_test"]
    k_pooled = matches.shape[1]
    handles = [
        Line2D([0], [0], color=_COLOR_TEST, linewidth=1.6,
               label=f"Held-out test set ($n$ = {n_test:,})"),
        # Plain text here, not LaTeX: matplotlib renders "_" literally outside
        # mathtext, so escaping it as "\_" would print the backslash.
        Patch(facecolor=_COLOR_RETRIEVED, alpha=0.55,
              label=(f"Retrieved alex_mp20 neighbours "
                     f"(top-{k_pooled // max(n_seeds, 1)}"
                     + (f" $\\times$ {n_seeds} seeds" if n_seeds > 1 else "") + ")")),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=2, frameon=False, fontsize=10, handletextpad=0.7,
               columnspacing=2.4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", facecolor=_SURFACE)
    logging.info(f"Wrote {out_path}")

    if write_png and out_path.suffix.lower() != ".png":
        png_path = out_path.with_suffix(".png")
        fig.savefig(str(png_path), dpi=400, bbox_inches="tight", facecolor=_SURFACE)
        logging.info(f"Wrote {png_path}")

    plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Manuscript-quality replot of the retrieval histograms "
                    "from cached embed_and_match_alex_v3.py artifacts."
    )
    p.add_argument("--seed-dirs", nargs="+", required=True,
                   help="One or more run dirs containing matches_v3.npy, "
                        "test_embeddings.npz and alex_embeddings_dedup.npz. "
                        "Multiple dirs are pooled into one retrieved set.")
    p.add_argument("--out", default="figures/retrieval_histograms.pdf",
                   help="Output path (.pdf recommended; a .png sibling is also "
                        "written unless --no-png).")
    p.add_argument("--no-png", action="store_true")
    args = p.parse_args()

    test_data, alex_data, matches = load_seed_dirs(args.seed_dirs)
    summary = plot(test_data, alex_data, matches, args.out,
                   n_seeds=len(args.seed_dirs), write_png=not args.no_png)

    logging.info("\n=== Summary ===")
    labels = {"y_eform": "Formation energy", "y_bandgap": "Band gap",
              "y_gvrh": "Bulk modulus"}
    for k, lab in labels.items():
        s = summary[k]
        logging.info(
            f"  {lab:18s} KL={s['kl']:.4f}  "
            f"test {s['test_mean']:+.3f}+/-{s['test_std']:.3f} (n={s['n_test']:,})  "
            f"retrieved {s['matched_mean']:+.3f}+/-{s['matched_std']:.3f} (n={s['n_matched']:,})"
        )


if __name__ == "__main__":
    main()
