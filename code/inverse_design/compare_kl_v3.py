"""
compare_kl_v3.py
================
Extended version of compare_kl_v2.py: instead of only ensembling MEX-Multi
across seeds, EVERY method (Random, Single-eform, Single-bandgap, Single-gvrh,
MEX-Multi) is ensembled across its available seeds.

  Seed ensemble for all methods (--random-dirs, --single-eform-dirs,
  --single-bandgap-dirs, --single-gvrh-dirs, --multi-dirs)
     For each method, merges the top-K matched structures from ALL provided
     seed directories into one pool (K * n_seeds matches per query).  This
     reduces the estimator variance caused by training/sampling randomness,
     for every method, not just MEX-Multi.

  Bootstrap confidence intervals  (--bootstrap 200)
     Re-samples the retrieved pool 200 times and reports the 95% CI on KL.
     This makes it explicit whether a win over random is real or within noise.

Each method therefore contributes exactly ONE bar per property in the final
comparison (the seed-ensembled result) instead of one bar per seed.

USAGE
-----
python inverse_design/compare_kl_v3.py \\
    --random-dirs         outputs/rand_mp20_1k_seed0,outputs/rand_mp20_1k_seed1,outputs/rand_mp20_1k_seed2 \\
    --single-eform-dirs   /path/to/mex_org/outputs/alex_match_1k_eform_seed0,.../alex_match_1k_eform_seed1,.../alex_match_1k_eform_seed2 \\
    --single-bandgap-dirs /path/to/mex_org/outputs/alex_match_1k_bandgap_seed0,...seed1,...seed2 \\
    --single-gvrh-dirs    /path/to/mex_org/outputs/alex_match_1k_gvrh_seed0,...seed1,...seed2 \\
    --multi-dirs          outputs/alex_match_1k_multi_seed0,outputs/alex_match_1k_multi_seed1,outputs/alex_match_1k_multi_seed2 \\
    --test-lmdb           data/mp20_1k/test/data.lmdb \\
    --out-dir             outputs/compare_kl_v3_1k \\
    --bootstrap           200

Outputs
-------
  kl_comparison_all_methods.png      — all methods (seed-ensembled), with 95% CI error bars
  kl_comparison_single_vs_multi.png  — single-prop vs multi (both seed-ensembled)
  kl_comparison.csv                  — mean, std, ci95_lo, ci95_hi per method
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_inv_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_inv_dir))

from random_retrieval_baseline import (  # noqa: E402
    kl_divergence,
    load_test_properties,
    PANELS,
)
from embed_and_match_alex import _load_cache  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Method registry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (display_label, match_filename, alex_cache_source)
#   alex_cache_source = "self"  → use seed_dir/alex_embeddings_dedup.npz
#   alex_cache_source = "multi" → fall back to the multi-dirs' alex cache
#                                 (random shares the same alex pool as MEX-Multi)
METHOD_SPECS = {
    "random":         ("Random (ensemble)",         "random_matches.npy",  "multi"),
    "single_eform":    ("Single-eform (ensemble)",   "matches_eform.npy",   "self"),
    "single_bandgap":  ("Single-bandgap (ensemble)", "matches_bandgap.npy", "self"),
    "single_gvrh":     ("Single-gvrh (ensemble)",    "matches_gvrh.npy",    "self"),
    "multi":           ("MEX-Multi (ensemble)",      "matches_v3.npy",      "self"),
}

# Validated categorical palette (CVD-safe adjacent pairs, OKLab ΔE >= 8):
# random is the null-model baseline, so it stays a neutral gray to visually
# separate "reference" from the four substantive retrieval methods, which use
# fixed-order categorical hues (blue / orange / aqua / violet).
METHOD_COLORS = {
    "random":         "#8f8d86",
    "single_eform":   "#2a78d6",
    "single_bandgap": "#eb6834",
    "single_gvrh":    "#1baf7a",
    "multi":          "#4a3aa7",
}

# Ink / chrome tokens shared by every chart in this module.
_INK_PRIMARY   = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED     = "#898781"
_GRIDLINE      = "#e1e0d9"
_AXIS_LINE     = "#c3c2b7"
_SURFACE       = "#ffffff"

METHOD_ORDER = ["random", "single_eform", "single_bandgap", "single_gvrh", "multi"]

PROP_KEYS      = ["y_eform",           "y_bandgap",   "y_gvrh"]
PROP_LABELS    = ["Formation\nEnergy", "Band\nGap",   "Bulk\nModulus"]
PROP_CSV_NAMES = ["formation_energy",  "band_gap",    "bulk_modulus_log10"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_seed_ensemble(method_id, seed_dirs, multi_dirs_list, random_alex_cache=None):
    """
    Merge matches from multiple seed directories of a single method into one
    pool, exactly like compare_kl_v2's load_multi_ensemble but generalised to
    any method (random / single-* / multi).

    Returns (matches_ensemble, alex_data) or (None, None).
    alex_data is taken from the first seed dir that has a cache (or, for the
    random method, from --random-alex-cache / the first multi-dirs seed as a
    fallback, since random matches index into the same alex pool as MEX-Multi).
    """
    _, match_filename, cache_source = METHOD_SPECS[method_id]

    all_matches = []
    alex_data   = None

    for d in seed_dirs:
        d = Path(d)
        mpath = d / match_filename
        if not mpath.exists():
            logging.warning(f"[{method_id}] Missing {mpath} — skipping this seed")
            continue

        if alex_data is None:
            cpath = d / "alex_embeddings_dedup.npz"
            if cpath.exists():
                alex_data = _load_cache(cpath)

        m = np.load(str(mpath))
        all_matches.append(m)
        logging.info(f"[{method_id}] Loaded {mpath.parent.name}  shape {m.shape}")

    if not all_matches:
        logging.warning(f"[{method_id}] No valid seed dirs found — skipping.")
        return None, None

    if alex_data is None:
        # Fall back to an explicit cache (random) or the multi-dirs' cache
        # (random shares the same de-duplicated alex pool as MEX-Multi).
        if method_id == "random" and random_alex_cache is not None:
            alex_data = _load_cache(Path(random_alex_cache))
        elif cache_source == "multi" and multi_dirs_list:
            for d in multi_dirs_list:
                cpath = Path(d) / "alex_embeddings_dedup.npz"
                if cpath.exists():
                    alex_data = _load_cache(cpath)
                    break

    if alex_data is None:
        logging.warning(f"[{method_id}] No alex cache found for any seed — skipping.")
        return None, None

    # Stack along the K axis: (n_test, K*n_seeds)
    ensemble = np.concatenate(all_matches, axis=1)
    logging.info(f"[{method_id}] Combined ensemble shape: {ensemble.shape} "
                 f"({len(all_matches)} seeds)")
    return ensemble, alex_data


# ─────────────────────────────────────────────────────────────────────────────
# KL computation with bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def compute_kl_for_method(matches, alex_data, test_data,
                          n_bootstrap=200, rng_seed=42):
    """
    Compute KL(P_test || P_retrieved) for all 3 properties.

    With n_bootstrap > 1 an additional bootstrap step resamples the retrieved
    pool to estimate the variance of the KL estimator.

    Returns:
        dict of shape {test_key: {"mean", "std", "ci95_lo", "ci95_hi"}}
    """
    rng      = np.random.default_rng(rng_seed)
    flat_idx = matches.ravel()
    kl       = {}

    for tkey, akey, _, xmin, xmax, nbins in PANELS:
        test_vals = test_data[tkey].ravel()
        test_vals = test_vals[~np.isnan(test_vals)]

        alex_prop = alex_data.get(akey, np.array([]))
        if not isinstance(alex_prop, np.ndarray):
            alex_prop = np.array(alex_prop, dtype=float)

        if len(alex_prop) == 0 or flat_idx.max() >= len(alex_prop):
            logging.warning(f"  {tkey}: index out of range — skipping")
            kl[tkey] = {"mean": float("nan"), "std": float("nan"),
                        "ci95_lo": float("nan"), "ci95_hi": float("nan")}
            continue

        matched_vals = alex_prop[flat_idx].astype(float)
        matched_vals = matched_vals[~np.isnan(matched_vals)]

        if n_bootstrap <= 1:
            point = kl_divergence(test_vals, matched_vals,
                                  n_bins=nbins, xmin=xmin, xmax=xmax)
            kl[tkey] = {"mean": point, "std": 0.0,
                        "ci95_lo": point, "ci95_hi": point}
        else:
            boot_kls = []
            for _ in range(n_bootstrap):
                idx_boot = rng.choice(len(matched_vals), size=len(matched_vals),
                                      replace=True)
                boot_kls.append(
                    kl_divergence(test_vals, matched_vals[idx_boot],
                                  n_bins=nbins, xmin=xmin, xmax=xmax)
                )
            boot_kls = np.array(boot_kls)
            boot_kls = boot_kls[np.isfinite(boot_kls)]
            kl[tkey] = {
                "mean":    float(np.mean(boot_kls)),
                "std":     float(np.std(boot_kls)),
                "ci95_lo": float(np.percentile(boot_kls, 2.5)),
                "ci95_hi": float(np.percentile(boot_kls, 97.5)),
            }
    return kl


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _kl_mean(kl_dict, prop_key):
    v = kl_dict.get(prop_key, float("nan"))
    return v.get("mean", float("nan")) if isinstance(v, dict) else float(v)


def _kl_err(kl_dict, prop_key):
    """Return (err_lo, err_hi) for asymmetric error bars, or (0, 0)."""
    v = kl_dict.get(prop_key, None)
    if isinstance(v, dict):
        mean = v.get("mean", float("nan"))
        return abs(mean - v.get("ci95_lo", mean)), abs(v.get("ci95_hi", mean) - mean)
    return 0.0, 0.0


def _apply_pub_style(plt):
    """Shared manuscript-figure rcParams: clean sans, muted chrome, no clutter."""
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


def _bar_group_chart(kl_by_method, ordered_methods, title, out_path,
                     show_errorbars=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_pub_style(plt)

    n = len(ordered_methods)
    x = np.arange(len(PROP_KEYS))

    # Slot = space allotted to one property group; bar_w < slot_w leaves a
    # visible gap between neighboring bars instead of a border (see dataviz
    # skill: gap-as-separator, never a stroke).
    slot_w = min(0.8 / n, 0.22)
    bar_w  = slot_w * 0.82

    fig, ax = plt.subplots(figsize=(max(10, 2.6 * n), 6))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=11)
    ax.tick_params(axis="y", length=3, labelsize=10, color=_AXIS_LINE)

    max_top = 0.0
    for i, method_id in enumerate(ordered_methods):
        label  = METHOD_SPECS[method_id][0]
        means  = [_kl_mean(kl_by_method[method_id], pk) for pk in PROP_KEYS]
        offset = (i - n / 2 + 0.5) * slot_w

        his = [_kl_err(kl_by_method[method_id], pk)[1] for pk in PROP_KEYS]
        yerr = None
        if show_errorbars:
            los = [_kl_err(kl_by_method[method_id], pk)[0] for pk in PROP_KEYS]
            yerr = [los, his]

        bars = ax.bar(x + offset, means, bar_w,
                      label=label, color=METHOD_COLORS[method_id],
                      zorder=3,
                      yerr=yerr,
                      error_kw={"elinewidth": 1.1, "capsize": 3,
                                "ecolor": _INK_SECONDARY, "zorder": 4})

        for bar, mean, hi in zip(bars, means, his):
            if np.isnan(mean):
                continue
            top = bar.get_height() + (hi if show_errorbars else 0.0)
            max_top = max(max_top, top)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    top + 0.012 * max(1.0, top),
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=8, color=_INK_SECONDARY, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(PROP_LABELS, fontsize=11.5, color=_INK_PRIMARY)
    ax.set_ylabel("KL Divergence  $D_{KL}(P_{test} \\Vert P_{retrieved})$   (lower is better)",
                  fontsize=11, color=_INK_PRIMARY)
    ax.set_title(title, fontsize=13, color=_INK_PRIMARY, pad=14)
    ax.set_ylim(bottom=0, top=max_top * 1.18 if max_top > 0 else 1.0)

    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
        ncol=min(n, 5), frameon=False, fontsize=10,
        handlelength=1.4, handleheight=1.4, columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(_INK_PRIMARY)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight",
                facecolor=_SURFACE)
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def plot_full_comparison(kl_by_method, active_methods, out_dir, bootstrap):
    ordered = [m for m in METHOD_ORDER if m in active_methods]
    suffix  = "  —  error bars = 95% bootstrap CI" if bootstrap else ""
    _bar_group_chart(
        kl_by_method, ordered,
        f"Retrieval quality: all methods, seed-ensembled{suffix}\n"
        "(lower KL = retrieved properties closer to test set)",
        out_dir / "kl_comparison_all_methods.png",
        show_errorbars=bootstrap,
    )


def plot_single_vs_multi(kl_by_method, out_dir, bootstrap):
    comparison = [m for m in ["single_eform", "single_bandgap", "single_gvrh", "multi"]
                  if m in kl_by_method]
    if len(comparison) < 2:
        logging.info("Fewer than 2 single/multi methods — skipping single-vs-multi chart.")
        return
    suffix = "  —  error bars = 95% bootstrap CI" if bootstrap else ""
    _bar_group_chart(
        kl_by_method, comparison,
        f"Single-property MEX vs MEX-Multi, seed-ensembled{suffix}\n"
        "(lower KL = retrieved properties closer to test set)",
        out_dir / "kl_comparison_single_vs_multi.png",
        show_errorbars=bootstrap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_kl_csv(kl_by_method, out_dir):
    out_path = out_dir / "kl_comparison.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "property", "kl_mean", "kl_std", "ci95_lo", "ci95_hi"])
        for method_id in METHOD_ORDER:
            if method_id not in kl_by_method:
                continue
            kl_dict = kl_by_method[method_id]
            for pk, pname in zip(PROP_KEYS, PROP_CSV_NAMES):
                v = kl_dict.get(pk, {})
                if isinstance(v, dict):
                    mean = v.get("mean", float("nan"))
                    std  = v.get("std",  float("nan"))
                    lo   = v.get("ci95_lo", float("nan"))
                    hi   = v.get("ci95_hi", float("nan"))
                else:
                    mean, std, lo, hi = float(v), 0.0, float(v), float(v)
                w.writerow([method_id, pname,
                            f"{mean:.6f}", f"{std:.6f}",
                            f"{lo:.6f}", f"{hi:.6f}"])
    logging.info(f"KL CSV saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _split_dirs(arg):
    if not arg:
        return []
    return [d.strip() for d in arg.split(",") if d.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="compare_kl_v3 — KL comparison with ALL methods seed-ensembled, "
                    "plus bootstrap CI."
    )
    p.add_argument("--random-dirs",         default=None,
                   help="Comma-separated dirs with random_matches.npy, one per seed "
                        "(e.g. seed0_dir,seed1_dir,seed2_dir)")
    p.add_argument("--single-eform-dirs",   default=None,
                   help="Comma-separated dirs with matches_eform.npy, one per seed")
    p.add_argument("--single-bandgap-dirs", default=None,
                   help="Comma-separated dirs with matches_bandgap.npy, one per seed")
    p.add_argument("--single-gvrh-dirs",    default=None,
                   help="Comma-separated dirs with matches_gvrh.npy, one per seed")
    p.add_argument("--multi-dirs",          default=None,
                   help="Comma-separated dirs with matches_v3.npy, one per seed")
    p.add_argument("--test-lmdb",           required=True,
                   help="Multi-property test LMDB (data/mp20_1k/test/data.lmdb)")
    p.add_argument("--out-dir",             required=True)
    p.add_argument("--random-alex-cache",   default=None,
                   help="alex_embeddings_dedup.npz for random baseline "
                        "(defaults to the first --multi-dirs seed's cache)")
    p.add_argument("--bootstrap",           type=int, default=200,
                   help="Bootstrap resamples for 95%% CI on KL "
                        "(default 200; set 0 or 1 to disable)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_bootstrap = (args.bootstrap > 1)

    # ── test set ──────────────────────────────────────────────────────────────
    test_data = load_test_properties(args.test_lmdb)
    logging.info(f"Test set: {len(test_data['material_id'])} structures")

    method_dirs_arg = {
        "random":         _split_dirs(args.random_dirs),
        "single_eform":   _split_dirs(args.single_eform_dirs),
        "single_bandgap": _split_dirs(args.single_bandgap_dirs),
        "single_gvrh":    _split_dirs(args.single_gvrh_dirs),
        "multi":          _split_dirs(args.multi_dirs),
    }
    multi_dirs_list = method_dirs_arg["multi"]

    kl_by_method   = {}
    active_methods = []

    for method_id in METHOD_ORDER:
        seed_dirs = method_dirs_arg[method_id]
        if not seed_dirs:
            continue

        matches, alex_data = load_seed_ensemble(
            method_id, seed_dirs, multi_dirs_list, args.random_alex_cache
        )
        if matches is None:
            continue

        kl = compute_kl_for_method(matches, alex_data, test_data,
                                   n_bootstrap=args.bootstrap)
        kl_by_method[method_id] = kl
        active_methods.append(method_id)

        logging.info(
            f"  [{METHOD_SPECS[method_id][0]}]  "
            f"eform={kl['y_eform']['mean']:.4f}  "
            f"bandgap={kl['y_bandgap']['mean']:.4f}  "
            f"gvrh={kl['y_gvrh']['mean']:.4f}"
        )

    if not active_methods:
        logging.error("No method directories found. Check your paths.")
        return

    logging.info(f"\nActive methods: {active_methods}")

    # ── plots + CSV ───────────────────────────────────────────────────────────
    plot_full_comparison(kl_by_method, active_methods, out_dir, use_bootstrap)
    plot_single_vs_multi(kl_by_method, out_dir, use_bootstrap)
    save_kl_csv(kl_by_method, out_dir)
    logging.info("compare_kl_v3 done.")


if __name__ == "__main__":
    main()
