"""
compare_kl_v2.py
================
Extended version of compare_kl.py with two additions:

  1. Seed ensemble  (--multi-dirs seed0,seed1,seed2)
     Merges the top-K matched structures from all MEX-Multi seeds into one
     bigger pool (K * n_seeds matches per query).  This reduces the estimator
     variance caused by training randomness.

  2. Bootstrap confidence intervals  (--bootstrap 200)
     Re-samples the retrieved pool 200 times and reports the 95% CI on KL.
     This makes it explicit whether a win over random is real or within noise.

Everything else is identical to compare_kl.py, so the two files can be run
side-by-side for direct comparison.

USAGE
-----
# Point estimates only (same as compare_kl.py, no new features active):
python inverse_design/compare_kl_v2.py \\
    --random-dir  outputs/rand_mp20_1k_seed0 \\
    --multi-dir   outputs/alex_match_1k_multi_seed0 \\
    --test-lmdb   data/mp20_1k/test/data.lmdb \\
    --out-dir     outputs/compare_kl_v2_1k_seed0 \\
    --bootstrap 0

# Full v2 mode — ensemble + bootstrap:
python inverse_design/compare_kl_v2.py \\
    --random-dir          outputs/rand_mp20_1k_seed0 \\
    --single-eform-dir    /path/to/mex_org/outputs/alex_match_1k_eform_seed0 \\
    --single-bandgap-dir  /path/to/mex_org/outputs/alex_match_1k_bandgap_seed0 \\
    --single-gvrh-dir     /path/to/mex_org/outputs/alex_match_1k_gvrh_seed0 \\
    --multi-dir           outputs/alex_match_1k_multi_seed0 \\
    --multi-dirs          outputs/alex_match_1k_multi_seed0,outputs/alex_match_1k_multi_seed1,outputs/alex_match_1k_multi_seed2 \\
    --test-lmdb           data/mp20_1k/test/data.lmdb \\
    --out-dir             outputs/compare_kl_v2_1k_seed0 \\
    --bootstrap           200

Outputs
-------
  kl_comparison_all_methods.png      — all methods, with 95% CI error bars
  kl_comparison_single_vs_multi.png  — single-prop vs multi (incl. ensemble)
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

METHOD_SPECS = {
    "random":         ("Random",               "random_matches.npy",  "multi"),
    "single_eform":   ("Single-eform",         "matches_eform.npy",   "self"),
    "single_bandgap": ("Single-bandgap",       "matches_bandgap.npy", "self"),
    "single_gvrh":    ("Single-gvrh",          "matches_gvrh.npy",    "self"),
    "multi":          ("MEX-Multi",            "matches_v3.npy",      "self"),
    "multi_ensemble": ("MEX-Multi (ensemble)", "matches_v3.npy",      "self"),
}

METHOD_COLORS = {
    "random":         "#aaaaaa",
    "single_eform":   "#2196F3",
    "single_bandgap": "#4CAF50",
    "single_gvrh":    "#FF9800",
    "multi":          "#9C27B0",
    "multi_ensemble": "#E91E63",
}

METHOD_ORDER = ["random", "single_eform", "single_bandgap", "single_gvrh",
                "multi", "multi_ensemble"]

PROP_KEYS      = ["y_eform",           "y_bandgap",   "y_gvrh"]
PROP_LABELS    = ["Formation\nEnergy", "Band\nGap",   "Bulk\nModulus"]
PROP_CSV_NAMES = ["formation_energy",  "band_gap",    "bulk_modulus_log10"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_method(method_id, method_dir, multi_dir, random_alex_cache=None):
    """Load (matches, alex_data) for a single-seed method directory."""
    if method_dir is None:
        return None, None

    method_dir = Path(method_dir)
    _, match_filename, cache_source = METHOD_SPECS[method_id]

    matches_path = method_dir / match_filename
    if not matches_path.exists():
        logging.warning(f"[{method_id}] Match file not found: {matches_path}")
        return None, None

    if method_id == "random" and random_alex_cache is not None:
        alex_cache_path = Path(random_alex_cache)
    elif cache_source == "multi" and multi_dir is not None:
        alex_cache_path = Path(multi_dir) / "alex_embeddings_dedup.npz"
    else:
        alex_cache_path = method_dir / "alex_embeddings_dedup.npz"

    if not alex_cache_path.exists():
        logging.warning(f"[{method_id}] Alex cache not found: {alex_cache_path}")
        return None, None

    matches   = np.load(str(matches_path))
    alex_data = _load_cache(alex_cache_path)
    logging.info(f"[{method_id}] matches {matches.shape}  cache {alex_cache_path.name}")
    return matches, alex_data


def load_multi_ensemble(multi_dirs_list):
    """
    Merge matches from multiple MEX-Multi seed directories into one pool.

    Each training seed produces a different (but valid) embedding space, so
    its top-K retrieved structures are different.  Concatenating them:
      - Gives K*n_seeds candidates per query  →  denser, more representative pool
      - Averages out per-seed biases caused by the random initialisation

    Returns (matches_ensemble, alex_data) or (None, None).
    alex_data is taken from the first available directory (all seeds use
    the same de-duplicated alex pool).
    """
    all_matches = []
    alex_data   = None

    for d in multi_dirs_list:
        d = Path(d)
        mpath = d / "matches_v3.npy"
        cpath = d / "alex_embeddings_dedup.npz"
        if not mpath.exists():
            logging.warning(f"[ensemble] Missing {mpath} — skipping")
            continue
        if alex_data is None and cpath.exists():
            alex_data = _load_cache(cpath)
        m = np.load(str(mpath))
        all_matches.append(m)
        logging.info(f"[ensemble] Loaded {mpath.parent.name}  shape {m.shape}")

    if not all_matches:
        logging.warning("[ensemble] No valid dirs found — skipping ensemble.")
        return None, None

    # Stack along the K axis: (n_test, K*n_seeds)
    ensemble = np.concatenate(all_matches, axis=1)
    logging.info(f"[ensemble] Combined shape: {ensemble.shape}")
    return ensemble, alex_data


# ─────────────────────────────────────────────────────────────────────────────
# KL computation with bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def compute_kl_for_method(matches, alex_data, test_data,
                          n_bootstrap=200, rng_seed=42):
    """
    Compute KL(P_test || P_retrieved) for all 3 properties.

    With n_bootstrap > 1 an additional bootstrap step resamples the retrieved
    pool to estimate the variance of the KL estimator (see module docstring
    for a full explanation of why this is necessary on small datasets).

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
            # Single point estimate — equivalent to what compare_kl.py computes
            point = kl_divergence(test_vals, matched_vals,
                                  n_bins=nbins, xmin=xmin, xmax=xmax)
            kl[tkey] = {"mean": point, "std": 0.0,
                        "ci95_lo": point, "ci95_hi": point}
        else:
            # Bootstrap: draw 200 resample-with-replacement variants of the
            # retrieved pool, recompute KL each time, report mean and 95% CI.
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


def _bar_group_chart(kl_by_method, ordered_methods, title, out_path,
                     show_errorbars=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Apply publication-ready global styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True, # Ensure grid lines stay behind the bars
    })

    n = len(ordered_methods)
    x = np.arange(len(PROP_KEYS))
    # Slightly widen bars for better visibility, keeping spacing proportional
    w = min(0.8 / n, 0.22)

    fig, ax = plt.subplots(figsize=(max(9, 2.8 * n), 5.5))

    # Add subtle horizontal gridlines to help the reader estimate bar heights
    ax.yaxis.grid(True, linestyle="--", color="#cccccc", alpha=0.7)

    for i, method_id in enumerate(ordered_methods):
        label  = METHOD_SPECS[method_id][0]
        means  = [_kl_mean(kl_by_method[method_id], pk) for pk in PROP_KEYS]
        offset = (i - n / 2 + 0.5) * w

        yerr = None
        if show_errorbars:
            lo = [_kl_err(kl_by_method[method_id], pk)[0] for pk in PROP_KEYS]
            hi = [_kl_err(kl_by_method[method_id], pk)[1] for pk in PROP_KEYS]
            yerr = [lo, hi]

        # Add black edgecolors to the bars for sharper contrast
        bars = ax.bar(x + offset, means, w,
                      label=label, color=METHOD_COLORS[method_id], alpha=0.85,
                      edgecolor="black", linewidth=0.8,
                      yerr=yerr,
                      error_kw={"elinewidth": 1.5, "capsize": 4, "markeredgewidth": 1.2})

        # Annotate bars with values
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                # Add a semi-transparent white background to text to prevent overlap with error bars
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{mean:.3f}", ha="center", va="bottom",
                        fontsize=9, rotation=45, color="black",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(PROP_LABELS, fontweight="bold")
    ax.set_ylabel("KL Divergence ($P_{test} \\parallel P_{retrieved}$) ↓", fontweight="bold")
    
    # Pad the title slightly so it doesn't crowd the top annotations
    ax.set_title(title, pad=15)
    
    # Remove the legend border for a cleaner look
    ax.legend(loc="upper right", frameon=False)
    
    fig.tight_layout()
    # Increase DPI to 300 (standard publication resolution)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def plot_full_comparison(kl_by_method, active_methods, out_dir, bootstrap):
    ordered = [m for m in METHOD_ORDER if m in active_methods]
    suffix  = "\nError bars indicate 95% bootstrap CI" if bootstrap else ""
    _bar_group_chart(
        kl_by_method, ordered,
        f"Retrieval Quality Across All Methods{suffix}",
        out_dir / "kl_comparison_all_methods.png",
        show_errorbars=bootstrap,
    )


def plot_single_vs_multi(kl_by_method, out_dir, bootstrap):
    comparison = [m for m in ["single_eform", "single_bandgap", "single_gvrh",
                               "multi", "multi_ensemble"]
                  if m in kl_by_method]
    if len(comparison) < 2:
        logging.info("Fewer than 2 single/multi methods — skipping single-vs-multi chart.")
        return
    suffix = "\nError bars indicate 95% bootstrap CI" if bootstrap else ""
    _bar_group_chart(
        kl_by_method, comparison,
        f"Single-Property MEX vs. MEX-Multi{suffix}",
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

def parse_args():
    p = argparse.ArgumentParser(
        description="compare_kl_v2 — KL comparison with ensemble and bootstrap CI."
    )
    p.add_argument("--random-dir",          default=None,
                   help="Dir with random_matches.npy (from random_retrieval_baseline.py)")
    p.add_argument("--single-eform-dir",    default=None,
                   help="Dir with matches_eform.npy (from embed_and_match_alex_single.py)")
    p.add_argument("--single-bandgap-dir",  default=None,
                   help="Dir with matches_bandgap.npy")
    p.add_argument("--single-gvrh-dir",     default=None,
                   help="Dir with matches_gvrh.npy")
    p.add_argument("--multi-dir",           default=None,
                   help="Dir with matches_v3.npy from ONE MEX-Multi seed")
    p.add_argument("--multi-dirs",          default=None,
                   help="Comma-separated MEX-Multi dirs for ensemble "
                        "(e.g. seed0_dir,seed1_dir,seed2_dir). "
                        "Adds a 'MEX-Multi (ensemble)' curve that merges all seeds.")
    p.add_argument("--test-lmdb",           required=True,
                   help="Multi-property test LMDB (data/mp20_1k/test/data.lmdb)")
    p.add_argument("--out-dir",             required=True)
    p.add_argument("--random-alex-cache",   default=None,
                   help="alex_embeddings_dedup.npz for random baseline "
                        "(defaults to --multi-dir/alex_embeddings_dedup.npz)")
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

    # ── single-seed methods ───────────────────────────────────────────────────
    method_dirs = {
        "random":         args.random_dir,
        "single_eform":   args.single_eform_dir,
        "single_bandgap": args.single_bandgap_dir,
        "single_gvrh":    args.single_gvrh_dir,
        "multi":          args.multi_dir,
    }

    kl_by_method   = {}
    active_methods = []

    for method_id in ["random", "single_eform", "single_bandgap", "single_gvrh", "multi"]:
        matches, alex_data = load_method(
            method_id, method_dirs[method_id], args.multi_dir, args.random_alex_cache
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

    # ── seed ensemble ─────────────────────────────────────────────────────────
    if args.multi_dirs:
        dirs_list = [d.strip() for d in args.multi_dirs.split(",") if d.strip()]
        ens_matches, ens_alex = load_multi_ensemble(dirs_list)
        if ens_matches is not None:
            kl_ens = compute_kl_for_method(ens_matches, ens_alex, test_data,
                                           n_bootstrap=args.bootstrap)
            kl_by_method["multi_ensemble"] = kl_ens
            active_methods.append("multi_ensemble")
            logging.info(
                f"  [MEX-Multi ensemble]  "
                f"eform={kl_ens['y_eform']['mean']:.4f}  "
                f"bandgap={kl_ens['y_bandgap']['mean']:.4f}  "
                f"gvrh={kl_ens['y_gvrh']['mean']:.4f}"
            )

    if not active_methods:
        logging.error("No method directories found. Check your paths.")
        return

    logging.info(f"\nActive methods: {active_methods}")

    # ── plots + CSV ───────────────────────────────────────────────────────────
    plot_full_comparison(kl_by_method, active_methods, out_dir, use_bootstrap)
    plot_single_vs_multi(kl_by_method, out_dir, use_bootstrap)
    save_kl_csv(kl_by_method, out_dir)
    logging.info("compare_kl_v2 done.")


if __name__ == "__main__":
    main()
