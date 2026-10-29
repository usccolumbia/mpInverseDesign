"""
compare_kl.py
=============
Compare KL divergences of retrieved property distributions across retrieval
methods:
  - Random retrieval baseline
  - Single-property MEX  (eform  / bandgap / gvrh)
  - Multi-property MEX   (MEX-Multi)

For each (method, property) pair this script loads the precomputed match
array (.npy), looks up the corresponding alex pool property values, and
computes KL(P_test || P_retrieved).

Produces:
  1. kl_comparison_all_methods.png  — full grouped bar chart (≤5 methods × 3 properties)
  2. kl_comparison_single_vs_multi.png — Single-MEX vs MEX-Multi only
  3. kl_comparison.csv              — raw KL numbers for all methods/properties

All method directories are optional; any combination present will be compared.

USAGE
-----
python inverse_design/compare_kl.py \\
    --random-dir          outputs/rand_mp20_1k_seed0 \\
    --single-eform-dir    /path/to/mex_org/outputs/alex_match_1k_eform_seed0 \\
    --single-bandgap-dir  /path/to/mex_org/outputs/alex_match_1k_bandgap_seed0 \\
    --single-gvrh-dir     /path/to/mex_org/outputs/alex_match_1k_gvrh_seed0 \\
    --multi-dir           outputs/alex_match_1k_multi_seed0 \\
    --test-lmdb           data/mp20_1k/test/data.lmdb \\
    --out-dir             outputs/compare_kl_1k_seed0

Notes
-----
* --test-lmdb should be the MULTI-PROPERTY LMDB that stores y_eform, y_bandgap,
  y_gvrh (e.g. data/mp20_1k/test/data.lmdb).  Single-property test LMDBs are
  *not* used here; the test reference is always the multi-prop split so that KL
  comparisons are on an equal footing.

* random matches index into the same alex pool order as MEX-Multi (both built
  from the same train/val CSVs with the same mp20 overlap removal).  The script
  therefore uses the multi-dir's alex_embeddings_dedup.npz for the random method.
  If --multi-dir is unavailable but --random-dir is, pass --random-alex-cache
  to explicitly specify an alex cache for the random baseline.
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

# Reuse helpers from sibling scripts (no model loading required).
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
#   alex_cache_source = "self"  → use method_dir/alex_embeddings_dedup.npz
#   alex_cache_source = "multi" → use multi_dir/alex_embeddings_dedup.npz
#                                 (random baseline shares the same alex pool as MEX-Multi)
METHOD_SPECS = {
    "random":         ("Random",         "random_matches.npy",  "multi"),
    "single_eform":   ("Single-eform",   "matches_eform.npy",   "self"),
    "single_bandgap": ("Single-bandgap", "matches_bandgap.npy", "self"),
    "single_gvrh":    ("Single-gvrh",    "matches_gvrh.npy",    "self"),
    "multi":          ("MEX-Multi",      "matches_v3.npy",      "self"),
}

METHOD_COLORS = {
    "random":         "#aaaaaa",
    "single_eform":   "#2196F3",
    "single_bandgap": "#4CAF50",
    "single_gvrh":    "#FF9800",
    "multi":          "#9C27B0",
}

# Canonical order for charts
METHOD_ORDER = ["random", "single_eform", "single_bandgap", "single_gvrh", "multi"]

# Property display order (must match PANELS)
PROP_KEYS       = ["y_eform",            "y_bandgap",    "y_gvrh"]
PROP_LABELS     = ["Formation\nEnergy",  "Band\nGap",    "Bulk\nModulus"]
PROP_CSV_NAMES  = ["formation_energy",   "band_gap",     "bulk_modulus_log10"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_method(method_id, method_dir, multi_dir, random_alex_cache=None):
    """
    Load (matches, alex_data) for a given method.
    Returns (None, None) if the required files are absent.
    """
    if method_dir is None:
        return None, None

    method_dir = Path(method_dir)
    label, match_filename, cache_source = METHOD_SPECS[method_id]

    matches_path = method_dir / match_filename
    if not matches_path.exists():
        logging.warning(f"[{method_id}] Match file not found: {matches_path}")
        return None, None

    # Resolve alex cache
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
    logging.info(f"[{method_id}] matches {matches.shape}  "
                 f"alex cache {alex_cache_path.name}")
    return matches, alex_data


# ─────────────────────────────────────────────────────────────────────────────
# KL computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_kl_for_method(matches, alex_data, test_data):
    """
    Compute KL(P_test || P_retrieved) for all 3 properties.
    Returns dict {test_key: kl_value}.
    """
    flat_idx = matches.ravel()
    kl = {}
    for tkey, akey, _, xmin, xmax, nbins in PANELS:
        test_vals = test_data[tkey].ravel()
        test_vals = test_vals[~np.isnan(test_vals)]

        alex_prop = alex_data.get(akey, np.array([]))
        if not isinstance(alex_prop, np.ndarray):
            alex_prop = np.array(alex_prop, dtype=float)

        if len(alex_prop) == 0 or flat_idx.max() >= len(alex_prop):
            logging.warning(
                f"  {tkey}: alex prop length {len(alex_prop)} < max index "
                f"{flat_idx.max()} – skipping"
            )
            kl[tkey] = float("nan")
            continue

        matched_vals = alex_prop[flat_idx].astype(float)
        matched_vals = matched_vals[~np.isnan(matched_vals)]
        kl[tkey] = kl_divergence(
            test_vals, matched_vals, n_bins=nbins, xmin=xmin, xmax=xmax
        )
    return kl


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _bar_group_chart(kl_by_method, ordered_methods, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ordered_methods)
    x = np.arange(len(PROP_KEYS))
    w = min(0.8 / n, 0.22)

    fig, ax = plt.subplots(figsize=(max(9, 2.5 * n), 5))

    for i, method_id in enumerate(ordered_methods):
        label = METHOD_SPECS[method_id][0]
        kl_vals = [kl_by_method[method_id].get(pk, float("nan")) for pk in PROP_KEYS]
        offset  = (i - n / 2 + 0.5) * w
        bars = ax.bar(x + offset, kl_vals, w,
                      label=label, color=METHOD_COLORS[method_id], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(PROP_LABELS, fontsize=11)
    ax.set_ylabel("KL Divergence (P_test ∥ P_retrieved) ↓", fontsize=10)
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {out_path}")


def plot_full_comparison(kl_by_method, active_methods, out_dir):
    ordered = [m for m in METHOD_ORDER if m in active_methods]
    _bar_group_chart(
        kl_by_method, ordered,
        "Retrieval quality: all methods\n"
        "(lower KL = retrieved properties closer to test set)",
        out_dir / "kl_comparison_all_methods.png",
    )


def plot_single_vs_multi(kl_by_method, out_dir):
    comparison = [m for m in ["single_eform", "single_bandgap", "single_gvrh", "multi"]
                  if m in kl_by_method]
    if len(comparison) < 2:
        logging.info("Fewer than 2 single/multi methods available — skipping "
                     "single-vs-multi chart.")
        return
    _bar_group_chart(
        kl_by_method, comparison,
        "Single-property MEX vs MEX-Multi retrieval quality\n"
        "(lower KL = retrieved properties closer to test set)",
        out_dir / "kl_comparison_single_vs_multi.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_kl_csv(kl_by_method, out_dir):
    out_path = out_dir / "kl_comparison.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "property", "kl_divergence"])
        for method_id in METHOD_ORDER:
            if method_id not in kl_by_method:
                continue
            kl_dict = kl_by_method[method_id]
            for pk, pname in zip(PROP_KEYS, PROP_CSV_NAMES):
                w.writerow([method_id, pname,
                             f"{kl_dict.get(pk, float('nan')):.6f}"])
    logging.info(f"KL CSV saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare KL retrieval quality across Random / Single-MEX / MEX-Multi."
    )
    p.add_argument("--random-dir",          default=None,
                   help="Dir containing random_matches.npy (from random_retrieval_baseline.py)")
    p.add_argument("--single-eform-dir",    default=None,
                   help="Dir containing matches_eform.npy (from embed_and_match_alex_single.py)")
    p.add_argument("--single-bandgap-dir",  default=None,
                   help="Dir containing matches_bandgap.npy")
    p.add_argument("--single-gvrh-dir",     default=None,
                   help="Dir containing matches_gvrh.npy")
    p.add_argument("--multi-dir",           default=None,
                   help="Dir containing matches_v3.npy (from embed_and_match_alex_v3.py)")
    p.add_argument("--test-lmdb",           required=True,
                   help="Multi-property test LMDB with y_eform, y_bandgap, y_gvrh "
                        "(e.g. data/mp20_1k/test/data.lmdb)")
    p.add_argument("--out-dir",             required=True)
    p.add_argument("--random-alex-cache",   default=None,
                   help="Explicit path to alex_embeddings_dedup.npz for the random baseline. "
                        "Defaults to --multi-dir/alex_embeddings_dedup.npz.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── test set ──────────────────────────────────────────────────────────────
    test_data = load_test_properties(args.test_lmdb)
    n_test = len(test_data["material_id"])
    logging.info(f"Test set: {n_test} structures from {args.test_lmdb}")

    # ── method directories ────────────────────────────────────────────────────
    method_dirs = {
        "random":         args.random_dir,
        "single_eform":   args.single_eform_dir,
        "single_bandgap": args.single_bandgap_dir,
        "single_gvrh":    args.single_gvrh_dir,
        "multi":          args.multi_dir,
    }

    # ── load + compute KL ────────────────────────────────────────────────────
    kl_by_method  = {}
    active_methods = []

    for method_id in METHOD_ORDER:
        method_dir = method_dirs[method_id]
        matches, alex_data = load_method(
            method_id, method_dir, args.multi_dir, args.random_alex_cache
        )
        if matches is None:
            continue

        kl = compute_kl_for_method(matches, alex_data, test_data)
        kl_by_method[method_id]  = kl
        active_methods.append(method_id)

        label = METHOD_SPECS[method_id][0]
        logging.info(
            f"  [{label}]  "
            f"eform={kl.get('y_eform', float('nan')):.4f}  "
            f"bandgap={kl.get('y_bandgap', float('nan')):.4f}  "
            f"gvrh={kl.get('y_gvrh', float('nan')):.4f}"
        )

    if not active_methods:
        logging.error("No method output directories found. Check your paths.")
        return

    logging.info(f"\nActive methods: {active_methods}")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_full_comparison(kl_by_method, active_methods, out_dir)
    plot_single_vs_multi(kl_by_method, out_dir)

    # ── CSV ───────────────────────────────────────────────────────────────────
    save_kl_csv(kl_by_method, out_dir)

    logging.info("compare_kl done.")


if __name__ == "__main__":
    main()
