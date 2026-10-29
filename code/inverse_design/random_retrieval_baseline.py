"""
random_retrieval_baseline.py
============================
Baseline for inverse design retrieval: for each test structure randomly
sample K alex_mp20 structures and compare property distributions against
the test set.

This is the lower-bound baseline for the embedding-space retrieval experiment.
If MEX-Multi's cosine-similarity retrieval (embed_and_match_alex_v3.py) shows
better distributional alignment than this random baseline, it confirms that
the learned embedding space encodes property-relevant structural information.

USAGE
-----
python inverse_design/random_retrieval_baseline.py \
    --alex-train data/alex_mp20_train.csv \
    --alex-val   data/alex_mp20_val.csv \
    --all-csv    data/all_with_180.csv \
    --test-lmdb  data/mp20_three_property/test/data.lmdb \
    --train-lmdb data/mp20_three_property/train/data.lmdb \
    --val-lmdb   data/mp20_three_property/val/data.lmdb \
    --out-dir    outputs/random_retrieval_baseline \
    --top-k      10 \
    --seed       42

# Compare KL divergences vs. embed_and_match_alex_v3 outputs:
#   python inverse_design/random_retrieval_baseline.py --compare-dir outputs/alex_matching_v3
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── reuse data-loading helpers from v3 infra (no model needed) ───────────────
_inv_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_inv_dir))
from embed_and_match_alex import _load_cache  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ocp"))


# ─────────────────────────────────────────────────────────────────────────────
# Load alex properties (no embedding required)
# ─────────────────────────────────────────────────────────────────────────────

def load_alex_properties(alex_train_csv, alex_val_csv, split_mids, all_csv=None):
    """
    Load alex_mp20 property columns only (no CIF embedding).
    Returns a dict with numpy arrays, same format as embed_and_match_alex_v2's
    alex_data dict (minus 'embeddings').
    """
    import pandas as pd

    df_t = pd.read_csv(alex_train_csv)
    df_v = pd.read_csv(alex_val_csv)
    df = pd.concat([df_t, df_v], ignore_index=True)

    # Remove overlap with MP20 splits
    before = len(df)
    df = df[~df["material_id"].isin(split_mids)].reset_index(drop=True)
    logging.info(f"Alex: {before} → {len(df)} after removing MP20 overlap")

    # Keep only rows with at least one property
    has_prop = df["dft_band_gap"].notna() | df["dft_bulk_modulus"].notna()
    df = df[has_prop].reset_index(drop=True)
    logging.info(f"Alex after property filter: {len(df)} rows")

    # Compute log10 bulk modulus
    df["log10_bulk_modulus"] = np.where(
        df["dft_bulk_modulus"].notna() & (df["dft_bulk_modulus"] > 0),
        np.log10(df["dft_bulk_modulus"].clip(lower=1e-6)),
        np.nan,
    )

    # Join formation energy from all_with_180.csv
    if all_csv and Path(all_csv).exists():
        df_eform = pd.read_csv(all_csv, usecols=["material_id", "formation_energy_per_atom"])
        df = df.merge(df_eform, on="material_id", how="left")
        n_with = df["formation_energy_per_atom"].notna().sum()
        logging.info(f"Formation energy joined for {n_with}/{len(df)} alex rows")
    else:
        df["formation_energy_per_atom"] = np.nan

    def _arr(col):
        return df[col].to_numpy(dtype=float)

    return {
        "material_id":              df["material_id"].tolist(),
        "formation_energy_per_atom": _arr("formation_energy_per_atom"),
        "dft_band_gap":             _arr("dft_band_gap"),
        "log10_bulk_modulus":       _arr("log10_bulk_modulus"),
        "energy_above_hull":        _arr("energy_above_hull"),
    }


def collect_split_material_ids(*lmdb_paths):
    """Return the set of material_ids in the given LMDB files."""
    import lmdb, pickle
    mids = set()
    for p in lmdb_paths:
        env = lmdb.open(str(p), readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            n = pickle.loads(txn.get(b"length"))
            for i in range(n):
                d = pickle.loads(txn.get(str(i).encode()))
                if hasattr(d, "material_id"):
                    mids.add(str(d.material_id))
                elif hasattr(d, "sid"):
                    mids.add(str(int(d.sid)))
                elif isinstance(d, dict):
                    sid = d.get("sid") or d.get("material_id") or d.get("id")
                    if sid:
                        mids.add(str(sid))
        env.close()
    logging.info(f"Collected {len(mids)} split material_ids")
    return mids


def load_test_properties(test_lmdb):
    """Read y_eform, y_bandgap, y_gvrh and material_id from test LMDB."""
    import lmdb, pickle
    env = lmdb.open(str(test_lmdb), readonly=True, lock=False, subdir=False)
    mids, eforms, bandgaps, gvrhs = [], [], [], []
    with env.begin() as txn:
        n = pickle.loads(txn.get(b"length"))
        for i in range(n):
            d = pickle.loads(txn.get(str(i).encode()))
            # material_id
            if hasattr(d, "material_id"):
                mids.append(str(d.material_id))
            elif hasattr(d, "sid"):
                mids.append(str(int(d.sid)))
            else:
                mids.append("")
            # properties
            eforms.append(float(d.y_eform)   if hasattr(d, "y_eform")   else
                          float(d.y_relaxed) if hasattr(d, "y_relaxed") else
                          float(d.y[0])      if hasattr(d, "y")         else float("nan"))
            bandgaps.append(float(d.y_bandgap) if hasattr(d, "y_bandgap") else float("nan"))
            gvrhs.append(float(d.y_gvrh)       if hasattr(d, "y_gvrh")    else float("nan"))
    env.close()
    return {
        "material_id": mids,
        "y_eform":   np.array(eforms),
        "y_bandgap": np.array(bandgaps),
        "y_gvrh":    np.array(gvrhs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Random matching
# ─────────────────────────────────────────────────────────────────────────────

def random_matches(n_test, n_alex, top_k, rng):
    """
    For each of the n_test queries, sample top_k indices uniformly at random
    from [0, n_alex) WITHOUT replacement per query.
    Returns int array of shape (n_test, top_k).
    """
    return rng.choice(n_alex, size=(n_test, top_k), replace=False)


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence helper
# ─────────────────────────────────────────────────────────────────────────────

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
# Histogram plot
# ─────────────────────────────────────────────────────────────────────────────

PANELS = [
    ("y_eform",   "formation_energy_per_atom",
     "Formation Energy (eV/atom)\nTest: y_eform  |  Alex: formation_energy_per_atom",
     -5.0, 1.0, 50),
    ("y_bandgap", "dft_band_gap",
     "Band Gap (eV)\nTest: y_bandgap  |  Alex: dft_band_gap",
     0.0, 10.0, 50),
    ("y_gvrh",    "log10_bulk_modulus",
     "log₁₀ Bulk Modulus (log₁₀ GPa)\nTest: y_gvrh  |  Alex: log10_bulk_modulus",
     -1.0, 3.5, 50),
]


def plot_histograms(test_data, alex_data, matches, out_dir, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = matches.shape[1]
    flat_idx = matches.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    kl_scores = {}
    summary = {}

    for ax, (tkey, akey, xlabel, xmin, xmax, nbins) in zip(axes, PANELS):
        test_vals = test_data[tkey].ravel()
        test_vals = test_vals[~np.isnan(test_vals)]

        alex_prop = alex_data[akey]
        matched_prop = alex_prop[flat_idx]
        matched_vals = matched_prop[~np.isnan(matched_prop)]

        bins = np.linspace(xmin, xmax, nbins + 1)
        ax.hist(test_vals, bins=bins, density=True,
                color="steelblue", alpha=0.55,
                label=f"Test set ({len(test_vals):,})")
        if len(matched_vals) > 0:
            ax.hist(matched_vals, bins=bins, density=True,
                    color="darkorange", alpha=0.70,
                    label=f"Random alex top-{K} pool ({len(matched_vals):,} values)")

        kl = kl_divergence(test_vals, matched_vals, n_bins=nbins, xmin=xmin, xmax=xmax)
        kl_scores[tkey] = kl
        ax.set_title(f"KL = {kl:.4f}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=8)

        summary[tkey] = {
            "test_mean":    float(np.mean(test_vals)) if len(test_vals) else float("nan"),
            "test_std":     float(np.std(test_vals))  if len(test_vals) else float("nan"),
            "matched_n":    len(matched_vals),
            "matched_mean": float(np.nanmean(matched_vals)) if len(matched_vals) else float("nan"),
            "matched_std":  float(np.nanstd(matched_vals))  if len(matched_vals) else float("nan"),
            "kl_div":       kl,
        }

    fig.suptitle(
        f"Test set vs RANDOM alex_mp20 retrieval (K={K}){title_suffix}\n"
        "(lower bound baseline — no embedding used)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = out_dir / "random_retrieval_histograms.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Histogram saved to {out_path}")
    return summary, kl_scores


# ─────────────────────────────────────────────────────────────────────────────
# KL comparison plot (random vs MEX-Multi)
# ─────────────────────────────────────────────────────────────────────────────

def plot_kl_comparison(kl_random, kl_mex, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    props = ["y_eform", "y_bandgap", "y_gvrh"]
    labels = ["Formation\nEnergy", "Band\nGap", "Bulk\nModulus"]
    x = np.arange(len(props))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_r = ax.bar(x - w/2, [kl_random.get(p, float("nan")) for p in props],
                    w, label="Random retrieval", color="tomato", alpha=0.85)
    bars_m = ax.bar(x + w/2, [kl_mex.get(p, float("nan")) for p in props],
                    w, label="MEX-Multi (ours)", color="steelblue", alpha=0.85)

    for bar in list(bars_r) + list(bars_m):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("KL Divergence (P_test || P_retrieved) ↓")
    ax.set_title("Retrieval quality: Random vs MEX-Multi embedding\n"
                 "(lower KL = retrieved properties closer to test set)")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "kl_comparison_random_vs_mex.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"KL comparison plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(summary, kl_scores, out_dir):
    out_path = out_dir / "random_retrieval_summary.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["property", "test_mean", "test_std",
                    "matched_n", "matched_mean", "matched_std", "kl_div"])
        for prop, s in summary.items():
            w.writerow([prop, s["test_mean"], s["test_std"],
                        s["matched_n"], s["matched_mean"], s["matched_std"],
                        s["kl_div"]])
    logging.info(f"Summary CSV saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Random retrieval baseline for inverse design comparison."
    )
    p.add_argument("--alex-train",  default="data/alex_mp20_train.csv")
    p.add_argument("--alex-val",    default="data/alex_mp20_val.csv")
    p.add_argument("--all-csv",     default="data/all_with_180.csv")
    p.add_argument("--test-lmdb",   default="data/mp20_three_property/test/data.lmdb")
    p.add_argument("--train-lmdb",  default="data/mp20_three_property/train/data.lmdb")
    p.add_argument("--val-lmdb",    default="data/mp20_three_property/val/data.lmdb")
    p.add_argument("--out-dir",     default="outputs/random_retrieval_baseline")
    p.add_argument("--top-k",       type=int, default=10)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--compare-dir", default=None,
                   help="Path to embed_and_match_alex_v3 output dir containing "
                        "matches_v3.npy and alex_embeddings_dedup.npz. "
                        "If provided, produces a side-by-side KL comparison plot.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── load split IDs for deduplication ────────────────────────────────────
    split_mids = collect_split_material_ids(
        args.train_lmdb, args.val_lmdb, args.test_lmdb
    )

    # ── load test properties (no model needed) ───────────────────────────────
    test_data = load_test_properties(args.test_lmdb)
    n_test = len(test_data["material_id"])
    logging.info(f"Test set: {n_test} structures")

    # ── load alex properties (no embedding needed) ───────────────────────────
    alex_data = load_alex_properties(
        args.alex_train, args.alex_val, split_mids, args.all_csv
    )
    n_alex = len(alex_data["material_id"])
    logging.info(f"Alex pool: {n_alex} structures")

    # ── random matching ───────────────────────────────────────────────────────
    matches = random_matches(n_test, n_alex, args.top_k, rng)
    np.save(str(out_dir / "random_matches.npy"), matches)
    logging.info(f"Random matches: {matches.shape}")

    # ── histograms ────────────────────────────────────────────────────────────
    summary, kl_random = plot_histograms(
        test_data, alex_data, matches, out_dir,
        title_suffix=f"  (seed={args.seed})"
    )

    # ── summary CSV ───────────────────────────────────────────────────────────
    save_summary_csv(summary, kl_random, out_dir)

    # ── KL comparison with MEX-Multi (if v3 output dir provided) ─────────────
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        v3_matches_path = compare_dir / "matches_v3.npy"
        alex_cache_path = compare_dir / "alex_embeddings_dedup.npz"

        if v3_matches_path.exists() and alex_cache_path.exists():
            v3_matches = np.load(str(v3_matches_path))
            alex_v3 = _load_cache(alex_cache_path)

            # Recompute KL for MEX-Multi using same test data
            kl_mex = {}
            flat_v3 = v3_matches.ravel()
            prop_ranges = {
                "y_eform":   (-5.0, 1.0),
                "y_bandgap": (0.0, 10.0),
                "y_gvrh":    (-1.0, 3.5),
            }
            for tkey, akey, _, xmin, xmax, nbins in PANELS:
                test_vals = test_data[tkey].ravel()
                test_vals = test_vals[~np.isnan(test_vals)]
                matched_vals = alex_v3[akey][flat_v3]
                matched_vals = matched_vals[~np.isnan(matched_vals)]
                kl_mex[tkey] = kl_divergence(
                    test_vals, matched_vals, n_bins=nbins, xmin=xmin, xmax=xmax
                )

            plot_kl_comparison(kl_random, kl_mex, out_dir)

            logging.info("\n=== KL Divergence Comparison (lower is better) ===")
            prop_labels = {
                "y_eform":   "Formation energy",
                "y_bandgap": "Band gap        ",
                "y_gvrh":    "Bulk modulus    ",
            }
            for tkey, label in prop_labels.items():
                logging.info(
                    f"  {label}  Random: {kl_random.get(tkey, float('nan')):.4f}  "
                    f"MEX-Multi: {kl_mex.get(tkey, float('nan')):.4f}"
                )
        else:
            logging.warning(
                f"--compare-dir provided but missing files in {compare_dir}. "
                f"Skipping comparison plot."
            )

    # ── print summary ─────────────────────────────────────────────────────────
    logging.info("\n=== Random Retrieval Summary ===")
    for prop, s in summary.items():
        logging.info(
            f"  {prop:12s}  test: {s['test_mean']:.3f}±{s['test_std']:.3f}  "
            f"matched: {s['matched_mean']:.3f}±{s['matched_std']:.3f}  "
            f"KL: {s['kl_div']:.4f}"
        )


if __name__ == "__main__":
    main()
