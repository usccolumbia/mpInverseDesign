"""
embed_and_match_alex_v3.py
==========================
Retrieval-based inverse design — property distribution comparison between
test structures and their top-K embedding-space neighbours in alex_mp20.

CONCEPTUAL IDEA
---------------
Each test structure has three known properties:
    y_eform   (formation energy, eV/atom)
    y_bandgap (band gap, eV)
    y_gvrh    (log10 bulk modulus, log10 GPa)

For each test structure we retrieve its top-K nearest neighbours in the
MEX-Multi embedding space from alex_mp20 (overlap-free).
Each retrieved alex structure has AT LEAST ONE of:
    formation_energy_per_atom  (comparable to y_eform)
    dft_band_gap               (comparable to y_bandgap)
    log10_bulk_modulus         (comparable to y_gvrh)

HISTOGRAM LOGIC (3 panels, one per property pair)
-------------------------------------------------
Panel 1 — Formation energy:
  Blue  : test set y_eform values  (all 981 samples)
  Orange: top-K matched alex formation_energy_per_atom values
          (flatten 981×K, drop NaN, keep with repetition)

Panel 2 — Band gap:
  Blue  : test set y_bandgap values
  Orange: matched alex dft_band_gap values

Panel 3 — Bulk modulus (log10):
  Blue  : test set y_gvrh values
  Orange: matched alex log10_bulk_modulus values

If the embedding space has captured property-relevant structure, the orange
distribution should broadly follow the blue one — structures that are
geometrically similar (close in embedding) should also have similar properties.

DIFFERENCES FROM v2
-------------------
- Histogram compares TEST properties vs MATCHED ALEX properties
  (not full-alex background vs matched-alex)
- Three panels only (one per corresponding property pair)
- All K matched entries per test sample are included (with repetition),
  giving up to N_test × K alex property values per panel

USAGE
-----
python inverse_design/embed_and_match_alex_v3.py \\
    --checkpoint outputs/.../best_checkpoint.pt \\
    --config     configs/mex-multi/mp20_three_property.yml \\
    --test-lmdb  data/mp20_three_property/test/data.lmdb \\
    --train-lmdb data/mp20_three_property/train/data.lmdb \\
    --val-lmdb   data/mp20_three_property/val/data.lmdb \\
    --alex-train data/alex_mp20_train.csv \\
    --alex-val   data/alex_mp20_val.csv \\
    --all-csv    data/all_with_180.csv \\
    --out-dir    outputs/alex_matching_v3 \\
    --top-k      10 \\
    --device     cuda

# Reuse cached embeddings from a v2 run (same cache format)
python inverse_design/embed_and_match_alex_v3.py \\
    --checkpoint ... --config ... \\
    --out-dir    outputs/alex_matching_v2 \\   # point at v2 cache dir
    --cache-dir  outputs/alex_matching_v2 \\
    --out-dir    outputs/alex_matching_v3 \\
    --top-k      10
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Func

# ── make ocp importable ───────────────────────────────────────────────────────
# Insert BEFORE any ocpmodels import and evict stale cluster site-packages.
REPO_ROOT = Path(__file__).resolve().parent.parent
_ocp_path = str(REPO_ROOT / "ocp")
if _ocp_path not in sys.path:
    sys.path.insert(0, _ocp_path)
for _mod in list(sys.modules):
    if _mod == "ocpmodels" or _mod.startswith("ocpmodels."):
        del sys.modules[_mod]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── reuse infrastructure from sibling files ───────────────────────────────────
_inv_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_inv_dir))
from inverse_design_mex import MEXInferenceEngine          # noqa: E402
from embed_and_match_alex import _save_cache, _load_cache  # noqa: E402
from embed_and_match_alex_v2 import (                       # noqa: E402
    collect_split_material_ids,
    embed_test_set,
    load_and_embed_alex,
    top_k_matches,
)


# ─────────────────────────────────────────────────────────────────────────────
# Histogram: test properties  vs  matched alex properties
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_vs_matched(
    test_data: dict,
    alex_data: dict,
    matches: np.ndarray,     # (N_test, K)
    out_dir: Path,
):
    """
    3-panel histogram comparing test set property distributions (blue)
    against the properties of their top-K alex_mp20 neighbours (orange).

    The matched alex set is the full top-K collection (N_test × K entries).
    If the same alex structure appears in multiple test samples' top-K lists
    it is kept as-is — no extra structures are fetched and no entries are
    removed.  Per-property NaN values are dropped before plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = matches.shape[1]
    N_test = matches.shape[0]

    # All match indices, N_test × K (repetitions kept as-is)
    flat_idx = matches.ravel()

    # Property pairs: (test_key, alex_key, xlabel, xmin, xmax, nbins)
    panels = [
        (
            "y_eform",
            "formation_energy_per_atom",
            "Formation Energy (eV/atom)\n"
            "Test: y_eform  |  Alex: formation_energy_per_atom (mp-* only)",
            -5.0, 1.0, 50,
        ),
        (
            "y_bandgap",
            "dft_band_gap",
            "Band Gap (eV)\n"
            "Test: y_bandgap  |  Alex: dft_band_gap",
            0.0, 10.0, 50,
        ),
        (
            "y_gvrh",
            "log10_bulk_modulus",
            "log₁₀ Bulk Modulus (log₁₀ GPa)\n"
            "Test: y_gvrh  |  Alex: log10_bulk_modulus",
            -1.0, 3.5, 50,
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    summary = {}

    for ax, (tkey, akey, xlabel, xmin, xmax, nbins) in zip(axes, panels):

        # ── test values (all N_test, always non-NaN) ─────────────────────────
        test_vals = test_data[tkey].ravel()           # (N_test,)

        # ── matched alex values (top-K pool, drop NaN per property) ──────────
        alex_prop = alex_data[akey]                   # (N_alex,)
        matched_prop = alex_prop[flat_idx]            # (N_test * K,)
        matched_vals = matched_prop[~np.isnan(matched_prop)]

        n_test    = len(test_vals)
        n_matched = len(matched_vals)

        bins = np.linspace(xmin, xmax, nbins + 1)

        ax.hist(test_vals,    bins=bins, density=True,
                color="steelblue", alpha=0.55,
                label=f"Test set ({n_test:,})")
        if n_matched > 0:
            ax.hist(matched_vals, bins=bins, density=True,
                    color="darkorange", alpha=0.70,
                    label=f"Matched alex top-{K} pool ({n_matched:,} values)")
        else:
            ax.text(0.5, 0.5, "No matched alex\nstructures with\nthis property",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="darkorange")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=8)

        summary[tkey] = {
            "test_mean":   float(np.mean(test_vals)),
            "test_std":    float(np.std(test_vals)),
            "matched_n":   n_matched,
            "matched_mean": float(np.nanmean(matched_vals)) if n_matched > 0 else float("nan"),
            "matched_std":  float(np.nanstd(matched_vals))  if n_matched > 0 else float("nan"),
        }

    fig.suptitle(
        f"Test set property distribution  vs  top-{K} alex_mp20 neighbours\n"
        "(cosine similarity in MEX-Multi embedding space · overlap-free alex)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    out_path = out_dir / "test_vs_matched_alex_histograms.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Histogram saved to {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CSV of matched alex entries (one row per test × match pair)
# ─────────────────────────────────────────────────────────────────────────────

def save_match_table(
    test_data: dict,
    alex_data: dict,
    matches: np.ndarray,   # (N_test, K)
    out_dir: Path,
):
    """
    Write a flat CSV: each row = one (test_structure, matched_alex) pair.
    Columns: test_material_id, test_y_eform, test_y_bandgap, test_y_gvrh,
             rank (1..K), alex_material_id, cos_sim (not stored — N/A here),
             alex_formation_energy, alex_dft_band_gap,
             alex_log10_bulk_modulus, alex_energy_above_hull
    """
    K = matches.shape[1]
    csv_path = out_dir / "test_alex_match_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "test_material_id",
            "test_y_eform_eV_atom",
            "test_y_bandgap_eV",
            "test_y_gvrh_log10GPa",
            "match_rank",
            "alex_material_id",
            "alex_formation_energy_eV_atom",
            "alex_dft_band_gap_eV",
            "alex_log10_bulk_modulus",
            "alex_energy_above_hull_eV_atom",
        ])
        for q_i, test_mid in enumerate(test_data["material_id"]):
            for rank, a_i in enumerate(matches[q_i], start=1):
                w.writerow([
                    test_mid,
                    round(float(test_data["y_eform"][q_i]),    6),
                    round(float(test_data["y_bandgap"][q_i]),  6),
                    round(float(test_data["y_gvrh"][q_i]),     6),
                    rank,
                    alex_data["material_id"][a_i],
                    _fmt(alex_data["formation_energy_per_atom"][a_i]),
                    _fmt(alex_data["dft_band_gap"][a_i]),
                    _fmt(alex_data["log10_bulk_modulus"][a_i]),
                    _fmt(alex_data["energy_above_hull"][a_i]),
                ])
    logging.info(f"Match table saved to {csv_path}  ({len(test_data['material_id']) * K} rows)")


def _fmt(v):
    return "" if (isinstance(v, float) and np.isnan(v)) else round(float(v), 6)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare test set property distributions with properties of their "
            "top-K nearest alex_mp20 neighbours in MEX-Multi embedding space."
        )
    )
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--test-lmdb",   default="data/mp20_three_property/test/data.lmdb")
    p.add_argument("--train-lmdb",  default="data/mp20_three_property/train/data.lmdb")
    p.add_argument("--val-lmdb",    default="data/mp20_three_property/val/data.lmdb")
    p.add_argument("--alex-train",  default="data/alex_mp20_train.csv")
    p.add_argument("--alex-val",    default="data/alex_mp20_val.csv")
    p.add_argument("--all-csv",     default="data/all_with_180.csv",
                   help="CSV with formation_energy_per_atom for mp-* IDs")
    p.add_argument("--out-dir",     default="outputs/alex_matching_v3")
    p.add_argument("--cache-dir",   default=None,
                   help="Read cached embeddings from this dir instead of re-embedding "
                        "(useful to reuse outputs from a v2 run)")
    p.add_argument("--top-k",       type=int, default=10)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--no-cache",    action="store_true",
                   help="Force re-embedding even if cache files exist")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir   = Path(args.out_dir)
    # cache_dir: where to LOOK for pre-existing caches (e.g. a v2 output dir).
    # New caches are always written to out_dir regardless.
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read locations (may point at v2 dir or out_dir)
    test_cache_read = cache_dir / "test_embeddings.npz"
    alex_cache_read = cache_dir / "alex_embeddings_dedup.npz"
    # Write locations (always in out_dir)
    test_cache_write = out_dir / "test_embeddings.npz"
    alex_cache_write = out_dir / "alex_embeddings_dedup.npz"

    # ── load model ────────────────────────────────────────────────────────────
    engine = MEXInferenceEngine(args.checkpoint, args.config, device=args.device)

    # ── split material_ids for deduplication ──────────────────────────────────
    split_mids = collect_split_material_ids(
        args.train_lmdb, args.val_lmdb, args.test_lmdb
    )

    # ── test embeddings ───────────────────────────────────────────────────────
    if not args.no_cache and test_cache_read.exists():
        logging.info(f"Loading test embeddings from cache: {test_cache_read}")
        test_data = _load_cache(test_cache_read)
    else:
        logging.info("Computing test set embeddings from scratch …")
        test_data = embed_test_set(engine, args.test_lmdb, args.batch_size)
        _save_cache(test_cache_write, test_data)
        logging.info(f"Test embeddings cached to {test_cache_write}")

    # ── alex embeddings ───────────────────────────────────────────────────────
    if not args.no_cache and alex_cache_read.exists():
        logging.info(f"Loading alex embeddings from cache: {alex_cache_read}")
        alex_data = _load_cache(alex_cache_read)
    else:
        logging.info("Computing alex_mp20 embeddings from scratch (this may take a while) …")
        alex_data = load_and_embed_alex(
            engine, args.alex_train, args.alex_val,
            split_mids, args.batch_size,
            all_csv=args.all_csv,
        )
        _save_cache(alex_cache_write, alex_data)
        logging.info(f"Alex embeddings cached to {alex_cache_write}")

    # ── top-K matching ────────────────────────────────────────────────────────
    matches = top_k_matches(
        test_data["embeddings"],
        alex_data["embeddings"],
        top_k=args.top_k,
    )
    np.save(str(out_dir / "matches_v3.npy"), matches)

    # ── histograms: test properties vs matched alex properties ────────────────
    summary = plot_test_vs_matched(test_data, alex_data, matches, out_dir)

    # ── flat match table CSV ──────────────────────────────────────────────────
    save_match_table(test_data, alex_data, matches, out_dir)

    # ── print summary ─────────────────────────────────────────────────────────
    logging.info("\n=== Property distribution comparison ===")
    prop_labels = {
        "y_eform":   "Formation energy (eV/atom)",
        "y_bandgap": "Band gap (eV)       ",
        "y_gvrh":    "log10 Bulk modulus  ",
    }
    for tkey, label in prop_labels.items():
        s = summary[tkey]
        logging.info(
            f"  {label}  "
            f"test: {s['test_mean']:.3f}±{s['test_std']:.3f}  |  "
            f"matched alex: {s['matched_mean']:.3f}±{s['matched_std']:.3f} "
            f"(n={s['matched_n']:,})"
        )


if __name__ == "__main__":
    main()
