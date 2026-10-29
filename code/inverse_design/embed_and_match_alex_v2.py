"""
embed_and_match_alex_v2.py
==========================
Retrieval-based inverse design — finding alex_mp20 analogues of test structures
in MEX-Multi embedding space, with per-property separate comparison.

KEY DIFFERENCES FROM embed_and_match_alex.py
--------------------------------------------
1. DEDUPLICATION: material_ids present in any of the mp20_three_property
   train / val / test splits are removed from alex_mp20 before embedding.

2. PARTIAL PROPERTIES ALLOWED: a structure is included if it has at least
   one of {dft_band_gap, dft_bulk_modulus}.  The full 675k alex rows are
   NOT embedded — only those with at least one "interesting" property
   (~43k rows; energy_above_hull is present for all of these as well).
   This keeps the embedding step practical (~35k after dedup).

   Note: energy_above_hull (thermodynamic convex-hull distance, ≥ 0) is a
   different quantity from formation energy (y_eform, can be negative).
   The plot labels make this explicit.

3. PER-PROPERTY SEPARATE COMPARISON: for each property k, histograms use
   only the subset of matched structures where property k is non-null —
   not the same intersection set for all three properties.

MATCHING STRATEGY
-----------------
  - Embed test set  →  X_test  (N_test × D)
  - Embed valid alex  →  X_alex  (N_alex × D)
  - Top-K cosine-similarity matches per test embedding  →  (N_test × K)
  - For each property: collect matched structures that have that property,
    compare their distribution to the full (non-overlapping) alex background

USAGE
-----
# Run from repo root
python inverse_design/embed_and_match_alex_v2.py \\
    --checkpoint outputs/.../best_checkpoint.pt \\
    --config     configs/mex-multi/mp20_three_property.yml \\
    --test-lmdb  data/mp20_three_property/test/data.lmdb \\
    --train-lmdb data/mp20_three_property/train/data.lmdb \\
    --val-lmdb   data/mp20_three_property/val/data.lmdb \\
    --alex-train data/alex_mp20_train.csv \\
    --all-csv    data/all_with_180.csv \\
    --alex-val   data/alex_mp20_val.csv \\
    --out-dir    outputs/alex_matching_v2 \\
    --top-k      10 \\
    --device     cuda

# Use cached embeddings from a previous run (skip re-embedding)
python inverse_design/embed_and_match_alex_v2.py \\
    --checkpoint ... --config ... \\
    --out-dir    outputs/alex_matching_v2 \\
    --top-k      10
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Func

# ── make ocp importable ───────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ocp"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── reuse inference engine and CIF → graph converter from sibling files ───────
_inv_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_inv_dir))
from inverse_design_mex import MEXInferenceEngine       # noqa: E402
from embed_and_match_alex import (                       # noqa: E402
    cif_to_graph,
    embed_records,
    read_lmdb,
    collate,
    _save_cache,
    _load_cache,
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 0 – collect material_ids from train / val / test splits
# ─────────────────────────────────────────────────────────────────────────────

def collect_split_material_ids(*lmdb_paths: str) -> set:
    """Return the set of material_id strings from one or more LMDB files."""
    mids = set()
    for lmdb_path in lmdb_paths:
        if not lmdb_path or not Path(lmdb_path).exists():
            logging.warning(f"LMDB path not found, skipping: {lmdb_path}")
            continue
        records = read_lmdb(lmdb_path)
        for r in records:
            if hasattr(r, "material_id"):
                v = r.material_id
                mids.add(v[0] if isinstance(v, list) else str(v))
    logging.info(f"Collected {len(mids)} unique material_ids from dataset splits")
    return mids


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – embed TEST set
# ─────────────────────────────────────────────────────────────────────────────

def embed_test_set(engine: MEXInferenceEngine, lmdb_path: str, batch_size: int) -> dict:
    logging.info(f"Reading test LMDB: {lmdb_path}")
    records = read_lmdb(lmdb_path)
    N = len(records)
    logging.info(f"  {N} test structures")

    logging.info("Embedding test structures …")
    embs = embed_records(engine, records, batch_size)

    def _prop(key):
        return np.array([float(r[key]) for r in records], dtype=np.float32)

    def _mid(r):
        if hasattr(r, "material_id"):
            v = r.material_id
            return v[0] if isinstance(v, list) else str(v)
        return str(int(r.sid))

    return {
        "embeddings":  embs,
        "y_eform":     _prop("y_eform"),
        "y_bandgap":   _prop("y_bandgap"),
        "y_gvrh":      _prop("y_gvrh"),
        "material_id": [_mid(r) for r in records],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – load, deduplicate, and embed alex_mp20
# ─────────────────────────────────────────────────────────────────────────────

def load_and_embed_alex(
    engine: MEXInferenceEngine,
    train_csv: str,
    val_csv: str,
    split_mids: set,
    batch_size: int,
    all_csv: str = "data/all_with_180.csv",
) -> dict:
    """
    Load alex CSVs, deduplicate against dataset splits, keep structures with at
    least one of {dft_band_gap, dft_bulk_modulus} (energy_above_hull is
    universally available for all rows), embed, and return result dict.

    Selection logic
    ---------------
    We require at least one of the two "challenging" properties so that the
    set is practical to embed (~43k rows before dedup, ~35k after).
    energy_above_hull is non-null for all rows, so it is always available for
    any structure that passes the bandgap/bulk_modulus filter.

    Formation energy
    ----------------
    Joined from data/all_with_180.csv on material_id.  Covers ~87% of the
    embedded set (the mp-* IDs).  alex<agm*> structures get NaN since no
    DFT formation energy is available for them without new calculations.

    Returns
    -------
    dict with keys:
      "embeddings"                – (N, D) float32
      "material_id"               – list[str]
      "dft_band_gap"              – (N,) float32, NaN where unavailable
      "dft_bulk_modulus"          – (N,) float32, NaN where unavailable
      "log10_bulk_modulus"        – (N,) float32, NaN where unavailable
      "energy_above_hull"         – (N,) float32  (always available)
      "formation_energy_per_atom" – (N,) float32, NaN for alex<agm*> IDs
    """
    import pandas as pd

    logging.info("Loading alex_mp20 CSVs …")
    df = pd.concat(
        [pd.read_csv(train_csv), pd.read_csv(val_csv)],
        ignore_index=True,
    )

    # ── join formation energy from all_with_180.csv ──────────────────────────
    if all_csv and Path(all_csv).exists():
        df_eform = pd.read_csv(
            all_csv, usecols=["material_id", "formation_energy_per_atom"]
        )
        df = df.merge(df_eform, on="material_id", how="left")
        n_eform = df["formation_energy_per_atom"].notna().sum()
        logging.info(
            f"  joined formation_energy_per_atom: {n_eform} rows "
            f"({n_eform/len(df)*100:.1f}% coverage)"
        )
    else:
        logging.warning(
            f"all_with_180.csv not found at '{all_csv}'; "
            "formation_energy_per_atom will be all NaN"
        )
        df["formation_energy_per_atom"] = float("nan")
    logging.info(f"  total rows loaded: {len(df)}")

    # ── deduplication ────────────────────────────────────────────────────────
    before = len(df)
    df = df[~df["material_id"].isin(split_mids)].reset_index(drop=True)
    logging.info(f"  removed {before - len(df)} rows overlapping with dataset splits")
    logging.info(f"  remaining: {len(df)}")

    # ── keep structures with at least one of band_gap / bulk_modulus ─────────
    has_prop = df["dft_band_gap"].notna() | df["dft_bulk_modulus"].notna()
    df = df[has_prop].reset_index(drop=True)
    logging.info(
        f"  rows with dft_band_gap or dft_bulk_modulus non-null: {len(df)}"
        f"  (bandgap: {df['dft_band_gap'].notna().sum()}, "
        f"bulk_mod: {df['dft_bulk_modulus'].notna().sum()})"
    )

    # ── CIF → graph ──────────────────────────────────────────────────────────
    logging.info("Parsing CIF strings to graphs …")
    records = []
    valid_idx = []
    for i, row in df.iterrows():
        try:
            g = cif_to_graph(row["cif"], sid=i)
            records.append(g)
            valid_idx.append(i)
        except Exception as e:
            logging.warning(f"  CIF parse failed for {row['material_id']}: {e}")

    df = df.loc[valid_idx].reset_index(drop=True)
    logging.info(f"  successfully parsed: {len(records)} graphs")

    # ── embed ─────────────────────────────────────────────────────────────────
    logging.info("Embedding alex structures …")
    embs = embed_records(engine, records, batch_size)

    # log10 of bulk modulus (NaN-safe)
    bm = df["dft_bulk_modulus"].values.astype(np.float32)
    log10_bm = np.where(
        ~np.isnan(bm) & (bm > 0),
        np.log10(bm),
        np.nan,
    ).astype(np.float32)

    return {
        "embeddings":                embs,
        "material_id":               df["material_id"].tolist(),
        "dft_band_gap":              df["dft_band_gap"].values.astype(np.float32),
        "dft_bulk_modulus":          bm,
        "log10_bulk_modulus":        log10_bm,
        "energy_above_hull":         df["energy_above_hull"].values.astype(np.float32),
        "formation_energy_per_atom": df["formation_energy_per_atom"].values.astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – cosine-similarity top-K matching
# ─────────────────────────────────────────────────────────────────────────────

def top_k_matches(
    test_embs: np.ndarray,
    alex_embs: np.ndarray,
    top_k: int,
    chunk_size: int = 128,
) -> np.ndarray:
    """
    Returns (N_test, top_k) int array — indices into alex_embs.
    Uses cosine similarity; computation done in float32 chunks to fit in RAM.
    """
    te = Func.normalize(torch.tensor(test_embs, dtype=torch.float32), dim=-1)
    ae = Func.normalize(torch.tensor(alex_embs, dtype=torch.float32), dim=-1)

    N_test = te.shape[0]
    matches = np.empty((N_test, top_k), dtype=np.int32)

    logging.info(
        f"Computing top-{top_k} cosine-similarity matches "
        f"({N_test} test × {ae.shape[0]} alex) …"
    )
    for start in range(0, N_test, chunk_size):
        end = min(start + chunk_size, N_test)
        sim = te[start:end] @ ae.T           # (chunk, N_alex)
        topk_ids = sim.topk(top_k, dim=-1).indices.cpu().numpy()
        matches[start:end] = topk_ids
        if (start // chunk_size) % 10 == 0:
            logging.info(f"  {end}/{N_test}")

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – per-property histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_property_histograms(
    alex_data: dict,
    matches: np.ndarray,   # (N_test, K)
    out_dir: Path,
):
    """
    For each property separately:
      background  = full (dedup'd) alex subset that has this property
      foreground  = matched alex structures that have this property

    Each property only uses the available subset — no intersection required.

    Note on energy_above_hull vs y_eform
    -------------------------------------
    energy_above_hull ≥ 0 is the thermodynamic convex-hull distance (0 = stable).
    y_eform is formation energy (can be negative).  They are related but different;
    the histogram shows how stable the retrieved structures are.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # All unique match indices (union across all test queries)
    match_idx_all = np.unique(matches.ravel())

    # Property specifications: (data_key, label, xmin, xmax, nbins)
    props = [
        ("dft_band_gap",              "DFT Band Gap (eV)",                 0,    10.0, 40),
        ("log10_bulk_modulus",        "log₁₀ Bulk Modulus (log₁₀ GPa)",  -1,     3.5, 40),
        ("formation_energy_per_atom", "Formation Energy (eV/atom)\n"
                                      "[mp-* IDs only, ~87% coverage]",   -5,     1.0, 40),
        ("energy_above_hull",         "Energy Above Hull (eV/atom)",       0,     0.1, 40),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    summary = {}
    for ax, (pkey, xlabel, xmin, xmax, nbins) in zip(axes, props):
        full_vals = alex_data[pkey]                          # (N_alex,)

        # Background: all non-NaN values for this property
        bg_mask = ~np.isnan(full_vals)
        bg_vals = full_vals[bg_mask]

        # Foreground: matched structures with this property non-NaN
        fg_mask = bg_mask[match_idx_all]
        fg_vals = full_vals[match_idx_all][fg_mask]

        n_bg = len(bg_vals)
        n_fg = len(fg_vals)

        bins = np.linspace(xmin, xmax, nbins + 1)
        ax.hist(bg_vals, bins=bins, density=True, color="steelblue",
                alpha=0.45, label=f"Full alex ({n_bg:,})")
        if n_fg > 0:
            ax.hist(fg_vals, bins=bins, density=True, color="darkorange",
                    alpha=0.75, label=f"Matched ({n_fg:,})")
        else:
            ax.text(0.5, 0.5, "No matched\nstructures with\nthis property",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="darkorange")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=9)
        ax.set_title(pkey, fontsize=10)

        summary[pkey] = {"n_background": n_bg, "n_matched_with_prop": n_fg}
        if n_fg > 0:
            summary[pkey].update({
                "matched_mean": float(np.nanmean(fg_vals)),
                "matched_std":  float(np.nanstd(fg_vals)),
                "bg_mean":      float(np.nanmean(bg_vals)),
                "bg_std":       float(np.nanstd(bg_vals)),
            })

    fig.suptitle(
        f"Top-{matches.shape[1]} alex_mp20 matches per test query\n"
        "(cosine similarity in MEX-Multi embedding space • overlapping samples removed)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = out_dir / "alex_match_histograms_v2.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Histogram saved to {out_path}")

    return summary


def save_matched_csv(alex_data: dict, matches: np.ndarray, out_dir: Path):
    """Save a CSV of all unique matched alex structures with their properties."""
    import csv
    match_idx = np.unique(matches.ravel())
    csv_path = out_dir / "matched_alex_properties_v2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "material_id",
            "dft_band_gap_eV",
            "dft_bulk_modulus_GPa",
            "log10_bulk_modulus",
            "energy_above_hull_eV_atom",
            "formation_energy_per_atom_eV",
        ])
        for i in match_idx:
            w.writerow([
                alex_data["material_id"][i],
                _fmt(alex_data["dft_band_gap"][i]),
                _fmt(alex_data["dft_bulk_modulus"][i]),
                _fmt(alex_data["log10_bulk_modulus"][i]),
                _fmt(alex_data["energy_above_hull"][i]),
                _fmt(alex_data["formation_energy_per_atom"][i]),
            ])
    logging.info(f"Matched properties CSV saved to {csv_path}")


def _fmt(v):
    return "" if (isinstance(v, float) and np.isnan(v)) else float(v)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Deduplicated per-property alex_mp20 embedding matching. "
            "Structures present in any dataset split are excluded."
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
                   help="CSV with formation_energy_per_atom (mp-* IDs); used for join")
    p.add_argument("--out-dir",     default="outputs/alex_matching_v2")
    p.add_argument("--top-k",       type=int, default=10)
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Embedding batch size — reduce to 4 if OOM")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--no-cache",    action="store_true",
                   help="Re-embed even if cached .npz files exist")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_cache = out_dir / "test_embeddings.npz"
    alex_cache = out_dir / "alex_embeddings_dedup.npz"

    # ── load model ────────────────────────────────────────────────────────────
    engine = MEXInferenceEngine(args.checkpoint, args.config, device=args.device)

    # ── collect split material_ids for deduplication ──────────────────────────
    split_mids = collect_split_material_ids(
        args.train_lmdb, args.val_lmdb, args.test_lmdb
    )

    # ── Step 1: test embeddings ───────────────────────────────────────────────
    if not args.no_cache and test_cache.exists():
        test_data = _load_cache(test_cache)
    else:
        test_data = embed_test_set(engine, args.test_lmdb, args.batch_size)
        _save_cache(test_cache, test_data)

    # ── Step 2: alex embeddings (dedup + partial property OK) ────────────────
    if not args.no_cache and alex_cache.exists():
        alex_data = _load_cache(alex_cache)
    else:
        alex_data = load_and_embed_alex(
            engine, args.alex_train, args.alex_val,
            split_mids, args.batch_size,
            all_csv=args.all_csv,
        )
        _save_cache(alex_cache, alex_data)

    # ── Step 3: top-K cosine matching ─────────────────────────────────────────
    matches = top_k_matches(
        test_data["embeddings"],
        alex_data["embeddings"],
        top_k=args.top_k,
    )
    np.save(str(out_dir / "matches_v2.npy"), matches)
    logging.info(f"Match index saved to {out_dir / 'matches_v2.npy'}")

    # ── Step 4: per-property histograms ───────────────────────────────────────
    summary = plot_per_property_histograms(alex_data, matches, out_dir)
    save_matched_csv(alex_data, matches, out_dir)

    # ── print summary ─────────────────────────────────────────────────────────
    logging.info("\n=== Summary (per-property, separate subsets) ===")
    for pkey, s in summary.items():
        logging.info(f"\n  {pkey}")
        logging.info(f"    Background (full alex):    {s['n_background']:,}")
        logging.info(f"    Matched w/ this property:  {s['n_matched_with_prop']:,}")
        if "matched_mean" in s:
            logging.info(
                f"    Background   mean±std:  {s['bg_mean']:.3f} ± {s['bg_std']:.3f}"
            )
            logging.info(
                f"    Matched      mean±std:  {s['matched_mean']:.3f} ± {s['matched_std']:.3f}"
            )

    logging.info(
        "\nNote: formation_energy_per_atom from data/all_with_180.csv "
        "(mp-* IDs only, ~87% coverage of embedded set)."
        " alex<agm*> structures receive NaN."
    )


if __name__ == "__main__":
    main()
