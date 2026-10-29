"""
embed_and_match_alex.py
========================
Retrieval-based inverse design via structural embedding similarity.

PIPELINE
--------
1. Load the trained MEX-Multi checkpoint and embed every structure in the
   mp20_three_property TEST set  →  X_test  (N_test × D)

2. Load alex_mp20_{train,val}.csv, keep rows that have all three properties
   (dft_band_gap, dft_bulk_modulus, energy_above_hull), parse each CIF →
   PyG graph → embed with the same backbone  →  X_alex  (N_alex × D)

3. For every test embedding find the top-K most cosine-similar alex
   embeddings  →  (N_test × K) match index

4. Collect the properties of all matched alex structures and plot histograms
   comparing:
     - full alex distribution (background, all valid rows)
     - matched alex distribution (foreground, top-K retrieved per test query)

   Properties shown: dft_band_gap (eV), dft_bulk_modulus (GPa),
                     energy_above_hull (eV/atom)

USAGE
-----
# Full run (embeds test + alex from scratch, saves intermediate results)
python inverse_design/embed_and_match_alex.py \\
    --checkpoint outputs/mp20_three_property_higher_patience_gvrh_infonce/checkpoints/.../best_checkpoint.pt \\
    --config     configs/mex-multi/mp20_three_property.yml \\
    --test-lmdb  data/mp20_three_property/test/data.lmdb \\
    --alex-train data/alex_mp20_train.csv \\
    --alex-val   data/alex_mp20_val.csv \\
    --out-dir    outputs/alex_matching \\
    --top-k      10 \\
    --device     cuda

# Skip re-embedding if cached .npz files already exist
python inverse_design/embed_and_match_alex.py \\
    --checkpoint ... \\
    --config     ... \\
    --out-dir    outputs/alex_matching \\   # reads cached .npz from here
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

# ─────────────────────────────────────────────────────────────────────────────
# Reuse MEXInferenceEngine from the sibling file
# ─────────────────────────────────────────────────────────────────────────────
_inv_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_inv_dir))
from inverse_design_mex import MEXInferenceEngine  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CIF → PyG Data (identical to files/build_mp20_three_property_lmdb.py)
# ─────────────────────────────────────────────────────────────────────────────
RADIUS    = 12.0
MAX_NEIGH = 20


def cif_to_graph(cif_str: str, sid: int):
    """Parse a CIF string to a torch_geometric Data object."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from torch_geometric.data import Data

    struct = Structure.from_str(cif_str, fmt="cif")
    atoms  = AseAtomsAdaptor.get_atoms(struct)

    _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
        r=RADIUS, numerical_tol=0, exclude_self=True
    )

    # Keep at most MAX_NEIGH nearest neighbours per centre atom
    _nonmax_idx = []
    for i in range(len(atoms)):
        idx_i = np.where(_c_index == i)[0]
        idx_sorted = np.argsort(n_distance[idx_i])[:MAX_NEIGH]
        _nonmax_idx.append(idx_i[idx_sorted])
    _nonmax_idx = np.concatenate(_nonmax_idx) if _nonmax_idx else np.array([], dtype=int)

    _c_index   = _c_index[_nonmax_idx]
    _n_index   = _n_index[_nonmax_idx]
    n_distance = n_distance[_nonmax_idx]
    _offsets   = _offsets[_nonmax_idx]

    edge_index   = torch.LongTensor(np.vstack((_n_index, _c_index)))
    cell_offsets = torch.LongTensor(_offsets)
    edge_dist    = torch.FloatTensor(n_distance)

    nonzero = torch.where(edge_dist >= 1e-8)[0]
    edge_index   = edge_index[:, nonzero]
    cell_offsets = cell_offsets[nonzero]

    natoms = len(atoms)
    return Data(
        cell=torch.tensor(np.array(atoms.get_cell()), dtype=torch.float).view(1, 3, 3),
        pos=torch.tensor(atoms.get_positions(), dtype=torch.float),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        natoms=natoms,
        tags=torch.zeros(natoms, dtype=torch.long),
        fixed=torch.zeros(natoms, dtype=torch.int),
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        sid=torch.LongTensor([sid]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LMDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_lmdb(lmdb_path: str):
    import lmdb as _lmdb
    env = _lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    records = []
    with env.begin() as txn:
        n = pickle.loads(txn.get(b"length"))
        for i in range(n):
            records.append(pickle.loads(txn.get(str(i).encode())))
    env.close()
    return records


def collate(records):
    import torch_geometric
    return torch_geometric.data.Batch.from_data_list(records)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_records(engine: MEXInferenceEngine, records: list, batch_size: int = 8) -> np.ndarray:
    """Embed a list of PyG Data records in batches. Returns (N, D) float32."""
    all_emb = []
    N = len(records)
    for start in range(0, N, batch_size):
        chunk = records[start: start + batch_size]
        batch = collate(chunk).to(engine.device)
        emb = engine.embed_batch(batch).cpu().float().numpy()
        all_emb.append(emb)
        if (start // batch_size) % 20 == 0:
            logging.info(f"  embedded {min(start + batch_size, N)}/{N}")
    return np.concatenate(all_emb, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – embed TEST set
# ─────────────────────────────────────────────────────────────────────────────

def embed_test_set(engine, lmdb_path: str, batch_size: int) -> dict:
    """
    Returns:
        {
          "embeddings": (N, D) float32,
          "y_eform":    (N,)  raw eV/atom,
          "y_bandgap":  (N,)  raw eV,
          "y_gvrh":     (N,)  raw log10 GPa,
          "material_id": list[str],
        }
    """
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
# Step 2 – preprocess alex_mp20 CSV files and embed
# ─────────────────────────────────────────────────────────────────────────────

def load_and_embed_alex(
    engine,
    train_csv: str,
    val_csv: str,
    batch_size: int,
) -> dict:
    """
    Load both alex_mp20 CSVs, keep rows with all three properties non-null,
    parse CIFs, embed, and return a result dict.

    Returns:
        {
          "embeddings":         (N, D) float32,
          "material_id":        list[str],
          "dft_band_gap":       (N,) eV,
          "dft_bulk_modulus":   (N,) GPa (raw),
          "log10_bulk_modulus": (N,) log10 GPa  (for comparison with y_gvrh),
          "energy_above_hull":  (N,) eV/atom,
        }
    """
    import pandas as pd

    logging.info(f"Loading alex_mp20 CSVs …")
    df_train = pd.read_csv(train_csv)
    df_val   = pd.read_csv(val_csv)
    df = pd.concat([df_train, df_val], ignore_index=True)
    logging.info(f"  total rows: {len(df)}")

    # Keep only rows with all three properties
    required = ["dft_band_gap", "dft_bulk_modulus", "energy_above_hull"]
    df = df.dropna(subset=required).reset_index(drop=True)
    logging.info(f"  rows with all 3 properties: {len(df)}")

    # Parse CIF → PyG graphs
    logging.info("Parsing CIF strings to graphs …")
    records = []
    failed  = []
    for idx, row in df.iterrows():
        try:
            g = cif_to_graph(row["cif"], sid=idx)
            records.append(g)
        except Exception as e:
            logging.warning(f"  CIF parse failed for {row['material_id']}: {e}")
            failed.append(idx)

    if failed:
        df = df.drop(index=failed).reset_index(drop=True)
    logging.info(f"  parsed {len(records)} graphs ({len(failed)} failed)")

    # Embed in batches
    logging.info("Embedding alex structures …")
    embs = embed_records(engine, records, batch_size)

    log10_bm = np.log10(
        np.clip(df["dft_bulk_modulus"].values.astype(np.float32), 1e-6, None)
    )

    return {
        "embeddings":         embs,
        "material_id":        df["material_id"].tolist(),
        "dft_band_gap":       df["dft_band_gap"].values.astype(np.float32),
        "dft_bulk_modulus":   df["dft_bulk_modulus"].values.astype(np.float32),
        "log10_bulk_modulus": log10_bm,
        "energy_above_hull":  df["energy_above_hull"].values.astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – cosine-similarity top-K matching
# ─────────────────────────────────────────────────────────────────────────────

def top_k_matches(
    test_embs: np.ndarray,   # (N_test, D)
    alex_embs: np.ndarray,   # (N_alex, D)
    top_k: int,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    Compute cosine similarity between every test embedding and every alex
    embedding, return the top-K alex indices per test query.

    Returns: (N_test, top_k) int array of indices into alex_embs.
    """
    # L2-normalise
    te = torch.tensor(test_embs, dtype=torch.float32)
    ae = torch.tensor(alex_embs, dtype=torch.float32)
    te = Func.normalize(te, dim=-1)
    ae = Func.normalize(ae, dim=-1)

    N_test = te.shape[0]
    matches = np.empty((N_test, top_k), dtype=np.int32)

    logging.info(f"Computing top-{top_k} matches for {N_test} test embeddings …")
    for start in range(0, N_test, chunk_size):
        end = min(start + chunk_size, N_test)
        sim = te[start:end] @ ae.T   # (chunk, N_alex)
        topk_ids = sim.topk(top_k, dim=-1).indices.cpu().numpy()
        matches[start:end] = topk_ids

        if (start // chunk_size) % 10 == 0:
            logging.info(f"  matched {end}/{N_test}")

    return matches


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – histogram plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(
    alex_data: dict,
    matches: np.ndarray,   # (N_test, K)
    out_dir: Path,
):
    """
    For each property, plot:
      - grey background: full alex distribution
      - orange foreground: properties of the matched structures (all top-K hits
        flattened across all test queries, with repetitions counted once)

    Saves one PNG per property plus a combined figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match_idx_unique = np.unique(matches.ravel())
    n_matched = len(match_idx_unique)
    n_total   = len(alex_data["material_id"])
    logging.info(f"Unique matched alex structures: {n_matched} / {n_total}")

    properties = {
        "dft_band_gap":      ("DFT Band Gap (eV)",       0,   10,  40),
        "dft_bulk_modulus":  ("DFT Bulk Modulus (GPa)",  0,  800,  40),
        "log10_bulk_modulus":("log₁₀ Bulk Modulus (GPa)", -1, 3.5, 40),
        "energy_above_hull": ("Energy Above Hull (eV/atom)", 0, 0.1, 40),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, (pkey, (xlabel, xmin, xmax, nbins)) in zip(axes, properties.items()):
        full_vals    = alex_data[pkey]
        matched_vals = full_vals[match_idx_unique]

        bins = np.linspace(xmin, xmax, nbins + 1)
        ax.hist(full_vals,    bins=bins, density=True, color="steelblue",
                alpha=0.5, label=f"Full alex ({n_total:,})")
        ax.hist(matched_vals, bins=bins, density=True, color="darkorange",
                alpha=0.7, label=f"Matched ({n_matched:,})")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=9)
        ax.set_title(xlabel, fontsize=11)

    fig.suptitle(
        f"Top-{matches.shape[1]} alex_mp20 matches per test query\n"
        f"(cos-sim in MEX-Multi embedding space)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = out_dir / "alex_match_histograms.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Histogram saved to {out_path}")

    # Also save the per-match property table as CSV
    import csv
    csv_path = out_dir / "matched_alex_properties.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["material_id", "dft_band_gap_eV",
                    "dft_bulk_modulus_GPa", "log10_bulk_modulus", "energy_above_hull_eV"])
        for i in match_idx_unique:
            w.writerow([
                alex_data["material_id"][i],
                float(alex_data["dft_band_gap"][i]),
                float(alex_data["dft_bulk_modulus"][i]),
                float(alex_data["log10_bulk_modulus"][i]),
                float(alex_data["energy_above_hull"][i]),
            ])
    logging.info(f"Matched properties CSV saved to {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_cache(path: Path, data: dict):
    np.savez_compressed(str(path), **{
        k: (np.array(v, dtype=object) if isinstance(v, list) else v)
        for k, v in data.items()
    })
    logging.info(f"Cached embeddings to {path}")


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
    logging.info(f"Loaded cached embeddings from {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Embed test + alex_mp20 structures and run top-K matching"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_checkpoint.pt")
    p.add_argument("--config",     required=True,
                   help="Path to configs/mex-multi/mp20_three_property.yml")
    p.add_argument("--test-lmdb",  default="data/mp20_three_property/test/data.lmdb",
                   help="Path to test set LMDB")
    p.add_argument("--alex-train", default="data/alex_mp20_train.csv",
                   help="Path to alex_mp20_train.csv")
    p.add_argument("--alex-val",   default="data/alex_mp20_val.csv",
                   help="Path to alex_mp20_val.csv")
    p.add_argument("--out-dir",    default="outputs/alex_matching",
                   help="Directory for cached embeddings and output figures")
    p.add_argument("--top-k",      type=int, default=10,
                   help="Number of alex matches per test embedding")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Embedding batch size (reduce if OOM)")
    p.add_argument("--device",     default="cpu",
                   help="'cpu' or 'cuda' or 'cuda:0'")
    p.add_argument("--no-cache",   action="store_true",
                   help="Re-embed even if .npz cache files exist")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_cache  = out_dir / "test_embeddings.npz"
    alex_cache  = out_dir / "alex_embeddings.npz"

    # ── load model ────────────────────────────────────────────────────────────
    engine = MEXInferenceEngine(args.checkpoint, args.config, device=args.device)

    # ── Step 1: test embeddings ───────────────────────────────────────────────
    if not args.no_cache and test_cache.exists():
        test_data = _load_cache(test_cache)
    else:
        test_data = embed_test_set(engine, args.test_lmdb, args.batch_size)
        _save_cache(test_cache, test_data)

    # ── Step 2: alex embeddings ───────────────────────────────────────────────
    if not args.no_cache and alex_cache.exists():
        alex_data = _load_cache(alex_cache)
    else:
        alex_data = load_and_embed_alex(
            engine, args.alex_train, args.alex_val, args.batch_size
        )
        _save_cache(alex_cache, alex_data)

    # ── Step 3: matching ──────────────────────────────────────────────────────
    matches = top_k_matches(
        test_data["embeddings"],
        alex_data["embeddings"],
        top_k=args.top_k,
    )
    np.save(str(out_dir / "matches.npy"), matches)
    logging.info(f"Match index saved to {out_dir / 'matches.npy'}")

    # ── Step 4: histograms ────────────────────────────────────────────────────
    plot_histograms(alex_data, matches, out_dir)

    # ── Summary stats ─────────────────────────────────────────────────────────
    match_idx = np.unique(matches.ravel())
    logging.info("\n=== Matched alex structures — property summary ===")
    for pkey, label in [
        ("dft_band_gap",       "Band gap (eV)      "),
        ("dft_bulk_modulus",   "Bulk modulus (GPa) "),
        ("log10_bulk_modulus", "log10 BM           "),
        ("energy_above_hull",  "E above hull (eV)  "),
    ]:
        vals = alex_data[pkey][match_idx]
        logging.info(
            f"  {label}  mean={vals.mean():.3f}  std={vals.std():.3f}"
            f"  min={vals.min():.3f}  max={vals.max():.3f}"
        )


if __name__ == "__main__":
    main()
