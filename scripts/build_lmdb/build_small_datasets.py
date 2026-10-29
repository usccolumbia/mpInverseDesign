"""
build_small_datasets.py
=======================
Creates small training subsets from data/mp20_three_property/ with the same
property distribution as the original MEX paper datasets (small range).

Generates:
  data/mp20_1k/{train,val,test}/         — 1000 structures, 90/5/5 split
  data/mp20_2k/{train,val,test}/         — 2000 structures, 90/5/5 split

Each split contains:
  - Multi-property LMDB: y_eform, y_bandgap, y_gvrh per structure
    → used by mex_multi_trainer

Single-property LMDBs (y attribute) are also written:
  data/mp20_1k_eform/{train,val,test}/
  data/mp20_1k_bandgap/{train,val,test}/
  data/mp20_1k_gvrh/{train,val,test}/
  data/mp20_2k_eform/{train,val,test}/
  data/mp20_2k_bandgap/{train,val,test}/
  data/mp20_2k_gvrh/{train,val,test}/

Run from repo root:
    python scripts/build_lmdb/build_small_datasets.py

After completion, copy the printed YAML blocks into the config files.
"""

import argparse
import copy
import math
import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import lmdb
except ImportError:
    print("ERROR: lmdb not installed.  Run: pip install lmdb")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent  # scripts/build_lmdb/ -> repo root
SRC_ROOT   = REPO_ROOT / "data" / "mp20_three_property"
OUT_ROOT   = REPO_ROOT / "data"
SPLITS     = ["train", "val", "test"]

# Dataset sizes to generate
SIZES = [5000]

# 90 / 5 / 5 split
TRAIN_FRAC = 0.90
VAL_FRAC   = 0.05
# TEST_FRAC  = 0.05  (remainder)

SEEDS = [42]  # Used for subsampling; training seeds are set via CLI --seed

PROPS = ["eform", "bandgap", "gvrh"]
SRC_ATTRS = {
    "eform":   "y_eform",
    "bandgap": "y_bandgap",
    "gvrh":    "y_gvrh",
}

# ─────────────────────────────────────────────────────────────────────────────
# LMDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_lmdb(path: Path) -> List:
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
    objs = []
    with env.begin() as txn:
        length = pickle.loads(txn.get(b"length"))
        for i in range(length):
            objs.append(pickle.loads(txn.get(str(i).encode())))
    env.close()
    return objs


def write_lmdb(path: Path, objects: list, map_size_mb: int = 512):
    os.makedirs(path.parent, exist_ok=True)
    env = lmdb.open(str(path), map_size=map_size_mb * 1024 * 1024, subdir=False)
    with env.begin(write=True) as txn:
        for i, obj in enumerate(objects):
            txn.put(str(i).encode(), pickle.dumps(obj))
        txn.put(b"length", pickle.dumps(len(objects)))
    env.close()


def write_metadata(path: Path, objects: list):
    natoms = np.array([o.natoms for o in objects], dtype=np.int32)
    os.makedirs(path.parent, exist_ok=True)
    np.savez(str(path / "metadata.npz"), natoms=natoms)


# ─────────────────────────────────────────────────────────────────────────────
# Load full mp20_three_property pool
# ─────────────────────────────────────────────────────────────────────────────

def load_full_pool() -> List:
    """Return all objects from all three splits of mp20_three_property."""
    pool = []
    for split in SPLITS:
        src = SRC_ROOT / split / "data.lmdb"
        print(f"  Reading {src} ...", end="", flush=True)
        objs = read_lmdb(src)
        pool.extend(objs)
        print(f" {len(objs)} objects")
    return pool


def is_valid(obj) -> bool:
    """Return True if all three property values are finite."""
    for attr in SRC_ATTRS.values():
        val = getattr(obj, attr, None)
        if val is None:
            return False
        try:
            if not math.isfinite(float(val)):
                return False
        except (TypeError, ValueError):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Build one size variant
# ─────────────────────────────────────────────────────────────────────────────

def build_variant(pool: List, n_total: int, seed: int = 42):
    """Sample n_total structures from pool and write multi + single LMDBs."""
    rng   = np.random.default_rng(seed)
    idxs  = rng.choice(len(pool), size=n_total, replace=False)

    n_train = int(TRAIN_FRAC * n_total)
    n_val   = int(VAL_FRAC   * n_total)
    n_test  = n_total - n_train - n_val

    split_idxs = {
        "train": idxs[:n_train],
        "val":   idxs[n_train : n_train + n_val],
        "test":  idxs[n_train + n_val :],
    }
    print(f"\n  [{n_total} structures]  "
          f"train={n_train}, val={n_val}, test={n_test}")

    size_tag = f"mp20_{n_total // 1000}k"

    # ── multi-property LMDBs ─────────────────────────────────────────────────
    multi_dir = OUT_ROOT / size_tag
    train_multi_vals = {p: [] for p in PROPS}

    for split, idx_arr in split_idxs.items():
        objs = []
        for raw_i in idx_arr:
            obj = copy.copy(pool[raw_i])
            objs.append(obj)
            if split == "train":
                for p, attr in SRC_ATTRS.items():
                    train_multi_vals[p].append(float(getattr(obj, attr)))

        out_path = multi_dir / split / "data.lmdb"
        write_lmdb(out_path, objs)
        write_metadata(multi_dir / split, objs)
        print(f"    multi/{split}: {len(objs)} → {out_path}")

    # ── single-property LMDBs ────────────────────────────────────────────────
    single_stats = {}
    for prop, src_attr in SRC_ATTRS.items():
        prop_dir = OUT_ROOT / f"{size_tag}_{prop}"
        train_vals = []

        for split, idx_arr in split_idxs.items():
            objs = []
            for raw_i in idx_arr:
                src_obj = pool[raw_i]
                val = float(getattr(src_obj, src_attr))
                obj = copy.copy(src_obj)
                obj.y = val
                objs.append(obj)
            if split == "train":
                train_vals = [o.y for o in objs]

            out_path = prop_dir / split / "data.lmdb"
            write_lmdb(out_path, objs)
            write_metadata(prop_dir / split, objs)
            print(f"    {prop}/{split}: {len(objs)} → {out_path}")

        arr = np.array(train_vals)
        single_stats[prop] = {
            "mean":      float(arr.mean()),
            "stdev":     float(arr.std()),
            "train_max": float(arr.max()),
            "train_min": float(arr.min()),
            "n":         len(arr),
        }

    # ── multi-property training stats ────────────────────────────────────────
    multi_stats = {}
    for prop in PROPS:
        arr = np.array(train_multi_vals[prop])
        multi_stats[prop] = {
            "mean":      float(arr.mean()),
            "stdev":     float(arr.std()),
            "train_max": float(arr.max()),
            "train_min": float(arr.min()),
        }

    return size_tag, single_stats, multi_stats


# ─────────────────────────────────────────────────────────────────────────────
# EBM bound configuration per property (covers expected test range with margin)
# ─────────────────────────────────────────────────────────────────────────────

PROP_BOUNDS = {
    "eform":   {"lower": -6,  "upper": 2},   # eV
    "bandgap": {"lower": -1,  "upper": 20},  # eV
    "gvrh":    {"lower": -2,  "upper": 12},  # log10(GPa)
}

# ─────────────────────────────────────────────────────────────────────────────
# Write YAML config files
# ─────────────────────────────────────────────────────────────────────────────

def write_single_config(repo_root: Path, size_tag: str, prop: str, stats: dict):
    # Guarded by --write-configs. The shipped configs differ from what this
    # generates (nce rather than info-nce, plus patience and infer_type
    # overrides), so regenerating them silently breaks the published setup.
    s = stats[prop]
    b = PROP_BOUNDS[prop]
    dname = f"{size_tag}_{prop}"
    out_path = repo_root / "configs" / "mex-single" / f"{dname}.yml"
    content = f"""\
includes:
- configs/mex-single/equiformer_v2/base_ebm.yml
- configs/mex-single/params/base_param.yml

loss_functions:
  - y:
      fn: info-nce
      coefficient: 1

dataset:
  train:
    format: lmdb
    src: data/{dname}/train
    weighted_sample: false
    train_max_y: {s['train_max']}
    train_min_y: {s['train_min']}
    lower_bound: {b['lower']}
    upper_bound: {b['upper']}
    infer_type: energy
    transforms:
      normalizer:
        y:
          mean: {s['mean']}
          stdev: {s['stdev']}

  val:
    src: data/{dname}/val

  test:
    src: data/{dname}/test

optim:
  patience: 10

logger:
  name: wandb
  project: mex_inverse_design
  entity: dipannoydip
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(f"    config → {out_path.relative_to(repo_root)}")


def write_multi_config(repo_root: Path, size_tag: str, multi_stats: dict):
    # Guarded by --write-configs; see write_single_config.
    out_path = repo_root / "configs" / "mex-multi" / f"{size_tag}.yml"
    content = f"""\
includes:
- configs/mex-multi/equiformer_v2/base_ebm_mp20_three_property.yml
- configs/mex-multi/params/base_param_multi.yml

dataset:
  property_names: [eform, bandgap, gvrh]
  train_max_y_eform:   {multi_stats['eform']['train_max']}
  train_min_y_eform:   {multi_stats['eform']['train_min']}
  train_max_y_bandgap: {multi_stats['bandgap']['train_max']}
  train_min_y_bandgap: {multi_stats['bandgap']['train_min']}
  train_max_y_gvrh:    {multi_stats['gvrh']['train_max']}
  train_min_y_gvrh:    {multi_stats['gvrh']['train_min']}

  train:
    format: lmdb
    src: data/{size_tag}/train
    weighted_sample: false
    transforms:
      normalizer:
        y_eform:
          mean:  {multi_stats['eform']['mean']}
          stdev: {multi_stats['eform']['stdev']}
        y_bandgap:
          mean:  {multi_stats['bandgap']['mean']}
          stdev: {multi_stats['bandgap']['stdev']}
        y_gvrh:
          mean:  {multi_stats['gvrh']['mean']}
          stdev: {multi_stats['gvrh']['stdev']}

  val:
    src: data/{size_tag}/val
  test:
    src: data/{size_tag}/test

logger:
  name: wandb
  project: mex_inverse_design
  entity: dipannoydip
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(f"    config → {out_path.relative_to(repo_root)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help=f"subsample sizes to build (default {SIZES}); the paper uses 5000")
    ap.add_argument("--write-configs", action="store_true",
                    help="regenerate configs/ from the build. OFF by default: the "
                         "repository's configs are curated (nce rather than info-nce, "
                         "plus patience and infer_type overrides) and would be overwritten.")
    args = ap.parse_args()
    sizes = args.sizes if args.sizes else SIZES

    print("Loading full mp20_three_property pool ...")
    raw_pool = load_full_pool()
    pool = [o for o in raw_pool if is_valid(o)]
    print(f"Total valid structures: {len(pool)}")

    all_stats = {}
    for n_total in sizes:
        if n_total > len(pool):
            print(f"WARNING: requested {n_total} but pool has only {len(pool)}. Skipping.")
            continue
        size_tag, single_stats, multi_stats = build_variant(pool, n_total, seed=42)
        all_stats[size_tag] = (single_stats, multi_stats)

    # ── Write config YAML files (opt-in) ──────────────────────────────────
    if args.write_configs:
        print("\n" + "="*70)
        print("Writing config files ...")
        print("="*70)
        for size_tag, (single_stats, multi_stats) in all_stats.items():
            print(f"\n  {size_tag}:")
            for prop in PROPS:
                write_single_config(REPO_ROOT, size_tag, prop, single_stats)
            write_multi_config(REPO_ROOT, size_tag, multi_stats)
    else:
        print("\nconfigs/ left untouched (pass --write-configs to regenerate).")

    print("\nDone.")


if __name__ == "__main__":
    main()
