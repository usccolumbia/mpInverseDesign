"""
build_single_property_lmdbs.py
================================
Reads data/mp20_three_property/{train,val,test}/data.lmdb and extracts
three single-property LMDB datasets for use with the original MEX
single-property training pipeline (trainer: mex, model: equiformer_v2-mex).

Outputs:
  data/mp20_eform/{train,val,test}/    — y = y_eform  (eV/atom)
  data/mp20_bandgap/{train,val,test}/  — y = y_bandgap (eV)
  data/mp20_gvrh/{train,val,test}/     — y = y_gvrh   (log10 GPa)

The MEX single-property trainer expects each data object to carry a `y`
attribute.  The existing mp20_three_property entries carry y_eform,
y_bandgap, and y_gvrh — this script copies each object and renames the
chosen property to `y`.

Run from the repo root (on the cluster, inside mex_env):
    python scripts/build_lmdb/build_single_property_lmdbs.py

After completion the script prints YAML blocks with the normalizer statistics
(mean / stdev / train_max / train_min) for each property; paste these into the
corresponding config files.
"""

import copy
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import lmdb
except ImportError:
    print("ERROR: lmdb not installed. Run: pip install lmdb")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/build_lmdb/ -> repo root
SRC_ROOT  = REPO_ROOT / "data" / "mp20_three_property"
OUT_ROOT  = REPO_ROOT / "data"
SPLITS    = ["train", "val", "test"]

# Property name → attribute on the Data object in mp20_three_property
PROPERTIES = {
    "mp20_eform":   "y_eform",
    "mp20_bandgap": "y_bandgap",
    "mp20_gvrh":    "y_gvrh",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_lmdb(path: Path):
    """Yield all Data objects from an LMDB (integer-keyed, b'length' header)."""
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        length = pickle.loads(txn.get(b"length"))
        for i in range(length):
            obj = pickle.loads(txn.get(str(i).encode()))
            yield obj
    env.close()


def write_lmdb(path: Path, objects: list):
    """Write a list of Data objects to *path* as an integer-keyed LMDB."""
    os.makedirs(path.parent, exist_ok=True)
    map_size = 512 * 1024 * 1024  # 512 MB — large enough for mp20 splits
    env = lmdb.open(str(path), map_size=map_size, subdir=False)
    with env.begin(write=True) as txn:
        for i, obj in enumerate(objects):
            txn.put(str(i).encode(), pickle.dumps(obj))
        txn.put(b"length", pickle.dumps(len(objects)))
    env.close()


def is_valid(val) -> bool:
    """Return True if val is a real, finite number (not None / NaN / inf)."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_single_property(dataset_name: str, src_attr: str):
    """Build one single-property dataset from mp20_three_property."""
    print(f"\n{'='*60}")
    print(f"Building {dataset_name}  (src_attr={src_attr})")
    print(f"{'='*60}")

    out_dir = OUT_ROOT / dataset_name
    train_values = []  # collect train-split values for stats

    for split in SPLITS:
        src_path = SRC_ROOT / split / "data.lmdb"
        dst_path = out_dir / split / "data.lmdb"

        print(f"  [{split}] reading {src_path} ...", end="", flush=True)
        objects = []
        skipped = 0

        for obj in read_lmdb(src_path):
            raw_val = getattr(obj, src_attr, None)

            # Handle tensor attributes (  common in PyG Data objects)
            if hasattr(raw_val, "item"):
                raw_val = raw_val.item()

            if not is_valid(raw_val):
                skipped += 1
                continue

            # Deep-copy so we don't mutate the cached object
            new_obj = copy.copy(obj)
            new_obj.y = float(raw_val)
            objects.append(new_obj)

        print(f" {len(objects)} kept, {skipped} skipped (invalid/NaN)")

        if split == "train":
            train_values = [o.y for o in objects]

        # Rebuild metadata (natoms array expected by OCP loaders)
        meta_path = out_dir / split / "metadata.npz"
        os.makedirs(meta_path.parent, exist_ok=True)
        natoms = np.array([o.natoms for o in objects], dtype=np.int32)
        np.savez(str(meta_path), natoms=natoms)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"         writing {dst_path} ...", end="", flush=True)
        write_lmdb(dst_path, objects)
        print(" done.")

    # Compute and print training statistics
    arr = np.array(train_values, dtype=np.float64)
    mean   = float(arr.mean())
    stdev  = float(arr.std())
    t_max  = float(arr.max())
    t_min  = float(arr.min())

    print(f"\n  ── Training statistics for {dataset_name} ──")
    print(f"  n_train  = {len(arr)}")
    print(f"  mean     = {mean:.16g}")
    print(f"  stdev    = {stdev:.16g}")
    print(f"  train_max_y = {t_max:.16g}")
    print(f"  train_min_y = {t_min:.16g}")
    print()
    print(f"  Paste into configs/mex-single/{dataset_name}.yml → dataset.train:")
    print(f"    train_max_y: {t_max}")
    print(f"    train_min_y: {t_min}")
    print(f"    transforms:")
    print(f"      normalizer:")
    print(f"        y:")
    print(f"          mean: {mean}")
    print(f"          stdev: {stdev}")

    return dict(mean=mean, stdev=stdev, train_max=t_max, train_min=t_min, n_train=len(arr))


def main():
    stats_all = {}
    for dset_name, src_attr in PROPERTIES.items():
        stats = build_single_property(dset_name, src_attr)
        stats_all[dset_name] = stats

    print("\n" + "="*60)
    print("All done.  Summary of training statistics:")
    print("="*60)
    for name, s in stats_all.items():
        print(f"  {name}: n={s['n_train']}, mean={s['mean']:.4f}, stdev={s['stdev']:.4f}, "
              f"min={s['train_min']:.4f}, max={s['train_max']:.4f}")


if __name__ == "__main__":
    main()
