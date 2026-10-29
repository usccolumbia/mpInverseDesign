"""
build_heldout_lmdb.py
=====================
Builds a held-out evaluation library from the structures that were in the
three-property pool but never drawn into the mp20_5k split.

WHY
---
The Pareto evaluation currently runs over the 250-structure mp20_5k test set,
whose true front is 18 structures.  One recovered structure is 5.6 points of
recall, three seeds resolve nothing finer, and recall saturates: ERM-single
already sits at 15-16 of 18, leaving two structures of headroom.  That is why
every comparison outside min-min-min came back INDISTINGUISHABLE -- the
measurement cannot resolve anything short of a huge effect.

The pool of structures carrying all three labels has 9,803 members.  5,000 were
drawn for mp20_5k (4,500 train / 250 val / 250 test).  The remaining 4,803 were
never trained on and never evaluated.  Running existing checkpoints over them
costs inference only and gives an evaluation library roughly 19x larger, whose
true front should hold on the order of 50-150 structures.  Recall stops
saturating and differences of a few percent become measurable.

These structures are held out in the strict sense: they appear in no split of
data/mp20_5k, which is the only data any of the four arms was trained on.  The
script verifies that rather than assuming it.

PROVENANCE
----------
Structures are copied from data/mp20_three_property/{train,val,test}, which is
the full 9,803-structure pool built by scripts/build_lmdb/build_mp20_three_property_lmdb.py
from data/all_with_180.csv and data/labels/mp20_bulk_modulus.csv.  Graphs are reused
as-is, so no CIF parsing happens here and the graph construction settings
(radius 12 A, max_neigh 20) are by construction identical to what every arm was
trained on.

A NOTE ON DUPLICATES
--------------------
The three mp20_three_property splits hold 10,783 entries but only 9,803 unique
material_ids, i.e. ~980 material_ids appear in more than one split.  This
script deduplicates by material_id and reports whether the duplicate copies
carry identical property values.  (That duplication is a property of the
existing pool, not something introduced here; it does not affect the mp20_5k
arms, whose 5,000 entries are 5,000 unique ids.)

OUTPUT
------
    data/mp20_heldout/test/data.lmdb
    data/mp20_heldout/test/metadata.npz

Written in the same format as every other split in this repo: keys "0".."N-1"
plus b"length", and a metadata.npz carrying natoms.  `sid` is renumbered to the
new index, because sids restart at 0 in each source split and would otherwise
collide -- predict() writes batch.sid into the npz 'ids' field, and every arm
must key off the same ids.  material_id is preserved so rows can be mapped back
to Materials Project.

USAGE
-----
    python build_heldout_lmdb.py
    python build_heldout_lmdb.py --verify-only     # check an existing build
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import lmdb
import numpy as np

POOL_ROOT = Path("data/mp20_three_property")
USED_ROOT = Path("data/mp20_5k")
OUT_ROOT = Path("data/mp20_heldout")
SPLITS = ["train", "val", "test"]
PROP_FIELDS = ["y_eform", "y_bandgap", "y_gvrh"]


def read_split(path: Path):
    """Yield (key, Data) for every structure entry in one lmdb, skipping b'length'."""
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
    try:
        with env.begin() as txn:
            for k, v in txn.cursor():
                key = k.decode()
                if key == "length":
                    continue
                yield key, pickle.loads(v)
    finally:
        env.close()


def collect_pool():
    """
    {material_id: Data} over every split of the pool, deduplicated.

    Iteration order is train, val, test and ascending integer key within each,
    so which copy of a duplicated material_id survives is deterministic rather
    than dependent on lmdb cursor order.
    """
    pool, dupes, conflicting = {}, Counter(), []
    for split in SPLITS:
        path = POOL_ROOT / split / "data.lmdb"
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. This script reads the 9,803-structure "
                     f"pool built by scripts/build_lmdb/build_mp20_three_property_lmdb.py.")
        entries = sorted(read_split(path), key=lambda kv: int(kv[0]))
        for _key, data in entries:
            mid = getattr(data, "material_id", None)
            if mid is None:
                sys.exit(f"ERROR: an entry in {path} has no material_id; cannot "
                         f"deduplicate or check overlap without it.")
            if mid in pool:
                dupes[mid] += 1
                old = [float(getattr(pool[mid], f)) for f in PROP_FIELDS]
                new = [float(getattr(data, f)) for f in PROP_FIELDS]
                if not np.allclose(old, new, rtol=0, atol=1e-9):
                    conflicting.append((mid, old, new))
                continue
            pool[mid] = data
    return pool, dupes, conflicting


def collect_used():
    """Set of material_ids appearing in ANY mp20_5k split (train, val or test)."""
    used = set()
    for split in SPLITS:
        path = USED_ROOT / split / "data.lmdb"
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Held-out membership is defined as "
                     f"'in the pool and in no mp20_5k split', so all three are needed.")
        for _key, data in read_split(path):
            mid = getattr(data, "material_id", None)
            if mid is None:
                sys.exit(f"ERROR: an entry in {path} has no material_id.")
            used.add(mid)
    return used


def write_lmdb(records, out_dir: Path, y_field=None):
    """
    records: list of (material_id, Data), already ordered.

    y_field selects what goes in `.y`:
      None            -> leave as-is (matches data/mp20_5k/*, read by the multi
                         arms, which key off y_eform / y_bandgap / y_gvrh)
      'y_eform' etc.  -> copy that property into `.y` (matches
                         data/mp20_5k_eform/*, read by the single-property arms,
                         whose head is keyed task_name: 'y')

    Row order and sids are identical across all four variants, so row i means
    the same structure in every arm's predictions and the merge-by-ids in
    load_erm_npzs has something consistent to align on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.lmdb"
    if out_path.exists():
        out_path.unlink()          # a stale file would be merged into, not replaced
    lock = out_dir / "data.lmdb-lock"
    if lock.exists():
        lock.unlink()

    # 4,803 structures at roughly the density of mp20_three_property (~200 MB for
    # 7,842) needs well under 1 GB; 4 GB leaves room without costing anything,
    # since lmdb map_size is reserved address space, not allocated bytes.
    env = lmdb.open(str(out_path), map_size=4 * 1024 ** 3, subdir=False)
    natoms_list = []
    try:
        with env.begin(write=True) as txn:
            for i, (_mid, data) in enumerate(records):
                # sid restarts at 0 in each source split, so the copies collide.
                # predict() writes batch.sid into the npz 'ids' field and every
                # arm must key off the same ids, so renumber to the new index.
                data.sid = type(data.sid)([i]) if hasattr(data.sid, "__len__") else i
                if y_field is not None:
                    data.y = float(getattr(data, y_field))
                txn.put(str(i).encode(), pickle.dumps(data))
                natoms_list.append(int(data.natoms))
            txn.put(b"length", pickle.dumps(len(records)))
    finally:
        env.close()

    np.savez(str(out_dir / "metadata.npz"), natoms=np.array(natoms_list, dtype=np.int32))
    return len(records), natoms_list


def verify(out_dir: Path, expected_ids: set):
    """Re-open the written lmdb and check it is what we intended to write."""
    path = out_dir / "data.lmdb"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")

    entries = sorted(read_split(path), key=lambda kv: int(kv[0]))
    ids_seen, sids, bad = [], [], []
    for key, data in entries:
        ids_seen.append(data.material_id)
        sid = data.sid
        sids.append(int(sid[0]) if hasattr(sid, "__len__") else int(sid))
        vals = [float(getattr(data, f, float("nan"))) for f in PROP_FIELDS]
        if not all(np.isfinite(vals)):
            bad.append((data.material_id, vals))

    used = collect_used()
    overlap = set(ids_seen) & used

    print(f"\n── verification ──")
    print(f"  entries written           : {len(entries)}")
    print(f"  unique material_ids       : {len(set(ids_seen))}")
    print(f"  overlap with any mp20_5k  : {len(overlap)}   <- must be 0")
    print(f"  sids are 0..N-1 in order  : {sids == list(range(len(entries)))}")
    print(f"  non-finite property values: {len(bad)}")
    if expected_ids is not None:
        print(f"  matches expected id set   : {set(ids_seen) == expected_ids}")

    ok = (len(overlap) == 0
          and len(set(ids_seen)) == len(entries)
          and sids == list(range(len(entries)))
          and not bad)
    if bad[:5]:
        print(f"  first bad rows: {bad[:5]}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--no-per-property", action="store_true",
                    help="skip the three single-property variants")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an existing build without rewriting it")
    args = ap.parse_args()

    out_dir = args.out_root / "test"

    if args.verify_only:
        return 0 if verify(out_dir, expected_ids=None) else 1

    print("Reading the three-property pool ...")
    pool, dupes, conflicting = collect_pool()
    print(f"  unique material_ids in pool : {len(pool)}")
    print(f"  duplicated material_ids     : {len(dupes)} "
          f"({sum(dupes.values())} extra copies discarded)")
    if conflicting:
        print(f"  WARNING: {len(conflicting)} duplicated ids carry DIFFERENT property "
              f"values between copies. First few:")
        for mid, old, new in conflicting[:5]:
            print(f"    {mid}: kept {old}  discarded {new}")
        print("  The first copy in (train, val, test) order was kept. If these ids "
              "matter to you, resolve them upstream in the pool builder.")

    print("\nReading mp20_5k (train + val + test) ...")
    used = collect_used()
    print(f"  material_ids used by mp20_5k: {len(used)}")

    stray = used - set(pool)
    if stray:
        print(f"  WARNING: {len(stray)} mp20_5k ids are not in the pool "
              f"(e.g. {sorted(stray)[:3]}). Held-out membership is still well "
              f"defined, but the two datasets are less related than assumed.")

    held = sorted(set(pool) - used)
    print(f"\nHeld-out structures: {len(held)}  "
          f"({len(pool)} pool - {len(set(pool) & used)} used)")
    if not held:
        sys.exit("ERROR: nothing held out. Check that the two roots are what you think.")

    # Deterministic order: sorted by material_id. Row i of every arm's
    # predictions then refers to the same structure regardless of when it ran.
    records = [(mid, pool[mid]) for mid in held]

    print(f"\nWriting {out_dir / 'data.lmdb'} ...")
    n, natoms = write_lmdb(records, out_dir)
    print(f"  wrote {n} entries, natoms min/median/max = "
          f"{min(natoms)}/{int(np.median(natoms))}/{max(natoms)}")

    ok = verify(out_dir, expected_ids=set(held))

    # Per-property variants for the single-property arms. Their model head is
    # keyed task_name: 'y' and their configs declare `outputs: y`, so they read
    # `.y` and cannot consume the multi-property lmdb above. Same structures,
    # same order, same sids -- only `.y` differs.
    if not args.no_per_property:
        print("\nWriting per-property variants for the single-property arms ...")
        for field in PROP_FIELDS:
            prop = field[2:]                       # y_eform -> eform
            sub = Path(f"{args.out_root}_{prop}") / "test"
            n_p, _ = write_lmdb(records, sub, y_field=field)
            # spot-check that .y really carries this property
            entries = sorted(read_split(sub / "data.lmdb"), key=lambda kv: int(kv[0]))
            mism = sum(1 for _k, d in entries
                       if not np.isclose(float(d.y), float(getattr(d, field))))
            same_order = [d.material_id for _k, d in entries] == held
            print(f"  {sub}: {n_p} entries, .y == {field} for all rows: "
                  f"{mism == 0}, order matches multi lmdb: {same_order}")
            ok = ok and mism == 0 and same_order and n_p == len(held)

    if ok:
        print(f"\nPoint a config's dataset.test.src at:  {out_dir}")
        print("Property ranges over the held-out library (raw units):")
        for f in PROP_FIELDS:
            vals = np.array([float(getattr(d, f)) for _m, d in records])
            print(f"  {f:10s} min={vals.min():9.4f}  max={vals.max():9.4f}  "
                  f"mean={vals.mean():8.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
