"""
download_bulk_modulus.py
========================
Downloads bulk modulus (K_VRH, log10 scale) from the Materials Project API
for all material IDs in data/all_with_180.csv, then saves the result to
data/labels/mp20_bulk_modulus.csv.

Run from the repo root:
    MP_API_KEY=your_key_here python scripts/build_lmdb/download_bulk_modulus.py

Requirements:
    pip install mp-api pymatgen

What is downloaded
------------------
The MP elasticity endpoint returns K_VRH (Voigt-Reuss-Hill bulk modulus) in GPa
for ~13,000 materials that have DFT-computed elastic constants.  We convert to
log10 scale (matching the gvrh convention in the original MEX paper and the
mp_gvrh datasets already in this repo).

Output columns:
    material_id, k_vrh_gpa, k_vrh_log10
"""

import os
import csv
import sys
import math
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/build_lmdb/ -> repo root
os.chdir(REPO_ROOT)

CSV_IN  = Path("data/all_with_180.csv")
CSV_OUT = Path("data/labels/mp20_bulk_modulus.csv")

# ── Check API key ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("MP_API_KEY", "")
if not API_KEY:
    print("ERROR: Set your Materials Project API key:\n"
          "    export MP_API_KEY=your_key_here\n"
          "Get a free key at https://materialsproject.org/api")
    sys.exit(1)

# ── Check dependencies ────────────────────────────────────────────────────────
try:
    from mp_api.client import MPRester
except ImportError:
    print("ERROR: mp-api not installed.  Run:\n    pip install mp-api pymatgen")
    sys.exit(1)


def read_material_ids(csv_path: Path) -> list:
    ids = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            mid = row.get("material_id", "").strip()
            if mid:
                ids.append(mid)
    # deduplicate while preserving order
    seen = set()
    unique = []
    for m in ids:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def main():
    logging.info(f"Reading material IDs from {CSV_IN} ...")
    all_ids = set(read_material_ids(CSV_IN))
    logging.info(f"  {len(all_ids)} unique material IDs")

    # ── Download ALL MP elasticity data, filter locally ───────────────────────
    # The elasticity endpoint in this mp-api version has no material_ids filter.
    # There are ~13,500 materials in MP with elasticity data — downloading all
    # is fast (~14 chunks of 1000) and we filter to our target IDs locally.
    # We bypass the broken pydantic document model (task_id vs material_id
    # mismatch) by setting use_document_model=False to get raw dicts.
    results = {}  # material_id -> k_vrh_gpa

    logging.info("Downloading all MP elasticity data (bypassing document model) ...")
    with MPRester(API_KEY) as mpr:
        rester = mpr.materials.elasticity
        rester.use_document_model = False
        try:
            docs = rester.search(chunk_size=1000)
            logging.info(f"  Retrieved {len(docs)} elasticity documents from MP")
        except Exception as e:
            logging.error(f"  Failed to download elasticity data: {e}")
            sys.exit(1)

    logging.info("Filtering and extracting k_vrh ...")
    for doc in docs:
        mid = doc.get("material_id")
        if not mid or mid not in all_ids:
            continue
        bm = doc.get("bulk_modulus")
        if not isinstance(bm, dict):
            continue
        k_vrh = bm.get("vrh")
        if k_vrh is not None and float(k_vrh) > 0:
            results[str(mid)] = float(k_vrh)

    found = len(results)
    logging.info(f"Bulk modulus found for {found} / {len(all_ids)} materials "
                 f"({100*found/len(all_ids):.1f}%)"
                 if all_ids else
                 f"Bulk modulus found for {found} materials")

    # ── Write output ───────────────────────────────────────────────────────────
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["material_id", "k_vrh_gpa", "k_vrh_log10"])
        for mid, k_gpa in sorted(results.items()):
            k_log10 = math.log10(k_gpa) if k_gpa > 0 else float("nan")
            w.writerow([mid, round(k_gpa, 4), round(k_log10, 8)])

    logging.info(f"Saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
