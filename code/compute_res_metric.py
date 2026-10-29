"""
compute_res_metric.py
---------------------
Compute MAE, Spearman r, and error geometric mean from single-property MEX
test-set predictions saved as a *_predictions.npz file.

The npz file is expected to contain:
  - 'y'     : model predictions  (N,)
  - 'y_gts' : ground-truth values (N,)

Usage (single file):
    python compute_res_metric.py --pred-file <path/to/*_predictions.npz>

Scan a directory for all prediction files:
    python compute_res_metric.py --scan-dir <results_dir/>

Output: prints a results table and writes results.csv to the same directory
as the prediction file (or scan-dir).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, gmean


def metrics_for_property(preds: np.ndarray, gts: np.ndarray) -> dict:
    """MAE, Spearman r, error geometric mean for one property."""
    errors = np.abs(preds - gts)
    mae = float(np.mean(errors))
    spear, _ = spearmanr(preds, gts)
    errors_nz = np.where(errors == 0, 1e-10, errors)
    egm = float(gmean(errors_nz))
    return {"mae": mae, "spearman": float(spear), "egm": egm}


def process_file(npz_path: Path) -> dict:
    """Load one predictions file and return metrics."""
    data = np.load(npz_path, allow_pickle=True)
    if "y" not in data or "y_gts" not in data:
        print(f"  [skip] 'y' or 'y_gts' not found in {npz_path.name}")
        return {}
    preds = np.array(data["y"]).reshape(-1)
    gts = np.array(data["y_gts"]).reshape(-1)
    return metrics_for_property(preds, gts)


def print_table(m: dict, label: str = "") -> None:
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(f"  {'MAE':>10}  {'Spearman':>10}  {'EGM':>10}")
    print("-" * 50)
    print(f"  {m['mae']:>10.4f}  {m['spearman']:>10.4f}  {m['egm']:>10.4f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute single-property test metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred-file", type=Path,
                       help="Path to a single *_predictions.npz file")
    group.add_argument("--scan-dir", type=Path,
                       help="Recursively scan this directory for *_predictions.npz files")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Where to write CSV (default: same dir as pred file / scan-dir)")
    args = parser.parse_args()

    if args.pred_file:
        pred_files = [args.pred_file]
        out_dir = args.pred_file.parent
    else:
        pred_files = sorted(args.scan_dir.rglob("*_predictions.npz"))
        out_dir = args.scan_dir
        if not pred_files:
            print(f"No *_predictions.npz files found under {args.scan_dir}")
            return

    out_csv = args.out_csv or (out_dir / "results.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "MAE", "Spearman", "EGM"])

        for pf in pred_files:
            print(f"\nProcessing: {pf}")
            m = process_file(pf)
            if not m:
                continue
            print_table(m, label=str(pf))
            writer.writerow([str(pf),
                              round(m["mae"], 6),
                              round(m["spearman"], 6),
                              round(m["egm"], 6)])

    print(f"Results written to: {out_csv}")


if __name__ == "__main__":
    main()