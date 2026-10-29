"""
compute_multi_res_metric.py
---------------------------
Compute per-property MAE, Spearman r, and error geometric mean from
multi-property MEX test-set predictions saved as ocp_predictions.npz.

Usage (after running predict_mex_multi.sh):
    python compute_multi_res_metric.py \
        --pred-file <run_dir>/results/<timestamp>/<identifier>_predictions.npz

Or scan a directory for all prediction files:
    python compute_multi_res_metric.py --scan-dir record_multi/

Output: prints a results table and writes results.csv to the scan-dir / pred-file dir.
"""
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, gmean


PROPERTIES = ["eform", "bandgap", "gvrh"]  # must match training config
  # must match training config


def metrics_for_property(preds: np.ndarray, gts: np.ndarray) -> dict:
    """MAE, Spearman r, error geometric mean for one property."""
    errors = np.abs(preds - gts)
    mae = float(np.mean(errors))
    spear, _ = spearmanr(preds, gts)
    errors_nz = np.where(errors == 0, 1e-10, errors)
    egm = float(gmean(errors_nz))
    return {"mae": mae, "spearman": float(spear), "egm": egm}


def process_file(npz_path: Path) -> dict:
    """Load one predictions file and return per-property metrics."""
    data = np.load(npz_path, allow_pickle=True)
    results = {}
    for prop in PROPERTIES:
        key_pred = f"y_{prop}"
        key_gt   = f"y_{prop}_gts"
        if key_pred not in data or key_gt not in data:
            print(f"  [skip] {prop} not found in {npz_path.name}")
            continue
        preds = np.array(data[key_pred]).reshape(-1)
        gts   = np.array(data[key_gt]).reshape(-1)
        results[prop] = metrics_for_property(preds, gts)
    return results


def print_table(results: dict, label: str = "") -> None:
    header = f"{'Property':<12}  {'MAE':>10}  {'Spearman':>10}  {'EGM':>10}"
    print(f"\n{'='*50}")
    if label:
        print(f"  {label}")
    print(header)
    print("-" * 50)
    maes = []
    for prop, m in results.items():
        print(f"  {prop:<10}  {m['mae']:>10.4f}  {m['spearman']:>10.4f}  {m['egm']:>10.4f}")
        maes.append(m['mae'])
    if maes:
        print("-" * 50)
        print(f"  {'avg_MAE':<10}  {np.mean(maes):>10.4f}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Compute multi-property test metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred-file", type=Path,
                       help="Path to a single *_predictions.npz file")
    group.add_argument("--scan-dir", type=Path,
                       help="Recursively scan this directory for *_predictions.npz files")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Where to write CSV (default: same dir as pred file / scan-dir)")
    args = parser.parse_args()

    # Collect files to process
    if args.pred_file:
        pred_files = [args.pred_file]
        out_dir = args.pred_file.parent
    else:
        pred_files = sorted(args.scan_dir.rglob("*_predictions.npz"))
        out_dir = args.scan_dir
        if not pred_files:
            print(f"No *_predictions.npz files found under {args.scan_dir}")
            return

    out_csv = args.out_csv or (out_dir / "multi_results.csv")

    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "property", "MAE", "Spearman", "EGM"])

        for pf in pred_files:
            print(f"\nProcessing: {pf}")
            res = process_file(pf)
            if not res:
                continue
            print_table(res, label=str(pf))
            for prop, m in res.items():
                writer.writerow([str(pf), prop,
                                 round(m["mae"], 6),
                                 round(m["spearman"], 6),
                                 round(m["egm"], 6)])

    print(f"Results written to: {out_csv}")


if __name__ == "__main__":
    main()
