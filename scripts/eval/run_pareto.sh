#!/bin/bash
# Pareto-front recovery on the held-out library, for all four variants, both
# unfiltered and restricted to physical bulk moduli.
#
# Run from the repository root after training and prediction have finished.
# Variants or seeds whose held-out predictions are missing are skipped and
# reported rather than silently dropped.
#
#   bash scripts/eval/run_pareto.sh
#   KMIN=1 KMAX=600 bash scripts/eval/run_pareto.sh     # defaults
#   SEEDS="0" bash scripts/eval/run_pareto.sh           # fewer than three seeds
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-python}
# predict.sbatch writes into a timestamped subdirectory, so this is a glob.
H='results_heldout/results/*/ocp_predictions.npz'

"$PY" scripts/eval/pareto_filtered.py --repo code \
  --arm "MEX-Single=triple:runs/mex_single_{prop}_seed{seed}/$H" \
  --arm "MEX-Multi=single:runs/mex_multi_seed{seed}/$H" \
  --arm "ERM-Single=triple:runs/erm_single_{prop}_seed{seed}/$H" \
  --arm "ERM-Multi=single:runs/erm_multi_seed{seed}/$H" \
  --seeds ${SEEDS:-0 1 2} \
  --kmin-gpa "${KMIN:-1}" --kmax-gpa "${KMAX:-600}" \
  --out-dir results/pareto_rerun
