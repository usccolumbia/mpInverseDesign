"""
Multi-property plain-regression (ERM) trainer.

This is the ERM control for MEXMultiTrainer.  OCPTrainer is already generic
over `self.output_targets`: _forward, _compute_loss, _compute_metrics and
predict all loop over the declared output keys, so training a model with three
regression heads needs no new training logic.  The one thing missing is the
checkpoint-selection criterion.

MEXMultiTrainer selects on `avg_mae`, a single scalar pooled over the
per-property MAEs (mex_multi_trainer.py:660-668).  OCPTrainer does not compute
it, so `primary_metric: avg_mae` would raise a KeyError there.  This subclass
adds exactly that key, computed the same way -- sum of the per-property error
totals over the sum of their counts -- so that:

  * best-checkpoint selection,
  * early stopping, and
  * the ReduceLROnPlateau schedule

all key off the same quantity in the multi-property ERM arm as in the
multi-property MEX arm.  Without this the two arms would be stopped and
selected by different rules and the comparison would not be controlled.

Note on units: _compute_metrics denormalises predictions before evaluating, so
the pooled quantity mixes raw units across properties (eV/atom, eV, log10 GPa)
and is dominated by whichever property has the largest absolute error.  That is
a real limitation of the criterion, but it is MEXMultiTrainer's criterion, and
matching it is what keeps the comparison honest.

Nothing else is overridden.
"""

import logging

from ocpmodels.common.registry import registry
from ocpmodels.trainers.ocp_trainer import OCPTrainer


@registry.register_trainer("ocp_multi")
class OCPMultiTrainer(OCPTrainer):
    """OCPTrainer plus the pooled `avg_mae` validation metric."""

    def validate(self, split: str = "val", disable_tqdm: bool = False):
        metrics = super().validate(split=split, disable_tqdm=disable_tqdm)

        # Pool the per-property MAEs exactly as MEXMultiTrainer does.  Keys come
        # from the config's `outputs:` block, so this stays correct if the set
        # of properties changes.
        mae_keys = [
            f"{target}_mae"
            for target in self.config["outputs"]
            if f"{target}_mae" in metrics
        ]

        if len(mae_keys) < 2:
            # Single-target run, or MAE not among the evaluation metrics.
            # Nothing to pool; leave the metrics dict untouched so that a
            # `primary_metric` naming one property still works.
            return metrics

        total = sum(metrics[k]["total"] for k in mae_keys)
        numel = sum(metrics[k]["numel"] for k in mae_keys)
        metrics["avg_mae"] = {
            "total": total,
            "numel": numel,
            "metric": total / numel,
        }

        # super().validate() already logged, so avg_mae needs its own line to
        # appear in the run log and in W&B alongside the per-property MAEs.
        if self.logger is not None:
            self.logger.log(
                {"avg_mae": metrics["avg_mae"]["metric"], "epoch": self.epoch},
                step=self.step,
                split=split,
            )
        logging.info(
            f"avg_mae: {metrics['avg_mae']['metric']:.4f} "
            f"(pooled over {', '.join(mae_keys)})"
        )

        return metrics
