# Why `mex_trainer.py` differs from upstream

This repository applies a one-line fix to the single-property energy-head
trainer. Without it, single-property MEX models collapse to nearly constant
predictions. The training loss is unaffected; only model selection is.

## The cause

`DerivativeFreeOptimizer.infer` returns predictions of shape `(B,)` — its
comment says `(B, 1)`, but it does not — and `MEXTrainer._forward` passed them
through unchanged. `_compute_metrics` reshapes the targets to `(B, 1)`, and the
evaluator computes `torch.abs(target - prediction)`, so `(B, 1) - (B,)`
broadcasts to a `B x B` matrix. The reported validation "MAE" was therefore the
mean of `|y_j - yhat_i|` over **every pair** of structures in the evaluation
batch, not the per-structure error.

The multi-property trainer reshapes with `.view(-1, 1)`
(`mex_multi_trainer.py:406`) and was never affected. Neither was the regression
trainer, which reshapes in `ocp_trainer.py`.

## Why it causes collapse

Mean pairwise `|y_j - yhat_i|` decomposes per structure into
`sum_j |y_j - yhat_i|`, which is minimized at `yhat_i = median(y)`. The metric's
global optimum is therefore the constant median predictor. On the band-gap
validation split:

| Predictions | Broken metric | True MAE |
|---|---|---|
| perfect (`yhat = y`) | 0.844 | 0 |
| constant 0 eV (the median) | 0.502 | 0.504 |

Checkpoint selection, `ReduceLROnPlateau`, and early stopping all consumed this
quantity, so training kept the checkpoint whose predictions were most
concentrated near the median, cut the learning rate as the model learned real
spread, and stopped shortly after.

## Evidence

With the unpatched trainer, validation "MAE" bottoms out early and then *rises*
while the training loss keeps falling — for band gap, 0.575 at epoch 12 rising
to 0.925 by epoch 43, with early stopping at epoch 45. The resulting predictions
have a standard deviation of 0.19 eV against 1.74 eV for the labels, and their
error is within 3-12% of simply predicting the median.

With the fix, validation MAE falls monotonically and plateaus, and the three
seeds agree to the fourth decimal instead of varying by ±0.02.

Ruled out as causes: the data (the single-property LMDBs match the
three-property LMDB in order and labels), the model architecture, the loss, the
derivative-free optimizer, the search bounds, EMA, the scheduler, and the
training loop — all equivalent between the two trainers.

## The fix

`code/ocp/ocpmodels/trainers/mex_trainer.py`:

```python
return {target_name: y_pred.view(-1, 1)}
```

Do not replace this file with an upstream copy.
