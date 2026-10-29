"""
Multi-Property MEX Trainer
==========================
Extends MEXTrainer to support simultaneous prediction of multiple material
properties.  Key ideas (following the MEX paper):

  * Each property y_k has its own label encoder  fc_y_k: R -> D
  * At training time the molecule embedding x is augmented with ALL property
    embeddings:  x_aug = x + sum_k  W_k * fc_y_k(y_k)
    (similar to how MEX conditions on a single y, but now multi-dimensional)
  * Each property k has its own EBM head (fc_x_k, fc_xy_k, …) so inference
    can optimise each property independently.
  * At inference we use the Derivative-Free Optimizer (DFO) per property to
    produce a Pareto-optimal solution set via NSGA-II style non-dominated
    sorting.

Copyright (c) Facebook, Inc. and its affiliates. / Extended by user.
"""

import logging
import os
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with mex_trainer
# ─────────────────────────────────────────────────────────────────────────────

def gauss_density_centered(x, std):
    return torch.exp(-0.5 * (x / std) ** 2) / (math.sqrt(2 * math.pi) * std)


def gmm_density_centered(x, std):
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError("Last dimension must be the gmm stds.")
    return gauss_density_centered(x, std).prod(-2).mean(-1)


def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components
    std = std.view(1, num_dims, num_components)
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0, :, k].t()
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)
    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)
    return x_centered, prob_dens, prob_dens_zero


# ─────────────────────────────────────────────────────────────────────────────
# Per-property Derivative-Free Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer (DFO) for a single property."""

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        device: torch.device,
        noise_scale: float = 0.5,
        noise_shrink: float = 0.5,
        iters: int = 5,
        train_samples: int = 256,
        inference_samples: int = 500,
    ):
        self.device = device
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples

    def _sample(self, num_samples: int) -> torch.Tensor:
        size = (num_samples, 1)
        samples = np.random.uniform(self.lower_bound, self.upper_bound, size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, batch_size: int) -> torch.Tensor:
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples)

    @torch.no_grad()
    def infer(self, x_embed: torch.Tensor, model_cls: nn.Module, prop_name: str) -> torch.Tensor:
        """
        Infer best y for property prop_name given molecular embedding x_embed.

        x_embed : (B, D)
        returns : (B,)   – normalised predicted property values
        """
        noise_scale = self.noise_scale
        samples = self._sample(x_embed.size(0) * self.inference_samples)
        samples = samples.view(x_embed.size(0), self.inference_samples)  # (B, S)

        for _ in range(self.iters):
            scores = model_cls.get_energy_for_prop(x_embed, samples, prop_name)  # (B, S)
            scores = scores.double()
            probs = F.softmax(scores, dim=-1).cpu()
            probs[torch.isnan(probs)] = 1e-6
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=self.lower_bound, max=self.upper_bound)
            noise_scale *= self.noise_shrink

        scores = model_cls.get_energy_for_prop(x_embed, samples, prop_name)
        best_idxs = scores.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs]  # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Pareto utilities
# ─────────────────────────────────────────────────────────────────────────────

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution `a` Pareto-dominates `b` (minimisation)."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(costs: np.ndarray) -> List[List[int]]:
    """
    NSGA-II non-dominated sorting.

    costs : (N, K)  – K objectives, minimisation assumed
    returns list of fronts (each front is a list of indices)
    """
    n = len(costs)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(costs[i], costs[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(costs[j], costs[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Property MEX Trainer
# ─────────────────────────────────────────────────────────────────────────────

@registry.register_trainer("mex_multi")
class MEXMultiTrainer(BaseTrainer):
    """
    Trainer for multi-property MEX.

    Config additions vs single-property MEX
    ----------------------------------------
    dataset:
      property_names: [dielectric, gvrh, phonons]   # names of y_* fields in batch
      transforms:
        normalizer:
          y_dielectric: {mean: ..., stdev: ...}
          y_gvrh:       {mean: ..., stdev: ...}
          y_phonons:    {mean: ..., stdev: ...}
      train_max_y_dielectric: ...   train_min_y_dielectric: ...
      train_max_y_gvrh:       ...   train_min_y_gvrh:       ...
      train_max_y_phonons:    ...   train_min_y_phonons:    ...

    outputs:
      y_dielectric: {level: system, prediction_dtype: float32}
      y_gvrh:       {level: system, prediction_dtype: float32}
      y_phonons:    {level: system, prediction_dtype: float32}

    loss_functions:
      - y_dielectric: {fn: nce, coefficient: 1}
      - y_dielectric: {fn: euc, coefficient: 0.25}
      - y_gvrh:  ...
      - y_phonons: ...
    """

    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="wandb",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        name="ocp",
        early_stop=30,
        linear_probe=False,
    ):
        # Save the full dataset dict before base_trainer splits it into
        # self.config["dataset"] = dataset["train"], which drops top-level keys
        # like property_names, train_max_y_*, train_min_y_*.
        self._raw_dataset: Dict = dataset if isinstance(dataset, dict) else {}

        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_fns=loss_fns,
            eval_metrics=eval_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            name=name,
            early_stop=early_stop,
            linear_probe=linear_probe,
        )

        # ── property names ─────────────────────────────────────────────────
        self.property_names: List[str] = self._raw_dataset.get(
            "property_names", []
        )
        assert len(self.property_names) >= 2, (
            "MEXMultiTrainer requires at least 2 properties. "
            "Set dataset.property_names in your config."
        )
        logging.info(f"[MEXMulti] Properties: {self.property_names}")

        # ── per-property normalised bounds ─────────────────────────────────
        self.train_max_y: Dict[str, float] = {}
        self.train_min_y: Dict[str, float] = {}
        self.lower_bound: Dict[str, float] = {}
        self.upper_bound: Dict[str, float] = {}

        for pname in self.property_names:
            nkey = f"y_{pname}"
            norm = self.normalizers.get(nkey)
            assert norm is not None, (
                f"No normalizer found for '{nkey}'. "
                "Add it under dataset.transforms.normalizer in your config."
            )

            raw_max = self._raw_dataset.get(f"train_max_y_{pname}", None)
            raw_min = self._raw_dataset.get(f"train_min_y_{pname}", None)
            assert raw_max is not None and raw_min is not None, (
                f"Missing train_max_y_{pname} / train_min_y_{pname} in dataset config."
            )

            self.train_max_y[pname] = norm.norm(raw_max).item()
            self.train_min_y[pname] = norm.norm(raw_min).item()

            # lb = norm.norm(-10).item()
            # ub = norm.norm(10).item()
            # # ensure the interval contains the training range
            # lb = min(lb, self.train_min_y[pname] - 1.0)
            # ub = max(ub, self.train_max_y[pname] + 1.0)
            lb = self.train_min_y[pname] - 2.0
            ub = self.train_max_y[pname] + 2.0
            self.lower_bound[pname] = lb
            self.upper_bound[pname] = ub

        # ── per-property stochastic optimizers ────────────────────────────
        self.neg_samples = 500
        self.infer_samples = 2000

        self.stochastic_optimizers: Dict[str, DerivativeFreeOptimizer] = {}
        for pname in self.property_names:
            self.stochastic_optimizers[pname] = DerivativeFreeOptimizer(
                lower_bound=self.lower_bound[pname],
                upper_bound=self.upper_bound[pname],
                device=self.device,
                noise_scale=0.5,
                noise_shrink=0.5,
                iters=5,
                train_samples=self.neg_samples,
                inference_samples=self.infer_samples,
            )
            logging.info(
                f"[MEXMulti] DFO for {pname}: "
                f"[{self.lower_bound[pname]:.3f}, {self.upper_bound[pname]:.3f}]"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Loss loading (mirrors mex_trainer but per-property)
    # ──────────────────────────────────────────────────────────────────────

    def load_loss(self) -> None:
        """Parse loss config and store per-property loss flags.

        NOTE: called by base_trainer.__init__ -> self.load() *before* our own
        __init__ body runs, so self.property_names may not exist yet.
        We therefore read property_names straight from self.config here.
        """
        # Read from _raw_dataset (set before super().__init__()) because
        # base_trainer replaces self.config["dataset"] with only the train sub-dict,
        # discarding top-level keys like property_names.
        property_names = self._raw_dataset.get("property_names", [])
        assert len(property_names) >= 2, (
            "dataset.property_names must list at least 2 properties."
        )

        # per-property loss flags:  {pname: {use_nce, nce_coeff, use_euc, ...}}
        self.loss_cfg: Dict[str, Dict] = {
            p: {"use_nce": False, "use_euc": False, "use_infonce": False}
            for p in property_names
        }

        for loss in self.config["loss_fns"]:
            for target in loss:
                # target is e.g. "y_dielectric"
                pname = target.replace("y_", "", 1)
                if pname not in self.loss_cfg:
                    # fallback: treat target itself as property name
                    pname = target
                if pname not in self.loss_cfg:
                    logging.warning(f"[MEXMulti] Unknown target '{target}' in loss config, skipping.")
                    continue

                fn_name = loss[target].get("fn", None)
                coeff = loss[target].get("coefficient", 1.0)

                if fn_name == "nce":
                    self.loss_cfg[pname]["use_nce"] = True
                    self.loss_cfg[pname]["nce_coeff"] = coeff
                elif fn_name == "info-nce":
                    self.loss_cfg[pname]["use_infonce"] = True
                    self.loss_cfg[pname]["infonce_coeff"] = coeff
                elif fn_name == "euc":
                    self.loss_cfg[pname]["use_euc"] = True
                    self.loss_cfg[pname]["euc_coeff"] = coeff
                else:
                    raise ValueError(f"Unknown loss fn '{fn_name}' for target '{target}'")

                assert not (
                    self.loss_cfg[pname]["use_nce"]
                    and self.loss_cfg[pname]["use_infonce"]
                ), f"Cannot use nce and info-nce together for property {pname}"

    # ──────────────────────────────────────────────────────────────────────
    # Forward  (inference time)
    # ──────────────────────────────────────────────────────────────────────

    def _forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Returns per-property predictions in the *normalised* space.
        Shape: {pname: (B,)}
        """
        model_cls = self.model.module if hasattr(self.model, "module") else self.model
        x = model_cls.get_mol_rep(batch.to(self.device))  # (B, D)

        preds = {}
        for pname in self.property_names:
            opt = self.stochastic_optimizers[pname]
            y_pred = opt.infer(x, model_cls, pname)  # (B,)
            preds[f"y_{pname}"] = y_pred.view(-1, 1)

        return preds

    # ──────────────────────────────────────────────────────────────────────
    # Loss computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        total_loss = 0.0
        loss_info: Dict[str, torch.Tensor] = {}

        model_cls = self.model.module if hasattr(self.model, "module") else self.model
        batch = batch.to(self.device)

        # Shared molecule embedding
        mol_embeddings = model_cls.get_mol_rep(batch)  # (B, D)

        for pname in self.property_names:
            cfg = self.loss_cfg[pname]
            nkey = f"y_{pname}"
            norm = self.normalizers[nkey]

            ys_raw = batch[nkey].view(-1, 1)   # (B, 1) unnormalised
            ys = norm.norm(ys_raw)             # (B, 1) normalised

            scores_gt, x_feat, y_feat = model_cls.get_energy_for_prop(
                mol_embeddings, ys, pname, ret_rep=True
            )
            scores_gt = scores_gt.view(-1)  # (B,)

            # ── euclidean ────────────────────────────────────────────────
            if cfg["use_euc"]:
                xn = F.normalize(x_feat, dim=-1)
                yn = F.normalize(y_feat, dim=-1)
                euc = torch.pow(xn - yn, 2).sum(1)
                euc = euc[torch.isfinite(euc)]
                euc = self._ddp_loss(euc)
                assert hasattr(euc, "grad_fn")
                total_loss = total_loss + cfg["euc_coeff"] * euc
                loss_info[f"{pname}_euc"] = euc

            # ── NCE ──────────────────────────────────────────────────────
            if cfg["use_nce"]:
                stds = torch.zeros((1, 3))
                stds[0, 0] = 0.075
                stds[0, 1] = 0.15
                stds[0, 2] = 0.3
                y_zero, q_samples, q_ys = sample_gmm_centered(stds, num_samples=self.neg_samples)
                y_zero = y_zero.squeeze(1).to(self.device)
                y_samp = ys + y_zero.unsqueeze(0)             # (B, neg)
                scores_samp = model_cls.get_energy_for_prop(
                    mol_embeddings, y_samp, pname
                )                                              # (B, neg)
                q_samp = q_samples.to(self.device).unsqueeze(0) * torch.ones(y_samp.size(), device=self.device)
                q_pos = q_ys[0] * torch.ones(ys.size(0), device=self.device)

                # Numerically stable NCE via log-sum-exp trick.
                # Old formula used raw torch.exp which overflows for large scores.
                log_p_pos = scores_gt - torch.log(q_pos)          # (B,)
                log_p_neg = scores_samp - torch.log(q_samp)        # (B, neg)
                log_denom = torch.logsumexp(
                    torch.cat([log_p_pos.unsqueeze(1), log_p_neg], dim=1), dim=1
                )                                                  # (B,)
                nce = -(log_p_pos - log_denom)                     # (B,)
                nce = nce[torch.isfinite(nce)]
                if nce.numel() == 0:
                    logging.warning(f"[MEXMulti] All NCE samples non-finite for '{pname}', skipping step.")
                else:
                    nce = self._ddp_loss(nce)
                    assert hasattr(nce, "grad_fn")
                    total_loss = total_loss + cfg["nce_coeff"] * nce
                    loss_info[f"{pname}_nce"] = nce

            # ── InfoNCE ──────────────────────────────────────────────────
            if cfg["use_infonce"]:
                opt = self.stochastic_optimizers[pname]
                negatives = opt.sample(ys.shape[0])           # (B, neg)
                targets = torch.cat([ys, negatives], dim=1)   # (B, 1+neg)
                perm = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
                targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), perm]
                gt_cls = (perm == 0).nonzero()[:, 1].to(self.device)
                energy = model_cls.get_energy_for_prop(mol_embeddings, targets, pname)
                infonce = F.cross_entropy(energy, gt_cls, reduction="none")
                infonce = infonce[torch.isfinite(infonce)]
                if infonce.numel() == 0:
                    logging.warning(f"[MEXMulti] All InfoNCE samples non-finite for '{pname}', skipping step.")
                else:
                    infonce = self._ddp_loss(infonce)
                    assert hasattr(infonce, "grad_fn")
                    total_loss = total_loss + cfg["infonce_coeff"] * infonce
                    loss_info[f"{pname}_infonce"] = infonce

        assert hasattr(total_loss, "grad_fn"), "Loss has no grad_fn – check loss config."
        loss_info["loss"] = total_loss
        return loss_info

    # ──────────────────────────────────────────────────────────────────────
    # DDP loss helper
    # ──────────────────────────────────────────────────────────────────────

    def _ddp_loss(self, loss_local: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(loss_local)
        bs = loss_local.shape[0]
        bs = distutils.all_reduce(bs, device=loss_local.device)
        loss = loss * distutils.get_world_size() / bs
        return loss

    # ──────────────────────────────────────────────────────────────────────
    # Training loop  (follows mex_trainer exactly)
    # ──────────────────────────────────────────────────────────────────────

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )

        if not hasattr(self, "primary_metric") or self.primary_metric != primary_metric:
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric

        if not hasattr(self, "early_stop") or self.early_stop is None:
            self.early_stop = self.config["early_stop"]

        self.metrics = {}
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            skip_steps = self.step % len(self.train_loader)
            self.train_sampler.set_epoch_and_start_iteration(epoch_int, skip_steps)
            train_loader_iter = iter(self.train_loader)
            epoch_metrics = {}  # accumulates loss/metrics over the full epoch

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                batch = next(train_loader_iter)
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    loss_dict = self._compute_loss(batch)
                    loss = loss_dict["loss"]

                if not torch.isfinite(loss):
                    logging.warning(
                        f"[MEXMulti] Non-finite loss ({loss.item()}) at step {self.step}, skipping update."
                    )
                    self.optimizer.zero_grad()
                    continue

                for k, v in loss_dict.items():
                    self.metrics = self.evaluator.update(k, v.item(), self.metrics)
                    epoch_metrics = self.evaluator.update(k, v.item(), epoch_metrics)

                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)

                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update({"lr": self.scheduler.get_lr(), "epoch": self.epoch, "step": self.step})

                if self.step % self.config["cmd"]["print_every"] == 0 and distutils.is_master():
                    log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(log_dict, step=self.step, split="train")

                if checkpoint_every != -1 and self.step % checkpoint_every == 0:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)

                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(split="val", disable_tqdm=disable_eval_tqdm)
                        self.update_best(primary_metric, val_metrics, disable_eval_tqdm=disable_eval_tqdm)

                    if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                        if self.step % eval_every == 0:
                            self.scheduler.step(metrics=val_metrics[primary_metric]["metric"])
                    else:
                        self.scheduler.step()

                torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

            # ── epoch-level training log ──────────────────────────────────
            if self.logger is not None and distutils.is_master() and epoch_metrics:
                epoch_log = {k: epoch_metrics[k]["metric"] for k in epoch_metrics}
                epoch_log["epoch"] = epoch_int + 1
                self.logger.log(epoch_log, step=self.step, split="train_epoch")

            if self.early_stop == 0:
                logging.info(f"Epoch {epoch_int}: early stop triggered.")
                break

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    # ──────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        )
        rank = distutils.get_rank()
        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                batch.to(self.device)
                out = self._forward(batch)
            metrics = self._compute_metrics(out, batch, evaluator, metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(metrics[k]["total"], average=False, device=self.device),
                "numel": distutils.all_reduce(metrics[k]["numel"], average=False, device=self.device),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        # Combined average MAE across all properties for use as primary_metric.
        mae_keys = [f"y_{p}_mae" for p in self.property_names]
        if all(k in aggregated_metrics for k in mae_keys):
            avg_total = sum(aggregated_metrics[k]["total"] for k in mae_keys)
            avg_numel = sum(aggregated_metrics[k]["numel"] for k in mae_keys)
            aggregated_metrics["avg_mae"] = {
                "total": avg_total,
                "numel": avg_numel,
                "metric": avg_total / avg_numel,
            }
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        if self.logger is not None:
            self.logger.log(log_dict, step=self.step, split=split)

        if self.ema:
            self.ema.restore()

        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Metrics computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_metrics(self, out, batch, evaluator, metrics={}):
        out = {k: v.clone() for k, v in out.items()}

        natoms = batch.natoms
        batch_size = natoms.numel()

        fixed = batch.fixed
        mask = fixed == 0
        s_idx = 0
        natoms_free = []
        for _n in natoms:
            natoms_free.append(torch.sum(mask[s_idx: s_idx + _n]).item())
            s_idx += _n
        natoms = torch.LongTensor(natoms_free).to(self.device)

        targets = {}
        for pname in self.property_names:
            tkey = f"y_{pname}"
            target = batch[tkey].view(batch_size, -1)
            targets[tkey] = target  # unnormalised

            norm = self.normalizers.get(tkey)
            if norm:
                out[tkey] = norm.denorm(out[tkey])

        targets["natoms"] = natoms
        out["natoms"] = natoms

        metrics = evaluator.eval(out, targets, prev_metrics=metrics)
        return metrics

    # ──────────────────────────────────────────────────────────────────────
    # Pareto-optimal inference
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def pareto_predict(
        self,
        data_loader,
        n_pareto_samples: int = 200,
        results_file: Optional[str] = None,
        disable_tqdm: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        For each molecule, sample `n_pareto_samples` candidate property
        vectors, then return the Pareto-front (non-dominated solutions).

        Returns a dict with keys:
          "ids"           – structure ids (N_pareto,)
          "y_<prop>"      – denormalised predictions (N_pareto,)  per property
          "pareto_front"  – bool mask over all samples (N_total,)
          "front_rank"    – Pareto front rank per sample
        """
        ensure_fitted(self._unwrapped_model, warn=True)
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        model_cls = self.model.module if hasattr(self.model, "module") else self.model
        rank = distutils.get_rank()

        all_ids: List[str] = []
        all_preds: Dict[str, List[float]] = {p: [] for p in self.property_names}
        all_front_rank: List[int] = []

        for batch in tqdm(
            data_loader,
            position=rank,
            desc=f"Pareto inference (device {rank})",
            disable=disable_tqdm,
        ):
            batch.to(self.device)
            x = model_cls.get_mol_rep(batch)  # (B, D)
            B = x.size(0)

            # For each molecule produce n_pareto_samples candidate points
            # by sampling the DFO independently per property
            batch_candidates: Dict[str, np.ndarray] = {}
            for pname in self.property_names:
                opt = self.stochastic_optimizers[pname]
                # Draw multiple restarts
                raw_samples = []
                for _ in range(n_pareto_samples):
                    y_p = opt.infer(x, model_cls, pname).cpu().numpy()  # (B,)
                    raw_samples.append(y_p)
                batch_candidates[pname] = np.stack(raw_samples, axis=1)  # (B, n_pareto_samples)

            # Denormalise
            for pname in self.property_names:
                norm = self.normalizers[f"y_{pname}"]
                for b in range(B):
                    vals_norm = torch.tensor(batch_candidates[pname][b], dtype=torch.float32)
                    vals_denorm = norm.denorm(vals_norm.view(-1, 1)).view(-1).numpy()
                    batch_candidates[pname][b] = vals_denorm

            # Non-dominated sorting per molecule
            for b in range(B):
                costs = np.stack(
                    [batch_candidates[p][b] for p in self.property_names], axis=1
                )  # (n_pareto_samples, K)
                fronts = non_dominated_sort(costs)
                ranks = np.zeros(n_pareto_samples, dtype=int)
                for r, front in enumerate(fronts):
                    for idx in front:
                        ranks[idx] = r

                # Keep Pareto front (rank 0)
                front0 = fronts[0] if fronts else list(range(n_pareto_samples))
                for fi in front0:
                    for pname in self.property_names:
                        all_preds[pname].append(float(batch_candidates[pname][b][fi]))
                    all_front_rank.append(0)

                    sid = batch.sid[b].item() if hasattr(batch, "sid") else f"{b}"
                    all_ids.append(str(sid))

        results: Dict[str, np.ndarray] = {"ids": np.array(all_ids)}
        for pname in self.property_names:
            results[f"y_{pname}"] = np.array(all_preds[pname])
        results["front_rank"] = np.array(all_front_rank)

        # if results_file:
        #     np.savez(results_file, **results)
        #     logging.info(f"Saved Pareto predictions to {results_file}")
            
        if results_file:
            results_file_path = os.path.join(
                self.config["cmd"]["results_dir"], f"{results_file}.npz"
            )
            np.savez(results_file_path, **results)
            logging.info(f"Saved Pareto predictions to {results_file_path}")

        if self.ema:
            self.ema.restore()

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Prediction (standard, per-property)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: Optional[str] = None,
        disable_tqdm: bool = False,
    ):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")

        rank = distutils.get_rank()
        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [data_loader]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        predictions: Dict[str, List] = defaultdict(list)

        for batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            _, batch = batch
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            for pname in self.property_names:
                tkey = f"y_{pname}"
                pred = out[tkey]
                norm = self.normalizers.get(tkey)
                if norm:
                    pred = norm.denorm(pred)
                if per_image:
                    pred = pred.cpu().detach().to(torch.float32).numpy().reshape(-1).tolist()
                    predictions[tkey].extend(pred)
                    gt = batch[tkey].cpu().numpy().reshape(-1).tolist()
                    predictions[f"{tkey}_gts"].extend(gt)
                else:
                    predictions[tkey] = pred.detach()

            if not per_image:
                return predictions

            sids = batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid
            predictions["ids"].extend([str(s) for s in sids])

        for key in predictions:
            if isinstance(predictions[key], list):
                predictions[key] = np.array(predictions[key])

        if results_file:
            self.save_results(predictions, results_file)

        if self.ema:
            self.ema.restore()

        return predictions