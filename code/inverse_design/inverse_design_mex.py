"""
inverse_design_mex.py
=====================
Inverse design using a trained MEX multi-property model.

BACKGROUND — Why MEX beats the Li-conductor paper's approach
------------------------------------------------------------
Sendek et al. (Nature 2019) discover Li-ion conductors via:
  1. Hand-craft ~20 structural/compositional descriptors X.
  2. Cluster in X-space (KNN / logistic regression).
  3. Hope that clusters correlating with known good Y also contain new good Y.

Limitation: clustering in X does NOT equal clustering in Y.
Features unrelated to conductivity add noise; expert engineering is mandatory.

MEX multi-property inverse design:
  1. Equiformer-V2 learns x = get_mol_rep(structure) — a D-dimensional
     SE(3)-equivariant embedding capturing all structural information.
     No manual feature engineering.
  2. For each property k, a 3-layer energy head f_k(x, y_k) is trained so
     that f_k is HIGH when structure x is COMPATIBLE with property value y_k,
     and LOW otherwise.  This is the joint X-Y latent space.
  3. Inverse design: given target property values (y_dielectric*, y_gvrh*),
     score every structure in a candidate database by
         score_i = sum_k  f_k(x_i, y_k*)
     and return the top-k structures — no feature engineering needed.

Limitations compared to a fully generative approach:
  · Retrieval-based: can only return structures that are already in the DB.
  · Extrapolation quality depends on training distribution coverage.
  · Property targets must be within the trained range (extended by DFO bounds).

Future extension — gradient-based structure optimisation:
  Since get_mol_rep is differentiable w.r.t. atomic positions, you could do
  gradient ascent on score(x(pos), y*) w.r.t. positions for generative design.
  That is left as future work.

Usage
-----
# 1. Build the embedding index once after training:
python inverse_design/inverse_design_mex.py build-index \\
    --checkpoint <best_checkpoint.pt> \\
    --config configs/mex-multi/dielectric_gvrh.yml \\
    --lmdb data/dielectric_gvrh/train/data.lmdb \\
    --out index/train_embeddings.npz

# 2. Query: find top-k structures compatible with target properties:
python inverse_design/inverse_design_mex.py query \\
    --checkpoint <best_checkpoint.pt> \\
    --config configs/mex-multi/dielectric_gvrh.yml \\
    --index index/train_embeddings.npz \\
    --targets "dielectric=3.5,gvrh=2.5" \\
    --top-k 20

# 3. Visualise the latent space (requires umap-learn):
python inverse_design/inverse_design_mex.py visualize \\
    --index index/train_embeddings.npz \\
    --out-fig figs/latent_space.png
"""

import sys
import os
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── make ocp importable from repo root ───────────────────────────────────────
# The cluster run scripts set PYTHONPATH=$CONDA_ENV_PATH/lib/python3.10/site-packages
# (no conflicting ocpmodels), so sys.path.insert(0, ocp/) is sufficient.
# code/ holds ocp/; the repository root holds configs/ and data/. These were the
# same directory only while code/configs and code/data were symlinks.
CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent
_ocp_path = str(CODE_ROOT / "ocp")
if _ocp_path not in sys.path:
    sys.path.insert(0, _ocp_path)
# Purge any stale ocpmodels that may have been loaded before this point
for _mod in list(sys.modules):
    if _mod == "ocpmodels" or _mod.startswith("ocpmodels."):
        del sys.modules[_mod]

import yaml
from ocpmodels.common.utils import load_state_dict
from ocpmodels.modules.normalizer import Normalizer
# Import model classes directly from the file that defines them —
# equiformer_v2_oc20.py is present on every copy of the repo.
from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import (
    EquiformerV2MEXMulti,
    EquiformerV2MEX,
)

from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import (  # noqa: E402
    EquiformerV2Direct,
    EquiformerV2DirectMulti,
)

# Retrieval only ever calls get_mol_rep, which every one of these inherits from
# EquiformerV2_OC20 -- the energy head, the label encoder and the derivative-free
# search play no part in embedding. So the plain-regression (ERM) arms can reuse
# this engine unchanged; they just need their class names to resolve here.
_MODEL_REGISTRY = {
    "equiformer_v2-mex-multi":    EquiformerV2MEXMulti,
    "equiformer_v2_mex_multi":    EquiformerV2MEXMulti,
    "equiformer_v2-mex":          EquiformerV2MEX,
    "equiformer_v2_mex":          EquiformerV2MEX,
    "equiformer_v2-direct-multi": EquiformerV2DirectMulti,
    "equiformer_v2_direct_multi": EquiformerV2DirectMulti,
    "equiformer_v2-direct":       EquiformerV2Direct,
    "equiformer_v2_direct":       EquiformerV2Direct,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_includes(cfg: dict, base_dir: Path) -> dict:
    """Recursively merge YAML 'includes' lists (shallow, later files win)."""
    merged: dict = {}
    for inc in cfg.get("includes", []):
        inc_path = REPO_ROOT / inc
        with open(inc_path) as f:
            inc_cfg = yaml.safe_load(f)
        inc_cfg = _resolve_includes(inc_cfg, inc_path.parent)
        _deep_update(merged, inc_cfg)
    cfg.pop("includes", None)
    _deep_update(merged, cfg)
    return merged


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(yml_path: str) -> dict:
    yml_path = Path(yml_path)
    with open(yml_path) as f:
        raw = yaml.safe_load(f)
    return _resolve_includes(raw, yml_path.parent)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

class MEXInferenceEngine:
    """
    Lightweight wrapper around a trained MEX multi-property model.

    Only loads the model and normalizers — no trainer, no dataloaders.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.config = load_config(config_path)

        self.property_names = self.config["dataset"]["property_names"]

        # ── normalizers ─────────────────────────────────────────────────────
        normalizer_cfg = (
            self.config["dataset"]
            .get("train", {})
            .get("transforms", {})
            .get("normalizer", {})
        )
        self.normalizers: dict[str, Normalizer] = {}
        for k, v in normalizer_cfg.items():
            self.normalizers[k] = Normalizer(
                mean=v["mean"], std=v["stdev"], device=str(self.device)
            )

        # ── model ────────────────────────────────────────────────────────────
        model_attrs = self.config.get("model_attributes",
                                      self.config.get("model", {}))
        if "name" in model_attrs:
            model_name = model_attrs.pop("name")
        else:
            # model config key is a dict with 'name' at top-level
            model_name = self.config["model"]["name"]
            model_attrs = {k: v for k, v in self.config["model"].items()
                          if k != "name"}

        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Supported: {list(_MODEL_REGISTRY)}"
            )
        model_cls = _MODEL_REGISTRY[model_name]
        self.model: torch.nn.Module = model_cls(
            None, None, 1, **model_attrs
        ).to(self.device)
        self.model.eval()

        # ── load checkpoint ─────────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt["state_dict"]
        # Strip DDP "module." prefixes
        n_module = next(iter(state)).count("module.")
        if n_module:
            state = {k[len("module.") * n_module:]: v for k, v in state.items()}
        load_state_dict(self.model, state, strict=False)

        # Override normalizers from checkpoint if available
        if "normalizers" in ckpt and ckpt["normalizers"]:
            for k, sd in ckpt["normalizers"].items():
                if k in self.normalizers:
                    self.normalizers[k].load_state_dict(sd)

        # Apply EMA weights if present
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema_state = ckpt["ema"]["shadow_params"]
            ema_keys  = list(self.model.state_dict().keys())
            ema_dict  = {k: v for k, v in zip(ema_keys, ema_state)}
            self.model.load_state_dict(ema_dict, strict=False)

        logging.info(
            f"Loaded {model_name} ({self.model.num_params:,} params) "
            f"| properties: {self.property_names}"
        )

    # ── core inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_batch(self, batch) -> torch.Tensor:
        """Return structural embeddings x ∈ R^D for a batch. Shape: (B, D)."""
        batch = batch.to(self.device)
        return self.model.get_mol_rep(batch)   # (B, D)

    @torch.no_grad()
    def score_batch(
        self,
        embeddings: torch.Tensor,      # (B, D)
        y_targets_norm: dict[str, torch.Tensor],  # {pname: scalar tensor}
    ) -> torch.Tensor:
        """
        Compute multi-property EBM compatibility score for a batch of embeddings.

        score_i = sum_k  f_k(x_i, y_k*)

        Higher = more compatible with ALL target property values simultaneously.
        Returns: (B,) tensor.
        """
        B = embeddings.shape[0]
        total = torch.zeros(B, device=self.device)
        for pname in self.property_names:
            y_t = y_targets_norm[pname].to(self.device).view(1, 1).expand(B, 1)
            s = self.model.get_energy_for_prop(
                embeddings, y_t, pname
            ).view(B)                         # (B,)
            total += s
        return total


# ─────────────────────────────────────────────────────────────────────────────
# LMDB reader (standalone, no dataloader)
# ─────────────────────────────────────────────────────────────────────────────

import lmdb

def read_all_from_lmdb(lmdb_path: str) -> list:
    """Return list of torch_geometric Data objects from an lmdb file."""
    import lmdb as _lmdb
    env = _lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    records = []
    with env.begin() as txn:
        n = pickle.loads(txn.get(b"length"))
        for i in range(n):
            obj = pickle.loads(txn.get(str(i).encode()))
            records.append(obj)
    env.close()
    return records


def collate_pyg(records: list, indices: list):
    """Minimal collate for equiformer_v2: returns a torch_geometric Batch."""
    import torch_geometric
    subset = [records[i] for i in indices]
    return torch_geometric.data.Batch.from_data_list(subset)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: build-index
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build_index(args):
    """
    Embed all structures in an LMDB and save:
      - embeddings   : (N, D) float32
      - property values : {pname: (N,) float32}  (raw, denormalised)
      - sids          : (N,) int
    """
    engine = MEXInferenceEngine(args.checkpoint, args.config, device=args.device)
    records = read_all_from_lmdb(args.lmdb)
    N = len(records)
    logging.info(f"Embedding {N} structures from {args.lmdb} ...")

    all_emb = []
    all_props = {p: [] for p in engine.property_names}
    all_sids  = []

    batch_size = args.batch_size
    for start in range(0, N, batch_size):
        idx = list(range(start, min(start + batch_size, N)))
        batch = collate_pyg(records, idx)
        emb = engine.embed_batch(batch).cpu().numpy()   # (B, D)
        all_emb.append(emb)

        for pname in engine.property_names:
            attr = f"y_{pname}"
            vals = np.array([float(records[i][attr]) for i in idx])
            all_props[pname].append(vals)

        sids = [int(getattr(records[i], "sid", i)) for i in idx]
        all_sids.extend(sids)

        if (start // batch_size) % 10 == 0:
            logging.info(f"  {min(start + batch_size, N)}/{N}")

    all_emb = np.concatenate(all_emb, axis=0).astype(np.float32)
    save_dict = {
        "embeddings": all_emb,
        "sids": np.array(all_sids, dtype=np.int32),
        "property_names": np.array(engine.property_names),
    }
    for pname in engine.property_names:
        save_dict[f"y_{pname}"] = np.concatenate(all_props[pname]).astype(np.float32)

    out_path = Path(args.out_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **save_dict)
    logging.info(f"Saved index to {out_path}  (shape: {all_emb.shape})")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: query
# ─────────────────────────────────────────────────────────────────────────────

def parse_targets(targets_str: str) -> dict:
    """Parse 'dielectric=3.5,gvrh=2.5' → {'dielectric': 3.5, 'gvrh': 2.5}."""
    out = {}
    for item in targets_str.split(","):
        k, v = item.strip().split("=")
        out[k.strip()] = float(v.strip())
    return out


def cmd_query(args):
    """
    Given target property values, retrieve the top-k database structures
    whose joint EBM score is highest.

    Multi-property EBM scoring replaces KNN-in-X-space from the Li paper:
      · No feature engineering — the GNN embedding IS the feature.
      · Directly target specific Y values, not clusters.
      · Score reflects joint compatibility with ALL requested properties.
    """
    engine = MEXInferenceEngine(args.checkpoint, args.config, device=args.device)
    targets_raw = parse_targets(args.targets)

    # Validate property names
    for p in targets_raw:
        if p not in engine.property_names:
            raise ValueError(
                f"Property '{p}' not in model's property_names: {engine.property_names}"
            )

    # Normalise targets
    y_targets_norm = {}
    for pname in engine.property_names:
        if pname not in targets_raw:
            raise ValueError(
                f"Target for property '{pname}' not provided. "
                f"Provide all of: {engine.property_names}"
            )
        raw_val = torch.tensor(targets_raw[pname], dtype=torch.float32)
        norm    = engine.normalizers[f"y_{pname}"]
        y_targets_norm[pname] = norm.norm(raw_val)

    # Load index
    idx_data = np.load(args.index, allow_pickle=True)
    embeddings = torch.tensor(idx_data["embeddings"], dtype=torch.float32)
    sids       = idx_data["sids"]
    prop_names = idx_data["property_names"].tolist()
    N = embeddings.shape[0]
    logging.info(f"Loaded index: {N} structures, dim={embeddings.shape[1]}")

    # Score in batches
    all_scores = []
    bs = args.batch_size
    for start in range(0, N, bs):
        emb_batch = embeddings[start:start+bs].to(engine.device)
        s = engine.score_batch(emb_batch, y_targets_norm).cpu()
        all_scores.append(s)
    scores = torch.cat(all_scores).numpy()   # (N,)

    # Top-k
    k = min(args.top_k, N)
    top_idx = np.argsort(scores)[::-1][:k]

    print(f"\n{'='*65}")
    print(f"  Top-{k} candidates for targets: {targets_raw}")
    print(f"{'='*65}")
    header = f"{'Rank':>4}  {'sid':>6}  {'Score':>8}"
    for pname in prop_names:
        header += f"  {pname:>12}"
    print(header)
    print("-" * 65)
    for rank, i in enumerate(top_idx):
        sid = sids[i]
        sc  = scores[i]
        row = f"{rank+1:>4}  {sid:>6}  {sc:>8.4f}"
        for pname in prop_names:
            yv = idx_data[f"y_{pname}"][i]
            row += f"  {yv:>12.4f}"
        print(row)
    print(f"{'='*65}\n")

    # Save results
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "sid", "ebm_score"] + [f"y_{p}" for p in prop_names])
            for rank, i in enumerate(top_idx):
                row = [rank+1, int(sids[i]), float(scores[i])]
                for pname in prop_names:
                    row.append(float(idx_data[f"y_{pname}"][i]))
                w.writerow(row)
        logging.info(f"Results written to {args.out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: pareto-query  (no explicit targets — find Pareto-optimal front)
# ─────────────────────────────────────────────────────────────────────────────

def _dfo_predict_for_embedding(engine, emb_single: torch.Tensor, pname: str) -> float:
    """
    Run DFO on a single embedding to get the optimal predicted property value.
    emb_single: (D,) tensor on engine.device
    Returns the denormalised predicted value.
    """
    opt = engine.model  # EquiformerV2MEXMulti
    # DFO: sample y candidates and pick argmax of energy score
    lb = engine.config["dataset"].get(f"train_min_y_{pname}", -5.0)
    ub = engine.config["dataset"].get(f"train_max_y_{pname}", 5.0)
    norm = engine.normalizers[f"y_{pname}"]
    lb_n = norm.norm(torch.tensor(lb)).item()
    ub_n = norm.norm(torch.tensor(ub)).item()

    n_samples = 500
    y_cands = torch.linspace(lb_n, ub_n, n_samples, device=engine.device).view(1, -1)
    x = emb_single.unsqueeze(0)   # (1, D)
    scores = opt.get_energy_for_prop(x, y_cands, pname)  # (1, n_samples)
    best_yn = y_cands[0, scores[0].argmax()].unsqueeze(0)
    best_raw = norm.denorm(best_yn).item()
    return best_raw


def _pareto_rank(costs: np.ndarray) -> np.ndarray:
    """
    Return Pareto rank (0 = non-dominated front) for a (N, K) cost matrix.
    We MINIMISE costs — to maximise property values, pass in -values.
    """
    N = costs.shape[0]
    ranks = np.full(N, -1, dtype=int)
    remaining = set(range(N))
    rank = 0
    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            ranks[i] = rank
            remaining.remove(i)
        rank += 1
    return ranks


def cmd_pareto_query(args):
    """
    Find materials on the Pareto-optimal front for simultaneous maximisation
    of all trained properties.

    Steps:
      1. For each structure in the index, run per-property DFO
         to predict optimal property values the model assigns to it.
      2. Build a cost matrix (-predicted_value) and compute Pareto fronts.
      3. Return the rank-0 (non-dominated) structures — these represent
         the best trade-off across all properties.

    This is the structural analogue of NSGA-II used in pareto_predict():
    instead of optimising y* in property space given a fixed x, we fix
    candidate y-values to their DFO-predicted optima and rank by Pareto dom.
    """
    engine = MEXInferenceEngine(args.checkpoint, args.config, device=args.device)

    idx_data = np.load(args.index, allow_pickle=True)
    embeddings = torch.tensor(idx_data["embeddings"], dtype=torch.float32)
    sids       = idx_data["sids"]
    prop_names = idx_data["property_names"].tolist()
    N          = embeddings.shape[0]

    logging.info(f"Running DFO predictions for {N} structures ...")
    pred_vals = {p: np.zeros(N) for p in prop_names}

    for i in range(N):
        emb = embeddings[i].to(engine.device)
        for pname in prop_names:
            pred_vals[pname][i] = _dfo_predict_for_embedding(engine, emb, pname)
        if i % max(1, N // 10) == 0:
            logging.info(f"  {i}/{N}")

    # Cost matrix  — minimise negative predicted values (= maximise values)
    cost_mat = np.stack([-pred_vals[p] for p in prop_names], axis=1)  # (N, K)
    ranks = _pareto_rank(cost_mat)

    # Print rank-0 (Pareto-optimal) structures
    front0_idx = np.where(ranks == 0)[0]
    logging.info(f"Pareto front (rank 0): {len(front0_idx)} structures")

    print(f"\n{'='*70}")
    print(f"  Pareto-optimal front ({len(front0_idx)} structures)")
    print(f"{'='*70}")
    header = f"{'sid':>6}  {'rank':>4}"
    for p in prop_names:
        header += f"  {'pred_'+p:>14}  {'actual_'+p:>14}"
    print(header)
    print("-" * 70)

    # Sort front by sum of predicted values (descending)
    sort_key = np.array(
        [sum(pred_vals[p][i] for p in prop_names) for i in front0_idx]
    )
    order = np.argsort(sort_key)[::-1]

    for i in front0_idx[order]:
        row = f"{int(sids[i]):>6}  {int(ranks[i]):>4}"
        for p in prop_names:
            actual = idx_data[f"y_{p}"][i]
            row += f"  {pred_vals[p][i]:>14.4f}  {actual:>14.4f}"
        print(row)
    print(f"{'='*70}\n")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["sid", "pareto_rank"]
                + [f"pred_{p}" for p in prop_names]
                + [f"actual_{p}" for p in prop_names]
            )
            for i in range(N):
                row = [int(sids[i]), int(ranks[i])]
                for p in prop_names:
                    row.append(float(pred_vals[p][i]))
                for p in prop_names:
                    row.append(float(idx_data[f"y_{p}"][i]))
                w.writerow(row)
        logging.info(f"Full Pareto results written to {args.out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: visualize
# ─────────────────────────────────────────────────────────────────────────────

def cmd_visualize(args):
    """
    UMAP of the structural embedding space, coloured by property values.

    This shows WHY MEX finds good candidates:
    - The X-embedding space is structured by crystal chemistry.
    - The EBM head maps compatible (X, Y) pairs to high energies.
    - Unlike clustering in hand-crafted X-space, the embedding captures
      what is actually relevant for the target properties.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    idx_data = np.load(args.index, allow_pickle=True)
    embeddings  = idx_data["embeddings"]           # (N, D)
    prop_names  = idx_data["property_names"].tolist()
    N = embeddings.shape[0]

    logging.info(f"Running UMAP on {N} structures (dim={embeddings.shape[1]}) ...")
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        random_state=42,
        metric="cosine",
    )
    umap_emb = reducer.fit_transform(embeddings)   # (N, 2)

    n_props = len(prop_names)
    fig, axes = plt.subplots(1, n_props, figsize=(5 * n_props, 4.5))
    if n_props == 1:
        axes = [axes]

    for ax, pname in zip(axes, prop_names):
        vals = idx_data[f"y_{pname}"]
        sc = ax.scatter(
            umap_emb[:, 0], umap_emb[:, 1],
            c=vals, cmap="viridis", s=6, alpha=0.7
        )
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(f"$y_{{{pname}}}$", fontsize=13)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "MEX Structural Embedding Space (Equiformer-V2 get_mol_rep)",
        fontsize=12
    )
    plt.tight_layout()
    out = Path(args.out_fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    logging.info(f"Saved figure to {out}")
    plt.close(fig)

    # Also save UMAP coordinates alongside index for future use
    umap_out = out.with_suffix(".umap.npz")
    np.savez_compressed(str(umap_out), umap=umap_emb.astype(np.float32))
    logging.info(f"Saved UMAP coordinates to {umap_out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="MEX multi-property inverse design pipeline"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── build-index ──────────────────────────────────────────────────────────
    p_idx = sub.add_parser("build-index", help="Embed all structures in an LMDB")
    p_idx.add_argument("--checkpoint", required=True)
    p_idx.add_argument("--config",     required=True)
    p_idx.add_argument("--lmdb",       required=True,
                       help="Path to data.lmdb to embed (can be train/val/test)")
    p_idx.add_argument("--out-index",  required=True,
                       help="Output .npz file for the embedding index")
    p_idx.add_argument("--batch-size", type=int, default=32)

    # ── query ────────────────────────────────────────────────────────────────
    p_qry = sub.add_parser("query", help="Retrieve top-k candidates for target properties")
    p_qry.add_argument("--checkpoint", required=True)
    p_qry.add_argument("--config",     required=True)
    p_qry.add_argument("--index",      required=True, help="Index .npz from build-index")
    p_qry.add_argument("--targets",    required=True,
                       help="Comma-separated targets e.g. 'dielectric=3.5,gvrh=2.5'")
    p_qry.add_argument("--top-k",      type=int, default=20)
    p_qry.add_argument("--batch-size", type=int, default=512)
    p_qry.add_argument("--out-csv",    default=None, help="Optional CSV output path")

    # ── pareto-query ─────────────────────────────────────────────────────────
    p_par = sub.add_parser("pareto-query",
                           help="Pareto-optimal retrieval (no explicit targets)")
    p_par.add_argument("--checkpoint", required=True)
    p_par.add_argument("--config",     required=True)
    p_par.add_argument("--index",      required=True)
    p_par.add_argument("--batch-size", type=int, default=32)
    p_par.add_argument("--out-csv",    default=None)

    # ── visualize ────────────────────────────────────────────────────────────
    p_viz = sub.add_parser("visualize", help="UMAP of the latent space")
    p_viz.add_argument("--index",       required=True, help="Index .npz from build-index")
    p_viz.add_argument("--out-fig",     required=True, help="Output figure path (.png)")
    p_viz.add_argument("--n-neighbors", type=int, default=30)
    p_viz.add_argument("--min-dist",    type=float, default=0.1)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.cmd == "build-index":
        cmd_build_index(args)
    elif args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "pareto-query":
        cmd_pareto_query(args)
    elif args.cmd == "visualize":
        cmd_visualize(args)


if __name__ == "__main__":
    main()
