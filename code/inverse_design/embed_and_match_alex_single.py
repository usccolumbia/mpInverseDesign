"""
embed_and_match_alex_single.py
==============================
Retrieval-based inverse design for a SINGLE-PROPERTY MEX model.

Mirrors embed_and_match_alex_v3.py but works with the mex_trainer
(equiformer_v2-mex) and single-property LMDBs that store the target
property as 'y' (not y_eform / y_bandgap / y_gvrh).

Architecture:
    - embed_test_set   reads `y` from single-prop LMDB
    - 1-panel histogram: test y  vs  top-K matched alex property
    - CSV match table with the relevant alex column

Usage
-----
python inverse_design/embed_and_match_alex_single.py \\
    --checkpoint  outputs/mp20_1k_eform_seed0/checkpoints/best_checkpoint.pt \\
    --config      configs/mex-single/mp20_1k_eform.yml \\
    --prop        eform \\
    --test-lmdb   data/mp20_1k_eform/test/data.lmdb \\
    --train-lmdb  data/mp20_1k_eform/train/data.lmdb \\
    --val-lmdb    data/mp20_1k_eform/val/data.lmdb \\
    --alex-train  data/alex_mp20_train.csv \\
    --alex-val    data/alex_mp20_val.csv \\
    --all-csv     data/all_with_180.csv \\
    --out-dir     outputs/alex_match_1k_eform_seed0 \\
    --top-k       10 \\
    --device      cpu
"""

import argparse
import csv
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ── make ocp importable ───────────────────────────────────────────────────────
CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent          # configs/ and data/ live here, ocp/ under code/
sys.path.insert(0, str(CODE_ROOT / "ocp"))

# ── purge stale ocpmodels from sys.modules ────────────────────────────────────
for _mod in list(sys.modules):
    if _mod == "ocpmodels" or _mod.startswith("ocpmodels."):
        del sys.modules[_mod]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_inv_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_inv_dir))

# ── reuse utilities from sibling scripts ──────────────────────────────────────
from embed_and_match_alex import (          # noqa: E402
    read_lmdb, embed_records, collate,
    _save_cache, _load_cache,
)
from embed_and_match_alex_v2 import (       # noqa: E402
    collect_split_material_ids,
    load_and_embed_alex,
    top_k_matches,
)

# ─────────────────────────────────────────────────────────────────────────────
# Property metadata: maps --prop name → (alex column, axis label, x range)
# ─────────────────────────────────────────────────────────────────────────────

PROP_META = {
    "eform": {
        "alex_col":   "formation_energy_per_atom",
        "multi_attr": "y_eform",
        "xlabel":     "Formation Energy (eV/atom)\nTest y  |  Alex: formation_energy_per_atom",
        "xmin": -5.0, "xmax": 1.0, "nbins": 50,
    },
    "bandgap": {
        "alex_col":   "dft_band_gap",
        "multi_attr": "y_bandgap",
        "xlabel":     "Band Gap (eV)\nTest y  |  Alex: dft_band_gap",
        "xmin": 0.0, "xmax": 10.0, "nbins": 50,
    },
    "gvrh": {
        "alex_col":   "log10_bulk_modulus",
        "multi_attr": "y_gvrh",
        "xlabel":     "log₁₀ Bulk Modulus (log₁₀ GPa)\nTest y  |  Alex: log10_bulk_modulus",
        "xmin": -1.0, "xmax": 3.5, "nbins": 50,
    },
}

# Ordered list used for the 3-panel histogram
ALL_PROPS = ["eform", "bandgap", "gvrh"]

# ─────────────────────────────────────────────────────────────────────────────
# Single-property inference engine
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Resolve a config and its `includes:` chain into one flat dict.

    Includes are resolved RECURSIVELY. A per-property config such as
    configs/mex-single/mp20_5k_gvrh.yml carries no `model:` block of its own --
    it inherits one from equiformer_v2/base_ebm.yml. If a config includes
    another config that in turn includes the base, a single-level resolver
    silently returns no `model:` block at all, EquiformerV2MEX falls back to its
    class defaults (lmax 6, ffn 512, no grid_resolution) and loading a
    checkpoint trained at lmax 4 / ffn 128 / grid 18 dies in a wall of size
    mismatches. Resolving the whole chain avoids that.
    """
    import re
    import yaml

    def _resolve(cfg_path: Path, seen: set) -> dict:
        cfg_path = cfg_path.resolve()
        if cfg_path in seen:          # guard against an include cycle
            return {}
        seen.add(cfg_path)

        raw = cfg_path.read_text()
        merged: dict = {}

        # Collect the `includes:` list items only: every line after `includes:`
        # that is blank, a comment, or a `-` item, stopping at the next
        # top-level key.
        in_block = False
        for line in raw.splitlines():
            if not in_block:
                in_block = line.strip() == "includes:" and not line[:1].isspace()
                continue
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if not line.lstrip().startswith("-"):
                break
            m = re.match(r'\s*-\s+(.+\.yml)\s*$', line)
            if not m:
                continue
            inc_path = m.group(1).strip().strip('"\'')
            for cand in (cfg_path.parent.parent.parent / inc_path,
                         REPO_ROOT / inc_path,
                         Path(inc_path)):
                if cand.exists():
                    merged.update(_resolve(cand, seen))
                    break
            else:
                raise FileNotFoundError(
                    f"{cfg_path}: cannot resolve include {inc_path!r}"
                )

        own = yaml.safe_load(raw) or {}
        own.pop("includes", None)
        merged.update(own)
        return merged

    cfg = _resolve(Path(path), set())

    # Without this the model is built from class defaults and the mismatch only
    # surfaces later, as hundreds of size-mismatch lines.
    if not cfg.get("model"):
        raise ValueError(
            f"{path} resolved to a config with no 'model:' block. The model "
            f"hyperparameters normally come from an included base config "
            f"(equiformer_v2/base_ebm.yml); check the `includes:` paths."
        )
    return cfg


class MEXSingleInferenceEngine:
    """
    Lightweight wrapper for a single-property MEX model (equiformer_v2-mex).
    Does NOT require property_names in config — reads 'y' from LMDB.
    """

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cpu"):
        from ocpmodels.common.utils import load_state_dict
        from ocpmodels.modules.normalizer import Normalizer
        from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2MEX

        self.device = torch.device(device)
        self.config = load_config(config_path)

        # Normalizer
        norm_cfg = (
            self.config.get("dataset", {})
            .get("train", {})
            .get("transforms", {})
            .get("normalizer", {})
            .get("y", {})
        )
        from ocpmodels.modules.normalizer import Normalizer
        self.normalizer = Normalizer(
            mean=norm_cfg.get("mean", 0.0),
            std=norm_cfg.get("stdev", 1.0),
            device=str(self.device),
        )

        # Model. Chosen by the config's model.name rather than hardcoded, so
        # the plain-regression (ERM) single-property arm works here too:
        # embedding only calls get_mol_rep, which both classes inherit from
        # EquiformerV2_OC20, and the energy head plays no part in retrieval.
        from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2Direct
        _CLASSES = {
            "equiformer_v2-mex":    EquiformerV2MEX,
            "equiformer_v2_mex":    EquiformerV2MEX,
            "equiformer_v2-direct": EquiformerV2Direct,
            "equiformer_v2_direct": EquiformerV2Direct,
        }
        model_name = self.config.get("model", {}).get("name", "equiformer_v2-mex")
        if model_name not in _CLASSES:
            raise ValueError(f"Unknown single-property model '{model_name}'. "
                             f"Supported: {sorted(_CLASSES)}")
        model_cls = _CLASSES[model_name]
        model_cfg = {k: v for k, v in self.config.get("model", {}).items() if k != "name"}
        self.model = model_cls(None, None, 1, **model_cfg).to(self.device)
        self.model.eval()

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt["state_dict"]
        n_module = next(iter(state)).count("module.")
        if n_module:
            state = {k[len("module.") * n_module:]: v for k, v in state.items()}
        load_state_dict(self.model, state, strict=False)

        if "normalizers" in ckpt and ckpt["normalizers"]:
            for k, sd in ckpt["normalizers"].items():
                if k == "y":
                    self.normalizer.load_state_dict(sd)

        if "ema" in ckpt and ckpt["ema"] is not None:
            ema_state = ckpt["ema"]["shadow_params"]
            ema_keys  = list(self.model.state_dict().keys())
            ema_dict  = {k: v for k, v in zip(ema_keys, ema_state)}
            self.model.load_state_dict(ema_dict, strict=False)

        logging.info(f"Loaded {model_cls.__name__} ({self.model.num_params:,} params)")

    @torch.no_grad()
    def embed_batch(self, batch) -> torch.Tensor:
        batch = batch.to(self.device)
        return self.model.get_mol_rep(batch)   # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Test-set embedding (reads 'y', not y_eform/y_bandgap/y_gvrh)
# ─────────────────────────────────────────────────────────────────────────────

def embed_test_set_single(engine: MEXSingleInferenceEngine,
                          lmdb_path: str, batch_size: int) -> dict:
    logging.info(f"Reading test LMDB: {lmdb_path}")
    records = read_lmdb(lmdb_path)
    logging.info(f"  {len(records)} test structures")
    logging.info("Embedding test structures …")
    embs = embed_records(engine, records, batch_size)

    def _mid(r):
        if hasattr(r, "material_id"):
            v = r.material_id
            return v[0] if isinstance(v, list) else str(v)
        return str(int(r.sid))

    y_vals = np.array([float(r.y) for r in records], dtype=np.float32)

    return {
        "embeddings":  embs,
        "y":           y_vals,
        "material_id": [_mid(r) for r in records],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load all-three-property test values from multi-property LMDB
# ─────────────────────────────────────────────────────────────────────────────

def load_multi_test_props(multi_test_lmdb: str) -> dict:
    """
    Read y_eform, y_bandgap, y_gvrh from a multi-property LMDB.
    Returns dict {prop_name: np.ndarray}.
    """
    records = read_lmdb(multi_test_lmdb)
    result = {}
    for prop, meta in PROP_META.items():
        attr = meta["multi_attr"]
        try:
            result[prop] = np.array([float(getattr(r, attr)) for r in records],
                                    dtype=np.float32)
        except AttributeError:
            result[prop] = None
            logging.warning(f"multi-test LMDB has no attribute '{attr}' — panel will show matched only")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3-panel histogram: all properties, trained prop highlighted
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_vs_matched(
    test_data: dict,
    alex_data: dict,
    matches: np.ndarray,         # (N_test, K)
    trained_prop: str,           # which property the model was trained on
    out_dir: Path,
    multi_props: dict = None,    # {prop: np.ndarray} from multi-prop LMDB, or None
):
    """
    3-panel histogram (eform | bandgap | gvrh).

    Blue bars (test set):
      - trained_prop panel: always available via test_data["y"]
      - other panels: shown only if multi_props is provided
    Orange bars (matched alex top-K): always shown for each property.
    The panel corresponding to the trained property has a bold title marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = matches.shape[1]
    flat_idx = matches.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    summary = {}

    for ax, prop in zip(axes, ALL_PROPS):
        meta = PROP_META[prop]
        bins = np.linspace(meta["xmin"], meta["xmax"], meta["nbins"] + 1)

        # ── matched alex values ───────────────────────────────────────────────
        alex_prop    = alex_data[meta["alex_col"]]
        matched_prop = alex_prop[flat_idx]
        matched_vals = matched_prop[~np.isnan(matched_prop)]

        # ── test values ───────────────────────────────────────────────────────
        if prop == trained_prop:
            test_vals = test_data["y"]
        elif multi_props is not None and multi_props.get(prop) is not None:
            test_vals = multi_props[prop]
        else:
            test_vals = None

        if test_vals is not None:
            ax.hist(test_vals, bins=bins, density=True,
                    color="steelblue", alpha=0.55,
                    label=f"Test set ({len(test_vals):,})")

        if len(matched_vals) > 0:
            ax.hist(matched_vals, bins=bins, density=True,
                    color="darkorange", alpha=0.70,
                    label=f"Matched alex top-{K} ({len(matched_vals):,})")
        else:
            ax.text(0.5, 0.5, "No matched alex\nstructures with\nthis property",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="darkorange")

        # Bold border for the trained property panel
        if prop == trained_prop:
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
                spine.set_edgecolor("black")

        # Label: mark trained property
        xlabel = meta["xlabel"]
        if prop == trained_prop:
            xlabel = f"★ {xlabel}  [trained]"
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(meta["xmin"], meta["xmax"])
        ax.legend(fontsize=8)

        summary[prop] = {
            "test_mean":    float(np.mean(test_vals))       if test_vals is not None else float("nan"),
            "test_std":     float(np.std(test_vals))        if test_vals is not None else float("nan"),
            "matched_n":    len(matched_vals),
            "matched_mean": float(np.nanmean(matched_vals)) if len(matched_vals) > 0 else float("nan"),
            "matched_std":  float(np.nanstd(matched_vals))  if len(matched_vals) > 0 else float("nan"),
        }

    fig.suptitle(
        f"Test set  vs  top-{K} alex_mp20 neighbours\n"
        f"(cosine similarity in MEX-Single [{trained_prop}] embedding space  ·  ★ = trained property)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    out_path = out_dir / f"test_vs_matched_alex_{trained_prop}_all_props.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Histogram saved to {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CSV match table
# ─────────────────────────────────────────────────────────────────────────────

def save_match_table(
    test_data: dict,
    alex_data: dict,
    matches: np.ndarray,
    trained_prop: str,
    out_dir: Path,
    multi_props: dict = None,
):
    """CSV with one row per (test, matched-alex) pair, all three alex properties included."""
    K = matches.shape[1]
    csv_path = out_dir / f"test_alex_match_table_{trained_prop}.csv"

    def _fmt(v):
        return "" if (isinstance(v, float) and np.isnan(v)) else round(float(v), 6)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # Header: include all three test props where available
        header = ["test_material_id", f"test_y_{trained_prop}"]
        for p in ALL_PROPS:
            if p != trained_prop:
                if multi_props is not None and multi_props.get(p) is not None:
                    header.append(f"test_y_{p}")
        header += [
            "match_rank", "alex_material_id",
            "alex_formation_energy_per_atom",
            "alex_dft_band_gap",
            "alex_log10_bulk_modulus",
            "alex_energy_above_hull_eV_atom",
        ]
        w.writerow(header)

        for q_i, mid in enumerate(test_data["material_id"]):
            for rank, a_i in enumerate(matches[q_i], start=1):
                row = [mid, round(float(test_data["y"][q_i]), 6)]
                for p in ALL_PROPS:
                    if p != trained_prop:
                        if multi_props is not None and multi_props.get(p) is not None:
                            row.append(round(float(multi_props[p][q_i]), 6))
                row += [
                    rank,
                    alex_data["material_id"][a_i],
                    _fmt(alex_data["formation_energy_per_atom"][a_i]),
                    _fmt(alex_data["dft_band_gap"][a_i]),
                    _fmt(alex_data["log10_bulk_modulus"][a_i]),
                    _fmt(alex_data["energy_above_hull"][a_i]),
                ]
                w.writerow(row)
    logging.info(f"Match table saved to {csv_path}  ({len(test_data['material_id']) * K} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inverse design matching for single-property MEX models."
    )
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--prop",        required=True, choices=list(PROP_META),
                   help="Which property this model was trained on (eform|bandgap|gvrh)")
    p.add_argument("--test-lmdb",        required=True)
    p.add_argument("--train-lmdb",       required=True)
    p.add_argument("--val-lmdb",         required=True)
    p.add_argument("--multi-test-lmdb",  default=None,
                   help="Multi-property LMDB (e.g. data/mp20_1k/test/data.lmdb) containing "
                        "y_eform, y_bandgap, y_gvrh for all test structures. "
                        "Enables test-set blue bars in the two non-trained property panels.")
    p.add_argument("--alex-train",       default="data/alex_mp20_train.csv")
    p.add_argument("--alex-val",         default="data/alex_mp20_val.csv")
    p.add_argument("--all-csv",          default="data/all_with_180.csv")
    p.add_argument("--out-dir",          default="outputs/alex_match_single")
    p.add_argument("--cache-dir",        default=None,
                   help="Look for pre-built embeddings here; write new ones to --out-dir")
    p.add_argument("--top-k",            type=int, default=10)
    p.add_argument("--batch-size",       type=int, default=8)
    p.add_argument("--device",           default="cpu")
    p.add_argument("--no-cache",         action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir   = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    test_cache_read  = cache_dir / "test_embeddings.npz"
    alex_cache_read  = cache_dir / "alex_embeddings_dedup.npz"
    test_cache_write = out_dir   / "test_embeddings.npz"
    alex_cache_write = out_dir   / "alex_embeddings_dedup.npz"

    engine = MEXSingleInferenceEngine(args.checkpoint, args.config, device=args.device)

    split_mids = collect_split_material_ids(
        args.train_lmdb, args.val_lmdb, args.test_lmdb
    )

    if not args.no_cache and test_cache_read.exists():
        logging.info(f"Loading test embeddings from cache: {test_cache_read}")
        test_data = _load_cache(test_cache_read)
    else:
        logging.info("Computing test set embeddings …")
        test_data = embed_test_set_single(engine, args.test_lmdb, args.batch_size)
        _save_cache(test_cache_write, test_data)

    if not args.no_cache and alex_cache_read.exists():
        logging.info(f"Loading alex embeddings from cache: {alex_cache_read}")
        alex_data = _load_cache(alex_cache_read)
    else:
        logging.info("Computing alex_mp20 embeddings …")
        alex_data = load_and_embed_alex(
            engine, args.alex_train, args.alex_val,
            split_mids, args.batch_size,
            all_csv=args.all_csv,
        )
        _save_cache(alex_cache_write, alex_data)

    matches = top_k_matches(
        test_data["embeddings"],
        alex_data["embeddings"],
        top_k=args.top_k,
    )
    np.save(str(out_dir / f"matches_{args.prop}.npy"), matches)

    # ── load all-three-property test values if multi-lmdb is provided ─────────
    multi_props = None
    if args.multi_test_lmdb:
        logging.info(f"Loading multi-property test values from {args.multi_test_lmdb}")
        multi_props = load_multi_test_props(args.multi_test_lmdb)

    summary = plot_test_vs_matched(
        test_data, alex_data, matches, args.prop, out_dir, multi_props=multi_props
    )
    save_match_table(
        test_data, alex_data, matches, args.prop, out_dir, multi_props=multi_props
    )

    logging.info("\n=== Property distribution summary ===")
    for prop in ALL_PROPS:
        s = summary[prop]
        trained_marker = " [★ trained]" if prop == args.prop else ""
        logging.info(
            f"  {prop:<8}{trained_marker:<14}  "
            f"test: {s['test_mean']:.3f}±{s['test_std']:.3f}  |  "
            f"matched alex: {s['matched_mean']:.3f}±{s['matched_std']:.3f}  (n={s['matched_n']:,})"
        )


if __name__ == "__main__":
    main()
