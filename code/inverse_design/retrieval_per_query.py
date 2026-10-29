"""
retrieval_per_query.py
======================
Per-query retrieval evaluation, replacing the pooled marginal-KL measure.

WHAT WAS WRONG WITH MARGINAL KL
-------------------------------
compare_kl_v3.py does `flat_idx = matches.ravel()` and then compares the
distribution of retrieved property values against the distribution of test
property values.  Flattening throws away which neighbour was retrieved for
which query.  Take any set of matches, permute the rows so every query gets
some other query's neighbours, and the KL is unchanged -- the retrieval is now
worthless and the metric cannot see it.  Marginal KL asks "does the retrieved
pool look like the test set overall", when the claim being made is "structures
near query q in embedding space have properties near q's".

It also bootstraps by resampling the flattened neighbour pool, which treats
n_queries x K neighbours as independent draws.  They are not: the K neighbours
of one query are highly dependent.  That makes the error bars far too narrow.

WHAT THIS COMPUTES INSTEAD
--------------------------
For each query q with measured value y_q and retrieved neighbours N(q):

    yhat_q = mean over N(q) of the neighbour's measured property

then over the queries:

    MAE       = mean_q |y_q - yhat_q|         (lower better)
    Spearman  = rank correlation of y_q vs yhat_q across queries
                                              (higher better)

Both are per-query by construction: permuting the rows destroys them, as it
should.  Confidence intervals come from bootstrapping the QUERIES -- the
independent unit, and there are 250 of them -- not the pooled neighbours.

THREE REFERENCE LEVELS
----------------------
A per-query MAE is only interpretable against what you would get without a
useful embedding, so every property reports three:

  model     the embedding under test
  random    random neighbours from the same pool, same K.  The honest null:
            it carries the pool's marginal distribution but no query-specific
            information.
  constant  predict the pool mean for every query.  This is what random
            retrieval converges to as K grows, and a model sitting at this
            level has an embedding that says nothing about the property.

`skill` = 1 - MAE_model / MAE_random.  Positive means better than random
retrieval; zero or below means the embedding is not helping for that property.

WHY THIS MATTERS FOR BULK MODULUS
---------------------------------
If a method's MAE lands at the constant-predictor level, its neighbour means
have collapsed towards the global mean -- retrieval is returning structures
whose modulus is uninformative about the query's.  The scatter plots this
script writes (y_q on x, yhat_q on y) show that directly: an uninformative
embedding gives a horizontal band at the pool mean rather than a diagonal.
That is the diagnostic to look at when a property loses to random.

USAGE
-----
Same directory arguments as compare_kl_v3.py, so existing command lines carry
over:

    python inverse_design/retrieval_per_query.py \\
        --test-lmdb data/mp20_5k/test/data.lmdb \\
        --random-dirs        a,b,c \\
        --single-eform-dirs  a,b,c \\
        --single-bandgap-dirs a,b,c \\
        --single-gvrh-dirs   a,b,c \\
        --multi-dirs         a,b,c \\
        --out-dir figures/retrieval_per_query

Extra arms (e.g. the regression-head retrieval, once it exists) need no code
change:

    --add-method 'erm_multi:ERM-Multi:matches_v3.npy:d0,d1,d2'
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _deps():
    """Imported lazily: these pull in torch/lmdb, which a replot does not need."""
    from random_retrieval_baseline import load_test_properties, PANELS  # noqa: E402
    from embed_and_match_alex import _load_cache  # noqa: E402
    return load_test_properties, PANELS, _load_cache


# (display label, match filename, alex-cache source) -- ids and filenames match
# compare_kl_v3.py so the same seed directories work unchanged.
#
# DISPLAY NAMES follow the manuscript's convention (main.tex, Introduction):
# the head prefix is MEX or ERM, and what follows says what the arm saw during
# training. A single-property arm is really three separately trained models, so
# in this evaluation -- where the distinction matters -- each is named by its
# own property: MEX-E_form, MEX-E_g, MEX-K_VRH, and likewise ERM-*.  The
# jointly trained arms stay MEX-Multi and ERM-Multi.  Never print the internal
# ids (single_gvrh, multi, ...) on an axis; that is what ARM_LABELS is for.
METHOD_SPECS = {
    "random":          ("Random", "random_matches.npy", "multi"),
    "single_eform":    (r"MEX-$E_\mathrm{form}$", "matches_eform.npy", "self"),
    "single_bandgap":  (r"MEX-$E_g$", "matches_bandgap.npy", "self"),
    "single_gvrh":     (r"MEX-$K_\mathrm{VRH}$", "matches_gvrh.npy", "self"),
    "multi":           ("MEX-Multi", "matches_v3.npy", "self"),
}
METHOD_ORDER = ["random", "single_eform", "single_bandgap", "single_gvrh", "multi"]

# The match FILENAMES are the same for the energy (MEX) and regression (ERM)
# arms -- only the directories and the display names differ -- so --arm just
# relabels. Keeps the two comparisons separate rather than mixing eight arms
# into one panel.  Both arms are listed in full rather than relying on the
# METHOD_SPECS defaults, so that renaming one arm cannot silently leave the
# other pointing at a stale label.
ARM_LABELS = {
    "mex": {"random":         "Random",
            "single_eform":   r"MEX-$E_\mathrm{form}$",
            "single_bandgap": r"MEX-$E_g$",
            "single_gvrh":    r"MEX-$K_\mathrm{VRH}$",
            "multi":          "MEX-Multi"},
    "erm": {"random":         "Random",
            "single_eform":   r"ERM-$E_\mathrm{form}$",
            "single_bandgap": r"ERM-$E_g$",
            "single_gvrh":    r"ERM-$K_\mathrm{VRH}$",
            "multi":          "ERM-Multi"},
}
METHOD_COLORS = {
    "random": "#8f8d86",
    "single_eform": "#2a78d6",
    "single_bandgap": "#eb6834",
    "single_gvrh": "#1baf7a",
    "multi": "#4a3aa7",
}
_EXTRA_COLORS = ["#c2185b", "#0f7d78", "#8a5a00", "#5a3fa0"]

PROP_LABELS = {
    "y_eform": r"$E_\mathrm{form}$ (eV/atom)",
    "y_bandgap": r"$E_g$ (eV)",
    "y_gvrh": r"$K_\mathrm{VRH}$ ($\log_{10}$ GPa)",
}

_INK = "#0b0b0b"
_GRID = "#e1e0d9"
_SURFACE = "#ffffff"


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 1.6,
        "figure.facecolor": _SURFACE,
        "axes.facecolor": _SURFACE,
        "savefig.facecolor": _SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_method(method_id, spec, seed_dirs, multi_dirs, random_alex_cache):
    """
    Returns (per_seed_matches, alex_data).

    per_seed_matches is a list of (n_test, K) arrays, one per seed, kept
    SEPARATE rather than concatenated. compare_kl_v3 concatenates along K,
    which is fine for a pooled measure but hides seed-to-seed variation in a
    per-query one. Both are reported here: each seed on its own, and the
    ensemble formed by pooling all seeds' neighbours per query.
    """
    _, _, _load_cache = _deps()
    _, match_filename, cache_source = spec

    per_seed, alex_data = [], None
    for d in seed_dirs:
        d = Path(d)
        mpath = d / match_filename
        if not mpath.exists():
            logging.warning(f"[{method_id}] missing {mpath} - skipping this seed")
            continue
        if alex_data is None:
            cpath = d / "alex_embeddings_dedup.npz"
            if cpath.exists():
                alex_data = _load_cache(cpath)
        per_seed.append(np.load(str(mpath)))
        logging.info(f"[{method_id}] loaded {mpath.parent.name} shape {per_seed[-1].shape}")

    if not per_seed:
        return None, None

    if alex_data is None:
        if method_id == "random" and random_alex_cache:
            alex_data = _load_cache(Path(random_alex_cache))
        elif cache_source == "multi" and multi_dirs:
            for d in multi_dirs:
                cpath = Path(d) / "alex_embeddings_dedup.npz"
                if cpath.exists():
                    alex_data = _load_cache(cpath)
                    break
    if alex_data is None:
        logging.warning(f"[{method_id}] no alex cache found - skipping")
        return None, None
    return per_seed, alex_data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(a, b):
    """Rank correlation without a scipy dependency; ties get average ranks."""
    def ranks(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        # average ranks within tied groups, which matters for band gap: a large
        # block of structures sits at exactly 0 eV.
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(np.arange(i, j + 1))
            i = j + 1
        return r
    ra, rb = ranks(np.asarray(a, float)), np.asarray(ranks(np.asarray(b, float)))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def neighbour_means(matches, alex_prop):
    """
    yhat_q = mean measured property over the neighbours retrieved for query q.

    NaN neighbours are ignored per query rather than poisoning the whole row;
    a query with no finite neighbour yields NaN and is dropped downstream.
    """
    vals = np.asarray(alex_prop, float)[matches]          # (n_test, K)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(vals)
        counts = finite.sum(axis=1)
        sums = np.where(finite, vals, 0.0).sum(axis=1)
        out = np.full(len(vals), np.nan)
        nz = counts > 0
        out[nz] = sums[nz] / counts[nz]
    return out


def metrics_with_ci(y, yhat, n_boot, rng):
    """
    MAE and Spearman with CIs from bootstrapping the QUERIES.

    The query is the independent unit. Resampling the pooled neighbours -- what
    the KL version did -- treats the K neighbours of one query as K independent
    observations and produces error bars several times too narrow.
    """
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    n = len(y)
    if n < 3:
        nanres = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        return {"n": n, "mae": dict(nanres), "spearman": dict(nanres)}

    point_mae = float(np.mean(np.abs(y - yhat)))
    point_rho = _spearman(y, yhat)

    maes, rhos = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        maes.append(np.mean(np.abs(y[idx] - yhat[idx])))
        rhos.append(_spearman(y[idx], yhat[idx]))
    maes = np.array(maes)
    rhos = np.array(rhos)
    rhos = rhos[np.isfinite(rhos)]

    return {
        "n": n,
        "mae": {"mean": point_mae,
                "lo": float(np.percentile(maes, 2.5)),
                "hi": float(np.percentile(maes, 97.5))},
        "spearman": {"mean": point_rho,
                     "lo": float(np.percentile(rhos, 2.5)) if len(rhos) else float("nan"),
                     "hi": float(np.percentile(rhos, 97.5)) if len(rhos) else float("nan")},
    }


def constant_baseline(y, pool_prop):
    """MAE of predicting the alex-pool mean for every query.

    This is the level random retrieval converges to as K grows. A method that
    lands here has an embedding carrying no information about the property.
    """
    ok = np.isfinite(y)
    pool = np.asarray(pool_prop, float)
    mu = float(np.nanmean(pool)) if np.isfinite(pool).any() else float("nan")
    return float(np.mean(np.abs(y[ok] - mu))), mu


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _bar_panel(ax, results, prop, metric, methods, labels, colors,
               ref_lines, ylabel, show_ylabel=True):
    xs = np.arange(len(methods))
    means = [results[m][prop][metric]["mean"] for m in methods]
    los = [results[m][prop][metric]["lo"] for m in methods]
    his = [results[m][prop][metric]["hi"] for m in methods]
    err = np.abs(np.vstack([np.array(means) - np.array(los),
                            np.array(his) - np.array(means)]))

    ax.bar(xs, means, color=[colors[m] for m in methods], width=0.68,
           edgecolor="none", zorder=2)
    ax.errorbar(xs, means, yerr=err, fmt="none", ecolor=_INK,
                elinewidth=1.6, capsize=4, zorder=3)

    # Reserve an empty strip on the right so the reference-line labels sit in
    # clear space instead of on top of the tallest bar.
    ax.set_xlim(-0.65, len(methods) - 1 + 1.25)

    for name, val, style in ref_lines:
        if np.isfinite(val):
            # Draw the line explicitly rather than with axhline so it stops
            # short of its label instead of striking through it.
            ax.plot([-0.65, len(methods) - 1 + 0.40], [val, val],
                    color=_INK, linestyle=style, linewidth=1.3,
                    zorder=1, alpha=0.75, clip_on=False)
            ax.text(len(methods) - 1 + 0.48, val, name,
                    va="center", ha="left", fontsize=9, color=_INK, zorder=4)

    # When every bar is positive the rho=0 line lands on the bottom spine and
    # its label collides with the tick labels; drop the floor a little.
    lo_y, hi_y = ax.get_ylim()
    span = hi_y - lo_y
    for _, val, _ in ref_lines:
        if np.isfinite(val) and val - lo_y < 0.04 * span:
            ax.set_ylim(val - 0.06 * span, hi_y)
            break

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    # Only the leftmost panel carries the label: it is longer than the panel is
    # tall, so repeating it on every panel pushes it past the figure edge and
    # tight_layout clips the ends off.
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(PROP_LABELS.get(prop, prop), fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_bars(results, methods, labels, colors, props, extras, out_png, metric,
              ylabel, random_null=None):
    _style()
    fig, axes = plt.subplots(1, len(props), figsize=(5.2 * len(props), 4.9))
    axes = np.atleast_1d(axes)
    for i, (ax, prop) in enumerate(zip(axes, props)):
        refs = []
        if metric == "mae":
            refs.append(("pool mean", extras[prop]["constant_mae"], "--"))
        else:
            refs.append((r"$\rho = 0$", 0.0, "--"))
        # Random is drawn as a reference line at the multi-draw mean rather than
        # as a bar: a single draw is too noisy to plot beside the trained
        # embeddings (one draw put band-gap rho at 0.168 against a null mean of
        # 0.001), and this keeps the figure consistent with the table.
        if random_null:
            key = prop[2:] if prop.startswith("y_") else prop
            stats = random_null.get("properties", {}).get(key, {})
            val = stats.get("mae_mean") if metric == "mae" else stats.get("spearman_mean")
            if val is not None and np.isfinite(val):
                refs.append(("random", float(val), "-."))
        _bar_panel(ax, results, prop, metric, methods, labels, colors, refs,
                   ylabel, show_ylabel=(i == 0))

    # The dashed line marks the uninformative-embedding level; what that means
    # is explained in the manuscript caption rather than printed on the figure.
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    print(f"  wrote {out_png}  (+ .pdf)")


def plot_scatter(pairs, props, extras, out_png, colors, labels):
    """
    y_q (measured) against yhat_q (neighbour mean), one row per method.

    This is the diagnostic for a property that loses to random: an embedding
    carrying no information about it produces a horizontal band at the pool
    mean instead of a diagonal, and that is visible here in a way no summary
    statistic makes obvious.

    `labels` maps method id -> display name. The row labels must use it: the
    raw ids (single_gvrh, multi, ...) are internal and do not match the names
    used in the manuscript or in the other two figures.
    """
    _style()
    methods = list(pairs.keys())
    fig, axes = plt.subplots(len(methods), len(props),
                             figsize=(4.4 * len(props), 3.9 * len(methods)),
                             squeeze=False)
    for r, m in enumerate(methods):
        for c, prop in enumerate(props):
            ax = axes[r][c]
            y, yhat = pairs[m][prop]
            ok = np.isfinite(y) & np.isfinite(yhat)
            ax.scatter(y[ok], yhat[ok], s=14, alpha=0.55, linewidths=0,
                       color=colors.get(m, "#444444"), zorder=3)
            if ok.any():
                lo = float(min(np.min(y[ok]), np.min(yhat[ok])))
                hi = float(max(np.max(y[ok]), np.max(yhat[ok])))
                ax.plot([lo, hi], [lo, hi], color=_INK, linewidth=1.2,
                        linestyle="-", alpha=0.7, zorder=2)
            mu = extras[prop]["pool_mean"]
            if np.isfinite(mu):
                ax.axhline(mu, color=_INK, linestyle=":", linewidth=1.2,
                           alpha=0.7, zorder=1)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(PROP_LABELS.get(prop, prop), fontsize=12,
                             fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{labels.get(m, m)}\nneighbour mean", fontsize=10,
                              fontweight="bold")
            if r == len(methods) - 1:
                ax.set_xlabel("measured value", fontsize=11, fontweight="bold")
    fig.suptitle("Solid line: perfect retrieval.  Dotted: pool mean "
                 "(where an uninformative embedding collapses).",
                 fontsize=11, y=1.002)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    print(f"  wrote {out_png}  (+ .pdf)")


# ─────────────────────────────────────────────────────────────────────────────

def _split(arg):
    return [s for s in (arg or "").split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-lmdb", required=True)
    ap.add_argument("--random-dirs", default=None)
    ap.add_argument("--single-eform-dirs", default=None)
    ap.add_argument("--single-bandgap-dirs", default=None)
    ap.add_argument("--single-gvrh-dirs", default=None)
    ap.add_argument("--multi-dirs", default=None)
    ap.add_argument("--random-json", default=None,
                    help="random_retrieval_multidraw.json; draws the random null as a "
                         "reference line at its multi-draw mean")
    ap.add_argument("--random-alex-cache", default=None)
    ap.add_argument("--add-method", action="append", default=[],
                    help="extra arm as 'id:label:match_filename:dir1,dir2,...' "
                         "(repeatable; needs no code change for new arms)")
    ap.add_argument("--k", type=int, default=None,
                    help="use only the first K neighbours per query "
                         "(default: all present in the match file)")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="bootstrap resamples OVER QUERIES (default 2000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", choices=sorted(ARM_LABELS), default="mex",
                    help="which comparison this is; relabels the plots and "
                         "tables (default mex). The directories you pass decide "
                         "what is actually read -- this only names it.")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    load_test_properties, PANELS, _ = _deps()
    test_data = load_test_properties(args.test_lmdb)
    props = [p[0] for p in PANELS]
    alex_keys = {p[0]: p[1] for p in PANELS}

    dirs = {
        "random": _split(args.random_dirs),
        "single_eform": _split(args.single_eform_dirs),
        "single_bandgap": _split(args.single_bandgap_dirs),
        "single_gvrh": _split(args.single_gvrh_dirs),
        "multi": _split(args.multi_dirs),
    }
    specs = dict(METHOD_SPECS)
    for mid, label in ARM_LABELS[args.arm].items():
        specs[mid] = (label,) + specs[mid][1:]
    order = list(METHOD_ORDER)
    colors = dict(METHOD_COLORS)
    for i, extra in enumerate(args.add_method):
        try:
            mid, label, fname, dlist = extra.split(":", 3)
        except ValueError:
            ap.error(f"--add-method must be 'id:label:file:dirs', got {extra!r}")
        specs[mid] = (label, fname, "self")
        dirs[mid] = _split(dlist)
        order.append(mid)
        colors.setdefault(mid, _EXTRA_COLORS[i % len(_EXTRA_COLORS)])

    random_null = None
    if args.random_json:
        with open(args.random_json) as fh:
            random_null = json.load(fh)
        logging.info(f"random null loaded from {args.random_json}")
    rng = np.random.default_rng(args.seed)
    results, per_seed_results, pairs, labels = {}, {}, {}, {}
    extras = {p: {} for p in props}
    alex_ref = None

    for mid in order:
        if not dirs.get(mid):
            continue
        per_seed, alex_data = load_method(mid, specs[mid], dirs[mid],
                                          dirs.get("multi"), args.random_alex_cache)
        if per_seed is None:
            continue
        alex_ref = alex_ref or alex_data
        labels[mid] = specs[mid][0]

        # Ensemble = pool every seed's neighbours for the same query. Row
        # correspondence is preserved, unlike the KL version's ravel().
        ens = np.concatenate(per_seed, axis=1)
        if args.k:
            ens = ens[:, :args.k]

        results[mid], pairs[mid], per_seed_results[mid] = {}, {}, {}
        for prop in props:
            akey = alex_keys[prop]
            alex_prop = alex_data.get(akey, np.array([]))
            y = np.asarray(test_data[prop], float).ravel()
            if len(alex_prop) == 0 or ens.max() >= len(alex_prop):
                logging.warning(f"[{mid}] {prop}: alex index out of range - skipping")
                continue
            yhat = neighbour_means(ens, alex_prop)
            results[mid][prop] = metrics_with_ci(y, yhat, args.bootstrap, rng)
            pairs[mid][prop] = (y, yhat)
            per_seed_results[mid][prop] = [
                float(np.nanmean(np.abs(y - neighbour_means(
                    m[:, :args.k] if args.k else m, alex_prop))))
                for m in per_seed
            ]
            if "constant_mae" not in extras[prop]:
                cmae, mu = constant_baseline(y, alex_prop)
                extras[prop]["constant_mae"] = cmae
                extras[prop]["pool_mean"] = mu

    if not results:
        print("No methods loaded. Check the --*-dirs arguments.")
        return 1

    methods = [m for m in order if m in results]

    # skill = 1 - MAE_model / MAE_random, reported only if random was loaded.
    for prop in props:
        rnd = results.get("random", {}).get(prop, {}).get("mae", {}).get("mean")
        extras[prop]["random_mae"] = rnd if rnd is not None else float("nan")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "retrieval_per_query.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "property", "n_queries",
                    "mae", "mae_lo", "mae_hi",
                    "spearman", "spearman_lo", "spearman_hi",
                    "skill_vs_random", "constant_mae", "per_seed_mae"])
        for m in methods:
            for prop in props:
                if prop not in results[m]:
                    continue
                r = results[m][prop]
                rnd = extras[prop]["random_mae"]
                skill = (1 - r["mae"]["mean"] / rnd) if (rnd and np.isfinite(rnd)
                                                         and rnd > 0) else float("nan")
                w.writerow([m, prop, r["n"],
                            f"{r['mae']['mean']:.6f}", f"{r['mae']['lo']:.6f}",
                            f"{r['mae']['hi']:.6f}",
                            f"{r['spearman']['mean']:.6f}", f"{r['spearman']['lo']:.6f}",
                            f"{r['spearman']['hi']:.6f}",
                            f"{skill:.6f}", f"{extras[prop]['constant_mae']:.6f}",
                            " ".join(f"{v:.6f}" for v in per_seed_results[m][prop])])
    print(f"  wrote {csv_path}")

    lbls = [labels[m] for m in methods]
    plot_bars(results, methods, lbls, colors, props, extras,
              args.out_dir / "retrieval_per_query_mae.png", "mae",
              "per-query MAE  (lower better)", random_null=random_null)
    plot_bars(results, methods, lbls, colors, props, extras,
              args.out_dir / "retrieval_per_query_spearman.png", "spearman",
              "Spearman, query vs neighbour mean\n(higher better)", random_null=random_null)
    plot_scatter(pairs, props, extras,
                 args.out_dir / "retrieval_per_query_scatter.png", colors,
                 labels)

    # ── console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PER-QUERY RETRIEVAL  (CIs bootstrap the queries, not the neighbours)")
    print("=" * 78)
    for prop in props:
        print(f"\n{prop}   constant-predictor MAE = {extras[prop]['constant_mae']:.4f}"
              f"   (pool mean {extras[prop]['pool_mean']:.4f})")
        print(f"  {'method':18s} {'MAE [95% CI]':>28s} {'skill':>8s} "
              f"{'Spearman [95% CI]':>28s}")
        for m in methods:
            if prop not in results[m]:
                continue
            r = results[m][prop]
            rnd = extras[prop]["random_mae"]
            skill = (1 - r["mae"]["mean"] / rnd) if (rnd and np.isfinite(rnd)
                                                     and rnd > 0) else float("nan")
            print(f"  {m:18s} {r['mae']['mean']:8.4f} "
                  f"[{r['mae']['lo']:.4f}, {r['mae']['hi']:.4f}] "
                  f"{skill:8.3f} "
                  f"{r['spearman']['mean']:9.4f} "
                  f"[{r['spearman']['lo']:.4f}, {r['spearman']['hi']:.4f}]")
        # Flag the failure mode explicitly rather than leaving it to be noticed.
        for m in methods:
            if m == "random" or prop not in results[m]:
                continue
            mae = results[m][prop]["mae"]["mean"]
            if np.isfinite(extras[prop]["random_mae"]) and mae >= extras[prop]["random_mae"]:
                print(f"    NOTE: {m} is NOT better than random retrieval here.")
            if mae >= extras[prop]["constant_mae"]:
                print(f"    NOTE: {m} is at or worse than predicting the pool mean "
                      f"-- the embedding carries no usable signal for this property.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
