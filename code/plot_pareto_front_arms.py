"""
plot_pareto_front_arms.py
=========================
Pareto-front comparison figures and statistics for the 2x2 of arms:

    MEX-Single    3 energy-head models, one per property
    MEX-Multi     1 model, 3 energy heads
    ERM-Single    3 regression models, one per property
    ERM-Multi     1 model, 3 regression heads

Two comparison SETS are produced by default, each isolating the effect of joint
training within one model family:

    --set mex     MEX-Single  vs  MEX-Multi
    --set erm     ERM-Single  vs  ERM-Multi

Each set gives one figure: 3 rows (seeds) x 3 columns (pairwise projections of
the joint eform / band gap / K_VRH space), with the true DFT front and both
arms' model fronts overlaid; plus a per-seed statistics table written as CSV
and as LaTeX.

WHY MODEL FRONTS ARE PLOTTED AT DFT COORDINATES
-----------------------------------------------
The existing figure plots the model front in PREDICTED coordinates and the true
front in DFT coordinates on the same axes.  That makes recall impossible to
read: a structure the model correctly puts on its front appears at its
predicted position, not on top of the orange true-front point it corresponds
to, so "no blue point near that orange one" looks like a coverage failure when
it is nothing of the sort.

Here the default (--coords dft) places every model-front structure at its
MEASURED property vector.  A recovered structure then lands exactly on its
true-front point, so recall is visible directly: filled marker inside an open
orange circle = recovered; open orange circle alone = missed.  Front membership
is still decided in predicted space -- only the plotting position changes.

Pass --coords pred to reproduce the old convention.

--coords IS A PLOTTING FLAG AND NOTHING ELSE.  Every metric here -- recall,
precision, enrichment, IGD, IGD+, hypervolume -- is computed at DFT
coordinates whatever this flag says, because the question the paper asks is
"you proposed these structures, how good did they actually turn out".  Scoring
a front at its predicted coordinates instead would reward a model for being
optimistic about its own selections, which is precisely the pathology a design
evaluation has to exclude.  So a --coords pred run produces a different figure
and identical numbers, and for that reason it writes no CSV or LaTeX table --
two stats files with the same contents under different names are an invitation
to cite them as two results.

WHAT IS AND IS NOT HELD FIXED ACROSS ARMS
-----------------------------------------
The MEX arms reach their point estimates by derivative-free search (S=2000
candidates, T=5 iterations, noise 0.5 shrinking x0.5, bounds anchored to
training-set extremes); those settings are shared between the two MEX arms.
The regression arms have no search stage at all -- one forward pass, read the
heads.  The four arms therefore cannot share DFO settings, because two of them
have no DFO, and any writeup of these figures has to say that the regression
arms use a different inference procedure, not merely a different read-out head.

What IS identical across all four: the same 250 test structures, one
point-estimate vector per structure, the same non-dominated sorting code, the
same objective directions, and the same DFT-derived true front.

READING THE NUMBERS
-------------------
The statistics tables give per-seed counts first and mean +/- std second.  At an
18-structure true front one recovered structure is 5.6 points of recall, and
three seeds resolve nothing finer than that, so a one- or two-structure gap
between arms is noise.  The tables are built to make that legible rather than
to hide it behind a mean.

DOMINATION LOGIC AND LOADERS
----------------------------
Imported from plot_pareto_front_structurelevel.py and
compare_pareto_erm_vs_mex.py rather than reimplemented, so the figures, the
tables and the command-line comparison can never disagree about what
"non-dominated" means or about how a 3-file single-property arm is merged.

USAGE
-----
    python plot_pareto_front_arms.py \\
        --mex-single-root  /path/to/single_checkout/outputs \\
        --erm-single-root  /path/to/single_checkout/outputs \\
        --mex-multi-root   /path/to/multi_checkout/outputs \\
        --erm-multi-root   /path/to/multi_checkout/outputs \\
        --out-dir figures --seeds 0 1 2
"""

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Single source of truth for domination, property keys and axis labels.
# non_dominated_front and PROP_NAMES are load-bearing: if they are missing, the
# module on the path is not the one the figures are built from and there is no
# safe fallback, so fail loudly with an actionable message rather than quietly
# reimplementing domination here.
try:
    from plot_pareto_front_structurelevel import (  # noqa: E402
        PROP_NAMES,
        non_dominated_front,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"Cannot import the domination logic: {exc}\n"
        "plot_pareto_front_structurelevel.py must be importable from the "
        "working directory (it is untracked, so check it was actually copied "
        "into this checkout)."
    ) from exc

# Everything below is cosmetic. Older copies of
# plot_pareto_front_structurelevel.py predate PROP_LABELS and some of the
# palette constants, and a stale copy on a cluster should degrade to default
# styling rather than abort a 20-minute job at import time.
try:
    from plot_pareto_front_structurelevel import PROP_LABELS  # noqa: E402
except ImportError:
    PROP_LABELS = [
        r"$E_\mathrm{form}$ (eV/atom)",
        r"$E_g$ (eV)",
        r"$K_\mathrm{VRH}$ (log$_{10}$ GPa)",
    ]
    print("note: PROP_LABELS not found in plot_pareto_front_structurelevel.py "
          "(stale copy?); using built-in axis labels.")

try:
    from plot_pareto_front_structurelevel import (  # noqa: E402
        _apply_pub_style,
        _COLOR_GROUND_TRUTH,
        _GRIDLINE,
        _INK_PRIMARY,
        _SURFACE,
    )
except ImportError:
    _COLOR_GROUND_TRUTH, _GRIDLINE = "#eb6834", "#e1e0d9"
    _INK_PRIMARY, _SURFACE = "#0b0b0b", "#ffffff"

    def _apply_pub_style():
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.linewidth": 1.8,
            "figure.facecolor": _SURFACE,
            "axes.facecolor": _SURFACE,
            "savefig.facecolor": _SURFACE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    print("note: style constants not found in plot_pareto_front_structurelevel.py "
          "(stale copy?); using built-in styling.")

# Single source of truth for loading and for the recovery metrics.
from compare_pareto_erm_vs_mex import (  # noqa: E402
    apply_directions,
    find_erm,
    find_mex,
    front_metrics,
    load_erm_npzs,
    load_multi_npz,
)

# ─────────────────────────────────────────────────────────────────────────────
# Arms
#
# kind "triple" -> one npz per property, merged by 'ids' (load_erm_npzs)
# kind "single" -> one npz holding all three targets (load_multi_npz)
#
# default_pattern is a glob relative to that arm's root; {prop} and {seed} are
# substituted.  Override any of them from the command line when your run-dir
# naming differs -- the script prints the resolved path for every arm and seed,
# so a wrong pattern shows up as "not found" rather than as a silently absent
# curve.
# ─────────────────────────────────────────────────────────────────────────────

ARM_SPECS = {
    "MEX-Single": {
        "kind": "triple",
        "default_pattern": "mp20_5k_{prop}_energy_patience_10_seed{seed}/**/ocp_predictions.npz",
    },
    "MEX-Multi": {
        "kind": "single",
        "default_pattern": "*_seed{seed}/results/*structurelevel_pareto*/ocp_predictions.npz",
    },
    "ERM-Single": {
        "kind": "triple",
        "default_pattern": "mp20_5k_erm_{prop}_seed{seed}/**/ocp_predictions.npz",
    },
    "ERM-Multi": {
        "kind": "single",
        "default_pattern": "mp20_5k_erm_multi_seed{seed}/**/ocp_predictions.npz",
    },
}

# Comparison sets.  Each becomes one figure and one statistics table.
SETS = {
    "mex": ("MEX-Single", "MEX-Multi"),
    "erm": ("ERM-Single", "ERM-Multi"),
}

SET_TITLES = {
    "mex": "Energy-head arms: single-property vs.\\ multi-property MEX",
    "erm": "Regression arms: single-property vs.\\ multi-property ERM",
}

# One colour per arm, stable across both figures so the two can be read side by
# side.  The true front keeps the orange of the existing figure.
ARM_COLORS = {
    "MEX-Single": "#7b5cd6",
    "MEX-Multi": "#2a78d6",
    "ERM-Single": "#c2185b",
    "ERM-Multi": "#1b9e77",
}
ARM_MARKERS = {
    "MEX-Single": "s",
    "MEX-Multi": "o",
    "ERM-Single": "^",
    "ERM-Multi": "D",
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def resolve_arm(arm, root, pattern, seed):
    """Locate one arm's prediction file(s) for one seed, or None."""
    if root is None:
        return None
    spec = ARM_SPECS[arm]
    pat = pattern or spec["default_pattern"]
    finder = find_erm if spec["kind"] == "triple" else find_mex
    return finder(root, seed, pat)


def load_arm(arm, src):
    """(preds, gts, ids) for one arm, each (N, 3) in PROP_NAMES order."""
    loader = load_erm_npzs if ARM_SPECS[arm]["kind"] == "triple" else load_multi_npz
    return loader(src)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style_axis(ax, xlabel, ylabel):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, color=_GRIDLINE, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(labelsize=13, width=1.4, length=5, color=_INK_PRIMARY)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", color=_INK_PRIMARY)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", color=_INK_PRIMARY)


def plot_seed_row(axrow, true_pts, arm_pts, seed_label):
    """
    One seed's row of three pairwise projections.

    true_pts : (T, 3) DFT vectors of the true-front structures
    arm_pts  : ordered {arm name: (M, 3) plotting coordinates of its front}
    """
    for (i, j), ax in zip(list(combinations(range(3), 2)), axrow):
        _style_axis(ax, PROP_LABELS[j], PROP_LABELS[i])

        # True front underneath, drawn as open rings so a model marker sitting
        # on one is visible as "recovered" rather than hidden by it.
        ax.scatter(
            true_pts[:, j], true_pts[:, i],
            s=150, facecolors="none", edgecolors=_COLOR_GROUND_TRUTH,
            linewidths=2.0, zorder=2,
        )
        # Marker size decreases with draw order so that when two arms recover
        # the same structure -- which at DFT coordinates puts them at exactly
        # the same point -- the later, smaller marker sits inside the earlier
        # one instead of hiding it. Without this an arm can vanish entirely
        # wherever the two agree, which is precisely where agreement matters.
        for k, (arm, pts) in enumerate(arm_pts.items()):
            ax.scatter(
                pts[:, j], pts[:, i],
                s=max(70 - 30 * k, 16), marker=ARM_MARKERS[arm],
                c=ARM_COLORS[arm], alpha=0.85, linewidths=0, zorder=3 + k,
            )
    axrow[0].set_title(seed_label, fontsize=14, fontweight="bold",
                       color=_INK_PRIMARY, loc="left", pad=10)


def make_figure(set_name, per_seed, coords, out_png):
    """per_seed: list of (seed_label, true_pts, {arm: pts})."""
    _apply_pub_style()
    n = len(per_seed)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5.0 * n), squeeze=False)

    for row, (seed_label, true_pts, arm_pts) in enumerate(per_seed):
        plot_seed_row(axes[row], true_pts, arm_pts, seed_label)

    arms = list(per_seed[0][2].keys())
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=12,
               markerfacecolor="none", markeredgecolor=_COLOR_GROUND_TRUTH,
               markeredgewidth=2.0, label="True front (DFT)"),
    ] + [
        Line2D([0], [0], marker=ARM_MARKERS[a], linestyle="none", markersize=9,
               markerfacecolor=ARM_COLORS[a], markeredgecolor="none",
               alpha=0.85, label=a)
        for a in arms
    ]
    legend = fig.legend(handles=handles, loc="upper center",
                        bbox_to_anchor=(0.5, 1.005), ncol=len(handles),
                        frameon=False, fontsize=14, handletextpad=0.7,
                        columnspacing=2.0)
    for text in legend.get_texts():
        text.set_color(_INK_PRIMARY)
        text.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight", facecolor=_SURFACE)
    fig.savefig(str(out_png.with_suffix(".pdf")), bbox_inches="tight",
                facecolor=_SURFACE)
    plt.close(fig)
    where = "DFT" if coords == "dft" else "predicted"
    print(f"  wrote {out_png}  (+ .pdf)   [model fronts at {where} coordinates]")


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "arm", "seed", "n", "front_size", "true_front",
                    "recovered", "recall_pct", "precision_pct", "enrichment",
                    "igd", "igd_plus", "hv_ratio"])
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def write_latex(set_name, rows, directions, path, tag=""):
    """
    LaTeX table in the style of tab:pareto_erm_vs_mex, but with per-seed rows
    kept visible above the mean.  Three seeds cannot support a hypothesis test
    at this front size, so the per-seed counts are the evidence and the mean is
    a convenience.

    `tag` distinguishes runs over different evaluation libraries or objective
    conventions.  It goes into the \\label, so a held-out table and an mp20_5k
    table can sit in the same document without one silently shadowing the
    other's cross-references.
    """
    arms = []
    for r in rows:
        if r[1] not in arms:
            arms.append(r[1])
    seeds = sorted({r[2] for r in rows}, key=str)
    n_true = {r[5] for r in rows}
    n_true_str = str(n_true.pop()) if len(n_true) == 1 else "varies"

    # Library size comes from the data, not a hardcoded 250: the same table is
    # produced for the mp20_5k test set (250) and for the held-out library
    # (4,803), and a caption naming the wrong one is a caption that lies.
    n_lib = {r[3] for r in rows}
    n_lib_str = (f"{n_lib.pop():,}".replace(",", "{,}")
                 if len(n_lib) == 1 else "varies")

    dir_str = ", ".join(str(d) for d in directions)
    label = f"tab:pareto_arms_{set_name}" + (f"_{tag}" if tag else "")

    lines = [
        r"\begin{table}[htbp]",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + SET_TITLES[set_name] + r". Pareto-front recovery over the "
        + n_lib_str + r"-structure evaluation library under objective directions ("
        + dir_str + r") in the order ($E_\mathrm{form}$, $E_g$, "
        r"$K_\mathrm{VRH}$). The true front is computed from DFT values and is "
        r"identical for both arms, with $|F^\star|{=}" + n_true_str + r"$. "
        r"Per-seed counts are given because one recovered structure is "
        + (f"{100.0 / int(n_true_str):.1f}" if n_true_str.isdigit() else "several")
        + r"\% of recall at this front size, so gaps of one or two structures "
        r"are within seed noise. Recall also has a ceiling; IGD$^+$ and the "
        r"hypervolume ratio do not. Both are computed on the structures each "
        r"arm places on its front, scored at their \emph{measured} property "
        r"vectors, with objectives min--max scaled over the library so all "
        r"three contribute comparably. IGD$^+$ is the mean distance from each "
        r"true-front point to the nearest selected structure, counting only the "
        r"component by which the selection is worse; HV ratio is the dominated "
        r"hypervolume of the selection over that of the true front.}",
        r"\label{" + label + r"}",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\hline\hline",
        r"\textbf{Arm} & \textbf{Seed} & \textbf{Front} & \textbf{Rec.} & "
        r"\textbf{Recall (\%)} & \textbf{Enrich.} & \textbf{IGD$^+\downarrow$} & "
        r"\textbf{HV ratio$\uparrow$} \\",
        r"\hline",
    ]

    for a_i, arm in enumerate(arms):
        arm_rows = [r for r in rows if r[1] == arm]
        span = len(arm_rows) + 1
        lines.append(r"\multirow{" + str(span) + r"}{*}{" + arm + r"}")
        for r in arm_rows:
            lines.append(
                f"  & {r[2]} & {r[4]} & {r[6]} & {r[7]:.1f} & {r[9]:.1f} & "
                f"{r[11]:.4f} & {r[12]:.4f} \\\\"
            )
        fs = np.array([r[4] for r in arm_rows], float)
        hit = np.array([r[6] for r in arm_rows], float)
        rc = np.array([r[7] for r in arm_rows], float)
        en = np.array([r[9] for r in arm_rows], float)
        ip = np.array([r[11] for r in arm_rows], float)
        hv = np.array([r[12] for r in arm_rows], float)
        lines.append(
            f"  & mean & ${fs.mean():.1f} \\pm {fs.std():.1f}$ & "
            f"${hit.mean():.1f} \\pm {hit.std():.1f}$ & "
            f"${rc.mean():.1f} \\pm {rc.std():.1f}$ & "
            f"${en.mean():.1f} \\pm {en.std():.1f}$ & "
            f"${ip.mean():.4f} \\pm {ip.std():.4f}$ & "
            f"${hv.mean():.4f} \\pm {hv.std():.4f}$ \\\\"
        )
        lines.append(r"\hline" if a_i < len(arms) - 1 else r"\hline\hline")

    lines += [r"\end{tabular}", r"\end{table}", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def verdict(rows, arms, n_true):
    """
    Plain-language read on whether the two arms' fronts differ, using the same
    blunt rule as compare_pareto_erm_vs_mex.py: overlapping per-seed ranges, a
    gap under one structure, or a gap inside the cross-seed spread all count as
    indistinguishable.  Deliberately not a hypothesis test -- three seeds
    cannot support one here.
    """
    a, b = arms
    ha = np.array([r[6] for r in rows if r[1] == a], float)
    hb = np.array([r[6] for r in rows if r[1] == b], float)
    if len(ha) == 0 or len(hb) == 0:
        return "  (only one arm present; no comparison)"
    delta = ha.mean() - hb.mean()
    spread = max(ha.std(), hb.std())
    overlap = (ha.min() <= hb.max()) and (hb.min() <= ha.max())
    detail = (f"{a} {[int(x) for x in ha]} vs {b} {[int(x) for x in hb]} "
              f"out of {n_true}")
    if overlap or abs(delta) < 1.0 or abs(delta) <= spread:
        why = ("ranges overlap" if overlap
               else "gap under one structure" if abs(delta) < 1.0
               else "gap within cross-seed spread")
        head = f"  → recall: INDISTINGUISHABLE ({why}). {detail}"
    else:
        ahead, behind = (a, b) if delta > 0 else (b, a)
        head = (f"  → recall: {ahead} ahead of {behind} by {abs(delta):.1f} "
                f"structures on average. {detail}")

    # Recall saturates; IGD+ and HV do not. Report them separately rather than
    # folding them into one verdict, so a recall tie broken only by the
    # continuous metrics is visible as exactly that.
    out = [head]
    for key, better, label in (("igd_plus", "lower", "IGD+"),
                               ("hv_ratio", "higher", "HV ratio")):
        va = np.array([r[10 + ["igd", "igd_plus", "hv_ratio"].index(key)]
                       for r in rows if r[1] == a], float)
        vb = np.array([r[10 + ["igd", "igd_plus", "hv_ratio"].index(key)]
                       for r in rows if r[1] == b], float)
        ov = (va.min() <= vb.max()) and (vb.min() <= va.max())
        a_wins = (va.mean() < vb.mean()) if better == "lower" else (va.mean() > vb.mean())
        winner = a if a_wins else b
        verdict_txt = ("ranges overlap across seeds" if ov
                       else f"{winner} better on every seed")
        out.append(f"    {label:9s} {a} {va.mean():.4f}±{va.std():.4f}  "
                   f"{b} {vb.mean():.4f}±{vb.std():.4f}   ({verdict_txt})")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="+", default=["mex", "erm"], choices=sorted(SETS),
                    help="which comparison sets to produce (default: both)")
    for arm in ARM_SPECS:
        flag = arm.lower().replace("-", "-")
        ap.add_argument(f"--{flag}-root", default=None,
                        help=f"root to search for {arm} predictions")
        ap.add_argument(f"--{flag}-pattern", default=None,
                        help=f"glob override for {arm}; may use {{prop}} and {{seed}}")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--directions", nargs=3, default=["min", "min", "min"],
                    metavar=("EFORM", "BANDGAP", "KVRH"),
                    help="'min', 'max', or a target value per property")
    ap.add_argument("--coords", choices=["dft", "pred"], default="dft",
                    help="plot model-front structures at their DFT (default) or "
                         "predicted coordinates")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--tag", default="",
                    help="optional suffix for output filenames")
    args = ap.parse_args()

    def arg_for(arm, suffix):
        return getattr(args, f"{arm.lower().replace('-', '_')}_{suffix}")

    print(f"\nobjective directions: " +
          ", ".join(f"{p}={d}" for p, d in zip(PROP_NAMES, args.directions)))
    print(f"model fronts PLOTTED at {args.coords.upper()} coordinates")
    print("METRICS are always computed at DFT coordinates: the front is selected\n"
          "      in predicted space, then scored at the selected structures' measured\n"
          "      properties. Scoring in predicted space would reward a model for being\n"
          "      optimistic. So --coords changes the figure, never the numbers.")
    print("NOTE: MEX arms reach point estimates by derivative-free search; "
          "regression\n      arms by a single forward pass. Different "
          "procedures, same downstream sorting.\n")

    exit_code = 0
    for set_name in args.set:
        arms = SETS[set_name]
        print(f"═══ set '{set_name}': {arms[0]} vs {arms[1]}")

        per_seed = []
        rows = []
        for seed in args.seeds:
            true_pts = None
            arm_pts = {}
            for arm in arms:
                src = resolve_arm(arm, arg_for(arm, "root"),
                                  arg_for(arm, "pattern"), seed)
                if src is None:
                    print(f"  seed {seed}: {arm} NOT FOUND under "
                          f"{arg_for(arm, 'root')}")
                    exit_code = 1
                    continue
                try:
                    preds, gts, _ids = load_arm(arm, src)
                except Exception as exc:
                    print(f"  seed {seed}: {arm} FAILED to load: {exc}")
                    exit_code = 1
                    continue

                m = front_metrics(preds, gts, args.directions)
                rows.append([set_name, arm, f"seed{seed}", m["n"], m["n_model"],
                             m["n_true"], m["hit"], m["recall"] * 100,
                             m["precision"] * 100, m["enrichment"],
                             m["igd"], m["igd_plus"], m["hv_ratio"]])

                # Front membership is decided in predicted space; only the
                # plotting position depends on --coords.
                coords_mat = gts if args.coords == "dft" else preds
                arm_pts[arm] = coords_mat[m["model_front"]]
                if true_pts is None:
                    true_pts = gts[m["true_front"]]

                print(f"  seed {seed}  {arm:11s} front={m['n_model']:3d}  "
                      f"recovered={m['hit']:3d}/{m['n_true']:<3d} "
                      f"recall={m['recall']*100:5.1f}%  "
                      f"prec={m['precision']*100:5.1f}%  "
                      f"enrich={m['enrichment']:4.1f}x")
                print(f"              IGD={m['igd']:.4f}  IGD+={m['igd_plus']:.4f}  "
                      f"HV ratio={m['hv_ratio']:.4f}")
                # A single-property arm is THREE files merged by structure id,
                # one per property. Print all of them: showing only the first
                # made it look as though the whole arm came from the eform run.
                if isinstance(src, list):
                    for prop, path in zip(PROP_NAMES, src):
                        print(f"              [{prop:7s}] {path}")
                else:
                    print(f"              {src}")

            if arm_pts:
                per_seed.append((f"seed {seed}", true_pts, arm_pts))

        if not per_seed:
            print(f"  nothing found for set '{set_name}', skipping\n")
            exit_code = 1
            continue

        suffix = f"_{args.tag}" if args.tag else ""
        stem = f"pareto_arms_{set_name}_{args.coords}{suffix}"
        make_figure(set_name, per_seed, args.coords,
                    args.out_dir / f"{stem}.png")

        # Statistics are written only for --coords dft. Every metric --
        # recall, precision, enrichment, IGD, IGD+, HV -- is defined at DFT
        # coordinates regardless of this flag, so a --coords pred run would
        # emit a byte-identical CSV and LaTeX table under a different name.
        # Two files with the same numbers and different names invite exactly
        # one mistake: citing them as two results. --coords pred changes the
        # figure and nothing else.
        if args.coords == "dft":
            write_csv(rows, args.out_dir / f"{stem}_stats.csv")
            write_latex(set_name, rows, args.directions,
                        args.out_dir / f"{stem}_stats.tex", tag=args.tag)
        else:
            print("  (no stats written: metrics are defined at DFT coordinates "
                  "and are identical to the --coords dft run; this flag changes "
                  "the figure only)")

        n_true_set = {r[5] for r in rows}
        n_true = n_true_set.pop() if len(n_true_set) == 1 else "?"
        print(verdict(rows, arms, n_true))
        if isinstance(n_true, int):
            print(f"  scale: one recovered structure = {100.0/n_true:.1f} points "
                  f"of recall; 3 seeds resolve nothing finer.")
        print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
