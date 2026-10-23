"""Create all report figures from the CSVs written by run_all.py.

    python experiments/make_figures.py [results_dir] [figures_dir]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

from q_adapt.codes import STRATEGIES, strategy_cost                     # noqa: E402
from q_adapt.selector import pick_cheapest_near_optimal                   # noqa: E402

# colour-blind-safe (Okabe-Ito) palette, fixed per strategy; markers/linestyles add a 2nd cue
COL = {"none": "#444444", "readout_mitigation": "#E69F00", "bitflip3": "#0072B2",
       "phaseflip3": "#D55E00", "shor9": "#009E73"}
MRK = {"none": "o", "readout_mitigation": "x", "bitflip3": "s", "phaseflip3": "^", "shor9": "D"}
LS = {"none": "-", "readout_mitigation": "--", "bitflip3": "-", "phaseflip3": "-", "shor9": "-"}
LAB = {"none": "No correction", "readout_mitigation": "Readout mitigation",
       "bitflip3": "3-qubit bit-flip code", "phaseflip3": "3-qubit phase-flip code",
       "shor9": "Shor 9-qubit code"}
PROFILE_TITLE = {"bitflip": "Bit-flip noise", "phaseflip": "Phase-flip noise",
                 "depolarizing": "Depolarizing noise", "amp_damping": "Amplitude damping (T1)",
                 "mixed": "Mixed noise + readout", "readout": "Readout-dominated"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight"})
IDLE = 100


def style(ax):
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# Check required data files
required_csvs = ["benchmark_sweep.csv", "adaptive_summary.csv", "idle_sensitivity.csv", "budget_summary.csv"]
missing = [f for f in required_csvs if not os.path.exists(os.path.join(RES, f))]
if missing:
    sys.exit(f"Error: Missing results file(s): {', '.join(missing)}\nPlease run 'python experiments/run_all.py' first to generate results.")

# ---------------------------------------------------------------- fig 1: LER vs p
sw = pd.read_csv(os.path.join(RES, "benchmark_sweep.csv"))
profiles = [p for p in PROFILE_TITLE if p in sw.profile.unique()]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
for ax, prof in zip(axes.ravel(), profiles):
    d = sw[sw.profile == prof]
    for s in STRATEGIES:
        e = d[d.strategy == s].sort_values("p")
        ax.plot(e.p, e.logical_error_rate.clip(lower=1e-4), color=COL[s], ls=LS[s], marker=MRK[s],
                ms=4.5, lw=1.8, label=LAB[s])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(PROFILE_TITLE[prof]); style(ax)
for ax in axes[1]:
    ax.set_xlabel("physical error probability p per step")
for ax in axes[:, 0]:
    ax.set_ylabel("logical error rate")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), frameon=False)
fig.suptitle("")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_logical_error_vs_noise.png")); plt.close(fig)

# ---------------------------------------------------------------- fig 2: best-strategy heatmap
ps = sorted(sw.p.unique())
grid = np.empty((len(profiles), len(ps)), dtype=object)
for i, prof in enumerate(profiles):
    for j, p in enumerate(ps):
        d = sw[(sw.profile == prof) & (np.isclose(sw.p, p))]
        ler = dict(zip(d.strategy, d.logical_error_rate))
        grid[i, j] = pick_cheapest_near_optimal(ler, IDLE, STRATEGIES)
code = {s: k for k, s in enumerate(STRATEGIES)}
from matplotlib.colors import ListedColormap
cmap = ListedColormap([COL[s] for s in STRATEGIES])
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.imshow(np.vectorize(code.get)(grid), cmap=cmap, vmin=-0.5, vmax=len(STRATEGIES) - 0.5, aspect="auto")
SHORT = {"none": "none", "readout_mitigation": "RO-mit", "bitflip3": "BF3", "phaseflip3": "PF3", "shor9": "Shor9"}
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, SHORT[grid[i, j]], ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
ax.set_xticks(range(len(ps))); ax.set_xticklabels([f"{p:.0e}" for p in ps], rotation=45, fontsize=8)
ax.set_yticks(range(len(profiles))); ax.set_yticklabels([PROFILE_TITLE[p] for p in profiles])
ax.set_xlabel("physical error probability p per step"); ax.spines[:].set_visible(False)
ax.set_title(f"Best strategy per noise condition ({IDLE}-step logical memory; cheapest within noise margin)", loc="left")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[s]) for s in STRATEGIES],
          labels=[LAB[s] for s in STRATEGIES], ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
fig.savefig(os.path.join(FIG, "fig2_best_strategy_heatmap.png")); plt.close(fig)

# ---------------------------------------------------------------- fig 3: adaptive vs fixed
sm = pd.read_csv(os.path.join(RES, "adaptive_summary.csv"))
order = [p for p in ["Uncorrected", "Fixed: bitflip3", "Fixed: phaseflip3", "Fixed: shor9",
                     "Q-Adapt (rule)", "Q-Adapt (lookup)", "Q-Adapt (tree)", "Oracle (best in hindsight)"]
         if p in set(sm.policy)]
sm = sm.set_index("policy").loc[order]
pcol = [("#9a9a9a" if p.startswith("Uncorr") else "#c9a7a0" if p.startswith("Fixed") else
         "#1f6f8b" if p.startswith("Q-Adapt") else "#2a2a2a") for p in order]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, col, ttl, fmt in [(axes[0], "mean_logical_error", "Mean logical error rate (lower is better)", "{:.3f}"),
                          (axes[1], "mean_qubits", "Mean physical qubits used", "{:.1f}"),
                          (axes[2], "mean_depth", "Mean circuit depth", "{:.0f}")]:
    b = ax.barh(range(len(order)), sm[col], color=pcol, height=0.62)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order if ax is axes[0] else [])
    ax.invert_yaxis(); ax.set_title(ttl, fontsize=10, loc="left"); style(ax); ax.grid(axis="y", visible=False)
    for r, v in zip(b, sm[col]):
        ax.text(v, r.get_y() + r.get_height() / 2, " " + fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_xlim(0, sm[col].max() * 1.18)
fig.suptitle("Unseen random noise conditions: Uncorrected vs fixed strategies vs Q-Adapt", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_adaptive_vs_fixed.png")); plt.close(fig)

# ---------------------------------------------------------------- fig 4: resource overhead
costs = {s: strategy_cost("none" if s == "readout_mitigation" else s, IDLE) for s in STRATEGIES}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, key, ttl in [(axes[0], "n_qubits", "Physical qubits (overhead vs 1 logical)"),
                     (axes[1], "depth", "Transpiled circuit depth"),
                     (axes[2], "cx_count", "Two-qubit (CNOT) gates")]:
    vals = [costs[s][key] for s in STRATEGIES]
    b = ax.bar(range(len(STRATEGIES)), vals, color=[COL[s] for s in STRATEGIES], width=0.62)
    ax.set_xticks(range(len(STRATEGIES))); ax.set_xticklabels([SHORT[s] for s in STRATEGIES], fontsize=8)
    ax.set_title(ttl, fontsize=10, loc="left"); style(ax); ax.grid(axis="x", visible=False)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_resource_overhead.png")); plt.close(fig)

# ---------------------------------------------------------------- fig 5: idle-length break-even
isn = pd.read_csv(os.path.join(RES, "idle_sensitivity.csv"))
profs = list(isn.profile.unique())
fig, axes = plt.subplots(1, len(profs), figsize=(6 * len(profs), 3.9), sharey=True)
axes = np.atleast_1d(axes)
for ax, prof in zip(axes, profs):
    d = isn[isn.profile == prof]
    for s in ["none", "bitflip3", "phaseflip3", "shor9"]:
        e = d[d.strategy == s].sort_values("idle")
        ax.plot(e.idle, e.logical_error_rate.clip(lower=1e-4), color=COL[s], marker=MRK[s], ms=4.5,
                lw=1.8, label=LAB[s])
    ax.set_xscale("log"); ax.set_yscale("log"); style(ax)
    ax.set_title(f"{PROFILE_TITLE[prof]}, p = 3e-4", fontsize=10, loc="left")
    ax.set_xlabel("logical memory length (idle steps)")
axes[0].set_ylabel("logical error rate"); axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_workload_length_breakeven.png")); plt.close(fig)

# ---------------------------------------------------------------- fig 6: resource budget
bud = pd.read_csv(os.path.join(RES, "budget_summary.csv"))
fig, ax = plt.subplots(figsize=(7, 4))
for pol, c, ls in [("Fixed: shor9", "#c9a7a0", "--"), ("Uncorrected", "#9a9a9a", "--"),
                   ("Q-Adapt (rule)", "#7fb3c7", "-"), ("Q-Adapt (lookup)", "#1f6f8b", "-"),
                   ("Oracle (best in hindsight)", "#2a2a2a", "-")]:
    d = bud[bud.policy == pol].sort_values("max_qubits")
    if len(d):
        ax.plot(d.max_qubits, d.mean_logical_error, marker="o", color=c, ls=ls, lw=1.8, label=pol)
ax.set_xticks([1, 3, 9]); ax.set_xlabel("qubit budget"); ax.set_ylabel("mean logical error rate")
ax.set_title("Resource-aware selection: error vs qubit budget", loc="left"); style(ax)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_resource_budget.png")); plt.close(fig)
print("figures written to", FIG)
