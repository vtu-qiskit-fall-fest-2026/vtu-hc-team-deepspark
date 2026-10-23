"""Headless benchmark: writes real experiment output to results/ (CSV, Markdown, PNG).

Usage:  python run_benchmark.py [--shots 5000] [--trials 6] [--seed 2026] [--budget 5]
"""
import argparse, json, platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qiskit, qiskit_aer
from qadapt.noise_models import NoiseConfig
from qadapt.strategies import TASK_QUANTUM, TASK_CLASSICAL, STRATEGIES
from qadapt.benchmarks import run_sweep_raw, sweep_tables, run_map_raw, map_choices

ap = argparse.ArgumentParser()
ap.add_argument("--shots", type=int, default=5000)
ap.add_argument("--trials", type=int, default=6)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--budget", type=int, default=5)
ap.add_argument("--lam", type=float, default=0.005)
ap.add_argument("--alpha", type=float, default=0.05)
a = ap.parse_args()
out = Path("results"); out.mkdir(exist_ok=True)
COL = {"none": "#64748b", "bit3_q": "#0ea5e9", "phase3_q": "#8b5cf6", "bit5_q": "#10b981",
       "phase5_q": "#f59e0b", "bit3_v": "#ec4899", "bit5_v": "#ef4444"}

experiments = [
    ("bitflip_sweep", "p_x", np.linspace(0, 0.25, 11), NoiseConfig(), TASK_QUANTUM),
    ("phaseflip_sweep", "p_z", np.linspace(0, 0.25, 11), NoiseConfig(), TASK_QUANTUM),
    ("gate_noise_sweep_px0.05", "p_gate", np.linspace(0, 0.06, 11), NoiseConfig(p_x=0.05), TASK_QUANTUM),
    ("classical_readout_sweep_px0.05", "p_meas", np.linspace(0, 0.25, 11), NoiseConfig(p_x=0.05), TASK_CLASSICAL),
]
md = [f"# Q-Adapt benchmark output\n\nshots/circuit/trial={a.shots}, trials={a.trials}, seed={a.seed}, budget={a.budget}, "
      f"lambda={a.lam}, alpha={a.alpha}\n\nqiskit {qiskit.__version__}, qiskit-aer {qiskit_aer.__version__}, python {platform.python_version()}\n\n"
      "All noise is SIMULATED. Adaptive rows use held-out validation trials.\n"]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (name, param, vals, base, task) in zip(axes.ravel(), experiments):
    print("running", name, flush=True)
    raw = run_sweep_raw(param, vals, base, task, a.shots, a.trials, a.seed)
    df, ad = sweep_tables(raw, param, a.budget, a.lam, a.alpha)
    df.to_csv(out / f"{name}_all_strategies.csv", index=False)
    ad.to_csv(out / f"{name}_adaptive.csv", index=False)
    for k in df["strategy"].unique():
        if STRATEGIES[k].n > a.budget: continue
        d = df[df.strategy == k]
        ax.errorbar(d[param], d.logical_failure_rate, yerr=[np.clip(d.logical_failure_rate - d.ci95_low, 0, None), np.clip(d.ci95_high - d.logical_failure_rate, 0, None)],
                    label=k, color=COL[k], ls="--" if k == "none" else "-", marker="o", ms=3, lw=1.5, capsize=2)
    ax.plot(ad[param], ad.heldout_failure_rate, color="gold", lw=3.5, marker="*", ms=11, label="ADAPTIVE (held-out)", zorder=5)
    ax.set_title(f"{name}  [{task}]"); ax.set_xlabel(param); ax.set_ylabel("logical failure rate"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    md.append(f"\n## {name}  (task={task})\n\nAdaptive choices:\n\n```\n" +
              ad[[param, "adaptive_choice", "heldout_failure_rate", "heldout_baseline_rate", "confident"]].to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```\n")
plt.tight_layout(); plt.savefig(out / "benchmark_overview.png", dpi=130)

print("running regime map", flush=True)
xs = np.linspace(0, 0.2, 6); ys = np.linspace(0, 0.2, 6)
mraw = run_map_raw("p_x", xs, "p_z", ys, NoiseConfig(), TASK_QUANTUM, 2000, 4, a.seed)
mdf = map_choices(mraw, a.budget, a.lam, a.alpha)
mdf.drop(columns=["i", "j"]).to_csv(out / "regime_map_px_vs_pz.csv", index=False)
grid = mdf.pivot(index="p_z", columns="p_x", values="best_strategy").sort_index(ascending=False)
md.append("\n## Regime map (rows p_z, columns p_x), budget %d\n\n```\n%s\n```\n" % (a.budget, grid.to_string()))
(out / "BENCHMARK_REPORT.md").write_text("\n".join(md))
json.dump(vars(a), open(out / "run_settings.json", "w"), indent=2)
print("done ->", out.resolve())
