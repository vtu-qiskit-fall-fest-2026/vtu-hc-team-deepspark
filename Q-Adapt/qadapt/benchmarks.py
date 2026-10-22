"""Benchmark engine: runs REAL Qiskit-Aer simulations and turns counts into metrics.

Metric definitions
------------------
* logical failure rate  = (# shots whose decoded logical bit != expected bit) / (# shots)
                          pooled over the tested logical states (|0>, and |+> for the quantum task)
* success probability   = 1 - failure rate
* 95% CI                = Wilson score interval on the pooled counts
* improvement           = baseline rate - strategy rate   (absolute)  and  relative reduction
* qubit overhead        = physical qubits used / 1 logical qubit
* gate count / depth    = measured on the transpiled circuit (basis: id,h,x,sx,rz,cx)

Calibration / validation split
------------------------------
Trials alternate between a CALIBRATION set (used by the adaptive engine to choose) and a
held-out VALIDATION set (used to report how the chosen strategy really performs). This avoids
the "winner's curse" of judging a selection on the same data that produced it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator

from .noise_models import NoiseConfig, build_noise_model
from .strategies import STRATEGIES, BASELINE_KEY, Strategy, strategies_for_task
from .circuits import transpiled_circuit, circuit_metrics, logical_outcome
from .stats import wilson_interval, improvement_p_value
from .adaptive import recommend


@dataclass
class StrategyResult:
    key: str
    label: str
    n_qubits: int
    cx_count: int
    gate_count: int
    depth: int
    cal_fail: int = 0
    cal_shots: int = 0
    val_fail: int = 0
    val_shots: int = 0
    basis_fail: dict = field(default_factory=dict)   # basis -> [fail, shots]
    trial_rates: list = field(default_factory=list)

    @property
    def fail(self) -> int:
        return self.cal_fail + self.val_fail

    @property
    def shots(self) -> int:
        return self.cal_shots + self.val_shots

    @property
    def rate(self) -> float:
        return self.fail / self.shots if self.shots else float("nan")

    @property
    def success(self) -> float:
        return 1 - self.rate

    @property
    def ci(self) -> tuple:
        return wilson_interval(self.fail, self.shots)

    @property
    def cal_rate(self) -> float:
        return self.cal_fail / self.cal_shots if self.cal_shots else float("nan")

    @property
    def val_rate(self) -> float:
        return self.val_fail / self.val_shots if self.val_shots else float("nan")

    @property
    def trial_std(self) -> float:
        return float(np.std(self.trial_rates, ddof=1)) if len(self.trial_rates) > 1 else float("nan")

    def basis_rate(self, b: str) -> float:
        f, s = self.basis_fail.get(b, (0, 0))
        return f / s if s else float("nan")


def _seed(base: int, *parts: int) -> int:
    x = int(base) & 0x7FFFFFFF
    for p in parts:
        x = (x * 1_000_003 + int(p) + 1) & 0x7FFFFFFF
    return x


def make_simulator(cfg: NoiseConfig) -> AerSimulator:
    return AerSimulator(method="density_matrix", noise_model=build_noise_model(cfg))


def run_circuit_counts(key: str, basis: str, cfg: NoiseConfig, shots: int, seed: int) -> dict:
    """Run ONE strategy circuit on the noisy simulator and return raw Aer counts."""
    sim = make_simulator(cfg)
    res = sim.run(transpiled_circuit(key, basis), shots=shots, seed_simulator=_seed(seed, 1)).result()
    return res.get_counts()


def count_failures(strategy: Strategy, counts: dict) -> int:
    return sum(c for bs, c in counts.items() if logical_outcome(strategy, bs) == 1)


def run_point(cfg: NoiseConfig, task: str, shots: int, trials: int, seed: int,
              point_id: int = 0) -> list[StrategyResult]:
    """Benchmark all strategies of `task` at ONE noise configuration."""
    if trials < 2:
        raise ValueError("trials must be >= 2 (calibration + validation split)")
    strats = strategies_for_task(task)
    sim = make_simulator(cfg)
    jobs = [(s, b) for s in strats for b in s.bases]
    circs = [transpiled_circuit(s.key, b) for s, b in jobs]
    results = {}
    for s in strats:
        m = circuit_metrics(s.key)
        results[s.key] = StrategyResult(s.key, s.label, m["qubits"], m["cx_count"],
                                        m["gate_count"], m["depth"],
                                        basis_fail={b: [0, 0] for b in s.bases})
    for t in range(trials):
        res = sim.run(circs, shots=shots, seed_simulator=_seed(seed, point_id, t) ).result()
        t_fail = {s.key: 0 for s in strats}
        t_shots = {s.key: 0 for s in strats}
        for idx, (s, b) in enumerate(jobs):
            f = count_failures(s, res.get_counts(idx))
            r = results[s.key]
            r.basis_fail[b][0] += f
            r.basis_fail[b][1] += shots
            if t % 2 == 0:
                r.cal_fail += f; r.cal_shots += shots
            else:
                r.val_fail += f; r.val_shots += shots
            t_fail[s.key] += f; t_shots[s.key] += shots
        for k in results:
            results[k].trial_rates.append(t_fail[k] / t_shots[k])
    return list(results.values())


# ----------------------------------------------------------------------------- tables
def results_to_df(results: list[StrategyResult]) -> pd.DataFrame:
    base = next(r for r in results if r.key == BASELINE_KEY)
    rows = []
    for r in results:
        lo, hi = r.ci
        p = None if r.key == BASELINE_KEY else improvement_p_value(base.fail, base.shots, r.fail, r.shots)
        rows.append({
            "strategy": r.key, "label": r.label, "physical_qubits": r.n_qubits,
            "cx_count": r.cx_count, "gate_count": r.gate_count, "depth": r.depth,
            "failures": r.fail, "total_shots": r.shots,
            "logical_failure_rate": r.rate, "ci95_low": lo, "ci95_high": hi,
            "success_probability": r.success, "trial_std": r.trial_std,
            "Z_basis_failure": r.basis_rate("Z"), "X_basis_failure": r.basis_rate("X"),
            "abs_improvement_vs_baseline": base.rate - r.rate,
            "rel_reduction_vs_baseline": (base.rate - r.rate) / base.rate if base.rate > 0 else float("nan"),
            "p_value_better_than_baseline": p,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- sweeps
SWEEP_PARAMS = ["p_x", "p_z", "p_gate", "p_meas"]


def run_sweep_raw(param: str, values, base_cfg: NoiseConfig, task: str, shots: int,
                  trials: int, seed: int, progress=None, runner=None) -> list[dict]:
    """Run run_point for each value of one noise parameter (other params held at base_cfg)."""
    run = runner or (lambda f, *a: f(*a))
    raw = []
    for i, v in enumerate(values):
        cfg = base_cfg.with_param(param, float(v))
        raw.append({"value": float(v), "cfg": cfg,
                    "results": run(run_point, cfg, task, shots, trials, seed, i)})
        if progress:
            progress((i + 1) / len(values))
    return raw


def sweep_tables(raw: list[dict], param: str, budget: int, lam: float, alpha: float):
    """Derive (all-strategies table, adaptive table) from raw sweep data. Cheap, so it
    is recomputed instantly when the budget / lambda / alpha controls change."""
    rows, arows = [], []
    for pt in raw:
        res = pt["results"]
        df = results_to_df(res)
        df.insert(0, param, pt["value"])
        rows.append(df)
        rec = recommend(res, budget, lam, alpha)
        by = {r.key: r for r in res}
        ch, base = by[rec.chosen], by[BASELINE_KEY]
        lo, hi = wilson_interval(ch.val_fail, ch.val_shots)
        arows.append({param: pt["value"], "adaptive_choice": rec.chosen,
                      "adaptive_label": ch.label, "qubits": ch.n_qubits,
                      "heldout_failure_rate": ch.val_rate, "heldout_ci95_low": lo,
                      "heldout_ci95_high": hi, "heldout_baseline_rate": base.val_rate,
                      "confident": rec.confident})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(arows)


# ----------------------------------------------------------------------------- regime map
def run_map_raw(x_param: str, xs, y_param: str, ys, base_cfg: NoiseConfig, task: str,
                shots: int, trials: int, seed: int, progress=None, runner=None) -> dict:
    run = runner or (lambda f, *a: f(*a))
    cells, total, n = {}, len(xs) * len(ys), 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            cfg = base_cfg.with_param(x_param, float(x)).with_param(y_param, float(y))
            cells[(i, j)] = run(run_point, cfg, task, shots, trials, seed, 1000 + j * len(xs) + i)
            n += 1
            if progress:
                progress(n / total)
    return {"x_param": x_param, "xs": list(map(float, xs)), "y_param": y_param,
            "ys": list(map(float, ys)), "cells": cells}


def map_choices(raw: dict, budget: int, lam: float, alpha: float) -> pd.DataFrame:
    rows = []
    for (i, j), res in raw["cells"].items():
        rec = recommend(res, budget, lam, alpha)
        by = {r.key: r for r in res}
        rows.append({raw["x_param"]: raw["xs"][i], raw["y_param"]: raw["ys"][j],
                     "best_strategy": rec.chosen, "label": by[rec.chosen].label,
                     "heldout_failure_rate": by[rec.chosen].val_rate,
                     "heldout_baseline_rate": by[BASELINE_KEY].val_rate,
                     "confident": rec.confident, "i": i, "j": j})
    return pd.DataFrame(rows)
