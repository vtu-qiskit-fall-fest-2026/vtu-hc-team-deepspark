"""Adaptive decision engine (transparent, rule-based; no machine learning).

Inputs : measured benchmark results (calibration counts), qubit budget, resource weight
         lambda, significance level alpha.
Output : a Recommendation with the chosen strategy and a plain-language explanation.

Algorithm
---------
1. ELIGIBILITY   drop strategies with physical qubits > budget.
2. EVIDENCE      a non-baseline strategy is a CANDIDATE only if a one-sided two-proportion
                 z-test shows it has a lower failure rate than the uncorrected baseline
                 (p < alpha). Otherwise "no demonstrated improvement".
3. OBJECTIVE     J = failure_rate + lambda * (physical_qubits - 1)
                 lambda = how much failure probability you would give up per extra qubit.
4. SELECTION     best = argmin J over {baseline} U candidates.
5. TIES          strategies whose J exceeds best's by less than z_crit * sqrt(var_a + var_b)
                 are statistically tied; the cheapest tied option (fewest qubits, then fewer
                 CX gates, then depth) is chosen.
6. FALLBACK      if no candidate beats the baseline, the baseline is recommended and the
                 engine says so explicitly (correction can hurt, e.g. wrong code for the
                 noise type, or gate noise larger than the protection gained).
The chosen strategy is finally CHECKED on held-out validation trials the engine never used.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from .stats import improvement_p_value, smoothed_variance, wilson_interval

BASELINE = "none"
Z_TIE = 1.96


@dataclass
class Recommendation:
    chosen: str
    chosen_label: str
    confident: bool
    headline: str
    explanation: list = field(default_factory=list)
    table: list = field(default_factory=list)
    tied: list = field(default_factory=list)
    excluded_budget: list = field(default_factory=list)
    unconstrained_best: str = ""
    heldout: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def recommend(results, budget: int, lam: float = 0.005, alpha: float = 0.05) -> Recommendation:
    by = {r.key: r for r in results}
    base = by[BASELINE]
    rows = []
    for r in results:
        k, n = r.cal_fail, r.cal_shots
        rate = k / n
        lo, hi = wilson_interval(k, n)
        p = None if r.key == BASELINE else improvement_p_value(base.cal_fail, base.cal_shots, k, n)
        within = r.n_qubits <= budget
        if r.key == BASELINE:
            status = "baseline"
        elif not within:
            status = "over qubit budget"
        elif p is not None and p < alpha and rate < base.cal_rate:
            status = "candidate"
        else:
            status = "no significant improvement"
        rows.append({"key": r.key, "label": r.label, "qubits": r.n_qubits, "cx": r.cx_count,
                     "depth": r.depth, "rate": rate, "ci_lo": lo, "ci_hi": hi, "p": p,
                     "J": rate + lam * (r.n_qubits - 1), "var": smoothed_variance(k, n),
                     "status": status, "within_budget": within})

    def select(pool):
        best = min(pool, key=lambda x: x["J"])
        tied = [x for x in pool if x is best or
                (x["J"] - best["J"]) < Z_TIE * math.sqrt(x["var"] + best["var"])]
        return min(tied, key=lambda x: (x["qubits"], x["cx"], x["depth"], x["J"])), tied

    pool = [x for x in rows if x["status"] in ("baseline", "candidate")]
    chosen, tied = select(pool)
    for x in rows:
        x["chosen"] = x is chosen

    over = [x for x in rows if x["status"] == "over qubit budget"]
    free_pool = [x for x in rows if x["key"] == BASELINE or
                 (x["p"] is not None and x["p"] < alpha and x["rate"] < base.cal_rate)]
    unconstrained, _ = select(free_pool)

    ch_res = by[chosen["key"]]
    exp, warn = [], []
    exp.append(f"Measured on calibration runs ({base.cal_shots:,} shots per strategy): "
               f"uncorrected failure rate {_pct(base.cal_rate)}.")
    if chosen["key"] == BASELINE:
        cands = [x for x in rows if x["status"] == "candidate"]
        if not cands:
            exp.append("No eligible correction strategy showed a statistically significant "
                       f"improvement over the baseline (needed p < {alpha}). "
                       "Adding qubits/gates is not justified by the evidence at this noise level.")
            weak = [x for x in rows if x["status"] == "no significant improvement"]
            for x in weak:
                if x["rate"] >= base.cal_rate:
                    exp.append(f"• {x['label']}: failure {_pct(x['rate'])} is NOT better than the baseline "
                               "(this code does not protect against the dominant error here, or its circuit "
                               "adds more error than it removes).")
            if base.cal_fail == 0:
                warn.append("The baseline had zero failures: noise is too weak to measure with this many shots.")
        else:
            exp.append("Corrections do help, but the resource penalty "
                       f"(lambda = {lam} per extra qubit) outweighs the gain, so the cheapest option wins.")
        headline = "Recommendation: no error correction (uncorrected qubit)"
    else:
        gain = base.cal_rate - chosen["rate"]
        rel = gain / base.cal_rate if base.cal_rate > 0 else float("nan")
        exp.append(f"Chosen: {chosen['label']} — failure {_pct(chosen['rate'])} "
                   f"(95% CI {_pct(chosen['ci_lo'])}–{_pct(chosen['ci_hi'])}), "
                   f"an absolute drop of {_pct(gain)} ({rel:.0%} relative), "
                   f"one-sided p = {chosen['p']:.2g} vs baseline.")
        exp.append(f"It uses {chosen['qubits']} physical qubits (budget {budget}) and "
                   f"{chosen['cx']} CX gates / depth {chosen['depth']}; objective J = "
                   f"{chosen['J']:.4f} (lambda = {lam}).")
        headline = f"Recommendation: {chosen['label']}"
    others = [x for x in tied if x is not chosen]
    if others:
        exp.append("Statistically tied within noise: " + ", ".join(x["label"] for x in others) +
                   f". The cheapest option ({chosen['label']}) was selected.")
    if over:
        best_over = [x for x in over if x["p"] is not None and x["p"] < alpha and x["rate"] < base.cal_rate]
        names = ", ".join(x["label"] for x in over)
        exp.append(f"Excluded by the {budget}-qubit budget: {names}.")
        if best_over:
            bo = min(best_over, key=lambda x: x["rate"])
            if bo["rate"] < chosen["rate"]:
                exp.append(f"Trade-off: {bo['label']} would reach {_pct(bo['rate'])} failure but needs "
                           f"{bo['qubits']} qubits.")
    if unconstrained["key"] != chosen["key"]:
        exp.append(f"With an unlimited budget the engine would pick: {unconstrained['label']}.")

    # held-out validation of the choice
    from .stats import improvement_p_value as ipv
    hv = {"chosen_rate": ch_res.val_rate, "baseline_rate": base.val_rate, "shots": ch_res.val_shots}
    if chosen["key"] != BASELINE:
        p_val = ipv(base.val_fail, base.val_shots, ch_res.val_fail, ch_res.val_shots)
        hv["p_value"] = p_val
        hv["holds"] = bool(ch_res.val_rate < base.val_rate and p_val < alpha)
        exp.append(f"Held-out check ({ch_res.val_shots:,} fresh shots): chosen {_pct(ch_res.val_rate)} vs "
                   f"baseline {_pct(base.val_rate)} -> improvement "
                   f"{'CONFIRMED' if hv['holds'] else 'NOT confirmed (treat with caution)'}.")
        if not hv["holds"]:
            warn.append("The improvement was not confirmed on held-out data; increase shots/trials.")
    confident = chosen["key"] != BASELINE and hv.get("holds", False)
    if chosen["key"] == BASELINE:
        confident = not warn
    exp.append("Noise values are SIMULATED settings chosen by the user, not measured from hardware.")
    return Recommendation(chosen["key"], chosen["label"], confident, headline, exp, rows,
                          [x["key"] for x in tied], [x["key"] for x in over],
                          unconstrained["key"], hv, warn)


def pareto_frontier(results) -> list:
    """Strategies not dominated in (physical qubits, failure rate): the reliability-vs-resource frontier."""
    pts = sorted(results, key=lambda r: (r.n_qubits, r.rate))
    front, best = [], float("inf")
    for r in pts:
        if r.rate < best - 1e-12:
            front.append(r.key)
            best = r.rate
    return front
