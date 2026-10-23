"""Adaptive strategy selectors for Q-Adapt.

All selectors consume a ``NoiseProfile`` (the analyzer's *measured* estimate of the noise,
never the hidden NoiseSpec) plus optional resource ``Constraints``, and return the name of a
strategy.

* ``RuleBasedSelector``  - transparent if/else rules on dominant noise type and noise level.
* ``LookupSelector``     - resource-aware k-nearest-neighbour lookup into the benchmark table:
                           averages the measured logical error of every strategy over the
                           most similar benchmarked noise conditions, drops strategies that
                           violate the budget, then picks the *cheapest near-optimal* one.
* ``TreeSelector``       - small decision tree trained on the same table (optional ML).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .analyzer import NoiseProfile
from .codes import STRATEGIES, strategy_cost

EPS = 1e-5


@dataclass(frozen=True)
class Constraints:
    max_qubits: Optional[int] = None
    max_depth: Optional[int] = None

    def allows(self, strategy: str, idle: int) -> bool:
        c = strategy_cost(strategy if strategy != "readout_mitigation" else "none", idle)
        if self.max_qubits is not None and c["n_qubits"] > self.max_qubits:
            return False
        if self.max_depth is not None and c["depth"] > self.max_depth:
            return False
        return True


def feasible(constraints: Optional[Constraints], idle: int,
             pool: Sequence[str] = STRATEGIES) -> List[str]:
    cons = constraints or Constraints()
    out = [s for s in pool if cons.allows(s, idle)]
    return out or ["none"]          # the bare qubit is always available


def features(profile: NoiseProfile) -> np.ndarray:
    """3-D log-scale feature vector: bit-type error, phase-type error, readout error
    (all per workload block)."""
    ro = 0.5 * (profile.ro0 + profile.ro1)
    return np.log10(np.maximum([profile.bit_flip + profile.decay, profile.phase_flip, ro], EPS))


def _cost_key(strategy: str, idle: int):
    c = strategy_cost(strategy if strategy != "readout_mitigation" else "none", idle)
    # cheapest = fewest qubits, then fewest gates; mitigation ranks just after the bare qubit
    return (c["n_qubits"], c["cx_count"], strategy == "readout_mitigation")


def pick_cheapest_near_optimal(ler: Dict[str, float], idle: int, allowed: Sequence[str],
                               rel_tol: float = 0.10, abs_tol: float = 0.0015) -> str:
    """Among allowed strategies whose error is within a noise-sized margin of the best,
    return the cheapest.  Avoids paying qubits for gains smaller than shot noise."""
    allowed = [s for s in allowed if s in ler]
    best = min(ler[s] for s in allowed)
    margin = max(abs_tol, rel_tol * best)
    near = [s for s in allowed if ler[s] <= best + margin]
    return min(near, key=lambda s: _cost_key(s, idle))


# --------------------------------------------------------------------------------------
# 1. rule based
# --------------------------------------------------------------------------------------
@dataclass
class RuleBasedSelector:
    """Hand-written policy.  Thresholds are *block-level* error probabilities."""
    low: float = 0.004       # below this, a code's own overhead costs more than it saves
    high: float = 0.20       # above this, noise is past break-even: nothing helps
    ratio: float = 3.0       # bias needed to call noise bit- or phase-dominated
    ro_gate: float = 0.003   # readout error worth mitigating
    ro_ratio: float = 1.0    # readout counts as dominant when ro >= ro_ratio * storage error

    name = "rule"

    def select(self, profile: NoiseProfile, idle: int = 100,
               constraints: Optional[Constraints] = None) -> str:
        ok = set(feasible(constraints, idle))
        bit = profile.bit_flip + profile.decay
        ph = profile.phase_flip
        ro = 0.5 * (profile.ro0 + profile.ro1)
        total = bit + ph

        def first(*cands):
            for c in cands:
                if c in ok:
                    return c
            return "none"

        mitigate = ro > self.ro_gate
        bare = first("readout_mitigation", "none") if mitigate else "none"

        if total < self.low or total > self.high or (mitigate and ro >= self.ro_ratio * total):
            return bare
        if bit > self.ratio * max(ph, EPS):
            return first("bitflip3", "shor9", bare)
        if ph > self.ratio * max(bit, EPS):
            return first("phaseflip3", "shor9", bare)
        return first("shor9", bare)


    def fit(self, profiles: Sequence[NoiseProfile], ler_table: List[Dict[str, float]],
            idle: int = 100) -> "RuleBasedSelector":
        """Tune the five thresholds by grid search to minimise mean logical error on the
        *training* conditions (the rule structure itself stays hand-written)."""
        import itertools
        best, best_cfg = float("inf"), None
        for low, high, ratio, rg, rr in itertools.product(
                [0.001, 0.002, 0.004, 0.008, 0.016], [0.05, 0.1, 0.2, 0.4],
                [1.5, 2.0, 3.0, 5.0], [0.002, 0.005], [0.5, 1.0, 2.0]):
            cand = RuleBasedSelector(low, high, ratio, rg, rr)
            err = np.mean([l[cand.select(pr, idle)] for pr, l in zip(profiles, ler_table)])
            if err < best - 1e-12:      # strict: first (simplest) config wins ties
                best, best_cfg = err, (low, high, ratio, rg, rr)
        self.low, self.high, self.ratio, self.ro_gate, self.ro_ratio = best_cfg
        return self


# --------------------------------------------------------------------------------------
# 2. lookup (resource-aware kNN on the benchmark table)
# --------------------------------------------------------------------------------------
class LookupSelector:
    name = "lookup"

    def __init__(self, k: int = 3, rel_tol: float = 0.10, abs_tol: float = 0.0015):
        self.k, self.rel_tol, self.abs_tol = k, rel_tol, abs_tol
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.strats: List[str] = []

    def fit(self, X: np.ndarray, ler_table: List[Dict[str, float]]):
        """X: (n,3) feature matrix; ler_table[i][strategy] = measured logical error."""
        self.strats = list(ler_table[0].keys())
        self.X = np.asarray(X, float)
        self.Y = np.array([[row[s] for s in self.strats] for row in ler_table])
        return self

    def predict_ler(self, profile: NoiseProfile) -> Dict[str, float]:
        d = np.linalg.norm(self.X - features(profile), axis=1)
        idx = np.argsort(d)[: self.k]
        w = 1.0 / (d[idx] + 0.05)
        mean = (self.Y[idx] * w[:, None]).sum(0) / w.sum()
        return dict(zip(self.strats, mean))

    def select(self, profile: NoiseProfile, idle: int = 100,
               constraints: Optional[Constraints] = None) -> str:
        pred = self.predict_ler(profile)
        return pick_cheapest_near_optimal(pred, idle, feasible(constraints, idle, self.strats),
                                          self.rel_tol, self.abs_tol)


# --------------------------------------------------------------------------------------
# 3. decision tree (optional ML)
# --------------------------------------------------------------------------------------
class TreeSelector:
    name = "tree"

    def __init__(self, max_depth: int = 5, random_state: int = 0):
        from sklearn.tree import DecisionTreeClassifier
        self.clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=3,
                                          random_state=random_state)

    def fit(self, X: np.ndarray, labels: Sequence[str]):
        self.clf.fit(np.asarray(X, float), list(labels))
        return self

    def select(self, profile: NoiseProfile, idle: int = 100,
               constraints: Optional[Constraints] = None) -> str:
        ok = feasible(constraints, idle)
        proba = self.clf.predict_proba(features(profile)[None, :])[0]
        ranked = sorted(zip(self.clf.classes_, proba), key=lambda t: -t[1])
        for s, _ in ranked:
            if s in ok:
                return s
        return "none"
