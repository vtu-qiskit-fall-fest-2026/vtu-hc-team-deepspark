"""Evaluation of one (strategy, noise) pair.

Logical error rate is averaged over four logical input states (|0>, |1>, |+>, |->) so that
both X-type and Z-type logical errors are counted fairly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from qiskit_aer import AerSimulator

from .codes import N_PHYS, STATES, build_full, compile_circuit, strategy_cost
from .noise import NoiseSpec, build_noise_model

DEFAULT_IDLE = 100        # number of noisy storage steps the logical qubit must survive
DEFAULT_SHOTS = 3000      # shots per logical state (x4 states = 12000 per evaluation)


@lru_cache(maxsize=None)
def _compiled(strategy: str, state: str, idle: int):
    return compile_circuit(build_full(strategy, state, idle))


def _mitigate(p_wrong_obs: float, ro0: float, ro1: float, expected: int) -> float:
    """Invert the 2x2 readout confusion matrix; return corrected P(correct outcome)."""
    M = np.array([[1 - ro0, ro1], [ro0, 1 - ro1]])        # columns = true value
    p_obs = np.array([1 - p_wrong_obs, p_wrong_obs]) if expected == 0 else np.array([p_wrong_obs, 1 - p_wrong_obs])
    p_true = np.linalg.solve(M, p_obs)
    return float(np.clip(p_true[expected], 0.0, 1.0))


def evaluate(strategy: str, spec: NoiseSpec, idle: int = DEFAULT_IDLE,
             shots: int = DEFAULT_SHOTS, seed: int = 1234,
             ro_estimate: Optional[tuple] = None) -> Dict[str, float]:
    """Simulate ``strategy`` under ``spec`` and return reliability + resource metrics.

    ``ro_estimate`` = (ro0, ro1) used by ``readout_mitigation``; defaults to the true values
    (i.e. a perfectly calibrated mitigation) when omitted.
    """
    base = "none" if strategy == "readout_mitigation" else strategy
    sim = AerSimulator(noise_model=build_noise_model(spec), method="automatic")
    if ro_estimate is not None:
        ro0, ro1 = ro_estimate
    elif isinstance(spec, NoiseSpec):
        ro0, ro1 = spec.ro0, spec.ro1
    else:  # a raw Aer NoiseModel carries no readable parameters: pass ro_estimate explicitly
        ro0 = ro1 = 0.0

    fails, succ_mit = [], []
    for k, st in enumerate(STATES):
        qc = _compiled(base, st, idle)
        counts = sim.run(qc, shots=shots, seed_simulator=seed + 17 * k).result().get_counts()
        expected = STATES[st]
        p_wrong = counts.get(str(1 - expected), 0) / shots
        fails.append(p_wrong)
        if strategy == "readout_mitigation":
            succ_mit.append(_mitigate(p_wrong, ro0, ro1, expected))

    if strategy == "readout_mitigation":
        success = float(np.mean(succ_mit))
        ler = 1.0 - success
    else:
        ler = float(np.mean(fails))
        success = 1.0 - ler

    cost = strategy_cost(base, idle)
    return {
        "logical_error_rate": ler,
        "success_probability": success,
        "depth": cost["depth"],
        "n_qubits": cost["n_qubits"],
        "qubit_overhead": cost["n_qubits"] / 1.0,   # logical qubits = 1
        "cx_count": cost["cx_count"],
        "gate_count": cost["gate_count"],
    }
