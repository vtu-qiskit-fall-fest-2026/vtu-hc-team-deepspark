"""Q-Adapt: adaptive quantum error-correction strategy selection (Qiskit + Aer)."""
from __future__ import annotations

from typing import Dict, Optional

from .analyzer import NoiseProfile, characterize
from .codes import DESCRIPTION, STRATEGIES, strategy_cost
from .metrics import evaluate
from .noise import NoiseSpec, build_noise_model, make_profile
from .selector import Constraints, LookupSelector, RuleBasedSelector, TreeSelector

__all__ = ["run", "characterize", "evaluate", "NoiseSpec", "make_profile", "build_noise_model",
           "Constraints", "RuleBasedSelector", "LookupSelector", "TreeSelector", "STRATEGIES"]


def run(noise, idle: int = 100, max_qubits: Optional[int] = None, max_depth: Optional[int] = None,
        selector=None, shots: int = 4000, seed: int = 7) -> Dict:
    """One-call demo: analyse ``noise`` -> choose a strategy -> run it and a baseline.

    ``noise`` may be a ``NoiseSpec`` or any Qiskit-Aer ``NoiseModel``.
    ``selector`` defaults to the transparent ``RuleBasedSelector``; pass a fitted
    ``LookupSelector`` / ``TreeSelector`` for the data-driven policies.
    """
    selector = selector or RuleBasedSelector()
    profile = characterize(noise, idle=idle, seed=seed)
    cons = Constraints(max_qubits=max_qubits, max_depth=max_depth)
    choice = selector.select(profile, idle, cons)
    ro = (profile.ro0, profile.ro1)
    chosen = evaluate(choice, noise, idle=idle, shots=shots, seed=seed, ro_estimate=ro)
    baseline = evaluate("none", noise, idle=idle, shots=shots, seed=seed, ro_estimate=ro)
    return {"noise_profile": profile.as_dict(), "strategy": choice,
            "description": DESCRIPTION[choice], "result": chosen, "uncorrected": baseline,
            "improvement": baseline["logical_error_rate"] - chosen["logical_error_rate"]}
