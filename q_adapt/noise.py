"""Noise models for Q-Adapt.

Everything is described by a small, serialisable ``NoiseSpec`` so that the same noise
condition can be rebuilt as a Qiskit-Aer ``NoiseModel`` anywhere (also inside worker
processes).

Conventions
-----------
* ``p`` is the *total* Pauli-error probability per single-qubit gate / idle step.
* Noise is attached to the gates that appear inside the protected region
  (``id`` = idle/storage step, ``h``, ``t``, ``tdg``, ``cx``).
* State preparation (``x``, ``ry``) and the final basis rotation are ideal, so the
  experiment isolates the effect of *memory* noise and of the *QEC overhead itself*.
* Two-qubit gates are ``cx_factor`` times noisier (per qubit) than one-qubit gates.
* Readout error is applied to measurements only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    amplitude_damping_error,
    pauli_error,
)

NOISY_1Q_GATES = ["id", "h", "t", "tdg"]
NOISY_2Q_GATES = ["cx"]

PROFILE_NAMES = ["bitflip", "phaseflip", "depolarizing", "amp_damping", "mixed", "readout"]


@dataclass(frozen=True)
class NoiseSpec:
    px: float = 0.0          # X (bit-flip) probability per 1q gate
    py: float = 0.0          # Y probability per 1q gate
    pz: float = 0.0          # Z (phase-flip) probability per 1q gate
    gamma: float = 0.0       # amplitude-damping (T1-like) strength per 1q gate
    ro0: float = 0.0         # P(read 1 | true 0)
    ro1: float = 0.0         # P(read 0 | true 1)
    cx_factor: float = 2.0   # 2q-gate error = cx_factor * 1q error (per qubit)
    label: str = "custom"

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    @property
    def gate_error(self) -> float:
        return self.px + self.py + self.pz + self.gamma


# --------------------------------------------------------------------------------------
# Spec -> Aer NoiseModel
# --------------------------------------------------------------------------------------
def _one_qubit_error(px: float, py: float, pz: float, gamma: float):
    err = None
    ptot = px + py + pz
    if ptot > 0:
        # keep the probabilities a valid distribution
        if ptot > 0.75:
            s = 0.75 / ptot
            px, py, pz = px * s, py * s, pz * s
            ptot = 0.75
        err = pauli_error([("X", px), ("Y", py), ("Z", pz), ("I", max(0.0, 1.0 - ptot))])
    if gamma > 0:
        ad = amplitude_damping_error(min(gamma, 1.0))
        err = ad if err is None else err.compose(ad)
    return err


def build_noise_model(spec) -> NoiseModel:
    """Turn a NoiseSpec into a Qiskit-Aer NoiseModel (an existing NoiseModel passes through)."""
    if isinstance(spec, NoiseModel):
        return spec
    nm = NoiseModel()

    e1 = _one_qubit_error(spec.px, spec.py, spec.pz, spec.gamma)
    if e1 is not None:
        nm.add_all_qubit_quantum_error(e1, NOISY_1Q_GATES)
        f = spec.cx_factor
        e1c = _one_qubit_error(spec.px * f, spec.py * f, spec.pz * f, spec.gamma * f)
        nm.add_all_qubit_quantum_error(e1c.tensor(e1c), NOISY_2Q_GATES)

    if spec.ro0 > 0 or spec.ro1 > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - spec.ro0, spec.ro0], [spec.ro1, 1 - spec.ro1]])
        )
    return nm


# --------------------------------------------------------------------------------------
# Named profiles (used for the structured benchmark sweep)
# --------------------------------------------------------------------------------------
def make_profile(name: str, p: float, cx_factor: float = 2.0) -> NoiseSpec:
    """Noise profile ``name`` at strength ``p``."""
    if name == "bitflip":
        return NoiseSpec(px=p, cx_factor=cx_factor, label=name)
    if name == "phaseflip":
        return NoiseSpec(pz=p, cx_factor=cx_factor, label=name)
    if name == "depolarizing":
        return NoiseSpec(px=p / 3, py=p / 3, pz=p / 3, cx_factor=cx_factor, label=name)
    if name == "amp_damping":
        return NoiseSpec(gamma=p, cx_factor=cx_factor, label=name)
    if name == "mixed":  # bit+phase flips plus moderate readout error
        return NoiseSpec(px=0.4 * p, py=0.2 * p, pz=0.4 * p, ro0=p / 2, ro1=p / 2,
                         cx_factor=cx_factor, label=name)
    if name == "readout":  # readout-dominated: weak gate noise, strong (asymmetric) readout error
        return NoiseSpec(px=p / 90, py=p / 90, pz=p / 90, ro0=p, ro1=min(0.4, 2 * p),
                         cx_factor=cx_factor, label=name)
    raise ValueError(f"unknown profile {name!r}")


# --------------------------------------------------------------------------------------
# Random continuous noise (used to train / test the adaptive selector on unseen conditions)
# --------------------------------------------------------------------------------------
def random_spec(rng: np.random.Generator, g_log10=(-4.5, -2.5)) -> NoiseSpec:
    """Sample a random noise condition from a broad continuous family.

    ``g_log10`` is the log10-range of the overall gate-noise scale.  The default spans the
    regime where correction is sometimes, but not always, worthwhile for a 100-step memory.
    """
    g = 10 ** rng.uniform(*g_log10)                   # overall gate-noise scale
    w = rng.dirichlet([0.3, 0.15, 0.3, 0.2])          # sparse, biased mix of X / Y / Z / T1
    ro_scale = 10 ** rng.uniform(-4.0, -2.0)          # readout scale (independent)
    ro0 = ro_scale
    ro1 = min(0.4, ro_scale * rng.uniform(1.0, 3.0))
    return NoiseSpec(
        px=g * w[0], py=g * w[1], pz=g * w[2], gamma=g * w[3],
        ro0=float(ro0), ro1=float(ro1),
        cx_factor=float(rng.uniform(1.0, 4.0)),
        label="random",
    )
