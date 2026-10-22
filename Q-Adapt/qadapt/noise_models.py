"""Configurable *simulated* noise for Q-Adapt (built on Qiskit Aer's NoiseModel).

IMPORTANT: every noise parameter here is a user-chosen simulation setting.
Nothing is estimated from real quantum hardware.

Noise channels
--------------
* p_x   : bit-flip probability   -> Pauli X applied to each qubit during the storage ("idle") step
* p_z   : phase-flip probability -> Pauli Z applied to each qubit during the storage step
          (X and Z flips are applied independently, so Y occurs with prob. ~ p_x * p_z)
* p_gate: two-qubit (CX) depolarizing parameter. Encoding/decoding gates are therefore
          NOT ideal when p_gate > 0, which exposes the cost of correction circuits.
* p_meas: symmetric classical read-out error on every measured qubit.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from qiskit_aer.noise import NoiseModel, pauli_error, depolarizing_error, ReadoutError

STORAGE_GATE = "id"  # the identity gate marks the noisy storage window


@dataclass(frozen=True)
class NoiseConfig:
    p_x: float = 0.0
    p_z: float = 0.0
    p_gate: float = 0.0
    p_meas: float = 0.0

    def __post_init__(self):
        for name, v in asdict(self).items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")

    def as_dict(self) -> dict:
        return asdict(self)

    def as_tuple(self) -> tuple:
        return (self.p_x, self.p_z, self.p_gate, self.p_meas)

    def with_param(self, name: str, value: float) -> "NoiseConfig":
        d = asdict(self)
        if name not in d:
            raise KeyError(name)
        d[name] = float(value)
        return NoiseConfig(**d)

    @property
    def is_noiseless(self) -> bool:
        return not any(self.as_tuple())


NOISE_PARAMS = {
    "p_x": "Bit-flip probability (X) per qubit",
    "p_z": "Phase-flip probability (Z) per qubit",
    "p_gate": "CX gate depolarizing parameter",
    "p_meas": "Read-out (measurement) error",
}


def build_noise_model(cfg: NoiseConfig) -> NoiseModel | None:
    """Translate a NoiseConfig into an Aer NoiseModel (None if noiseless)."""
    if cfg.is_noiseless:
        return None
    nm = NoiseModel(basis_gates=["id", "h", "x", "sx", "rz", "cx"])
    if cfg.p_x > 0 or cfg.p_z > 0:
        px, pz = cfg.p_x, cfg.p_z
        err = pauli_error([
            ("X", px * (1 - pz)),
            ("Z", pz * (1 - px)),
            ("Y", px * pz),
            ("I", (1 - px) * (1 - pz)),
        ])
        nm.add_all_qubit_quantum_error(err, [STORAGE_GATE])
    if cfg.p_gate > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(cfg.p_gate, 2), ["cx"])
    if cfg.p_meas > 0:
        q = cfg.p_meas
        nm.add_all_qubit_readout_error(ReadoutError([[1 - q, q], [q, 1 - q]]))
    return nm


def describe_noise(cfg: NoiseConfig) -> str:
    """Plain-language profile of which channel dominates (descriptive only; the
    adaptive engine does NOT use this text to make decisions)."""
    if cfg.is_noiseless:
        return "noiseless (ideal) simulation"
    ch = {"bit-flip": cfg.p_x, "phase-flip": cfg.p_z,
          "gate (CX)": cfg.p_gate, "read-out": cfg.p_meas}
    total = sum(ch.values())
    ranked = sorted(ch.items(), key=lambda kv: -kv[1])
    top, topv = ranked[0]
    if topv / total >= 0.6:
        return f"{top}-dominated noise ({topv/total:.0%} of total error strength)"
    active = [k for k, v in ranked if v > 0]
    return "mixed noise: " + ", ".join(active)
