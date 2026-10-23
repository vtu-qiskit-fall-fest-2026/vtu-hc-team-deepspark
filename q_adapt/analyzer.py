"""Noise analyzer: estimate a device's noise from small calibration circuits.

The analyzer only *sees the noisy backend through measurement counts* (like a real device
calibration would) and never reads the NoiseSpec parameters.

Calibration circuits (all use the same idle depth as the workload):
  |0> idle -> Z-measure   : flips of 0->1  = bit-flip (X/Y) errors + readout 0->1
  |1> idle -> Z-measure   : flips of 1->0  = bit-flip errors + T1 decay + readout 1->0
  |+> idle -> X-measure   : flips of + -> - = phase-flip (Z/Y) errors (+ readout)
  idle = 0 versions       : isolate pure readout error (no storage noise)

Output NoiseProfile fields are the *per-logical-step* effective rates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .codes import compile_circuit
from .noise import NoiseSpec, build_noise_model

CAL_SHOTS = 4000


@dataclass
class NoiseProfile:
    bit_flip: float        # net P(0->1) during idle block, readout removed
    decay: float           # extra P(1->0) beyond bit-flip + readout  (T1-like asymmetry)
    phase_flip: float      # net P(+ -> -) during idle block, readout removed
    ro0: float             # estimated readout error 0->1
    ro1: float             # estimated readout error 1->0
    gate_error: float      # total per-block error  ~ bit_flip + phase_flip - overlap
    dominant: str          # 'low' | 'bitflip' | 'phaseflip' | 'mixed' | 'readout'

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _cal_circuit(prep: str, idle: int) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    if prep == "1":
        qc.x(0)
    elif prep == "+":
        qc.ry(math.pi / 2, 0)
    qc.barrier()
    for _ in range(idle):
        qc.id(0)
        qc.barrier()
    if prep == "+":
        qc.ry(-math.pi / 2, 0)
    qc.measure(0, 0)
    return compile_circuit(qc)


def _p_one(sim, qc, shots, seed) -> float:
    c = sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    return c.get("1", 0) / shots


def _net(p_obs: float, ro_eff: float) -> float:
    """Remove a symmetric-ish readout contribution:  p_obs = p + ro - 2 p ro."""
    denom = 1 - 2 * ro_eff
    if denom <= 0.05:
        return max(0.0, p_obs)
    return min(1.0, max(0.0, (p_obs - ro_eff) / denom))


def characterize(spec, idle: int = 100, shots: int = CAL_SHOTS, seed: int = 99) -> NoiseProfile:
    """Probe a noisy backend (NoiseSpec or Aer NoiseModel) and return an estimated NoiseProfile."""
    sim = AerSimulator(noise_model=build_noise_model(spec), method="automatic")

    # 1. readout error, from idle=0 circuits  (state prep is ideal, only measurement is noisy)
    ro0 = _p_one(sim, _cal_circuit("0", 0), shots, seed)
    ro1 = 1 - _p_one(sim, _cal_circuit("1", 0), shots, seed + 1)
    ro_sym = 0.5 * (ro0 + ro1)

    # 2. storage noise, from idle=<workload idle> circuits
    p0 = _p_one(sim, _cal_circuit("0", idle), shots, seed + 2)             # 0 -> 1
    p1 = 1 - _p_one(sim, _cal_circuit("1", idle), shots, seed + 3)         # 1 -> 0
    pp = _p_one(sim, _cal_circuit("+", idle), shots, seed + 4)             # + -> -

    bit_flip = _net(p0, ro0)
    bit_flip_from_1 = _net(p1, ro1)
    decay = max(0.0, bit_flip_from_1 - bit_flip)        # T1 makes 1->0 more likely than 0->1
    phase_flip = _net(pp, ro_sym)

    gate_error = min(1.0, bit_flip + phase_flip + decay)
    return NoiseProfile(
        bit_flip=bit_flip, decay=decay, phase_flip=phase_flip,
        ro0=ro0, ro1=ro1, gate_error=gate_error,
        dominant=classify(bit_flip + decay, phase_flip, ro_sym),
    )


def classify(bit: float, phase: float, ro: float, low: float = 2e-3, ratio: float = 3.0) -> str:
    """Human-readable dominant noise type."""
    stor = bit + phase
    if stor < low and ro < low * 2:
        return "low"
    if ro > max(3 * stor, 0.02):
        return "readout"
    if bit > ratio * max(phase, 1e-9):
        return "bitflip"
    if phase > ratio * max(bit, 1e-9):
        return "phaseflip"
    return "mixed"
