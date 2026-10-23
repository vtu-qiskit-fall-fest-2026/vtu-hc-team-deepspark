"""Tests for the noise analyzer and selectors.   Run: python -m pytest -q tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q_adapt.analyzer import NoiseProfile, characterize
from q_adapt.noise import make_profile
from q_adapt.selector import Constraints, RuleBasedSelector, feasible, pick_cheapest_near_optimal


def prof(bit=0.0, phase=0.0, ro=0.0):
    return NoiseProfile(bit_flip=bit, decay=0.0, phase_flip=phase, ro0=ro, ro1=ro,
                        gate_error=bit + phase, dominant="x")


def test_rules_follow_dominant_noise():
    sel = RuleBasedSelector(low=0.004, high=0.2, ratio=3.0, ro_gate=0.003, ro_ratio=1.0)
    assert sel.select(prof(bit=0.05)) == "bitflip3"
    assert sel.select(prof(phase=0.05)) == "phaseflip3"
    assert sel.select(prof(bit=0.03, phase=0.03)) == "shor9"
    assert sel.select(prof(bit=0.0005)) == "none"            # too little noise to pay for a code
    assert sel.select(prof(bit=0.4)) == "none"                # hopelessly noisy
    assert sel.select(prof(bit=0.01, ro=0.05)) == "readout_mitigation"


def test_constraints_are_respected():
    sel = RuleBasedSelector()
    assert sel.select(prof(bit=0.03, phase=0.03), constraints=Constraints(max_qubits=3)) != "shor9"
    assert feasible(Constraints(max_qubits=1), 100) == ["none", "readout_mitigation"]
    assert feasible(Constraints(max_qubits=0), 100) == ["none"]        # bare qubit always allowed


def test_cheapest_near_optimal_prefers_fewer_qubits():
    ler = {"none": 0.0101, "bitflip3": 0.0100, "shor9": 0.0099}
    assert pick_cheapest_near_optimal(ler, 100, list(ler)) == "none"   # gains below shot noise
    ler = {"none": 0.10, "bitflip3": 0.02, "shor9": 0.019}
    assert pick_cheapest_near_optimal(ler, 100, list(ler)) == "bitflip3"


def test_analyzer_recovers_noise_type():
    bf = characterize(make_profile("bitflip", 3e-4), idle=100)
    pf = characterize(make_profile("phaseflip", 3e-4), idle=100)
    ro = characterize(make_profile("readout", 1e-2), idle=100)
    assert bf.bit_flip > 5 * bf.phase_flip
    assert pf.phase_flip > 5 * pf.bit_flip
    assert ro.ro0 > 0.005


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
