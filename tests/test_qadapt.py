"""Q-Adapt test-suite: ideal circuits, analytic noise checks, stats, decision logic."""
import math
import pytest
from qadapt.noise_models import NoiseConfig, build_noise_model, describe_noise
from qadapt.strategies import STRATEGIES, strategies_for_task, TASK_QUANTUM, TASK_CLASSICAL
from qadapt.circuits import transpiled_circuit, circuit_metrics, build_logical_circuit
from qadapt.benchmarks import run_point, results_to_df, run_circuit_counts
from qadapt.benchmarks import run_sweep_raw, sweep_tables, run_map_raw, map_choices
from qadapt.adaptive import recommend, pareto_frontier
from qadapt.stats import wilson_interval, improvement_p_value

SHOTS, TRIALS, SEED = 20000, 2, 123


def rate(results, key, basis=None):
    r = {x.key: x for x in results}[key]
    return r.basis_rate(basis) if basis else r.rate


def sigma(p, n):
    return math.sqrt(max(p * (1 - p), 1e-4) / n)


# ---------------------------------------------------------------- Stage 2: ideal circuits
@pytest.mark.parametrize("key", list(STRATEGIES))
def test_ideal_circuits_never_fail(key):
    for basis in STRATEGIES[key].bases:
        counts = run_circuit_counts(key, basis, NoiseConfig(), 500, SEED)
        from qadapt.benchmarks import count_failures
        assert count_failures(STRATEGIES[key], counts) == 0


def test_noise_model_none_when_noiseless():
    assert build_noise_model(NoiseConfig()) is None
    assert build_noise_model(NoiseConfig(p_x=0.1)) is not None


def test_invalid_noise_rejected():
    with pytest.raises(ValueError):
        NoiseConfig(p_x=1.5)


def test_overhead_metrics():
    assert circuit_metrics("none")["qubits"] == 1
    assert circuit_metrics("bit3_q")["qubits"] == 3
    assert circuit_metrics("bit5_q")["qubits"] == 5
    assert circuit_metrics("bit5_q")["cx_count"] > circuit_metrics("bit3_q")["cx_count"] > 0


# ---------------------------------------------------------------- Stage 3: baseline noise
@pytest.mark.parametrize("p", [0.05, 0.2])
def test_baseline_bitflip_rate_matches_p(p):
    res = run_point(NoiseConfig(p_x=p), TASK_QUANTUM, SHOTS, TRIALS, SEED)
    n = SHOTS * TRIALS
    assert abs(rate(res, "none", "Z") - p) < 5 * sigma(p, n)
    assert rate(res, "none", "X") == 0.0  # X flips do not change |+>


def test_baseline_phaseflip_hits_x_basis_only():
    p = 0.15
    res = run_point(NoiseConfig(p_z=p), TASK_QUANTUM, SHOTS, TRIALS, SEED)
    assert abs(rate(res, "none", "X") - p) < 5 * sigma(p, SHOTS * TRIALS)
    assert rate(res, "none", "Z") == 0.0


def test_readout_error_matches_q():
    q = 0.1
    res = run_point(NoiseConfig(p_meas=q), TASK_QUANTUM, SHOTS, TRIALS, SEED)
    assert abs(rate(res, "none") - q) < 5 * sigma(q, SHOTS * TRIALS * 2)


# ---------------------------------------------------------------- Stage 4: correction
def test_bit3_matches_analytic_formula():
    p = 0.1
    expected = 3 * p**2 - 2 * p**3       # majority fails when >= 2 of 3 qubits flip
    res = run_point(NoiseConfig(p_x=p), TASK_QUANTUM, SHOTS, TRIALS, SEED)
    assert abs(rate(res, "bit3_q", "Z") - expected) < 5 * sigma(expected, SHOTS * TRIALS)
    assert rate(res, "bit3_q", "X") == 0.0   # residual logical-X is harmless on |+>


def test_bit5_matches_analytic_formula():
    p = 0.15
    expected = sum(math.comb(5, k) * p**k * (1 - p)**(5 - k) for k in (3, 4, 5))
    res = run_point(NoiseConfig(p_x=p), TASK_QUANTUM, SHOTS, TRIALS, SEED)
    assert abs(rate(res, "bit5_q", "Z") - expected) < 5 * sigma(expected, SHOTS * TRIALS)


def test_phase3_matches_analytic_formula():
    p = 0.1
    expected = 3 * p**2 - 2 * p**3
    res = run_point(NoiseConfig(p_z=p), TASK_QUANTUM, SHOTS, TRIALS, SEED)
    # Majority decoding in the Hadamard frame fails (>=2 Z flips) as a residual logical
    # bit-flip, which the |0> (Z-readout) test sees; the |+> test is immune to it.
    assert abs(rate(res, "phase3_q", "Z") - expected) < 5 * sigma(expected, SHOTS * TRIALS)
    assert rate(res, "phase3_q", "X") == 0.0


def test_bitflip_code_is_worse_under_phase_noise():
    """A bit-flip code does NOT protect against phase flips - it amplifies them."""
    p = 0.1
    res = run_point(NoiseConfig(p_z=p), TASK_QUANTUM, SHOTS, TRIALS, SEED)
    expected = (1 - (1 - 2 * p) ** 3) / 2     # odd number of Z errors flips the logical phase
    assert abs(rate(res, "bit3_q", "X") - expected) < 5 * sigma(expected, SHOTS * TRIALS)
    assert rate(res, "bit3_q") > rate(res, "none")


def test_vote_decoder_corrects_bitflips_and_readout():
    res = run_point(NoiseConfig(p_x=0.1), TASK_CLASSICAL, SHOTS, TRIALS, SEED)
    expected = 3 * 0.1**2 - 2 * 0.1**3
    assert abs(rate(res, "bit3_v") - expected) < 5 * sigma(expected, SHOTS * TRIALS)
    res = run_point(NoiseConfig(p_meas=0.1), TASK_CLASSICAL, SHOTS, TRIALS, SEED)
    assert rate(res, "bit3_v") < rate(res, "none")  # voting also cancels read-out errors


def test_task_strategy_sets():
    q = {s.key for s in strategies_for_task(TASK_QUANTUM)}
    c = {s.key for s in strategies_for_task(TASK_CLASSICAL)}
    assert "bit3_v" not in q and "phase3_q" not in c   # votes are classical-only
    assert "none" in q and "none" in c


def test_gate_noise_can_make_correction_worse():
    res = run_point(NoiseConfig(p_x=0.02, p_gate=0.05), TASK_QUANTUM, 4000, 2, SEED)
    assert rate(res, "bit3_q") > rate(res, "none")


# ---------------------------------------------------------------- Stats
def test_wilson_and_pvalue():
    lo, hi = wilson_interval(50, 1000)
    assert lo < 0.05 < hi
    assert improvement_p_value(100, 1000, 10, 1000) < 1e-10
    assert improvement_p_value(10, 1000, 100, 1000) > 0.99
    assert improvement_p_value(0, 1000, 0, 1000) == 0.5


# ---------------------------------------------------------------- Stage 6: decision engine
def test_engine_respects_budget_and_picks_bit_code_for_bitflips():
    res = run_point(NoiseConfig(p_x=0.1), TASK_QUANTUM, 4000, 4, SEED)
    r5 = recommend(res, budget=5)
    assert r5.chosen in ("bit3_q", "bit5_q")
    r3 = recommend(res, budget=3)
    assert r3.chosen == "bit3_q" and "bit5_q" in r3.excluded_budget
    r1 = recommend(res, budget=1)
    assert r1.chosen == "none"


def test_engine_picks_phase_code_for_phase_noise():
    res = run_point(NoiseConfig(p_z=0.1), TASK_QUANTUM, 4000, 4, SEED)
    assert recommend(res, 3).chosen == "phase3_q"


def test_engine_falls_back_to_baseline_without_evidence():
    res = run_point(NoiseConfig(p_x=0.01, p_gate=0.1), TASK_QUANTUM, 4000, 4, SEED)
    rec = recommend(res, 5)
    assert rec.chosen == "none"
    assert any("NOT better" in line or "No eligible" in line for line in rec.explanation)


def test_engine_noiseless_returns_baseline_with_warning():
    res = run_point(NoiseConfig(), TASK_QUANTUM, 1000, 2, SEED)
    rec = recommend(res, 5)
    assert rec.chosen == "none" and rec.warnings


def test_resource_weight_changes_choice():
    res = run_point(NoiseConfig(p_x=0.1), TASK_QUANTUM, 4000, 4, SEED)
    assert recommend(res, 5, lam=0.0).chosen in ("bit3_q", "bit5_q")
    assert recommend(res, 5, lam=0.5).chosen == "none"      # absurdly expensive qubits


def test_engine_uses_measured_data_only():
    """Same results object -> same answer; different noise -> (can be) different answer."""
    a = run_point(NoiseConfig(p_x=0.1), TASK_QUANTUM, 2000, 4, SEED)
    b = run_point(NoiseConfig(p_z=0.1), TASK_QUANTUM, 2000, 4, SEED)
    assert recommend(a, 3).chosen != recommend(b, 3).chosen
    assert recommend(a, 3).chosen == recommend(a, 3).chosen


def test_reproducible_with_seed():
    a = run_point(NoiseConfig(p_x=0.1), TASK_QUANTUM, 1000, 2, 7)
    b = run_point(NoiseConfig(p_x=0.1), TASK_QUANTUM, 1000, 2, 7)
    assert [r.fail for r in a] == [r.fail for r in b]


def test_pareto_frontier_contains_baseline():
    res = run_point(NoiseConfig(p_x=0.1), TASK_QUANTUM, 2000, 2, SEED)
    assert pareto_frontier(res)[0] == "none"


# ---------------------------------------------------------------- Stage 5/7: sweeps + map
def test_sweep_and_adaptive_tables():
    raw = run_sweep_raw("p_x", [0.0, 0.05, 0.15], NoiseConfig(), TASK_QUANTUM, 1000, 2, SEED)
    df, ad = sweep_tables(raw, "p_x", budget=3, lam=0.005, alpha=0.05)
    assert len(ad) == 3 and set(df["p_x"]) == {0.0, 0.05, 0.15}
    assert ad.iloc[0]["adaptive_choice"] == "none"          # no noise -> no correction
    assert ad.iloc[2]["adaptive_choice"] == "bit3_q"


def test_regime_map():
    raw = run_map_raw("p_x", [0.0, 0.1], "p_z", [0.0, 0.1], NoiseConfig(), TASK_QUANTUM, 1000, 2, SEED)
    m = map_choices(raw, 3, 0.005, 0.05)
    got = {(r.p_x, r.p_z): r.best_strategy for r in m.itertuples()}
    assert got[(0.0, 0.0)] == "none"
    assert got[(0.1, 0.0)] == "bit3_q"
    assert got[(0.0, 0.1)] == "phase3_q"
