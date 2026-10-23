"""Run the full Q-Adapt experiment pipeline and write CSVs to results/.

    python experiments/run_all.py            # full run (~10-15 min on 2 cores)
    python experiments/run_all.py --quick    # tiny smoke run (~1 min)
"""
import argparse
import os
import sys
import time


import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from q_adapt.analyzer import NoiseProfile                      # noqa: E402
from q_adapt.benchmark import (random_jobs, run_many, structured_jobs,   # noqa: E402
                               to_long, to_wide)
from q_adapt.codes import STRATEGIES                           # noqa: E402
from q_adapt.noise import PROFILE_NAMES, make_profile          # noqa: E402
from q_adapt.selector import (Constraints, LookupSelector, RuleBasedSelector,   # noqa: E402
                              TreeSelector, features, feasible,
                              pick_cheapest_near_optimal)
from q_adapt.metrics import evaluate                           # noqa: E402
from q_adapt.codes import strategy_cost                        # noqa: E402

RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip simulation, re-fit/evaluate selectors from saved CSV tables")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    IDLE = 100
    SHOTS = 600 if args.quick else 3000
    PS = np.logspace(-4, np.log10(3e-2), 5 if args.quick else 13)
    N_TRAIN, N_TEST = (12, 8) if args.quick else (200, 120)
    t0 = time.time()


    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


    if not args.eval_only:
        # ---- 1. structured benchmark sweep (strategy x noise profile x strength) ------------------
        log("1/4 structured sweep")
        sweep = run_many(structured_jobs(PROFILE_NAMES, PS, IDLE, SHOTS), args.workers)
        to_long(sweep).to_csv(f"{RES}/benchmark_sweep.csv", index=False)

        # ---- 2. sensitivity to workload length (where is the break-even?) ---------------------------
        log("2/4 idle-length sensitivity")
        rows = []
        for prof in ["bitflip", "depolarizing"]:
            for idle in ([10, 100] if args.quick else [3, 10, 30, 100, 300, 1000]):
                for s in STRATEGIES:
                    m = evaluate(s, make_profile(prof, 3e-4), idle=idle, shots=SHOTS, seed=77)
                    m.update(profile=prof, idle=idle, strategy=s)
                    rows.append(m)
        pd.DataFrame(rows).to_csv(f"{RES}/idle_sensitivity.csv", index=False)

        # ---- 3. random train / held-out test noise ---------------------------------------------------
        log("3/4 random train + unseen test noise")
        train = run_many(random_jobs(N_TRAIN, IDLE, SHOTS, rng_seed=1, sim_seed=5000), args.workers)
        test = run_many(random_jobs(N_TEST, IDLE, SHOTS, rng_seed=2, sim_seed=90000), args.workers)
        train_w = pd.concat([to_wide(sweep), to_wide(train)], ignore_index=True)   # selectors see these only
        test_w = to_wide(test)
        train_w.to_csv(f"{RES}/selector_training_table.csv", index=False)
        test_w.to_csv(f"{RES}/test_table.csv", index=False)

    else:   # reuse the simulated tables from a previous run
        train_w = pd.read_csv(f"{RES}/selector_training_table.csv")
        test_w = pd.read_csv(f"{RES}/test_table.csv")

    # ---- 4. fit selectors, evaluate on unseen noise ---------------------------------------------
    log("4/4 fit + evaluate selectors")
    LER_COLS = [f"ler_{s}" for s in STRATEGIES]
    PROF_FIELDS = list(NoiseProfile.__dataclass_fields__)


    def row_profile(r) -> NoiseProfile:
        return NoiseProfile(**{k: r[k] for k in PROF_FIELDS})


    Xtr = np.array([features(row_profile(r)) for _, r in train_w.iterrows()])
    ler_tr = [{s: r[f"ler_{s}"] for s in STRATEGIES} for _, r in train_w.iterrows()]
    labels = [pick_cheapest_near_optimal(l, IDLE, STRATEGIES) for l in ler_tr]

    selectors = {
        "Q-Adapt (rule)": RuleBasedSelector().fit(
                [row_profile(r) for _, r in train_w.iterrows()], ler_tr, IDLE),
        "Q-Adapt (lookup)": LookupSelector(k=3).fit(Xtr, ler_tr),
    }
    try:   # optional ML selector (needs scikit-learn; see README note on libgomp)
        selectors["Q-Adapt (tree)"] = TreeSelector().fit(Xtr, labels)
    except ImportError as e:
        print("skipping tree selector:", e)
    fixed = {"Uncorrected": "none", "Fixed: bitflip3": "bitflip3",
             "Fixed: phaseflip3": "phaseflip3", "Fixed: shor9": "shor9"}


    def evaluate_policy(cons: Constraints):
        recs = []
        for _, r in test_w.iterrows():
            prof = row_profile(r)
            ler = {s: r[f"ler_{s}"] for s in STRATEGIES}
            ok = feasible(cons, IDLE)
            oracle = pick_cheapest_near_optimal(ler, IDLE, ok, rel_tol=0.0, abs_tol=0.0)
            near_opt_set = {s for s in ok if ler[s] <= ler[oracle] + max(0.0015, 0.10 * ler[oracle])}
            picks = {"Oracle (best in hindsight)": oracle}
            for name, sel in selectors.items():
                picks[name] = sel.select(prof, IDLE, cons)
            for name, s in fixed.items():
                picks[name] = s if s in ok else "none"
            for name, s in picks.items():
                c = strategy_cost("none" if s == "readout_mitigation" else s, IDLE)
                recs.append(dict(policy=name, chosen=s, ler=ler[s], oracle_ler=ler[oracle],
                                 exact=s == oracle, near_opt=s in near_opt_set,
                                 n_qubits=c["n_qubits"], depth=c["depth"], cx_count=c["cx_count"],
                                 none_ler=ler["none"], idx=r.name))
        return pd.DataFrame(recs)


    def summarize(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("policy", sort=False)
        out = g.agg(mean_logical_error=("ler", "mean"), mean_success=("ler", lambda x: 1 - x.mean()),
                    mean_depth=("depth", "mean"), mean_qubits=("n_qubits", "mean"),
                    mean_cx=("cx_count", "mean"), near_optimal_rate=("near_opt", "mean"),
                    exact_match_rate=("exact", "mean")).reset_index()
        out["mean_regret"] = out["mean_logical_error"] - df[df.policy.str.startswith("Oracle")]["ler"].mean()
        wins = df.assign(better=df.ler < df.none_ler - 0.0015, worse=df.ler > df.none_ler + 0.0015)
        out["pct_better_than_uncorrected"] = wins.groupby("policy", sort=False)["better"].mean().values
        out["pct_worse_than_uncorrected"] = wins.groupby("policy", sort=False)["worse"].mean().values
        return out


    unconstrained = evaluate_policy(Constraints())
    unconstrained.to_csv(f"{RES}/adaptive_eval_per_case.csv", index=False)
    summarize(unconstrained).to_csv(f"{RES}/adaptive_summary.csv", index=False)

    budget_rows = []
    for mq in [1, 3, 9]:
        s = summarize(evaluate_policy(Constraints(max_qubits=mq)))
        s.insert(0, "max_qubits", mq)
        budget_rows.append(s)
    pd.concat(budget_rows).to_csv(f"{RES}/budget_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print(summarize(unconstrained).round(4).to_string(index=False))
    log("done")



if __name__ == "__main__":
    main()
