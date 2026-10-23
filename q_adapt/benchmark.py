"""Benchmark drivers: simulate every strategy under many noise conditions."""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .analyzer import characterize
from .codes import STRATEGIES
from .metrics import evaluate
from .noise import NoiseSpec, make_profile, random_spec

LONG_COLS = ["profile", "p", "strategy", "logical_error_rate", "success_probability", "stderr",
             "depth", "n_qubits", "qubit_overhead", "cx_count", "gate_count"]


def _eval_one(args) -> Dict:
    """One noise condition: calibrate once, then simulate every strategy."""
    spec, idle, shots, seed, tag, p = args
    prof = characterize(spec, idle=idle, seed=seed + 5)
    rows = []
    for s in STRATEGIES:
        m = evaluate(s, spec, idle=idle, shots=shots, seed=seed,
                     ro_estimate=(prof.ro0, prof.ro1))
        m.update(profile=tag, p=p, strategy=s,
                 stderr=float(np.sqrt(max(m["logical_error_rate"], 1e-9)
                                      * (1 - m["logical_error_rate"]) / (4 * shots))))
        rows.append(m)
    return {"spec": spec, "tag": tag, "p": p, "profile": prof, "rows": rows}


def run_many(jobs: List[Tuple], workers: int = 2) -> List[Dict]:
    if workers <= 1:
        return [_eval_one(j) for j in jobs]
    # 'spawn' avoids fork-after-OpenMP deadlocks once Aer has run in the parent process
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_eval_one, jobs, chunksize=2))


def structured_jobs(profiles: Sequence[str], ps: Sequence[float], idle: int, shots: int,
                    seed: int = 1000) -> List[Tuple]:
    return [(make_profile(name, p), idle, shots, seed + i, name, float(p))
            for name in profiles for i, p in enumerate(ps)]


def random_jobs(n: int, idle: int, shots: int, rng_seed: int, sim_seed: int,
                tag: str = "random") -> List[Tuple]:
    rng = np.random.default_rng(rng_seed)
    return [(random_spec(rng), idle, shots, sim_seed + i, tag, float("nan")) for i in range(n)]


def to_long(results: Iterable[Dict]) -> pd.DataFrame:
    rows = [r for res in results for r in res["rows"]]
    return pd.DataFrame(rows)[LONG_COLS]


def to_wide(results: Iterable[Dict]) -> pd.DataFrame:
    """One row per noise condition: estimated profile + LER of every strategy."""
    out = []
    for res in results:
        d = {"tag": res["tag"], "p": res["p"], **res["profile"].as_dict()}
        d.update({f"spec_{k}": v for k, v in res["spec"].as_dict().items() if k != "label"})
        for r in res["rows"]:
            d[f"ler_{r['strategy']}"] = r["logical_error_rate"]
        out.append(d)
    return pd.DataFrame(out)
