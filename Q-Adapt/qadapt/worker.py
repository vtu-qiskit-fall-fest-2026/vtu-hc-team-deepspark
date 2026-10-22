"""Single long-lived worker thread for ALL Qiskit work when running inside Streamlit.

Why this exists (found while testing): in this environment (Qiskit 2.6 / Aer 0.18 /
Streamlit 1.66 / Python 3.13) constructing ANY QuantumCircuit inside a *fresh* Streamlit
script thread segfaults on the second script run. Streamlit starts a new script thread
for every rerun, so the real server died on the first widget interaction.
Running every Qiskit call on one persistent thread avoids the crash (verified with six
consecutive AppTest runs and with a real browser session).  Outside Streamlit (CLI, tests)
you do not need this module.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import threading

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qadapt-qiskit")
        return _pool


def call_in_worker(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) on the dedicated Qiskit thread and return its result."""
    return _get_pool().submit(fn, *args, **kwargs).result()
