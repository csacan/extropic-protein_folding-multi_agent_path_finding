"""Parallel (multiprocess) version of study_feasibility.py.

Each (K, instance) is independent -- CBS feasibility + a PT+FFBS solve -- so we farm
instances across a process pool. CBS is pure-Python and GIL-bound, so processes (not
threads) are what actually parallelize it; each worker is thread-capped (OMP/XLA = 1)
so N workers don't oversubscribe the cores. CPU-pinned, never touches the GPU.
"""

from __future__ import annotations

import os

# CPU-only, one thread per worker (set before any jax import, incl. spawned workers)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import jax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mapf_model as mapf
import pt_mapf as pt
import trajectory_sampler as ts
import cbs_mapf as cbs


def solve_one(job):
    """One instance: CBS verdict + PT+FFBS solve. Returns (K, verdict, solved)."""
    (K, starts, goals, rows, T, betas, n_systems, n_rounds, sweeps,
     lam_swap, cbs_budget, seed) = job
    model = pt.light_model(rows, starts, goals, T, lam_swap=lam_swap)
    verdict = cbs.feasibility(model, node_budget=cbs_budget)
    runner = ts.make_ffbs_runner(model, sweeps)
    res = pt.pt_solve(model, betas, n_systems, n_rounds, sweeps,
                      jax.random.key(seed), runner=runner)
    return (K, verdict, bool(res["solved_frac"] > 0))


def run(rows, T, K_values, n_instances, betas, n_systems, n_rounds, sweeps_per_round,
        lam_swap=1.0, cbs_budget=30000, seed=0, max_workers=16):
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    mid = len(rows) / 2.0
    sp = np.array([c for c in range(q) if coords[c][0] < mid])
    gp = np.array([c for c in range(q) if coords[c][0] > mid])
    rng = np.random.default_rng(seed)

    jobs, idx = [], 0
    for K in K_values:
        for _ in range(n_instances):
            s = rng.choice(sp, K, replace=False)
            g = rng.choice(gp, K, replace=False)
            starts = [tuple(int(x) for x in coords[c]) for c in s]
            goals = [tuple(int(x) for x in coords[c]) for c in g]
            jobs.append((K, starts, goals, rows, T, betas, n_systems, n_rounds,
                         sweeps_per_round, lam_swap, cbs_budget, 1000 + idx))
            idx += 1

    W = min(max_workers, len(jobs), os.cpu_count())
    print(f"dispatching {len(jobs)} instances across {W} workers (q={q}, T={T})", flush=True)
    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=W, mp_context=ctx) as ex:
        futs = [ex.submit(solve_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 10 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] completed", flush=True)

    rec = {K: [] for K in K_values}
    for K, verdict, solved in results:
        rec[K].append((verdict, solved))

    dens, feas_r, solv_r, cond_r, to_r = [], [], [], [], []
    print(f"\n  {'K':>3} {'dens':>6} {'feasible':>9} {'solved':>8} {'solve|feas':>11} {'timeouts':>9}")
    for K in K_values:
        pairs = rec[K]
        resolved = [(v, s) for v, s in pairs if v != "timeout"]
        feas = [v == "feasible" for v, _ in resolved]
        solved = [s for _, s in pairs]
        feas_only = [s for v, s in resolved if v == "feasible"]
        fr = float(np.mean(feas)) if feas else np.nan
        sr = float(np.mean(solved))
        cr = float(np.mean(feas_only)) if feas_only else np.nan
        tor = np.mean([v == "timeout" for v, _ in pairs])
        dens.append(K / q); feas_r.append(fr); solv_r.append(sr); cond_r.append(cr); to_r.append(tor)
        print(f"  {K:>3} {K/q:>6.3f} {fr:>9.2f} {sr:>8.2f} {cr:>11.2f} {tor:>9.2f}", flush=True)

    dens = np.array(dens)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(dens, feas_r, "o-", color="C2", label="feasible (CBS ground truth)")
    ax.plot(dens, solv_r, "s-", color="C0", label="sampler solved (PT+FFBS)")
    ax.plot(dens, cond_r, "^--", color="C1", label="findability  P(solved | feasible)")
    ax.fill_between(dens, solv_r, feas_r, color="red", alpha=0.12, label="missed-feasible gap")
    ax.plot(dens, to_r, ":", color="0.5", label="CBS timeout fraction")
    ax.set_xlabel("agent density  K / |cells|")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Feasible vs found  (grid {len(rows)}x{len(rows[0])}, T={T})")
    fig.tight_layout()
    fig.savefig("study_feasibility.png", dpi=130)
    print("\nsaved study_feasibility.png", flush=True)


if __name__ == "__main__":
    rows = [
        ".......",
        ".......",
        "###.###",
        ".......",
        ".......",
    ]
    run(rows, T=12, K_values=[2, 4, 6, 7, 8, 9, 10], n_instances=12,
        betas=[0.5, 1.0, 2.0, 4.0, 8.0], n_systems=64, n_rounds=20, sweeps_per_round=2,
        max_workers=16)
