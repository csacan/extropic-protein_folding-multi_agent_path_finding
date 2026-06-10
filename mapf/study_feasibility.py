"""Feasibility-decomposed solvability study: separate 'the instance is infeasible' from
'the sampler missed a feasible plan', using CBS as the complete ground-truth oracle.

For each random bottleneck instance we record:
  - CBS verdict (feasible / infeasible / timeout)   -- the ground-truth ceiling
  - PT+FFBS sampler success (best U == 0)            -- what the EBM sampler found

Then per agent density we plot three curves:
  - feasibility rate   P(feasible)            : how many instances admit ANY valid plan
  - sampler success    P(solved)              : how many the sampler actually cracked
  - findability        P(solved | feasible)   : the sampler's efficiency on solvable ones

The gap (feasibility - sampler) is the sampler's missed-feasible; the drop in
feasibility is genuine infeasibility. CBS is complete but worst-case exponential, so
it times out in the dense regime -- we report that fraction (it is itself a signal that
those instances are hard) and compute rates over CBS-resolved instances only.
"""

from __future__ import annotations

import numpy as np
import jax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mapf_model as mapf
import pt_mapf as pt
import trajectory_sampler as ts
import cbs_mapf as cbs


def run(rows, T, K_values, n_instances, betas, n_systems, n_rounds, sweeps_per_round,
        lam_swap=1.0, cbs_budget=40000, seed=0):
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    mid = len(rows) / 2.0
    sp = np.array([c for c in range(q) if coords[c][0] < mid])
    gp = np.array([c for c in range(q) if coords[c][0] > mid])
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    rec = {K: dict(feas=[], solved=[], timeout=0) for K in K_values}
    for K in K_values:
        for _ in range(n_instances):
            s = rng.choice(sp, K, replace=False)
            g = rng.choice(gp, K, replace=False)
            starts = [tuple(int(x) for x in coords[c]) for c in s]
            goals = [tuple(int(x) for x in coords[c]) for c in g]
            model = pt.light_model(rows, starts, goals, T, lam_swap=lam_swap)

            verdict = cbs.feasibility(model, node_budget=cbs_budget)
            runner = ts.make_ffbs_runner(model, sweeps_per_round)
            key, kpt = jax.random.split(key)
            res = pt.pt_solve(model, betas, n_systems, n_rounds, sweeps_per_round, kpt, runner=runner)
            solved = res["solved_frac"] > 0

            if verdict == "timeout":
                rec[K]["timeout"] += 1
            else:
                rec[K]["feas"].append(verdict == "feasible")
            rec[K]["solved"].append(solved)
            # cross-tab solved vs feasibility only for resolved instances
            rec[K].setdefault("pair", []).append((verdict, solved))

    dens, feas_r, solv_r, cond_r, to_r = [], [], [], [], []
    print(f"\ngrid {rows!r}  q={q}  T={T}  betas={list(betas)}  CBS_budget={cbs_budget}\n")
    print(f"  {'K':>3} {'dens':>6} {'feasible':>9} {'solved':>8} {'solve|feas':>11} {'timeouts':>9}")
    for K in K_values:
        pairs = rec[K]["pair"]
        resolved = [(v, s) for v, s in pairs if v != "timeout"]
        feas = [v == "feasible" for v, _ in resolved]
        solved = [s for _, s in pairs]
        feas_only = [s for v, s in resolved if v == "feasible"]
        fr = float(np.mean(feas)) if feas else np.nan
        sr = float(np.mean(solved))
        cr = float(np.mean(feas_only)) if feas_only else np.nan
        tor = rec[K]["timeout"] / n_instances
        dens.append(K / q); feas_r.append(fr); solv_r.append(sr); cond_r.append(cr); to_r.append(tor)
        print(f"  {K:>3} {K/q:>6.3f} {fr:>9.2f} {sr:>8.2f} {cr:>11.2f} {tor:>9.2f}")

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
    print("\nsaved study_feasibility.png")
    return dict(density=dens, feasible=feas_r, solved=solv_r, findability=cond_r, timeout=to_r)


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    rows = [
        ".......",
        ".......",
        "###.###",
        ".......",
        ".......",
    ]
    run(rows, T=12, K_values=[2, 4, 6, 7, 8, 9, 10], n_instances=12,
        betas=[0.5, 1.0, 2.0, 4.0, 8.0], n_systems=64, n_rounds=20, sweeps_per_round=2)
