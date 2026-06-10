"""Convergence / solvability study: success rate and PT swap-acceptance vs agent density.

For a fixed grid we draw random MAPF instances with an increasing number of agents K
(agent density rho = K/|cells|) and, for each, run parallel tempering over FFBS replicas
on the FULL motion+vertex+swap energy. We record:

  - PT success rate     : fraction of instances for which some PT ladder found a fully
                          collision-free plan (best unscaled energy U == 0).
  - FFBS-only baseline  : same budget but a single cold replica (no temperature swaps),
                          to isolate what parallel tempering buys.
  - mean swap-acceptance : averaged over adjacent temperature pairs -- the statistical-
                          physics diagnostic that probes the energy barrier and is
                          expected to vary across the solvability transition.

Output: a printed table and study_mapf.png. This is the characterization the project's
research story hinges on (convergence vs graph density / conflict rate).
"""

from __future__ import annotations

import numpy as np
import jax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mapf_model as mapf
import trajectory_sampler as ts
import pt_mapf as pt


def random_instance(coords, start_pool, goal_pool, K, rng):
    """Sample K starts from start_pool and K goals from goal_pool (distinct cells)."""
    s = rng.choice(start_pool, size=K, replace=False)
    g = rng.choice(goal_pool, size=K, replace=False)
    starts = [tuple(int(x) for x in coords[c]) for c in s]
    goals = [tuple(int(x) for x in coords[c]) for c in g]
    return starts, goals


def run_study(rows, T, K_values, n_instances, betas, n_systems, n_rounds,
              sweeps_per_round, lam_swap, seed=0):
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    # Bottleneck hardness: starts in the top half, goals in the bottom half, so every
    # agent must funnel through the 1-wide gap -> congestion grows with K.
    mid = len(rows) / 2.0
    start_pool = np.array([c for c in range(q) if coords[c][0] < mid])
    goal_pool = np.array([c for c in range(q) if coords[c][0] > mid])
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    rec = {k: dict(pt_solved=[], ffbs_solved=[], swap=[]) for k in K_values}
    for K in K_values:
        for _ in range(n_instances):
            starts, goals = random_instance(coords, start_pool, goal_pool, K, rng)
            model = pt.light_model(rows, starts, goals, T, lam_swap=lam_swap)
            runner = ts.make_ffbs_runner(model, sweeps_per_round)

            key, kpt, kff = jax.random.split(key, 3)
            res_pt = pt.pt_solve(model, betas, n_systems, n_rounds, sweeps_per_round,
                                 kpt, runner=runner)
            res_ff = pt.pt_solve(model, betas[-1:], n_systems, n_rounds, sweeps_per_round,
                                 kff, runner=runner)  # single cold replica, no swaps

            rec[K]["pt_solved"].append(res_pt["solved_frac"] > 0)
            rec[K]["ffbs_solved"].append(res_ff["solved_frac"] > 0)
            rec[K]["swap"].append(res_pt["mean_swap_accept"])

    dens = np.array([K / q for K in K_values])
    pt_succ = np.array([np.mean(rec[K]["pt_solved"]) for K in K_values])
    ff_succ = np.array([np.mean(rec[K]["ffbs_solved"]) for K in K_values])
    swap = np.array([np.mean(rec[K]["swap"]) for K in K_values])

    print(f"\ngrid {rows!r}  q={q}  T={T}  betas={list(betas)}  "
          f"n_systems={n_systems} n_rounds={n_rounds} sweeps={sweeps_per_round}\n")
    print(f"  {'K':>3} {'density':>8} {'PT_success':>11} {'FFBS_success':>13} {'swap_accept':>12}")
    for i, K in enumerate(K_values):
        print(f"  {K:>3} {dens[i]:>8.3f} {pt_succ[i]:>11.2f} {ff_succ[i]:>13.2f} {swap[i]:>12.3f}")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(dens, pt_succ, "o-", color="C0", label="PT success")
    ax1.plot(dens, ff_succ, "s--", color="C2", label="FFBS-only success")
    ax1.set_xlabel("agent density  K / |cells|")
    ax1.set_ylabel("success rate", color="C0")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left")
    ax2 = ax1.twinx()
    ax2.plot(dens, swap, "^-", color="C3", label="mean swap acceptance")
    ax2.set_ylabel("PT swap acceptance", color="C3")
    ax2.set_ylim(-0.02, 1.02)
    plt.title(f"MAPF solvability vs density  (grid {len(rows)}x{len(rows[0])}, T={T})")
    fig.tight_layout()
    fig.savefig("study_mapf.png", dpi=130)
    print("\nsaved study_mapf.png")
    return dict(density=dens, pt_success=pt_succ, ffbs_success=ff_succ, swap=swap)


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    # Two 5x7 rooms split by a wall with a single 1-wide gap (column 3); every agent
    # must cross the gap, so throughput is capped and a solvability transition appears
    # as the agent count grows past the gap's makespan-limited capacity.
    rows = [
        ".......",
        ".......",
        "###.###",
        ".......",
        ".......",
    ]
    run_study(
        rows=rows, T=12,
        K_values=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_instances=16,
        betas=[0.5, 1.0, 2.0, 4.0, 8.0],
        n_systems=64, n_rounds=24, sweeps_per_round=2,
        lam_swap=1.0, seed=0,
    )
