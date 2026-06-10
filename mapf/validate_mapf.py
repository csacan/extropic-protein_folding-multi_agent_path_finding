"""Validate the MAPF space-time Potts encoding against ground truth (run this first).

  1. matrix check : energy reconstructed from the THRML weight tensors must equal
                    the first-principles energy on random joint plans.
  2. sampling     : THRML's sampled histogram must match the exact enumerated
                    Boltzmann distribution on a tiny instance.
  3. solving      : on feasible instances THRML annealing must find a collision-free
                    plan (energy 0); on an infeasible instance its best energy must
                    equal the classical minimum number of violations.
"""

from __future__ import annotations

import jax
import numpy as np

from thrml import SamplingSchedule

import mapf_model as mapf
import classical_mapf as cls


def test_matrix_construction():
    print("[1] matrix vs first-principles energy")
    rows = ["....", ".##.", "...."]
    starts = [(0, 0), (0, 3)]
    goals = [(2, 3), (2, 0)]
    model = mapf.build_model(rows, starts, goals, T=4, beta=0.7, lam_move=2.3, lam_vertex=1.7)
    rng = np.random.default_rng(0)
    K, T, q = model["K"], model["T"], model["q"]
    pos = rng.integers(0, q, size=(5000, K, T + 1)).astype(np.int64)
    for k in range(K):  # respect the clamps
        pos[:, k, 0] = model["start_cells"][k]
        pos[:, k, T] = model["goal_cells"][k]
    e_w = mapf.energy_from_weights(model, pos)
    e_fp = mapf.energy_fp(model, pos)
    err = float(np.max(np.abs(e_w - e_fp)))
    print(f"    K={K} T={T} q={q} colors={model['n_colors']}")
    print(f"    max |E_weights - E_firstprinciples| = {err:.3e}")
    assert err < 1e-4, "weight tensors do not match the intended energy"
    print("    PASS\n")


def state_ids(model, pos):
    """Encode the free (unclamped) nodes as a base-q integer id."""
    q = model["q"]
    ids = np.zeros(pos.shape[:-2], dtype=np.int64)
    for node in model["free_nodes"]:
        k, t = model["kt_of"][node]
        ids = ids * q + pos[..., k, t]
    return ids


def test_sampling_matches_exact():
    print("[2] THRML sampling vs exact enumerated distribution")
    rows = ["...", "..."]            # 2x3 open map, q=6
    starts = [(0, 0), (0, 2)]
    goals = [(1, 2), (1, 0)]
    model = mapf.build_model(rows, starts, goals, T=2, beta=0.4, lam_move=1.5, lam_vertex=1.5)

    states = mapf.enumerate_states(model)            # (q^n_free, K, T+1)
    E = mapf.energy_fp(model, states)
    P_exact = np.exp(-(E - E.min()))
    P_exact /= P_exact.sum()

    schedule = SamplingSchedule(n_warmup=300, n_samples=300, steps_per_sample=3)
    pos = mapf.sample_positions(model, jax.random.key(1), n_chains=4000, schedule=schedule)
    ids = state_ids(model, pos).reshape(-1)
    counts = np.bincount(ids, minlength=states.shape[0]).astype(float)
    P_emp = counts / counts.sum()

    tv = 0.5 * np.abs(P_emp - P_exact).sum()
    max_abs = float(np.max(np.abs(P_emp - P_exact)))
    print(f"    n_free={len(model['free_nodes'])} states={states.shape[0]} samples={ids.size}")
    print(f"    total-variation distance = {tv:.4f}   max |dP| = {max_abs:.4f}")
    assert tv < 0.03, "sampled distribution does not match exact Boltzmann distribution"
    print("    PASS\n")


def test_solves_vs_classical():
    print("[3] THRML annealing vs classical feasibility (simulated annealing)")
    betas = np.linspace(0.2, 4.0, 28)
    schedule = SamplingSchedule(n_warmup=200, n_samples=40, steps_per_sample=4)
    instances = [
        # (name, rows, starts, goals, T)
        ("swap-2",  ["....."],                    [(0, 0), (0, 4)], [(0, 4), (0, 0)], 5),
        ("cross-2", ["...", "...", "..."],        [(0, 1), (1, 0)], [(2, 1), (1, 2)], 4),
        ("narrow-3",[".....", "##.##", "....."],  [(0, 0), (0, 4), (2, 2)],
                                                  [(2, 4), (2, 0), (0, 2)], 8),
    ]
    for name, rows, starts, goals, T in instances:
        pos, model = mapf.anneal(rows, starts, goals, T, betas, schedule,
                                 n_chains=4000, key=jax.random.key(2), return_model=True)
        plan = cls.solve_feasible(model)
        feasible = plan is not None

        valid = mapf.is_valid(model, pos)
        frac_valid = float(valid.mean())
        viol = (mapf.n_vertex_conflicts(model, pos) + mapf.n_illegal_moves(model, pos))
        thrml_min = int(viol.min())
        classical_min = 0 if feasible else _classical_min_violations(model)
        ok = (thrml_min == classical_min)
        print(f"    {name:9s} K={model['K']} T={T} q={model['q']}  feasible={feasible}  "
              f"THRML_min_viol={thrml_min} classical_min={classical_min}  "
              f"valid_frac={frac_valid:.3f}  {'PASS' if ok else 'FAIL'}")
        assert ok, f"THRML did not reach the classical optimum on {name}"
    print("    PASS\n")


def _classical_min_violations(model):
    """Min vertex-conflict count over all joint plans (tiny instances) -- for infeasible cases."""
    states = mapf.enumerate_states(model)
    viol = mapf.n_vertex_conflicts(model, states) + mapf.n_illegal_moves(model, states)
    return int(viol.min())


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}\n")
    test_matrix_construction()
    test_sampling_matches_exact()
    test_solves_vs_classical()
    print("all validation checks passed.")
