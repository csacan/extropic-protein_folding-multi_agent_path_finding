"""Validate the v2 swap/edge-conflict layer against ground truth (run after validate_mapf).

  1. matrix check : energy reconstructed from ALL weight tensors (pairwise + order-4
                    swap) must equal the first-principles energy on random plans.
  2. sampling     : THRML's sampled histogram must match the exact enumerated
                    Boltzmann distribution on a tiny instance (with swap active).
  3. corridor     : on the 1-wide corridor where two agents must cross, the swap
                    penalty makes a conflict-free plan impossible -- THRML's annealed
                    minimum total-violations must equal the brute-force minimum (>0),
                    in contrast to v1 where the same instance was feasible.
"""

from __future__ import annotations

import jax
import numpy as np

from thrml import SamplingSchedule

import mapf_model as mapf
import swap_conflicts as sw
import classical_swap as cls


def test_matrix_construction():
    print("[1] matrix vs first-principles energy (with swap term)")
    rows = ["....", ".##.", "...."]
    starts, goals = [(0, 0), (0, 3)], [(2, 3), (2, 0)]
    model = sw.build_model_swap(rows, starts, goals, T=4, beta=0.7,
                                lam_move=2.3, lam_vertex=1.7, lam_swap=1.9)
    rng = np.random.default_rng(0)
    K, T, q = model["K"], model["T"], model["q"]
    pos = rng.integers(0, q, size=(5000, K, T + 1)).astype(np.int64)
    for k in range(K):
        pos[:, k, 0] = model["start_cells"][k]
        pos[:, k, T] = model["goal_cells"][k]
    e_w = sw.energy_from_weights_swap(model, pos)
    e_fp = sw.energy_fp_swap(model, pos)
    err = float(np.max(np.abs(e_w - e_fp)))
    print(f"    K={K} T={T} q={q} colors={model['n_colors']} n_swap={len(model['swap_groups'])}")
    print(f"    max |E_weights - E_firstprinciples| = {err:.3e}")
    assert err < 1e-4, "weight tensors (incl. order-4 swap) do not match the intended energy"
    print("    PASS\n")


def state_ids(model, pos):
    q = model["q"]
    ids = np.zeros(pos.shape[:-2], dtype=np.int64)
    for node in model["free_nodes"]:
        k, t = model["kt_of"][node]
        ids = ids * q + pos[..., k, t]
    return ids


def test_sampling_matches_exact():
    print("[2] THRML sampling vs exact enumerated distribution (swap active)")
    rows = ["...", "..."]
    starts, goals = [(0, 0), (0, 2)], [(1, 2), (1, 0)]
    model = sw.build_model_swap(rows, starts, goals, T=2, beta=0.4,
                                lam_move=1.5, lam_vertex=1.5, lam_swap=1.5)
    states = mapf.enumerate_states(model)
    E = sw.energy_fp_swap(model, states)
    P_exact = np.exp(-(E - E.min()))
    P_exact /= P_exact.sum()

    schedule = SamplingSchedule(n_warmup=300, n_samples=300, steps_per_sample=3)
    pos = mapf.sample_positions(model, jax.random.key(1), n_chains=4000, schedule=schedule)
    ids = state_ids(model, pos).reshape(-1)
    counts = np.bincount(ids, minlength=states.shape[0]).astype(float)
    P_emp = counts / counts.sum()
    tv = 0.5 * np.abs(P_emp - P_exact).sum()
    print(f"    n_free={len(model['free_nodes'])} states={states.shape[0]} samples={ids.size}")
    print(f"    total-variation distance = {tv:.4f}")
    assert tv < 0.03, "sampled distribution does not match exact Boltzmann distribution"
    print("    PASS\n")


def test_corridor_no_crossing():
    print("[3] corridor: swap penalty forbids crossing (v1 feasible -> v2 infeasible)")
    rows = ["....."]
    starts, goals = [(0, 0), (0, 4)], [(0, 4), (0, 0)]
    T = 5
    betas = np.linspace(0.2, 4.0, 28)
    schedule = SamplingSchedule(n_warmup=200, n_samples=40, steps_per_sample=4)

    # v1 baseline (vertex-only): the crossing is feasible.
    pos_v1, m_v1 = mapf.anneal(rows, starts, goals, T, betas, schedule,
                               n_chains=4000, key=jax.random.key(2), return_model=True)
    v1_min = int((mapf.n_vertex_conflicts(m_v1, pos_v1) + mapf.n_illegal_moves(m_v1, pos_v1)).min())

    # v2 (swap active): the crossing now costs at least one violation.
    pos, model = sw.anneal_swap(rows, starts, goals, T, betas, schedule,
                                n_chains=4000, key=jax.random.key(2),
                                lam_swap=1.0, return_model=True)
    thrml_min = int(cls.total_violations(model, pos).min())
    classical_min = cls.min_violations(model)

    print(f"    v1 (vertex-only) min violations = {v1_min}  (feasible crossing)")
    print(f"    v2 (swap active) THRML_min = {thrml_min}  classical_min = {classical_min}")
    assert v1_min == 0, "v1 corridor should be feasible"
    assert classical_min > 0, "v2 corridor should be infeasible (swap forbidden)"
    assert thrml_min == classical_min, "THRML did not reach the classical swap-aware optimum"
    print("    PASS\n")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}\n")
    test_matrix_construction()
    test_sampling_matches_exact()
    test_corridor_no_crossing()
    print("all swap-layer validation checks passed.")
