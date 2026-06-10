"""Validate the FFBS whole-trajectory sampler (run after validate_mapf).

  [A] correctness : the FFBS Gibbs kernel must leave the vertex+motion Boltzmann
                    distribution invariant -- its empirical histogram must match the
                    exact enumerated distribution on a tiny instance (TV < 0.03).
  [B] mixing      : on a longer-horizon instance, FFBS must reach valid (collision-
                    free) configurations in far fewer sweeps than single-site Gibbs.
"""

from __future__ import annotations

import jax
import numpy as np

from thrml import SamplingSchedule

import mapf_model as mapf
import swap_conflicts as sw
import classical_swap as csw
import trajectory_sampler as ts


def state_ids(model, pos):
    q = model["q"]
    ids = np.zeros(pos.shape[:-2], dtype=np.int64)
    for node in model["free_nodes"]:
        k, t = model["kt_of"][node]
        ids = ids * q + pos[..., k, t]
    return ids


def test_ffbs_correctness():
    print("[A] FFBS stationary distribution vs exact enumeration")
    rows = ["...", "..."]
    starts, goals = [(0, 0), (0, 2)], [(1, 2), (1, 0)]
    model = mapf.build_model(rows, starts, goals, T=2, beta=0.4, lam_move=1.5, lam_vertex=1.5)

    states = mapf.enumerate_states(model)
    E = mapf.energy_fp(model, states)
    P_exact = np.exp(-(E - E.min()))
    P_exact /= P_exact.sum()

    trace = ts.ffbs_gibbs(model, jax.random.key(1), n_chains=2000, n_sweeps=200, beta=0.4)
    pos = trace[:, 50:]  # discard warmup sweeps
    ids = state_ids(model, pos).reshape(-1)
    counts = np.bincount(ids, minlength=states.shape[0]).astype(float)
    P_emp = counts / counts.sum()

    tv = 0.5 * np.abs(P_emp - P_exact).sum()
    print(f"    states={states.shape[0]}  samples={ids.size}")
    print(f"    total-variation distance = {tv:.4f}")
    assert tv < 0.03, "FFBS does not sample the correct Boltzmann distribution"
    print("    PASS\n")


def test_ffbs_mixing():
    print("[B] FFBS vs single-site mixing (valid_frac vs #sweeps)")
    rows = ["......", "......", "......"]              # 3x6 open grid, q=18
    starts = [(0, 0), (0, 5), (2, 2)]
    goals = [(2, 5), (2, 0), (0, 2)]
    T, beta = 10, 8.0   # cold enough that valid configs dominate the Boltzmann mass
    n_chains = 2000
    model = mapf.build_model(rows, starts, goals, T, beta, lam_move=1.0, lam_vertex=1.0)

    n_sweeps = 60
    # FFBS: one record per sweep
    ffbs_trace = ts.ffbs_gibbs(model, jax.random.key(3), n_chains, n_sweeps, beta=beta)
    ffbs_valid = mapf.is_valid(model, ffbs_trace)                      # (n_chains, n_sweeps)
    ffbs_vf = ffbs_valid.mean(0)

    # Single-site: one recorded sample per Gibbs sweep, from random init, same beta.
    schedule = SamplingSchedule(n_warmup=0, n_samples=n_sweeps, steps_per_sample=1)
    ss = mapf.sample_positions(model, jax.random.key(3), n_chains, schedule)
    ss_valid = mapf.is_valid(model, ss)
    ss_vf = ss_valid.mean(0)

    # Both samplers target the same Boltzmann distribution, so they share an
    # equilibrium valid-fraction; the contribution is mixing SPEED to that plateau.
    plateau = float(ffbs_vf[-20:].mean())
    target = 0.9 * plateau

    def sweeps_to(vf):
        return next((s + 1 for s in range(n_sweeps) if vf[s] >= target), None)

    print(f"    instance: K={model['K']} T={T} q={model['q']} beta={beta}  n_chains={n_chains}")
    print(f"    shared equilibrium valid_frac ~= {plateau:.3f}")
    print(f"    {'sweep':>6} {'FFBS valid':>12} {'single-site':>12}")
    for s in [1, 2, 3, 5, 10, 20, 40, 60]:
        i = s - 1
        print(f"    {s:>6} {ffbs_vf[i]:>12.3f} {ss_vf[i]:>12.3f}")

    f_first, s_first = sweeps_to(ffbs_vf), sweeps_to(ss_vf)
    print(f"    sweeps to reach 90% of equilibrium ({target:.3f}):  FFBS={f_first}  single-site={s_first}")
    speedup = (s_first / f_first) if (f_first and s_first) else None
    print(f"    mixing speedup (single-site sweeps / FFBS sweeps) = {speedup}")
    # FFBS should be at equilibrium almost immediately while single-site lags badly.
    assert ffbs_vf[2] > ss_vf[2] + 0.2, "FFBS did not show a clear early-sweep advantage"
    assert f_first is not None and (s_first is None or s_first >= 5 * f_first), \
        "FFBS did not reach equilibrium much faster than single-site"
    print("    PASS\n")


def test_ffbs_swap_correctness():
    print("[C] FFBS+swap stationary distribution vs exact enumeration (full energy)")
    rows = ["...", "..."]
    starts, goals = [(0, 0), (0, 2)], [(1, 2), (1, 0)]
    model = sw.build_model_swap(rows, starts, goals, T=2, beta=0.4,
                                lam_move=1.5, lam_vertex=1.5, lam_swap=1.5)
    states = mapf.enumerate_states(model)
    E = sw.energy_fp_swap(model, states)
    P_exact = np.exp(-(E - E.min()))
    P_exact /= P_exact.sum()

    trace = ts.ffbs_gibbs(model, jax.random.key(1), n_chains=2000, n_sweeps=200, beta=0.4)
    pos = trace[:, 50:]
    ids = state_ids(model, pos).reshape(-1)
    counts = np.bincount(ids, minlength=states.shape[0]).astype(float)
    P_emp = counts / counts.sum()
    tv = 0.5 * np.abs(P_emp - P_exact).sum()
    print(f"    states={states.shape[0]}  samples={ids.size}")
    print(f"    total-variation distance = {tv:.4f}")
    assert tv < 0.03, "FFBS+swap does not sample the correct full-energy Boltzmann distribution"
    print("    PASS\n")


def test_ffbs_swap_corridor():
    print("[D] FFBS+swap corridor: must reach the swap-aware optimum (no free crossing)")
    rows = ["....."]
    starts, goals = [(0, 0), (0, 4)], [(0, 4), (0, 0)]
    T, beta = 5, 6.0
    model = sw.build_model_swap(rows, starts, goals, T, beta,
                                lam_move=1.0, lam_vertex=1.0, lam_swap=1.0)
    trace = ts.ffbs_gibbs(model, jax.random.key(2), n_chains=2000, n_sweeps=40, beta=beta)
    thrml_min = int(csw.total_violations(model, trace).min())
    classical_min = csw.min_violations(model)
    print(f"    FFBS+swap min violations = {thrml_min}   classical_min = {classical_min}")
    assert classical_min > 0, "corridor with swap should be infeasible"
    assert thrml_min == classical_min, "FFBS+swap did not reach the swap-aware optimum"
    print("    PASS\n")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}\n")
    test_ffbs_correctness()
    test_ffbs_mixing()
    test_ffbs_swap_correctness()
    test_ffbs_swap_corridor()
    print("all FFBS validation checks passed.")
