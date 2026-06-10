"""Validate the HP-lattice Potts encoding against ground truth (run this first).

  1. matrix check : energy reconstructed from the THRML weight tensors must equal
                    the first-principles energy on random assignments.
  2. sampling     : THRML's sampled histogram must match the exact enumerated
                    Boltzmann distribution on a tiny model.
  3. physics      : THRML's best valid conformation must reach the exact SAW
                    ground-state H-H contact count.
"""

from __future__ import annotations

import jax
import numpy as np

from thrml import SamplingSchedule

import hp_model as hp


def test_matrix_construction():
    print("[1] matrix vs first-principles energy")
    seq, L, beta = "HPPHH", 4, 0.7
    model = hp.build_model(seq, L, beta, eps=1.0, lam=2.0)
    rng = np.random.default_rng(0)
    pos = rng.integers(0, model["q"], size=(5000, model["N"])).astype(np.int64)
    pos[:, 0] = model["center"]  # residue 0 clamped
    e_w = hp.energy_from_weights(model, pos)
    e_fp = hp.energy_fp(model, pos)
    err = float(np.max(np.abs(e_w - e_fp)))
    print(f"    max |E_weights - E_firstprinciples| = {err:.3e}")
    assert err < 1e-4, "weight matrices do not match the intended energy"
    print("    PASS\n")


def test_sampling_matches_exact():
    print("[2] THRML sampling vs exact enumerated distribution")
    seq, L, beta = "HPPH", 3, 0.3
    model = hp.build_model(seq, L, beta, eps=1.0, lam=1.5)

    states = hp.enumerate_states(model)           # (q^(N-1), N)
    E = hp.energy_fp(model, states)
    P_exact = np.exp(-(E - E.min()))
    P_exact /= P_exact.sum()

    schedule = SamplingSchedule(n_warmup=300, n_samples=300, steps_per_sample=3)
    pos = hp.sample_positions(model, jax.random.key(1), n_chains=4000, schedule=schedule)
    ids = state_ids(model, pos).reshape(-1)
    counts = np.bincount(ids, minlength=states.shape[0]).astype(float)
    P_emp = counts / counts.sum()

    tv = 0.5 * np.abs(P_emp - P_exact).sum()
    max_abs = float(np.max(np.abs(P_emp - P_exact)))
    print(f"    states={states.shape[0]}  samples={ids.size}")
    print(f"    total-variation distance = {tv:.4f}   max |dP| = {max_abs:.4f}")
    assert tv < 0.03, "sampled distribution does not match exact Boltzmann distribution"
    print("    PASS\n")


def test_recovers_ground_state():
    print("[3] THRML best valid fold vs exact SAW ground state (simulated annealing)")
    betas = np.linspace(0.1, 3.0, 24)
    schedule = SamplingSchedule(n_warmup=200, n_samples=30, steps_per_sample=4)
    for seq, L in [("HHHH", 4), ("HHPHHH", 5), ("HPHPHH", 5)]:
        gc, deg = hp.saw_ground_state(seq)
        pos, model = hp.anneal_positions(seq, L, betas, schedule, n_chains=4000,
                                         key=jax.random.key(2), eps=1.0, return_model=True)

        valid = hp.is_valid(model, pos)
        contacts = hp.n_contacts(model, pos)
        best = int(contacts[valid].max()) if valid.any() else -1
        frac_valid = float(valid.mean())
        hit = float((valid & (contacts == gc)).mean())
        ok = best == gc
        print(f"    {seq:8s} L={L}  exact_max_contacts={gc} (deg~{deg})  "
              f"THRML_best={best}  valid_frac={frac_valid:.2f}  hit_gs_frac={hit:.3f}  "
              f"{'PASS' if ok else 'FAIL'}")
        assert ok, f"did not reach ground state for {seq}"
    print("    PASS\n")


def state_ids(model, pos):
    """Encode residues 1..N-1 (residue 0 is the clamped center) as a base-q integer id."""
    q = model["q"]
    free = pos[..., 1:]
    ids = np.zeros(pos.shape[:-1], dtype=np.int64)
    for k in range(free.shape[-1]):
        ids = ids * q + free[..., k]
    return ids


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}\n")
    test_matrix_construction()
    test_sampling_matches_exact()
    test_recovers_ground_state()
    print("all validation checks passed.")
