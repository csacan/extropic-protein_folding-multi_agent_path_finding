"""Parallel tempering (PT) over FFBS replicas for the full motion+vertex+swap MAPF EBM.

Each replica runs whole-trajectory FFBS block-Gibbs (trajectory_sampler) at its own
inverse temperature; FFBS mixes the within-temperature chain quickly, while the PT
outer loop lets configurations migrate between temperatures so the cold replica can
escape a sub-optimal coordination basin (e.g. the wrong homotopy class) that a single
cold chain would be trapped in.

PT mechanics: replicas share the unscaled energy
    U(x) = lam_move * #illegal_moves + lam_vertex * #vertex_conflicts + lam_swap * #swaps
and differ only in beta. Adjacent replicas r, r+1 attempt a configuration swap with
the Metropolis acceptance
    A = min(1, exp( (beta_r - beta_{r+1}) * (U_r - U_{r+1}) )),
alternating even/odd adjacent pairs each round. The per-pair acceptance rate is the
publishable diagnostic: it measures the energy-barrier overlap between temperatures,
which sharpens near the solvability phase transition.

n_systems independent PT ladders run in parallel (vmapped) for statistics / restarts.
The lowest-U configuration ever seen in any replica of a ladder is that ladder's
answer; U == 0 means a fully collision-free plan was found.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import mapf_model as mapf
import swap_conflicts as sw
import trajectory_sampler as ts


def light_model(rows, starts, goals, T, beta=1.0,
                lam_move=1.0, lam_vertex=1.0, lam_swap=1.0):
    """A mapf/swap-compatible model dict WITHOUT the THRML program / q^4 swap tensor.

    FFBS and the numpy observables only need the potentials and bookkeeping, never the
    assembled factor graph, so this avoids building the dense order-4 swap weights
    (which would be q^4 per constraint) -- essential for the density study at larger q/K.
    """
    passable = mapf.parse_map(rows)
    coords, q, adj, allowed, cell_of = mapf.grid_graph(passable)
    sc = [cell_of[(int(r), int(c))] for r, c in starts]
    gc = [cell_of[(int(r), int(c))] for r, c in goals]
    return dict(allowed=allowed, adj=adj, coords=coords, q=q, K=len(starts), T=T,
                lam_move=lam_move, lam_vertex=lam_vertex, lam_swap=lam_swap, beta=beta,
                start_cells=sc, goal_cells=gc, starts=starts, goals=goals, rows=rows)


def unscaled_energy(model, pos):
    """U(pos) = lam_move*illegal + lam_vertex*vertex + lam_swap*swap (no beta)."""
    lam_swap = float(model.get("lam_swap", 0.0) or 0.0)
    U = (model["lam_move"] * mapf.n_illegal_moves(model, pos)
         + model["lam_vertex"] * mapf.n_vertex_conflicts(model, pos))
    if lam_swap > 0:
        U = U + lam_swap * sw.n_swap_conflicts(model, pos)
    return U.astype(np.float64)


def _random_init(model, n_systems, R, key):
    K, T, q = model["K"], model["T"], model["q"]
    pos = np.asarray(jax.random.randint(key, (n_systems, R, K, T + 1), 0, q)).astype(np.int64)
    for k in range(K):
        pos[:, :, k, 0] = model["start_cells"][k]
        pos[:, :, k, T] = model["goal_cells"][k]
    return pos


def pt_solve(model, betas, n_systems, n_rounds, sweeps_per_round, key,
             runner=None, record_history=False):
    """Run parallel tempering and return the best plan per ladder plus diagnostics.

    model : a single light/full model dict for the instance (energy is beta-independent;
            the runner applies beta). betas : ascending inverse temps (hot ... cold).
    Returns dict with best_U (n_systems,), best_pos (n_systems,K,T+1), swap_accept
    (R-1,) mean acceptance per adjacent pair, and solved fraction.
    """
    betas = np.asarray(betas, dtype=np.float64)
    R = len(betas)
    if runner is None:
        runner = ts.make_ffbs_runner(model, sweeps_per_round)

    key, k_init = jax.random.split(key)
    pos = _random_init(model, n_systems, R, k_init)  # (n_systems, R, K, T+1)

    best_U = np.full(n_systems, np.inf)
    best_pos = pos[:, -1].copy()
    swap_attempts = np.zeros(R - 1)
    swap_accepts = np.zeros(R - 1)
    history = []

    for rnd in range(n_rounds):
        # within-temperature FFBS on every replica (one compiled kernel, beta traced)
        for r in range(R):
            key, kr = jax.random.split(key)
            pos[:, r] = np.asarray(runner(kr, pos[:, r], jnp.asarray(betas[r]))).astype(np.int64)

        U = np.stack([unscaled_energy(model, pos[:, r]) for r in range(R)], axis=1)  # (n_systems, R)

        # track best over all replicas
        rmin = U.argmin(axis=1)
        umin = U[np.arange(n_systems), rmin]
        improve = umin < best_U
        best_U[improve] = umin[improve]
        best_pos[improve] = pos[np.arange(n_systems), rmin][improve]

        # adjacent swaps, alternating even/odd pairs
        for r in range(rnd % 2, R - 1, 2):
            key, ks = jax.random.split(key)
            delta = (betas[r] - betas[r + 1]) * (U[:, r] - U[:, r + 1])
            acc = np.asarray(jax.random.uniform(ks, (n_systems,))) < np.exp(np.minimum(delta, 0.0))
            swap_attempts[r] += n_systems
            swap_accepts[r] += int(acc.sum())
            if acc.any():
                tmp = pos[acc, r].copy()
                pos[acc, r] = pos[acc, r + 1]
                pos[acc, r + 1] = tmp
                U[acc, r], U[acc, r + 1] = U[acc, r + 1].copy(), U[acc, r].copy()

        if record_history:
            history.append((best_U <= 0).mean())

    swap_accept = np.divide(swap_accepts, swap_attempts,
                            out=np.zeros_like(swap_accepts), where=swap_attempts > 0)
    mean_swap = float(swap_accept.mean()) if swap_accept.size else 0.0
    return dict(
        best_U=best_U, best_pos=best_pos, solved_frac=float((best_U <= 0).mean()),
        swap_accept=swap_accept, mean_swap_accept=mean_swap,
        betas=betas, history=history,
    )
