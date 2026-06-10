"""Exact whole-trajectory block-Gibbs for the MAPF EBM via forward-filter/backward-sample.

Single-site Gibbs (mapf_model's CategoricalGibbsConditional) flips one (agent,time)
cell at a time, so information crawls along each agent's time-chain diffusively
(~O(T^2) sweeps to traverse a trajectory). But one agent's whole trajectory,
conditioned on all OTHER agents held fixed, is a 1-D chain MRF:

  log P(traj_k | others) = sum_t theta_k[t, x_t] + sum_t Pmove[x_t, x_{t+1}]

with unary  theta_k[t,c] = -beta*lam_vertex * (#other agents at cell c at time t)
(the vertex-conflict field from the frozen agents) and pairwise log-potential
Pmove = beta*M_move (M_move[c,c'] = -lam_move on an illegal one-step move). A chain
is sampled EXACTLY by forward-filter/backward-sample in O(T*q^2), giving a true
whole-trajectory move per agent. Resampling each agent in turn is a valid Gibbs
kernel whose stationary distribution is the vertex+motion Boltzmann distribution.

THRML cannot express this as a native custom sampler: its block routing reads tail
states from a global state snapshot taken once per sampling group, so intra-block
chain edges are seen at stale values (parallel single-site, not a joint draw), and
the head<->tail pairing / chain order inside a block is not recoverable by the
sampler. So this is an explicit outer loop that reuses only the model's potentials
(beta, lam_move, lam_vertex, allowed, start/goal clamps).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

import mapf_model as mapf

NEG = -1e9  # stand-in for -inf in clamp masks (safe under logsumexp / categorical)


# --------------------------------------------------------------------------- #
# FFBS for a single clamped chain
# --------------------------------------------------------------------------- #
def ffbs_sample_trajectory(key, unary, pair, clamp_first, clamp_last):
    """Exact forward-filter/backward-sample of one agent's chain.

    unary : (T+1, q) unary log-potentials.
    pair  : (q, q) shared OR (T, q, q) per-edge pairwise log-potential; pair[e][c, c']
            scores x_e=c, x_{e+1}=c'. The per-edge form carries the swap correction
            (a swap with a frozen neighbor on edge e is a pairwise term on this edge).
    clamp_first, clamp_last : cell indices the endpoints are pinned to.
    Returns a (T+1,) uint8 trajectory drawn from P(traj) ∝ exp(unary + pair sums).
    """
    T1, q = unary.shape
    T = T1 - 1
    if pair.ndim == 2:
        pair = jnp.broadcast_to(pair, (T, q, q))
    cells = jnp.arange(q)
    unary = unary.at[0].add(jnp.where(cells == clamp_first, 0.0, NEG))
    unary = unary.at[T].add(jnp.where(cells == clamp_last, 0.0, NEG))

    # forward filter: alpha[t, c] = log sum over prefixes ending in x_t=c
    def fwd(alpha_prev, inp):
        u_t, P_e = inp
        msg = jax.scipy.special.logsumexp(alpha_prev[:, None] + P_e, axis=0)  # (q,)
        return u_t + msg, u_t + msg

    _, alphas_rest = lax.scan(fwd, unary[0], (unary[1:], pair))
    alphas = jnp.concatenate([unary[0][None], alphas_rest], axis=0)  # (T+1, q)

    # backward sample: x_T ~ alpha[T]; x_t ~ alpha[t] + pair[t][:, x_{t+1}]
    keys = jax.random.split(key, T1)
    x_last = jax.random.categorical(keys[-1], alphas[-1])

    def bwd(x_next, inp):
        alpha_t, P_e, k_t = inp
        x_t = jax.random.categorical(k_t, alpha_t + P_e[:, x_next])
        return x_t, x_t

    _, xs_rev = lax.scan(bwd, x_last, (alphas[:-1][::-1], pair[::-1], keys[:-1][::-1]))
    xs = jnp.concatenate([xs_rev[::-1], x_last[None]], axis=0)
    return xs.astype(jnp.uint8)


# --------------------------------------------------------------------------- #
# Per-agent block-Gibbs sweep
# --------------------------------------------------------------------------- #
def _potentials(model, beta):
    q = model["q"]
    pair = jnp.asarray(beta * mapf.motion_matrix(q, model["allowed"], model["lam_move"]))
    lam_vertex = float(model["lam_vertex"])
    lam_swap = float(model.get("lam_swap", 0.0) or 0.0)
    return pair, beta * lam_vertex, beta * lam_swap, q


def _make_sweep(model, beta, K, T, q):
    """Build a single full per-agent FFBS sweep closure for the given model/beta.

    Each agent's conditional given the (frozen) others is an exact 1-D chain:
      - unary  : vertex-conflict field, theta[t,c] = -beta*lam_vertex * #others at (c,t)
      - pair   : base motion log-potential beta*M_move, PLUS a per-edge swap correction
                 -beta*lam_swap at (x_{k,t}=v, x_{k,t+1}=u) for every frozen neighbor that
                 occupies u at t and v at t+1 (i.e. the exchange agent k would complete).
    """
    base_pair, beta_lamv, beta_lams, _ = _potentials(model, beta)
    has_swap = beta_lams > 0
    cells = jnp.arange(q)
    sc = jnp.asarray(model["start_cells"], dtype=jnp.int32)
    gc = jnp.asarray(model["goal_cells"], dtype=jnp.int32)
    edges = jnp.arange(T)

    def agent_unary(pos, k):
        onehot = (pos[:, :, None] == cells[None, None, :]).astype(jnp.float32)  # (K,T+1,q)
        return -beta_lamv * (onehot.sum(0) - onehot[k])                          # (T+1,q)

    def agent_pair(pos, k):
        if not has_swap:
            return base_pair  # (q,q); ffbs broadcasts over edges
        P = jnp.broadcast_to(base_pair, (T, q, q)) + jnp.zeros((T, q, q), base_pair.dtype)
        for j in range(K):  # frozen neighbors (K static)
            if j == k:
                continue
            rows = pos[j, 1:].astype(jnp.int32)   # neighbor cell at t+1  -> agent k's x_{k,t}
            cols = pos[j, :-1].astype(jnp.int32)  # neighbor cell at t    -> agent k's x_{k,t+1}
            add = -beta_lams * (rows != cols).astype(base_pair.dtype)
            P = P.at[edges, rows, cols].add(add)
        return P

    def sweep(pos, k_sweep):
        keys = jax.random.split(k_sweep, K)
        for k in range(K):  # sequential agent updates
            traj = ffbs_sample_trajectory(keys[k], agent_unary(pos, k),
                                          agent_pair(pos, k), sc[k], gc[k])
            pos = pos.at[k].set(traj)
        return pos, pos

    return sweep


def ffbs_gibbs(model, key, n_chains, n_sweeps, beta=None, thin=1):
    """Run n_chains parallel chains, each doing n_sweeps per-agent FFBS block updates.

    Handles the full motion+vertex+swap energy when the model carries lam_swap.
    Returns cells (n_chains, n_record, K, T+1), compatible with the mapf observables.
    """
    beta = model["beta"] if beta is None else float(beta)
    K, T, q = model["K"], model["T"], model["q"]
    starts = jnp.asarray(model["start_cells"], dtype=jnp.uint8)
    goals = jnp.asarray(model["goal_cells"], dtype=jnp.uint8)
    sweep = _make_sweep(model, beta, K, T, q)

    def run_chain(k):
        k0, kr = jax.random.split(k)
        pos0 = jax.random.randint(k0, (K, T + 1), 0, q, dtype=jnp.uint8)
        pos0 = pos0.at[:, 0].set(starts).at[:, T].set(goals)
        _, trace = lax.scan(sweep, pos0, jax.random.split(kr, n_sweeps))
        return trace[thin - 1::thin]

    trace = jax.jit(jax.vmap(run_chain))(jax.random.split(key, n_chains))
    import numpy as np
    return np.asarray(trace).astype(np.int64)


def make_ffbs_runner(model, n_sweeps):
    """Compile a reusable FFBS runner run(key, init_pos, beta) for a fixed instance.

    beta is passed as a TRACED scalar (jnp.asarray(beta)), so a single compiled kernel
    is reused across every temperature and PT round -- only the instance (grid, K,T,q,
    start/goal) is baked in. Returns final positions (B, K, T+1). This is the hot path
    for parallel tempering / the density study.
    """
    K, T, q = model["K"], model["T"], model["q"]
    Mmove = jnp.asarray(mapf.motion_matrix(q, model["allowed"], model["lam_move"]))  # (q,q), no beta
    lam_vertex = float(model["lam_vertex"])
    lam_swap = float(model.get("lam_swap", 0.0) or 0.0)
    has_swap = lam_swap > 0
    sc = jnp.asarray(model["start_cells"], dtype=jnp.int32)
    gc = jnp.asarray(model["goal_cells"], dtype=jnp.int32)
    cells = jnp.arange(q)
    edges = jnp.arange(T)

    def sweep(pos, inp):
        beta, k_sweep = inp
        base_pair = beta * Mmove
        keys = jax.random.split(k_sweep, K)
        for k in range(K):
            onehot = (pos[:, :, None] == cells[None, None, :]).astype(jnp.float32)
            unary = -(beta * lam_vertex) * (onehot.sum(0) - onehot[k])
            if has_swap:
                P = jnp.broadcast_to(base_pair, (T, q, q)) + jnp.zeros((T, q, q), base_pair.dtype)
                for j in range(K):
                    if j == k:
                        continue
                    rows = pos[j, 1:].astype(jnp.int32)
                    cols = pos[j, :-1].astype(jnp.int32)
                    P = P.at[edges, rows, cols].add(
                        -(beta * lam_swap) * (rows != cols).astype(base_pair.dtype))
                pair = P
            else:
                pair = base_pair
            pos = pos.at[k].set(ffbs_sample_trajectory(keys[k], unary, pair, sc[k], gc[k]))
        return pos, None

    @jax.jit
    def run(key, init_pos, beta):
        init_pos = jnp.asarray(init_pos, dtype=jnp.uint8)
        B = init_pos.shape[0]
        betas_seq = jnp.full((n_sweeps,), beta)

        def run_chain(p0, k):
            pf, _ = lax.scan(sweep, p0, (betas_seq, jax.random.split(k, n_sweeps)))
            return pf

        return jax.vmap(run_chain)(init_pos, jax.random.split(key, B))

    return run


def ffbs_sweeps_from(model, key, init_pos, n_sweeps, beta=None):
    """Run n_sweeps FFBS block-Gibbs sweeps from a GIVEN initial state (no re-init).

    init_pos : (B, K, T+1) uint8/int array. Returns final (B, K, T+1) int64 array.
    Used by the parallel-tempering driver, which must carry replica state across rounds.
    """
    import numpy as np
    beta = model["beta"] if beta is None else float(beta)
    K, T, q = model["K"], model["T"], model["q"]
    sweep = _make_sweep(model, beta, K, T, q)
    init_pos = jnp.asarray(init_pos, dtype=jnp.uint8)
    B = init_pos.shape[0]

    def run_chain(p0, k):
        pf, _ = lax.scan(sweep, p0, jax.random.split(k, n_sweeps))
        return pf

    out = jax.jit(jax.vmap(run_chain))(init_pos, jax.random.split(key, B))
    return np.asarray(out).astype(np.int64)


def ffbs_anneal(model_builder, betas, n_chains, key, n_sweeps_per_beta=3):
    """Anneal beta low->high with FFBS sweeps, carrying state across stages.

    model_builder(beta) -> a mapf-compatible model dict at that beta. Returns the
    final (n_chains, K, T+1) cell array. (FFBS mixes so well that a short ramp
    usually suffices; provided mainly for parity with mapf.anneal.)
    """
    import numpy as np
    m0 = model_builder(float(betas[0]))
    K, T, q = m0["K"], m0["T"], m0["q"]
    starts = jnp.asarray(m0["start_cells"], dtype=jnp.uint8)
    goals = jnp.asarray(m0["goal_cells"], dtype=jnp.uint8)

    key, k0 = jax.random.split(key)
    pos = jax.random.randint(k0, (n_chains, K, T + 1), 0, q, dtype=jnp.uint8)
    pos = pos.at[:, :, 0].set(starts[None]).at[:, :, T].set(goals[None])

    for b in betas:
        model = model_builder(float(b))
        sweep = _make_sweep(model, float(b), K, T, q)
        key, kr = jax.random.split(key)

        def run_chain(p0, k):
            pf, _ = lax.scan(sweep, p0, jax.random.split(k, n_sweeps_per_beta))
            return pf

        pos = jax.jit(jax.vmap(run_chain))(pos, jax.random.split(kr, n_chains))

    return np.asarray(pos).astype(np.int64), model
