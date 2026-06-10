"""v2 swap/edge-conflict layer for the MAPF space-time Potts EBM (extends mapf_model).

The v1 model (mapf_model.py) penalizes only VERTEX conflicts -- two agents on the
same cell at the same time. It therefore permits two agents to pass *through* each
other on a 1-wide corridor (each one's move is legal, they are never co-located), an
EDGE / SWAP conflict. Real MAPF forbids that. A swap between agents k and k' across
t -> t+1 is the pattern

    x_{k,t}=u, x_{k,t+1}=v, x_{k',t}=v, x_{k',t+1}=u   with  u != v,

i.e. the two agents exchange cells. This is an order-4 interaction among the four
variables, encoded with THRML's CategoricalEBMFactor (node_groups of length 4,
weight tensor shape [n, q, q, q, q], energy E = -sum_b W[b, a, b', c, d]). A negative
entry penalizes (raises energy), matching the v1 sign convention.

Because the order-4 factor couples (k,t),(k,t+1),(k',t),(k',t+1) it adds two new
"diagonal" edges to the interaction graph -- (k,t)-(k',t+1) and (k,t+1)-(k',t) -- so
build_model_swap recolors the free sub-graph with those edges included. The returned
dict is key-compatible with mapf_model, so mapf.sample_positions and the observables
work on it unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, BlockGibbsSpec, CategoricalNode, FactorSamplingProgram, sample_states
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional, SquareCategoricalEBMFactor

import mapf_model as mapf


# --------------------------------------------------------------------------- #
# Order-4 swap weight tensor
# --------------------------------------------------------------------------- #
def swap_tensor(q, lam_swap):
    """q^4 reward tensor W[a,b,c,d] = -lam_swap where (a==d and b==c and a!=b), else 0.

    Indices are (a=x_{k,t}, b=x_{k,t+1}, c=x_{k',t}, d=x_{k',t+1}); the swap pattern is
    a->b for agent k and c->d for agent k' with c==b, d==a, a!=b.
    """
    idx = np.arange(q)
    a = idx[:, None, None, None]
    b = idx[None, :, None, None]
    c = idx[None, None, :, None]
    d = idx[None, None, None, :]
    mask = (a == d) & (b == c) & (a != b)
    return (-lam_swap * mask).astype(np.float32)


# --------------------------------------------------------------------------- #
# Model assembly (motion + vertex + swap)
# --------------------------------------------------------------------------- #
def build_model_swap(rows, starts, goals, T, beta,
                     lam_move=1.0, lam_vertex=1.0, lam_swap=1.0):
    """Assemble a THRML program with motion, vertex, and order-4 swap factors.

    Returns a dict key-compatible with mapf.build_model (so mapf.sample_positions /
    mapf._scatter_positions / the observables work on it) plus swap bookkeeping.
    """
    passable = mapf.parse_map(rows)
    coords, q, adj, allowed, cell_of = mapf.grid_graph(passable)
    K = len(starts)
    assert len(goals) == K
    start_cells = [cell_of[(int(r), int(c))] for r, c in starts]
    goal_cells = [cell_of[(int(r), int(c))] for r, c in goals]

    nodes2d = [[CategoricalNode() for _ in range(T + 1)] for _ in range(K)]
    kt_of = {nodes2d[k][t]: (k, t) for k in range(K) for t in range(T + 1)}
    flat = [nodes2d[k][t] for k in range(K) for t in range(T + 1)]

    clamped_nodes = [nodes2d[k][0] for k in range(K)] + [nodes2d[k][T] for k in range(K)]
    clamp_vals = start_cells + goal_cells
    clamped_set = set(clamped_nodes)

    # --- pairwise factors (identical to v1) ---------------------------------
    M_move = mapf.motion_matrix(q, allowed, lam_move)
    M_vertex = mapf.vertex_matrix(q, lam_vertex)

    move_heads = [nodes2d[k][t] for k in range(K) for t in range(T)]
    move_tails = [nodes2d[k][t + 1] for k in range(K) for t in range(T)]
    move_w = np.broadcast_to(M_move, (len(move_heads), q, q)).copy()

    vtx_pairs = [(nodes2d[k][t], nodes2d[kp][t])
                 for t in range(T + 1) for k in range(K) for kp in range(k + 1, K)]
    vtx_heads = [a for a, _ in vtx_pairs]
    vtx_tails = [b for _, b in vtx_pairs]
    vtx_w = np.broadcast_to(M_vertex, (len(vtx_heads), q, q)).copy()

    move_factor = SquareCategoricalEBMFactor(
        [Block(move_heads), Block(move_tails)], beta * jnp.asarray(move_w))
    vtx_factor = SquareCategoricalEBMFactor(
        [Block(vtx_heads), Block(vtx_tails)], beta * jnp.asarray(vtx_w))

    # --- order-4 swap factor ------------------------------------------------
    # one constraint per (k<k', t) over the step t -> t+1
    sw = [(nodes2d[k][t], nodes2d[k][t + 1], nodes2d[kp][t], nodes2d[kp][t + 1])
          for t in range(T) for k in range(K) for kp in range(k + 1, K)]
    sw_g1 = [g[0] for g in sw]
    sw_g2 = [g[1] for g in sw]
    sw_g3 = [g[2] for g in sw]
    sw_g4 = [g[3] for g in sw]
    W4 = swap_tensor(q, lam_swap)
    swap_w = np.broadcast_to(W4, (len(sw), q, q, q, q)).copy()
    swap_factor = CategoricalEBMFactor(
        [Block(sw_g1), Block(sw_g2), Block(sw_g3), Block(sw_g4)],
        beta * jnp.asarray(swap_w))

    factors = [move_factor, vtx_factor, swap_factor]

    # --- recolor the free sub-graph including swap-induced edges -------------
    free_nodes = [n for n in flat if n not in clamped_set]
    free_local = {n: i for i, n in enumerate(free_nodes)}
    edges = set()

    def add_edge(a, b):
        if a in free_local and b in free_local:
            i, j = free_local[a], free_local[b]
            edges.add((i, j) if i < j else (j, i))

    for h, t in zip(move_heads, move_tails):
        add_edge(h, t)
    for h, t in zip(vtx_heads, vtx_tails):
        add_edge(h, t)
    for g in sw:  # all 6 pairs among the 4 swap nodes (adds the 2 diagonals)
        for i in range(4):
            for j in range(i + 1, 4):
                add_edge(g[i], g[j])

    colors = mapf.dsatur_coloring(len(free_nodes), list(edges))
    n_colors = (max(colors) + 1) if colors else 0
    free_blocks = [Block([free_nodes[i] for i in range(len(free_nodes)) if colors[i] == c])
                   for c in range(n_colors)]
    free_blocks = [b for b in free_blocks if b.nodes]

    clamped_blocks = [Block([n]) for n in clamped_nodes]
    spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    sampler = CategoricalGibbsConditional(q)
    prog = FactorSamplingProgram(spec, [sampler for _ in free_blocks], factors, [])

    return dict(
        prog=prog, spec=spec, nodes2d=nodes2d, kt_of=kt_of, flat=flat,
        free_blocks=free_blocks, clamped_blocks=clamped_blocks, clamp_vals=clamp_vals,
        free_nodes=free_nodes, n_colors=len(free_blocks),
        q=q, K=K, T=T, coords=coords, adj=adj, allowed=allowed, cell_of=cell_of,
        starts=starts, goals=goals, start_cells=start_cells, goal_cells=goal_cells,
        lam_move=lam_move, lam_vertex=lam_vertex, lam_swap=lam_swap, beta=beta,
        move_heads=move_heads, move_tails=move_tails, move_w=np.asarray(beta * move_w),
        vtx_heads=vtx_heads, vtx_tails=vtx_tails, vtx_w=np.asarray(beta * vtx_w),
        swap_groups=sw, swap_w=np.asarray(beta * swap_w),
        rows=rows,
    )


# --------------------------------------------------------------------------- #
# Sampling / annealing that rebuild with the swap factor
# --------------------------------------------------------------------------- #
def anneal_swap(rows, starts, goals, T, betas, schedule, n_chains, key,
                lam_move=1.0, lam_vertex=1.0, lam_swap=1.0, return_model=False):
    """Simulated annealing with the swap factor active (mirrors mapf.anneal)."""
    key, k_init = jax.random.split(key)
    m0 = build_model_swap(rows, starts, goals, T, float(betas[0]), lam_move, lam_vertex, lam_swap)
    q = m0["q"]
    init = [jax.random.randint(k, (n_chains, len(b.nodes)), 0, q, dtype=jnp.uint8)
            for k, b in zip(jax.random.split(k_init, len(m0["free_blocks"])), m0["free_blocks"])]

    model, states = m0, None
    for b in betas:
        model = build_model_swap(rows, starts, goals, T, float(b), lam_move, lam_vertex, lam_swap)
        clamp = [jnp.array([v], dtype=jnp.uint8) for v in model["clamp_vals"]]
        free_blocks = model["free_blocks"]
        key, k_stage = jax.random.split(key)
        keys = jax.random.split(k_stage, n_chains)

        def one(init_c, k):
            return sample_states(k, model["prog"], schedule, init_c, clamp, free_blocks)

        states = jax.jit(jax.vmap(one))(init, keys)
        init = [s[:, -1, :] for s in states]

    pos = mapf._scatter_positions(model, states)
    return (pos, model) if return_model else pos


# --------------------------------------------------------------------------- #
# Observables / energy (swap-aware)
# --------------------------------------------------------------------------- #
def n_swap_conflicts(model, pos):
    """Number of (agent-pair, step) swap/edge conflicts in pos shaped (..., K, T+1)."""
    K, T = model["K"], model["T"]
    out = np.zeros(pos.shape[:-2], dtype=int)
    for k in range(K):
        for kp in range(k + 1, K):
            a = pos[..., k, :-1]
            b = pos[..., k, 1:]
            c = pos[..., kp, :-1]
            d = pos[..., kp, 1:]
            out += ((a != b) & (a == d) & (b == c)).sum(-1)
    return out


def energy_fp_swap(model, pos, beta=None):
    """First-principles energy E = beta*(lam_move*illegal + lam_vertex*vertex + lam_swap*swap)."""
    beta = model["beta"] if beta is None else beta
    return beta * (model["lam_move"] * mapf.n_illegal_moves(model, pos)
                   + model["lam_vertex"] * mapf.n_vertex_conflicts(model, pos)
                   + model["lam_swap"] * n_swap_conflicts(model, pos))


def energy_from_weights_swap(model, pos):
    """Energy reconstructed from all THRML weight tensors (pairwise + order-4 swap)."""
    kt = model["kt_of"]
    E = np.zeros(pos.shape[:-2], dtype=np.float64)
    for heads, tails, w in [
        (model["move_heads"], model["move_tails"], model["move_w"]),
        (model["vtx_heads"], model["vtx_tails"], model["vtx_w"]),
    ]:
        for p, (h, t) in enumerate(zip(heads, tails)):
            kh, th = kt[h]
            kt2, tt = kt[t]
            E -= w[p, pos[..., kh, th], pos[..., kt2, tt]]
    sw, sw_w = model["swap_groups"], model["swap_w"]
    for p, g in enumerate(sw):
        (ka, ta), (kb, tb), (kc, tc), (kd, td) = (kt[g[0]], kt[g[1]], kt[g[2]], kt[g[3]])
        E -= sw_w[p, pos[..., ka, ta], pos[..., kb, tb], pos[..., kc, tc], pos[..., kd, td]]
    return E


def is_valid_swap(model, pos):
    """True where pos has no vertex, illegal-move, or swap violations."""
    return ((mapf.n_vertex_conflicts(model, pos) == 0)
            & (mapf.n_illegal_moves(model, pos) == 0)
            & (n_swap_conflicts(model, pos) == 0))
