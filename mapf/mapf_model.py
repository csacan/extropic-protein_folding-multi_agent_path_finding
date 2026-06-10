"""Multi-Agent Pathfinding (MAPF) encoded as a space-time Potts EBM for THRML.

Each agent k at each timestep t is a categorical variable x_{k,t} in {0,...,q-1}
indexing a *passable* cell of the map (q = number of non-obstacle cells). A joint
assignment of all K*(T+1) variables is a set of K time-indexed trajectories. The
energy is built from pairwise q x q reward matrices in THRML's convention,
E = -sum_pairs W[x_a, x_b]  (a POSITIVE entry lowers energy / raises probability):

  * motion          (agent k, times t, t+1):  -lam_move    where cells are NOT
                                               adjacent-or-equal (waiting allowed)
  * vertex conflict (agents k<k', same t)   :  -lam_vertex  on the diagonal (same cell)

beta (inverse temperature) multiplies every matrix, exactly as in hp_model. Each
agent's start cell (t=0) and goal cell (t=T) is clamped, which both removes the
trivial all-agents-sit-still degeneracy and poses the actual coordination problem:
drive every agent from start to goal within the makespan T without ever colliding.

This is the standard "vertex-conflict" relaxation of MAPF (swap/edge conflicts are
order-4 factors, added in a later layer). Unlike the HP model -- whose excluded
volume couples every residue pair into a complete graph, forcing one residue per
color block -- the MAPF interaction graph is SPARSE: each timestep is a K-clique
(vertex conflict) stitched to the next by per-agent motion edges. So a graph
coloring yields genuinely parallel block-Gibbs updates, which is the regime
Extropic hardware is built for. We color the free (unclamped) sub-graph with a
small DSATUR coloring and vmap many independent replicas on top.

Maps onto THRML's SquareCategoricalEBMFactor / CategoricalGibbsConditional.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, BlockGibbsSpec, CategoricalNode, FactorSamplingProgram, sample_states
from thrml.models import CategoricalGibbsConditional, SquareCategoricalEBMFactor


# --------------------------------------------------------------------------- #
# Map / space graph
# --------------------------------------------------------------------------- #
def parse_map(rows):
    """Parse a list of strings ('.' free, '#'/'@'/'T' obstacle) into a passable bool grid."""
    blocked = set("#@T")
    grid = np.array([[ch not in blocked for ch in row] for row in rows], dtype=bool)
    return grid


def grid_graph(passable: np.ndarray):
    """Index the passable cells and build the von Neumann adjacency over them.

    Returns (coords, q, adj, allowed, cell_of) where coords[c] = (row, col) of cell
    c, adj[a,b] is True for edge-adjacent passable cells, allowed = adj | I (a legal
    one-step move is to a neighbor or to stay put), and cell_of maps (row,col) -> c.
    """
    rc = np.argwhere(passable)  # (q, 2) in row,col order of appearance
    q = len(rc)
    if q > 256:
        raise ValueError(f"q={q} passable cells exceeds 256 (categorical state is uint8).")
    cell_of = {(int(r), int(c)): i for i, (r, c) in enumerate(rc)}
    d = np.abs(rc[:, None, :] - rc[None, :, :]).sum(-1)
    adj = d == 1
    allowed = adj | np.eye(q, dtype=bool)
    return rc, q, adj, allowed, cell_of


# --------------------------------------------------------------------------- #
# Energy matrices (THRML sign: E = -sum W[x_a, x_b], positive entry => lower energy)
# --------------------------------------------------------------------------- #
def motion_matrix(q, allowed, lam_move):
    """q x q reward: penalize an illegal one-step move between consecutive times."""
    return (-lam_move * (~allowed)).astype(np.float32)  # 0 where allowed, -lam where not


def vertex_matrix(q, lam_vertex):
    """q x q reward: penalize two agents occupying the same cell at the same time."""
    return (-lam_vertex * np.eye(q)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Coloring of the free interaction graph (self-contained DSATUR, no networkx dep)
# --------------------------------------------------------------------------- #
def dsatur_coloring(n, edges):
    """Greedy DSATUR coloring of an undirected graph on nodes 0..n-1.

    Returns a list `color` of length n. DSATUR repeatedly colors the uncolored node
    with the highest saturation (number of distinct neighbor colors), breaking ties
    by plain degree -- near-optimal on the structured space-time graph here.
    """
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    color = [-1] * n
    deg = [len(adj[v]) for v in range(n)]
    for _ in range(n):
        best, best_key = -1, None
        for v in range(n):
            if color[v] != -1:
                continue
            sat = len({color[u] for u in adj[v] if color[u] != -1})
            key = (sat, deg[v])
            if best_key is None or key > best_key:
                best, best_key = v, key
        used = {color[u] for u in adj[best] if color[u] != -1}
        c = 0
        while c in used:
            c += 1
        color[best] = c
    return color


# --------------------------------------------------------------------------- #
# THRML model
# --------------------------------------------------------------------------- #
def build_model(rows, starts, goals, T, beta, lam_move=1.0, lam_vertex=1.0):
    """Assemble the THRML sampling program and bookkeeping for a MAPF instance.

    Arguments:
      rows    : list of map strings ('.' free, '#'/'@'/'T' obstacle).
      starts  : list of K (row,col) start cells, one per agent.
      goals   : list of K (row,col) goal cells, one per agent.
      T       : makespan (timesteps 0..T inclusive; trajectory length T+1).
      beta    : inverse temperature multiplying every weight matrix.
    """
    passable = parse_map(rows)
    coords, q, adj, allowed, cell_of = grid_graph(passable)
    K = len(starts)
    assert len(goals) == K, "starts and goals must have equal length"
    start_cells = [cell_of[(int(r), int(c))] for r, c in starts]
    goal_cells = [cell_of[(int(r), int(c))] for r, c in goals]

    # One categorical node per (agent, time); index2d[k][t] is the node object.
    nodes2d = [[CategoricalNode() for _ in range(T + 1)] for _ in range(K)]
    kt_of = {nodes2d[k][t]: (k, t) for k in range(K) for t in range(T + 1)}
    flat = [nodes2d[k][t] for k in range(K) for t in range(T + 1)]
    idx_of = {n: i for i, n in enumerate(flat)}

    # Clamped nodes: each agent's start (t=0) and goal (t=T).
    clamped_nodes = [nodes2d[k][0] for k in range(K)] + [nodes2d[k][T] for k in range(K)]
    clamp_vals = start_cells + goal_cells
    clamped_set = set(clamped_nodes)

    # --- factors -------------------------------------------------------------
    M_move = motion_matrix(q, allowed, lam_move)
    M_vertex = vertex_matrix(q, lam_vertex)

    # motion: agent k between t and t+1 (every temporal edge shares M_move)
    move_heads = [nodes2d[k][t] for k in range(K) for t in range(T)]
    move_tails = [nodes2d[k][t + 1] for k in range(K) for t in range(T)]
    move_w = np.broadcast_to(M_move, (len(move_heads), q, q)).copy()

    # vertex conflict: agents k<k' at the same time t (every pair shares M_vertex)
    vtx_pairs = [(nodes2d[k][t], nodes2d[kp][t])
                 for t in range(T + 1) for k in range(K) for kp in range(k + 1, K)]
    vtx_heads = [a for a, _ in vtx_pairs]
    vtx_tails = [b for _, b in vtx_pairs]
    vtx_w = np.broadcast_to(M_vertex, (len(vtx_heads), q, q)).copy()

    move_factor = SquareCategoricalEBMFactor(
        [Block(move_heads), Block(move_tails)], beta * jnp.asarray(move_w))
    vtx_factor = SquareCategoricalEBMFactor(
        [Block(vtx_heads), Block(vtx_tails)], beta * jnp.asarray(vtx_w))

    # --- coloring of the free sub-graph -------------------------------------
    free_nodes = [n for n in flat if n not in clamped_set]
    free_local = {n: i for i, n in enumerate(free_nodes)}
    edges = set()
    for h, t in zip(move_heads, move_tails):
        if h in free_local and t in free_local:
            edges.add((free_local[h], free_local[t]))
    for h, t in zip(vtx_heads, vtx_tails):
        if h in free_local and t in free_local:
            edges.add((free_local[h], free_local[t]))
    colors = dsatur_coloring(len(free_nodes), list(edges))
    n_colors = (max(colors) + 1) if colors else 0
    free_blocks = [Block([free_nodes[i] for i in range(len(free_nodes)) if colors[i] == c])
                   for c in range(n_colors)]
    free_blocks = [b for b in free_blocks if b.nodes]

    clamped_blocks = [Block([n]) for n in clamped_nodes]
    spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    sampler = CategoricalGibbsConditional(q)
    prog = FactorSamplingProgram(spec, [sampler for _ in free_blocks],
                                 [move_factor, vtx_factor], [])

    return dict(
        prog=prog, spec=spec, nodes2d=nodes2d, kt_of=kt_of, idx_of=idx_of, flat=flat,
        free_blocks=free_blocks, clamped_blocks=clamped_blocks, clamp_vals=clamp_vals,
        free_nodes=free_nodes, n_colors=len(free_blocks),
        q=q, K=K, T=T, coords=coords, adj=adj, allowed=allowed, cell_of=cell_of,
        starts=starts, goals=goals, start_cells=start_cells, goal_cells=goal_cells,
        lam_move=lam_move, lam_vertex=lam_vertex, beta=beta,
        # weight bookkeeping for the energy_from_weights cross-check
        move_heads=move_heads, move_tails=move_tails, move_w=np.asarray(beta * move_w),
        vtx_heads=vtx_heads, vtx_tails=vtx_tails, vtx_w=np.asarray(beta * vtx_w),
        rows=rows,
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _clamp_arrays(model):
    return [jnp.array([v], dtype=jnp.uint8) for v in model["clamp_vals"]]


def _scatter_positions(model, states):
    """Turn the per-free-block sampled states into a dense (..., K, T+1) cell array.

    `states` is the list returned by sample_states, parallel to model["free_blocks"].
    Clamped (start/goal) entries are filled from their fixed values.
    """
    K, T = model["K"], model["T"]
    lead = states[0].shape[:-1]  # (n_chains, n_samples) typically
    pos = np.empty((*lead, K, T + 1), dtype=np.int64)
    for k in range(K):
        pos[..., k, 0] = model["start_cells"][k]
        pos[..., k, T] = model["goal_cells"][k]
    for blk, arr in zip(model["free_blocks"], states):
        arr = np.asarray(arr)
        for i, node in enumerate(blk.nodes):
            k, t = model["kt_of"][node]
            pos[..., k, t] = arr[..., i]
    return pos


def sample_positions(model, key, n_chains, schedule):
    """Run n_chains parallel block-Gibbs chains; return cells (n_chains, n_samples, K, T+1)."""
    free_blocks, q = model["free_blocks"], model["q"]
    clamp = _clamp_arrays(model)
    k_init, k_run = jax.random.split(key)
    init = [jax.random.randint(k, (n_chains, len(b.nodes)), 0, q, dtype=jnp.uint8)
            for k, b in zip(jax.random.split(k_init, len(free_blocks)), free_blocks)]
    keys = jax.random.split(k_run, n_chains)

    def one(init_c, k):
        return sample_states(k, model["prog"], schedule, init_c, clamp, free_blocks)

    states = jax.jit(jax.vmap(one))(init, keys)
    return _scatter_positions(model, states)


def anneal(rows, starts, goals, T, betas, schedule, n_chains, key,
           lam_move=1.0, lam_vertex=1.0, return_model=False):
    """Simulated annealing: sweep beta low->high, carrying chain state across stages.

    Single-site Gibbs freezes if started cold, so we ramp temperature. Each stage
    rebuilds the program at the next beta (cheap) and re-inits from the previous
    stage's final state. Returns cells (n_chains, n_samples, K, T+1) from the final
    (coldest) stage.
    """
    key, k_init = jax.random.split(key)
    m0 = build_model(rows, starts, goals, T, float(betas[0]), lam_move, lam_vertex)
    q = m0["q"]
    init = [jax.random.randint(k, (n_chains, len(b.nodes)), 0, q, dtype=jnp.uint8)
            for k, b in zip(jax.random.split(k_init, len(m0["free_blocks"])), m0["free_blocks"])]

    model, states = m0, None
    for b in betas:
        model = build_model(rows, starts, goals, T, float(b), lam_move, lam_vertex)
        clamp = _clamp_arrays(model)
        free_blocks = model["free_blocks"]
        key, k_stage = jax.random.split(key)
        keys = jax.random.split(k_stage, n_chains)

        def one(init_c, k):
            return sample_states(k, model["prog"], schedule, init_c, clamp, free_blocks)

        states = jax.jit(jax.vmap(one))(init, keys)
        init = [s[:, -1, :] for s in states]  # carry final state into next stage

    pos = _scatter_positions(model, states)
    return (pos, model) if return_model else pos


# --------------------------------------------------------------------------- #
# Observables (numpy, vectorized over arbitrary leading batch axes)
# pos has shape (..., K, T+1) of cell indices
# --------------------------------------------------------------------------- #
def n_vertex_conflicts(model, pos):
    """Number of (agent-pair, time) collisions where two agents share a cell."""
    K = model["K"]
    out = np.zeros(pos.shape[:-2], dtype=int)
    for k in range(K):
        for kp in range(k + 1, K):
            out += (pos[..., k, :] == pos[..., kp, :]).sum(-1)
    return out


def n_illegal_moves(model, pos):
    """Number of agent-timesteps where a consecutive move is not adjacent-or-wait."""
    allowed = model["allowed"]
    a = pos[..., :-1]
    b = pos[..., 1:]
    legal = allowed[a, b]  # fancy-index the q x q allowed matrix
    return (~legal).sum((-1, -2))


def is_valid(model, pos):
    """True where every trajectory is a legal walk and there are no vertex collisions."""
    return (n_vertex_conflicts(model, pos) == 0) & (n_illegal_moves(model, pos) == 0)


def sum_of_costs(model, pos):
    """Sum over agents of timesteps until the agent first reaches and stays at its goal.

    A standard MAPF objective: an agent waiting at its goal for the tail of the
    horizon costs nothing for those waits. Counts the last timestep at which the
    agent is away from goal (0 if it is at goal the whole time).
    """
    K, T = model["K"], model["T"]
    out = np.zeros(pos.shape[:-2], dtype=int)
    for k in range(K):
        at_goal = pos[..., k, :] == model["goal_cells"][k]  # (..., T+1)
        away = ~at_goal
        # last index where the agent is away from goal (cost = that index)
        idx = np.arange(T + 1)
        last_away = np.where(away, idx, 0).max(-1)
        out += last_away
    return out


def energy_fp(model, pos, beta=None):
    """First-principles THRML energy E = beta*(lam_move*illegal + lam_vertex*collisions).

    Independent of the weight tensors, so agreement with energy_from_weights
    validates the matrix construction.
    """
    beta = model["beta"] if beta is None else beta
    return beta * (model["lam_move"] * n_illegal_moves(model, pos)
                   + model["lam_vertex"] * n_vertex_conflicts(model, pos))


def energy_from_weights(model, pos):
    """Energy reconstructed from the THRML weight tensors: E = -sum_p W[p, x_head, x_tail].

    pos is indexed by (k,t); map each factor's head/tail node to its (k,t) slot.
    """
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
    return E


# --------------------------------------------------------------------------- #
# Exact enumeration for validation (tiny instances only)
# --------------------------------------------------------------------------- #
def enumerate_states(model):
    """All assignments of the free (unclamped) nodes over q cells.

    Only tractable for tiny instances (q^n_free states). Returns array (#states, K, T+1)
    with clamped start/goal cells filled in.
    """
    q, K, T = model["q"], model["K"], model["T"]
    n_free = len(model["free_nodes"])
    free = np.array(list(itertools.product(range(q), repeat=n_free)), dtype=np.int64)
    pos = np.empty((free.shape[0], K, T + 1), dtype=np.int64)
    for k in range(K):
        pos[:, k, 0] = model["start_cells"][k]
        pos[:, k, T] = model["goal_cells"][k]
    for i, node in enumerate(model["free_nodes"]):
        k, t = model["kt_of"][node]
        pos[:, k, t] = free[:, i]
    return pos
