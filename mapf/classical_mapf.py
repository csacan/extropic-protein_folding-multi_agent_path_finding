"""Classical ground truth for the vertex-conflict MAPF relaxation (no THRML).

Brute-force feasibility on tiny instances: enumerate every legal single-agent walk
of makespan T from start to goal, then backtrack over agents pruning any timestep
where two agents share a cell. Returns one collision-free joint plan, or None if the
instance is infeasible within T. Tractable only for small maps / few agents, which is
exactly the regime where we can certify THRML's block-Gibbs solutions are correct.

(Vertex conflicts only, to match the v1 EBM. Swap/edge conflicts are added later in
both the EBM and here together.)
"""

from __future__ import annotations

import numpy as np

import mapf_model as mapf


def agent_walks(model, start_cell, goal_cell):
    """All legal walks [c_0=start, ..., c_T=goal] under the allowed (adj|wait) moves.

    Pruned by a reachability bound: drop any prefix that cannot still reach the goal
    in the remaining steps (BFS shortest-path distance on the move graph).
    """
    allowed = model["allowed"]
    q, T = model["q"], model["T"]
    nbrs = [np.flatnonzero(allowed[c]) for c in range(q)]

    # BFS distance from every cell to the goal (move graph is symmetric)
    dist = np.full(q, q + 1, dtype=int)
    dist[goal_cell] = 0
    frontier = [goal_cell]
    while frontier:
        nxt = []
        for c in frontier:
            for d in nbrs[c]:
                if dist[d] > dist[c] + 1:
                    dist[d] = dist[c] + 1
                    nxt.append(d)
        frontier = nxt

    walks = []
    path = [start_cell]

    def dfs(t):
        c = path[-1]
        if t == T:
            if c == goal_cell:
                walks.append(tuple(path))
            return
        for d in nbrs[c]:
            if dist[d] <= T - (t + 1):  # can still reach goal in time
                path.append(int(d))
                dfs(t + 1)
                path.pop()

    if dist[start_cell] <= T:
        dfs(0)
    return walks


def solve_feasible(model):
    """Return a collision-free joint plan (K, T+1) cell array, or None if infeasible."""
    K, T = model["K"], model["T"]
    walk_sets = [agent_walks(model, model["start_cells"][k], model["goal_cells"][k])
                 for k in range(K)]
    if any(len(w) == 0 for w in walk_sets):
        return None

    chosen = [None] * K

    def conflict(k, walk):
        for j in range(k):
            wj = chosen[j]
            for t in range(T + 1):
                if walk[t] == wj[t]:
                    return True
        return False

    def backtrack(k):
        if k == K:
            return True
        for walk in walk_sets[k]:
            if not conflict(k, walk):
                chosen[k] = walk
                if backtrack(k + 1):
                    return True
                chosen[k] = None
        return False

    if backtrack(0):
        return np.array(chosen, dtype=np.int64)  # (K, T+1)
    return None


def is_feasible(model):
    """True iff a collision-free joint plan exists within the makespan."""
    return solve_feasible(model) is not None
