"""Conflict-Based Search (CBS): a complete classical MAPF oracle for the fixed-horizon
vertex+swap formulation. Used as GROUND TRUTH to separate genuinely infeasible
instances from feasible-but-the-sampler-missed-it.

CBS is two-level:
  * low level  : time-expanded shortest path for one agent from (start,0) to (goal,T),
                 moves = neighbor-or-wait, subject to a set of vertex constraints
                 (cell,t forbidden) and edge constraints (move u->v at t forbidden).
                 Dijkstra over (cell,t); cost = #timesteps the agent is away from goal.
  * high level : best-first over a constraint tree. Find the first conflict between two
                 agents' paths (vertex: same cell same t; swap: they exchange cells over
                 t->t+1) and branch, adding the matching constraint to each agent. A
                 conflict-free node is a solution; an exhausted open list proves
                 infeasibility (within the makespan T). A node budget guards the
                 worst-case exponential blow-up (reported as 'timeout', not infeasible).

Complete and (sum-of-cost) optimal; here we only need the feasibility verdict.
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

import mapf_model as mapf


def _neighbors(model):
    allowed = model["allowed"]
    return [np.flatnonzero(allowed[c]).tolist() for c in range(model["q"])]


def low_level(model, nbrs, agent, vcons, econs):
    """Min-cost time-expanded path (cell list, length T+1) for one agent, or None.

    vcons: set of (cell, t) the agent may not occupy.
    econs: set of (u, v, t) -- the agent may not move u->v during t->t+1.
    """
    T = model["T"]
    s = model["start_cells"][agent]
    g = model["goal_cells"][agent]
    if (s, 0) in vcons:
        return None
    start = (s, 0)
    dist = {start: 0}
    par = {}
    pq = [(0, start)]
    while pq:
        d, (c, t) = heapq.heappop(pq)
        if d > dist.get((c, t), 1 << 60) or t == T:
            continue
        for c2 in nbrs[c]:
            if (c2, t + 1) in vcons or (c, c2, t) in econs:
                continue
            nd = d + (0 if c2 == g else 1)
            if nd < dist.get((c2, t + 1), 1 << 60):
                dist[(c2, t + 1)] = nd
                par[(c2, t + 1)] = (c, t)
                heapq.heappush(pq, (nd, (c2, t + 1)))
    if (g, T) not in dist:
        return None
    node, path = (g, T), []
    while node in par:
        path.append(node[0])
        node = par[node]
    path.append(s)
    return path[::-1]


def first_conflict(model, paths):
    """Return the first vertex or swap conflict, or None if the joint plan is valid."""
    K, T = len(paths), model["T"]
    for t in range(T + 1):
        seen = {}
        for k in range(K):
            c = paths[k][t]
            if c in seen:
                return ("v", seen[c], k, c, t)
            seen[c] = k
    for t in range(T):
        for i in range(K):
            for j in range(i + 1, K):
                if (paths[i][t] != paths[i][t + 1]
                        and paths[i][t] == paths[j][t + 1]
                        and paths[i][t + 1] == paths[j][t]):
                    return ("e", i, j, t, paths[i][t], paths[i][t + 1])
    return None


def _children(conf):
    """Two (agent, vertex_constraint, edge_constraint) branches resolving a conflict."""
    if conf[0] == "v":
        _, i, j, cell, t = conf
        return [(i, (cell, t), None), (j, (cell, t), None)]
    _, i, j, t, u, v = conf
    return [(i, None, (u, v, t)), (j, None, (v, u, t))]  # i forbidden u->v ; j forbidden v->u


def solve(model, node_budget=30000):
    """Return a conflict-free joint plan (K, T+1) ndarray, None (infeasible), or 'timeout'."""
    K = model["K"]
    nbrs = _neighbors(model)
    goals = model["goal_cells"]

    def cost(paths):
        return sum(sum(c != goals[k] for c in p) for k, p in enumerate(paths))

    vcons = [set() for _ in range(K)]
    econs = [set() for _ in range(K)]
    paths = []
    for k in range(K):
        p = low_level(model, nbrs, k, vcons[k], econs[k])
        if p is None:
            return None  # an agent cannot even reach its goal alone -> infeasible
        paths.append(p)

    cnt = itertools.count()
    openq = [(cost(paths), next(cnt), vcons, econs, paths)]
    nodes = 0
    while openq:
        nodes += 1
        if nodes > node_budget:
            return "timeout"
        _, _, vc, ec, paths = heapq.heappop(openq)
        conf = first_conflict(model, paths)
        if conf is None:
            return np.array(paths, dtype=np.int64)
        for k, addv, adde in _children(conf):
            nvc = [set(x) for x in vc]
            nec = [set(x) for x in ec]
            if addv:
                nvc[k].add(addv)
            if adde:
                nec[k].add(adde)
            p = low_level(model, nbrs, k, nvc[k], nec[k])
            if p is None:
                continue
            np_paths = list(paths)
            np_paths[k] = p
            heapq.heappush(openq, (cost(np_paths), next(cnt), nvc, nec, np_paths))
    return None  # open list exhausted -> provably infeasible within horizon T


def feasibility(model, node_budget=30000):
    """'feasible' / 'infeasible' / 'timeout' verdict from CBS."""
    r = solve(model, node_budget)
    if isinstance(r, str):  # 'timeout'
        return "timeout"
    return "feasible" if r is not None else "infeasible"
