"""Swap-aware classical ground truth for the v2 MAPF EBM (tiny instances only).

Brute-force minimum number of total violations (vertex + illegal-move + swap) over
every joint plan, by enumerating all free-node assignments. Used to certify that
THRML's annealed minimum on a tiny instance equals the true optimum -- in particular
that adding the swap penalty turns the 1-wide-corridor crossing from feasible (v1)
into a genuine >0 minimum.
"""

from __future__ import annotations

import mapf_model as mapf
import swap_conflicts as sw


def total_violations(model, pos):
    """Vertex + illegal-move + swap violation count for plans pos (..., K, T+1)."""
    return (mapf.n_vertex_conflicts(model, pos)
            + mapf.n_illegal_moves(model, pos)
            + sw.n_swap_conflicts(model, pos))


def min_violations(model):
    """Minimum total violations over all joint plans (enumerates q^n_free states)."""
    states = mapf.enumerate_states(model)
    return int(total_violations(model, states).min())


def is_feasible_swap(model):
    """True iff a fully conflict-free (vertex+move+swap) joint plan exists within T."""
    return min_violations(model) == 0
