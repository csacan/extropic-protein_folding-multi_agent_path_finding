"""Validate the CBS feasibility oracle against brute-force enumeration (run before use).

  [1] agreement : on tiny instances, CBS's feasible/infeasible verdict must match the
                  brute-force minimum-violations oracle (min_violations==0 iff feasible).
  [2] soundness : every plan CBS returns must be fully conflict-free (vertex+move+swap).
  [3] corridor  : the 1-wide corridor swap instance must be reported infeasible.
"""

from __future__ import annotations

import numpy as np

import mapf_model as mapf
import swap_conflicts as sw
import classical_swap as csw
import cbs_mapf as cbs


def _check(name, rows, starts, goals, T):
    model = sw.build_model_swap(rows, starts, goals, T, beta=1.0, lam_swap=1.0)
    bf_feasible = csw.min_violations(model) == 0
    verdict = cbs.feasibility(model)
    assert verdict != "timeout", f"{name}: CBS timed out on a tiny instance"
    cbs_feasible = verdict == "feasible"
    ok = cbs_feasible == bf_feasible
    sound = True
    if cbs_feasible:
        plan = cbs.solve(model)
        v = int(mapf.n_vertex_conflicts(model, plan) + mapf.n_illegal_moves(model, plan)
                + sw.n_swap_conflicts(model, plan))
        sound = (v == 0)
    print(f"    {name:12s} K={len(starts)} T={T} q={model['q']}  "
          f"CBS={'feas' if cbs_feasible else 'infeas'}  brute={'feas' if bf_feasible else 'infeas'}  "
          f"sound={sound}  {'PASS' if (ok and sound) else 'FAIL'}")
    assert ok, f"{name}: CBS disagrees with brute force"
    assert sound, f"{name}: CBS returned a plan with conflicts"


def test_agreement_and_soundness():
    print("[1+2] CBS vs brute force (agreement) + returned-plan soundness")
    _check("open2",  ["...", "..."],       [(0, 0), (0, 2)], [(1, 2), (1, 0)], 3)  # feasible
    _check("down3",  ["...", "..."],       [(0, 0), (0, 1), (0, 2)],
                                           [(1, 0), (1, 1), (1, 2)], 2)            # feasible
    _check("diag",   ["...", "...", "..."], [(0, 0), (2, 0)], [(2, 2), (0, 2)], 4) # feasible cross
    _check("tight2", ["...", "..."],       [(0, 0), (0, 1)], [(0, 2), (0, 0)], 2)  # infeasible (swap)
    # random tiny instances on a 2x3 grid with enough makespan
    rng = np.random.default_rng(0)
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(["...", "..."]))
    for r in range(4):
        ch = rng.choice(q, 4, replace=False)
        s = [tuple(int(x) for x in coords[c]) for c in ch[:2]]
        g = [tuple(int(x) for x in coords[c]) for c in ch[2:]]
        _check(f"rand{r}", ["...", "..."], s, g, 3)
    print("    PASS\n")


def test_corridor_infeasible():
    print("[3] corridor swap instance must be infeasible")
    rows, starts, goals, T = ["....."], [(0, 0), (0, 4)], [(0, 4), (0, 0)], 5
    model = sw.build_model_swap(rows, starts, goals, T, beta=1.0, lam_swap=1.0)
    verdict = cbs.feasibility(model)
    bf = csw.min_violations(model)
    print(f"    CBS={verdict}  brute_min_violations={bf}")
    assert verdict == "infeasible" and bf > 0
    print("    PASS\n")


if __name__ == "__main__":
    test_agreement_and_soundness()
    test_corridor_infeasible()
    print("all CBS validation checks passed.")
