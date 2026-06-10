"""Animate sampled MAPF plans: grid + obstacles + multi-agent trajectories.

Renders a plan (K, T+1) of cell indices as
  - an animated GIF (agents = colored disks moving cell-to-cell with fading trails,
    start = hollow square, goal = star), and
  - a static "filmstrip" PNG (snapshots at several timesteps) for slides / quick checks.

Uses pt_mapf.light_model so no q^4 swap tensor is built; FFBS+swap supplies a valid plan.
"""

from __future__ import annotations

import numpy as np
import jax

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle

from thrml import SamplingSchedule

import mapf_model as mapf
import swap_conflicts as sw
import trajectory_sampler as ts
import pt_mapf as pt


def _dims(model):
    rows = model["rows"]
    return len(rows), len(rows[0])


def _colors(K):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(K)]


def _cell_index(model):
    """(row, col) -> cell index, from model['coords']."""
    return {(int(r), int(c)): i for i, (r, c) in enumerate(model["coords"])}


def gap_cells(model):
    """Cell indices that sit in a 'wall row' (mostly blocked) -- the bottleneck throat."""
    rows = model["rows"]
    passable = mapf.parse_map(rows)
    R, C = len(rows), len(rows[0])
    idx = _cell_index(model)
    out = []
    for r in range(R):
        if passable[r].sum() <= C // 2:  # a wall row
            out += [idx[(r, c)] for c in range(C) if passable[r, c]]
    return out


def _collision_cells(plan_t):
    """Cells occupied by >= 2 agents in a single timestep slice plan_t (K,)."""
    vals, counts = np.unique(plan_t, return_counts=True)
    return vals[counts >= 2]


def _conflicts(model, plan):
    """Total vertex + illegal-move conflicts of a single plan (K, T+1)."""
    p = plan[None]
    return int(mapf.n_vertex_conflicts(model, p)[0] + mapf.n_illegal_moves(model, p)[0])


def _draw_grid(ax, model):
    R, C = _dims(model)
    passable = mapf.parse_map(model["rows"])
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(-0.5, R - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(R))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="0.88", lw=0.6)
    for r in range(R):
        for c in range(C):
            if not passable[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.45", zorder=1))


def sample_valid_plan(model, key, n_chains=4000, n_sweeps=50, beta=8.0):
    """Draw plans with FFBS+swap and return one fully valid plan (or the best found)."""
    trace = ts.ffbs_gibbs(model, key, n_chains, n_sweeps, beta=beta)
    last = trace[:, -1]  # (n_chains, K, T+1)
    valid = sw.is_valid_swap(model, last)
    if valid.any():
        return last[np.flatnonzero(valid)[0]], True
    viol = (mapf.n_vertex_conflicts(model, last) + mapf.n_illegal_moves(model, last)
            + sw.n_swap_conflicts(model, last))
    return last[int(viol.argmin())], False


def animate_plan(model, plan, path="mapf.gif", steps_per_move=8, fps=12,
                 highlight_conflicts=False):
    plan = np.asarray(plan)
    K, T1 = plan.shape
    T = T1 - 1
    coords = model["coords"].astype(float)
    rc = coords[plan]  # (K, T+1, 2) as (row, col)
    cols = _colors(K)
    R, C = _dims(model)

    fig, ax = plt.subplots(figsize=(C * 0.7 + 1, R * 0.7 + 1))
    _draw_grid(ax, model)
    for k in range(K):  # start (square) + goal (star)
        ax.scatter([rc[k, 0, 1]], [rc[k, 0, 0]], marker="s", s=130,
                   facecolors="none", edgecolors=[cols[k]], lw=2, zorder=3)
        ax.scatter([rc[k, -1, 1]], [rc[k, -1, 0]], marker="*", s=260,
                   color=[cols[k]], alpha=0.55, zorder=3)
    disks = [Circle((rc[k, 0, 1], rc[k, 0, 0]), 0.30, color=cols[k], zorder=5) for k in range(K)]
    for d in disks:
        ax.add_patch(d)
    tags = [ax.text(rc[k, 0, 1], rc[k, 0, 0], str(k), ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold", zorder=6) for k in range(K)]
    trails = [ax.plot([], [], color=cols[k], lw=1.8, alpha=0.4, zorder=2)[0] for k in range(K)]
    hops = [ax.plot([], [], color="red", lw=2.5, ls=(0, (2, 1.5)), zorder=4)[0] for k in range(K)]
    coll = ax.scatter([], [], s=520, facecolors="none", edgecolors="red", lw=3, zorder=7)
    title = ax.set_title("t = 0.0")
    # which moves are illegal (non-adjacent and not a wait) -> flagged as red teleports
    allowed = model["allowed"]
    legal = np.array([[bool(allowed[plan[k, t], plan[k, t + 1]]) for t in range(T)] for k in range(K)])

    def frame(f):
        seg = min(f // steps_per_move, T - 1)
        fr = (f - seg * steps_per_move) / steps_per_move
        cur = rc[:, seg] * (1 - fr) + rc[:, seg + 1] * fr  # (K, 2)
        for k in range(K):
            disks[k].center = (cur[k, 1], cur[k, 0])
            tags[k].set_position((cur[k, 1], cur[k, 0]))
            xs = list(rc[k, :seg + 1, 1]) + [cur[k, 1]]
            ys = list(rc[k, :seg + 1, 0]) + [cur[k, 0]]
            trails[k].set_data(xs, ys)
            if not legal[k, seg]:  # illegal hop: dashed red line + red disk outline
                hops[k].set_data([rc[k, seg, 1], rc[k, seg + 1, 1]],
                                 [rc[k, seg, 0], rc[k, seg + 1, 0]])
                disks[k].set_edgecolor("red"); disks[k].set_linewidth(2.5)
            else:
                hops[k].set_data([], [])
                disks[k].set_linewidth(0)
        if highlight_conflicts:
            tcur = seg if fr < 0.5 else seg + 1
            cc = _collision_cells(plan[:, tcur])
            coll.set_offsets(coords[cc][:, ::-1] if len(cc) else np.empty((0, 2)))
        title.set_text(f"t = {seg + fr:.1f}")
        return disks + tags + trails + hops + [coll, title]

    anim = animation.FuncAnimation(fig, frame, frames=T * steps_per_move + 1,
                                   interval=1000 / fps, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def filmstrip(model, plan, path="mapf_filmstrip.png", n_panels=6):
    plan = np.asarray(plan)
    K, T1 = plan.shape
    T = T1 - 1
    rc = model["coords"].astype(float)[plan]
    cols = _colors(K)
    idxs = np.unique(np.linspace(0, T, min(n_panels, T1)).round().astype(int))

    fig, axes = plt.subplots(1, len(idxs), figsize=(2.4 * len(idxs), 2.7))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, idxs):
        _draw_grid(ax, model)
        for k in range(K):
            ax.plot(rc[k, :t + 1, 1], rc[k, :t + 1, 0], color=cols[k], lw=1.3, alpha=0.35, zorder=2)
            ax.scatter([rc[k, -1, 1]], [rc[k, -1, 0]], marker="*", s=110, color=[cols[k]], alpha=0.5, zorder=3)
            ax.add_patch(Circle((rc[k, t, 1], rc[k, t, 0]), 0.30, color=cols[k], zorder=5))
            ax.text(rc[k, t, 1], rc[k, t, 0], str(k), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold", zorder=6)
        ax.set_title(f"t = {t}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _draw_plan_static(ax, model, plan, cols):
    """Draw one full candidate plan as agent paths; illegal (non-adjacent) hops in red.

    Each segment links an agent's consecutive-in-time cells: legal moves are
    horizontal/vertical (orthogonal or wait); a red dashed diagonal marks an illegal
    teleport, so an unconverged plan's violations are visible, not mysterious diagonals.
    """
    _draw_grid(ax, model)
    plan = np.asarray(plan)
    rc = model["coords"].astype(float)[plan]
    allowed = model["allowed"]
    K, T1 = plan.shape
    for k in range(K):
        for t in range(T1 - 1):
            x = [rc[k, t, 1], rc[k, t + 1, 1]]
            y = [rc[k, t, 0], rc[k, t + 1, 0]]
            if allowed[plan[k, t], plan[k, t + 1]]:
                ax.plot(x, y, color=cols[k], lw=2.0, alpha=0.6, zorder=2)
            else:
                ax.plot(x, y, color="red", lw=2.0, ls=(0, (2, 1.5)), alpha=0.9, zorder=3)
        ax.add_patch(Circle((rc[k, 0, 1], rc[k, 0, 0]), 0.26, color=cols[k], zorder=5))
        ax.scatter([rc[k, -1, 1]], [rc[k, -1, 0]], marker="*", s=180,
                   color=[cols[k]], alpha=0.5, zorder=3)


def animate_mixing(rows, starts, goals, T, path="mapf_mixing.gif",
                   beta=8.0, n_chains=512, n_sweeps=30, chain=0, fps=4):
    """Side-by-side FFBS vs single-site Gibbs: watch one chain's plan untangle per sweep.

    Title carries that chain's conflict count and the batch valid-fraction, so the
    visual (one chain) and the quantitative mixing speed (the batch) are both on screen.
    """
    model = mapf.build_model(rows, starts, goals, T, beta, lam_move=1.0, lam_vertex=1.0)
    cols = _colors(len(starts))

    ff = ts.ffbs_gibbs(model, jax.random.key(0), n_chains, n_sweeps, beta=beta)  # (n,sw,K,T+1)
    sched = SamplingSchedule(n_warmup=0, n_samples=n_sweeps, steps_per_sample=1)
    ss = mapf.sample_positions(model, jax.random.key(0), n_chains, sched)        # (n,sw,K,T+1)
    ff_vf = mapf.is_valid(model, ff).mean(0)
    ss_vf = mapf.is_valid(model, ss).mean(0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(2 * (_dims(model)[1] * 0.6 + 1), _dims(model)[0] * 0.6 + 1.6))

    def frame(s):
        axL.clear(); axR.clear()
        _draw_plan_static(axL, model, ff[chain, s], cols)
        _draw_plan_static(axR, model, ss[chain, s], cols)
        axL.set_title(f"FFBS whole-trajectory\nsweep {s}:  chain conflicts={_conflicts(model, ff[chain, s])}"
                      f"   valid={ff_vf[s]:.0%}", fontsize=10)
        axR.set_title(f"single-site Gibbs\nsweep {s}:  chain conflicts={_conflicts(model, ss[chain, s])}"
                      f"   valid={ss_vf[s]:.0%}", fontsize=10)

    anim = animation.FuncAnimation(fig, frame, frames=n_sweeps, interval=1000 / fps)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path, float(ff_vf[0]), float(ss_vf[-1])


def gap_utilization_figure(rows, K_values, T, path="gap_utilization.png",
                           n_instances=4, n_chains=512, beta=8.0, seed=0):
    """Expected #agents in the bottleneck gap vs time, per agent count K.

    As K grows the gap saturates (occupancy -> 1 across the whole horizon): the single
    cell can pass at most ~one agent per step, so beyond a makespan-limited capacity no
    schedule fits -- the mechanism behind the solvability transition.
    """
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    mid = len(rows) / 2.0
    start_pool = np.array([c for c in range(q) if coords[c][0] < mid])
    goal_pool = np.array([c for c in range(q) if coords[c][0] > mid])
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for ki, K in enumerate(K_values):
        occ = np.zeros(T + 1)
        n = 0
        for _ in range(n_instances):
            s = rng.choice(start_pool, K, replace=False)
            g = rng.choice(goal_pool, K, replace=False)
            starts = [tuple(int(x) for x in coords[c]) for c in s]
            goals = [tuple(int(x) for x in coords[c]) for c in g]
            model = pt.light_model(rows, starts, goals, T, lam_swap=1.0)
            gset = np.array(gap_cells(model))
            key, kk = jax.random.split(key)
            last = ts.ffbs_gibbs(model, kk, n_chains, 40, beta=beta)[:, -1]  # (n,K,T+1)
            inb = np.isin(last, gset)  # (n,K,T+1)
            occ += inb.sum(1).mean(0)  # expected #agents in gap per t
            n += 1
        ax.plot(np.arange(T + 1), occ / n, "-o", ms=3,
                color=cmap(ki / max(1, len(K_values) - 1)), label=f"K={K}")
    ax.axhline(1.0, color="0.6", ls="--", lw=1, label="1 agent (capacity)")
    ax.set_xlabel("time t")
    ax.set_ylabel("expected agents in gap")
    ax.set_title("Bottleneck gap utilization vs time")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_hard_instance(rows, K, T, path_gif="mapf_hard.gif",
                         path_strip="mapf_hard_filmstrip.png", n_chains=4000, beta=8.0, seed=1):
    """Sample a hard (K large) bottleneck instance and render a FAILED plan with the
    collisions highlighted in red -- coordination breaking down at the transition."""
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    mid = len(rows) / 2.0
    rng = np.random.default_rng(seed)
    sp = np.array([c for c in range(q) if coords[c][0] < mid])
    gp = np.array([c for c in range(q) if coords[c][0] > mid])
    s = rng.choice(sp, K, replace=False)
    g = rng.choice(gp, K, replace=False)
    starts = [tuple(int(x) for x in coords[c]) for c in s]
    goals = [tuple(int(x) for x in coords[c]) for c in g]
    model = pt.light_model(rows, starts, goals, T, lam_swap=1.0)

    last = ts.ffbs_gibbs(model, jax.random.key(seed), n_chains, 50, beta=beta)[:, -1]
    nvtx = mapf.n_vertex_conflicts(model, last)
    viol = nvtx + mapf.n_illegal_moves(model, last) + sw.n_swap_conflicts(model, last)
    # prefer a near-miss with a visible vertex collision; else the min-violation plan
    cand = np.flatnonzero((nvtx > 0) & (viol > 0))
    pick = int(cand[viol[cand].argmin()]) if len(cand) else int(viol.argmin())
    plan = last[pick]
    solved = float((viol == 0).mean())
    print(f"  hard K={K}: best-of-batch min_viol={int(viol.min())}  "
          f"shown FAIL plan viol={int(viol[pick])} (vertex={int(nvtx[pick])})  "
          f"solved_frac={solved:.2f}")

    # success counterpart on the SAME instance, if any chain solved it
    valid = np.flatnonzero(viol == 0)
    if len(valid):
        s_plan = last[int(valid[0])]
        animate_plan(model, s_plan, path_gif.replace(".gif", "_success.gif"))
        filmstrip(model, s_plan, path_strip.replace(".png", "_success.png"))
        print(f"  -> success counterpart ({len(valid)}/{n_chains} chains solved): "
              f"{path_gif.replace('.gif', '_success.gif')}")

    animate_plan(model, plan, path_gif, highlight_conflicts=True)
    # filmstrip with red rings on colliding cells; ensure a collision frame is shown
    rc = coords.astype(float)[plan]
    cols = _colors(K)
    coll_ts = [t for t in range(T + 1) if len(_collision_cells(plan[:, t]))]
    base = list(np.linspace(0, T, 5).round().astype(int))
    idxs = np.array(sorted(set(base) | set(coll_ts[:1])))
    fig, axes = plt.subplots(1, len(idxs), figsize=(2.4 * len(idxs), 2.7))
    for ax, t in zip(np.atleast_1d(axes), idxs):
        _draw_grid(ax, model)
        for k in range(K):
            ax.plot(rc[k, :t + 1, 1], rc[k, :t + 1, 0], color=cols[k], lw=1.1, alpha=0.3)
            ax.add_patch(Circle((rc[k, t, 1], rc[k, t, 0]), 0.28, color=cols[k], zorder=5))
        cc = _collision_cells(plan[:, t])
        if len(cc):
            ax.scatter(coords[cc][:, 1], coords[cc][:, 0], s=360, facecolors="none",
                       edgecolors="red", lw=2.5, zorder=7)
        ax.set_title(f"t = {t}")
    fig.tight_layout()
    fig.savefig(path_strip, dpi=130)
    plt.close(fig)
    return path_gif, path_strip


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    bottleneck = [
        ".......",
        ".......",
        "###.###",
        ".......",
        ".......",
    ]

    # (1) original: 4 agents negotiate the gap, valid plan
    starts = [(0, 1), (0, 5), (4, 1), (4, 5)]
    goals = [(4, 5), (4, 1), (0, 5), (0, 1)]
    model = pt.light_model(bottleneck, starts, goals, T=14, lam_swap=1.0)
    plan, ok = sample_valid_plan(model, jax.random.key(0))
    print(f"valid plan: {ok}  conflicts={_conflicts(model, plan)}")
    print("  ->", animate_plan(model, plan, "mapf.gif"))
    print("  ->", filmstrip(model, plan, "mapf_filmstrip.png"))

    # (2) side-by-side mixing: FFBS snaps, single-site crawls (3x6 open, K=3)
    p, ff0, ssN = animate_mixing(["......"] * 3, [(0, 0), (0, 5), (2, 2)],
                                 [(2, 5), (2, 0), (0, 2)], T=10, path="mapf_mixing.gif")
    print(f"  -> {p}   FFBS valid@sweep0={ff0:.0%}  single-site valid@last={ssN:.0%}")

    # (3) gap utilization vs time, per K -> saturation = throughput cap
    print("  ->", gap_utilization_figure(bottleneck, [2, 4, 6, 8, 10], T=12))

    # (4) hard instance: coordination breaks down, collisions in red
    print("  ->", render_hard_instance(bottleneck, K=8, T=12))
