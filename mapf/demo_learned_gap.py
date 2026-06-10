"""Learned policy on a BOTTLENECK: does it learn gap coordination and generalize?

Mini two-room map with a single 1-wide gap (center of the middle row). Agents start in
the top room and must reach the bottom room, so every plan has to funnel through the one
gap cell -- the exact coordination the solvability transition / throughput-cap study (A)
is about. We train the conditional learned policy (B2) on bottleneck instances and test
on held-out start/goal pairs: a success gif (agents time-share the gap on an UNSEEN
instance) and a failure gif (a collision at the gap).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import mapf_model as mapf
import pt_mapf as pt
import trajectory_sampler as ts
import learn_cond_mapf as lcm
import viz_mapf as viz


def make_gap_instances(rows, T, lay, key, n, top_pool, bot_pool, min_solutions=40, seed=0):
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    K = lay["K"]
    rng = np.random.default_rng(seed)
    inst, tries = [], 0
    while len(inst) < n and tries < n * 20:
        tries += 1
        s = rng.choice(top_pool, K, replace=False)       # starts in the top room
        g = rng.choice(bot_pool, K, replace=False)        # goals in the bottom room
        starts = [tuple(int(x) for x in coords[c]) for c in s]
        goals = [tuple(int(x) for x in coords[c]) for c in g]
        model = pt.light_model(rows, starts, goals, T, lam_swap=1.0)
        key, kk = jax.random.split(key)
        trace = ts.ffbs_gibbs(model, kk, 8000, 40, beta=4.0)[:, -1]
        valid = mapf.is_valid(model, trace)
        if valid.sum() < min_solutions:
            continue
        inst.append(dict(model=model, starts=starts, goals=goals,
                         start_cells=model["start_cells"], goal_cells=model["goal_cells"],
                         input_vec=lcm.encode_input(lay, model["start_cells"], model["goal_cells"]),
                         out_data=lcm.encode_output(lay, trace[valid])))
    return inst


def main():
    print(f"JAX devices: {jax.devices()}")
    rows = ["...", "#.#", "..."]          # 1-wide gap at (1,1)
    T = 5
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    top = np.array([c for c in range(q) if coords[c][0] == 0])
    bot = np.array([c for c in range(q) if coords[c][0] == len(rows) - 1])
    K = 2
    lay = lcm.build_layout(K=K, T=T, q=q, n_lat=128)
    print(f"gap map q={q}  gap cell=(1,1)  K={K} T={T}  n_in={lay['n_in']} n_out={lay['n_out']}")

    key = jax.random.key(0)
    key, kd = jax.random.split(key)
    allinst = make_gap_instances(rows, T, lay, kd, n=50, top_pool=top, bot_pool=bot, min_solutions=30)
    n_tr = 38
    train, test = allinst[:n_tr], allinst[n_tr:]
    print(f"instances: {len(train)} train, {len(test)} held-out (all cross the gap)")

    key, kw = jax.random.split(key)
    biases = jnp.zeros(len(lay["alln"]))
    weights = 0.01 * jax.random.normal(kw, (len(lay["edges"]),))
    trainer = lcm.make_trainer(lay, lr=0.02)
    rng = np.random.default_rng(0)
    for s in range(3000):
        ins = train[rng.integers(0, len(train))]
        od = ins["out_data"]; bidx = rng.integers(0, len(od), 128)
        key, ks = jax.random.split(key)
        biases, weights = trainer(biases, weights, jnp.asarray(od[bidx]),
                                  jnp.asarray(ins["input_vec"]), ks)
        if (s + 1) % 1000 == 0:
            print(f"  trained {s+1}/3000")

    res = []
    for ins in test:
        key, k = jax.random.split(key)
        plans, valid = lcm.sample_plans(lay, biases, weights, ins, k, n_chains=2000)
        res.append((ins, plans, valid))
    per_sample = float(np.mean([v.mean() for _, _, v in res]))
    solved = float(np.mean([v.any() for _, _, v in res]))
    vfracs = np.array([v.mean() for _, _, v in res])
    print(f"HELD-OUT (gap): solved={solved:.3f}  per_sample_valid={per_sample:.3f}")

    si = int(np.argmax(vfracs))
    ins_s, plans_s, valid_s = res[si]
    succ = plans_s[int(np.flatnonzero(valid_s)[0])]
    fi = int(np.argmin(vfracs))
    ins_f, plans_f, valid_f = res[fi]
    inv = np.flatnonzero(~valid_f)
    viol = mapf.n_vertex_conflicts(ins_f["model"], plans_f) + mapf.n_illegal_moves(ins_f["model"], plans_f)
    fail = plans_f[int(inv[np.argmin(viol[inv])])] if len(inv) else plans_f[0]

    viz.animate_plan(ins_s["model"], succ, "learned_gap_success.gif")
    viz.filmstrip(ins_s["model"], succ, "learned_gap_success_filmstrip.png")
    viz.animate_plan(ins_f["model"], fail, "learned_gap_fail.gif", highlight_conflicts=True)
    viz.filmstrip(ins_f["model"], fail, "learned_gap_fail_filmstrip.png")
    print(f"rendered gap gifs. success inst valid={vfracs[si]:.2f} start={ins_s['starts']} goal={ins_s['goals']}")
    print(f"                  failure inst valid={vfracs[fi]:.2f} start={ins_f['starts']} goal={ins_f['goals']}")


if __name__ == "__main__":
    main()
