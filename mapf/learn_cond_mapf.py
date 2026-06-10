"""B2: a CONDITIONAL learned MAPF solver -- generalize across instances, not memorize one.

A conditional RBM on a fixed grid:
  * input spins  (clamped at inference) = each agent's start + goal, one-hot.
  * output spins (sampled)              = the middle trajectory positions t=1..T-1.
  * latent spins                        = hidden coordination units.

Trained by conditional contrastive KL (estimate_kl_grad): for each instance, clamp the
input, clamp the output to a teacher solution (positive phase) vs sample it (negative
phase). conditioning_values are broadcast across the batch, so each step conditions on
ONE instance with a batch of its FFBS solutions; training cycles through many instances.

The test is GENERALIZATION: clamp the start/goal of HELD-OUT instances the model never
trained on, sample, prepend start / append goal, and measure how often it emits a valid
collision-free plan. High held-out validity = an amortized solver, the Extropic pitch
classical search can't make.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from thrml import Block, SpinNode, SamplingSchedule, sample_states
from thrml.models import (IsingEBM, IsingSamplingProgram, IsingTrainingSpec,
                          estimate_kl_grad, hinton_init)

import mapf_model as mapf
import pt_mapf as pt
import trajectory_sampler as ts


# --------------------------------------------------------------------------- #
# spin layout: input (start|goal) + output (middle) + latent
# --------------------------------------------------------------------------- #
def build_layout(K, T, q, n_lat):
    n_in = 2 * K * q            # start (K*q) then goal (K*q)
    n_out = K * (T - 1) * q     # middle times t=1..T-1
    inp = [SpinNode() for _ in range(n_in)]
    out = [SpinNode() for _ in range(n_out)]
    lat = [SpinNode() for _ in range(n_lat)]
    alln = inp + out + lat
    edges = [(v, l) for v in (inp + out) for l in lat]  # RBM: visible <-> latent
    return dict(inp=inp, out=out, lat=lat, alln=alln, edges=edges,
                K=K, T=T, q=q, n_in=n_in, n_out=n_out)


def encode_input(lay, start_cells, goal_cells):
    K, q = lay["K"], lay["q"]
    v = np.zeros(lay["n_in"], dtype=bool)
    for k in range(K):
        v[k * q + start_cells[k]] = True
        v[K * q + k * q + goal_cells[k]] = True
    return v


def encode_output(lay, plans):
    """plans (N,K,T+1) -> bool (N, n_out) over middle positions t=1..T-1."""
    K, T, q = lay["K"], lay["T"], lay["q"]
    N = plans.shape[0]
    d = np.zeros((N, lay["n_out"]), dtype=bool)
    for k in range(K):
        for t in range(1, T):
            base = (k * (T - 1) + (t - 1)) * q
            d[np.arange(N), base + plans[:, k, t]] = True
    return d


def decode_full(lay, out_spins, start_cells, goal_cells):
    """out_spins (N, n_out) -> full plans (N,K,T+1) with start/goal prepended/appended."""
    K, T, q = lay["K"], lay["T"], lay["q"]
    N = out_spins.shape[0]
    mid = out_spins.reshape(N, K, T - 1, q).astype(np.int8).argmax(-1)  # (N,K,T-1)
    plans = np.empty((N, K, T + 1), dtype=np.int64)
    for k in range(K):
        plans[:, k, 0] = start_cells[k]
        plans[:, k, T] = goal_cells[k]
        plans[:, k, 1:T] = mid[:, k]
    return plans


# --------------------------------------------------------------------------- #
# data: a family of instances, each with FFBS teacher solutions
# --------------------------------------------------------------------------- #
def make_instances(rows, T, lay, key, n_instances, min_solutions=40,
                   n_chains=6000, n_sweeps=30, beta=3.0, seed=0):
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    K = lay["K"]
    rng = np.random.default_rng(seed)
    inst = []
    tries = 0
    while len(inst) < n_instances and tries < n_instances * 12:
        tries += 1
        cells = rng.choice(q, 2 * K, replace=False)
        starts = [tuple(int(x) for x in coords[c]) for c in cells[:K]]
        goals = [tuple(int(x) for x in coords[c]) for c in cells[K:]]
        model = pt.light_model(rows, starts, goals, T, lam_swap=1.0)
        key, kk = jax.random.split(key)
        trace = ts.ffbs_gibbs(model, kk, n_chains, n_sweeps, beta=beta)[:, -1]
        valid = mapf.is_valid(model, trace)
        if valid.sum() < min_solutions:
            continue
        plans = trace[valid]
        inst.append(dict(
            starts=starts, goals=goals, model=model,
            start_cells=model["start_cells"], goal_cells=model["goal_cells"],
            input_vec=encode_input(lay, model["start_cells"], model["goal_cells"]),
            out_data=encode_output(lay, plans),
        ))
    return inst


# --------------------------------------------------------------------------- #
# conditional training
# --------------------------------------------------------------------------- #
def make_trainer(lay, lr=0.02, sched=SamplingSchedule(5, 20, 5)):
    alln, edges, inp, out, lat = lay["alln"], lay["edges"], lay["inp"], lay["out"], lay["lat"]
    data_blocks = [Block(out)]
    cond_blocks = [Block(inp)]
    pos_sample = [Block(lat)]                 # input+output clamped -> sample latents
    neg_sample = [Block(out), Block(lat)]     # input clamped -> sample output+latents
    beta = jnp.array(1.0)

    def step(biases, weights, out_batch, input_vec, key):
        model = IsingEBM(alln, edges, biases, weights, beta)
        spec = IsingTrainingSpec(model, data_blocks, cond_blocks, pos_sample, neg_sample, sched, sched)
        kp, kn, kg = jax.random.split(key, 3)
        batch = out_batch.shape[0]
        init_pos = hinton_init(kp, model, pos_sample, (1, batch))
        init_neg = hinton_init(kn, model, neg_sample, (batch,))
        gw, gb, _, _ = estimate_kl_grad(kg, spec, alln, edges, [out_batch], [input_vec],
                                        init_pos, init_neg)
        with jax.numpy_dtype_promotion("standard"):
            biases = biases - lr * gb
            weights = weights - lr * gw
        return biases, weights

    return jax.jit(step)


# --------------------------------------------------------------------------- #
# conditional inference: clamp input, sample output, decode, score validity
# --------------------------------------------------------------------------- #
def sample_plans(lay, biases, weights, instance, key, n_chains=2000,
                 sched=SamplingSchedule(250, 1, 10)):
    """Clamp the instance's start/goal, sample the trained model, decode full plans.

    Returns (plans (n,K,T+1), valid_mask (n,)).
    """
    alln, edges, inp, out, lat = lay["alln"], lay["edges"], lay["inp"], lay["out"], lay["lat"]
    model = IsingEBM(alln, edges, biases, weights, jnp.array(1.0))
    prog = IsingSamplingProgram(model, [Block(out), Block(lat)], [Block(inp)])
    clamp = [jnp.asarray(instance["input_vec"])]
    k_init, k_run = jax.random.split(key)
    init = hinton_init(k_init, model, [Block(out), Block(lat)], (n_chains,))

    def one(init_c, k):
        return sample_states(k, prog, sched, init_c, clamp, [Block(out)])

    states = jax.jit(jax.vmap(one))(init, jax.random.split(k_run, n_chains))
    plans = decode_full(lay, np.asarray(states[0][:, -1, :]),
                        instance["start_cells"], instance["goal_cells"])
    valid = mapf.is_valid(instance["model"], plans)
    return plans, valid


def solve(lay, biases, weights, instance, key, n_chains=2000,
          sched=SamplingSchedule(250, 1, 10)):
    plans, valid = sample_plans(lay, biases, weights, instance, key, n_chains, sched)
    n_distinct = (np.unique(plans[valid].reshape(valid.sum(), -1), axis=0).shape[0]
                  if valid.any() else 0)
    return float(valid.mean()), int(n_distinct)


def eval_set(lay, biases, weights, instances, key, n_chains=1500):
    vfs, nds = [], []
    for ins in instances:
        key, k = jax.random.split(key)
        vf, nd = solve(lay, biases, weights, ins, k, n_chains)
        vfs.append(vf); nds.append(nd)
    return float(np.mean(vfs)), float(np.mean([v > 0 for v in vfs])), float(np.mean(nds))


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    rows, T = ["...", "...", "..."], 4
    q = 9
    lay = build_layout(K=2, T=T, q=q, n_lat=256)
    print(f"layout: K={lay['K']} T={T} q={q}  n_in={lay['n_in']} n_out={lay['n_out']} "
          f"n_lat={len(lay['lat'])}  edges={len(lay['edges'])}")

    key = jax.random.key(0)
    key, kd = jax.random.split(key)
    allinst = make_instances(rows, T, lay, kd, n_instances=70, min_solutions=40)
    n_tr = 55
    train, test = allinst[:n_tr], allinst[n_tr:]
    print(f"instances: {len(train)} train, {len(test)} held-out test "
          f"(each >= 40 FFBS solutions)")

    key, kb, kw = jax.random.split(key, 3)
    biases = jnp.zeros(len(lay["alln"]))
    weights = 0.01 * jax.random.normal(kw, (len(lay["edges"]),))
    trainer = make_trainer(lay, lr=0.02)

    rng = np.random.default_rng(0)
    n_steps, batch = 3000, 128
    for s in range(n_steps):
        ins = train[rng.integers(0, len(train))]
        od = ins["out_data"]
        bidx = rng.integers(0, len(od), batch)
        key, ks = jax.random.split(key)
        biases, weights = trainer(biases, weights, jnp.asarray(od[bidx]),
                                  jnp.asarray(ins["input_vec"]), ks)
        if (s + 1) % 750 == 0:
            key, ke1, ke2 = jax.random.split(key, 3)
            tr = eval_set(lay, biases, weights, train[:15], ke1)
            te = eval_set(lay, biases, weights, test, ke2)
            print(f"  step {s+1:4d}: TRAIN valid={tr[0]:.3f} solved={tr[1]:.3f} | "
                  f"HELD-OUT valid={te[0]:.3f} solved={te[1]:.3f} distinct={te[2]:.1f}")

    key, ke1, ke2 = jax.random.split(key, 3)
    tr = eval_set(lay, biases, weights, train[:15], ke1)
    te = eval_set(lay, biases, weights, test, ke2)
    print(f"\nFINAL  TRAIN: valid={tr[0]:.3f} solved={tr[1]:.3f}  |  "
          f"HELD-OUT: valid={te[0]:.3f} solved={te[1]:.3f} distinct={te[2]:.1f}")
    print("held-out solved>0 = fraction of unseen instances the learned sampler can crack")
