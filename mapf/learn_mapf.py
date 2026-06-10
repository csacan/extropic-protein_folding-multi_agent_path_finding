"""B0 prototype: self-distillation of a MAPF solver into a Boltzmann machine (RBM).

The hand-derived EBM *encodes* MAPF; here we instead *learn* it. We:
  1. generate valid plans for a fixed instance with FFBS (the "teacher" / data),
  2. one-hot encode each (agent,time) position as q binary spins (the hardware-native
     substrate THRML can train),
  3. train an RBM (visible position spins <-> latent coordination units) by contrastive
     KL (estimate_kl_grad) to match that solution distribution,
  4. sample the trained RBM and decode -> measure how often it emits VALID, collision-
     free plans, vs the untrained model.

If a random RBM emits ~0% valid plans and the trained one emits many (and diverse),
the chip has learned to *sample solutions* -- the use case classical search can't claim.
Uses THRML's Ising training stack (IsingTrainingSpec / estimate_kl_grad), plain SGD.
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
# one-hot encoding  plan (K,T+1) <-> visible spins (K*(T+1)*q)
# --------------------------------------------------------------------------- #
def encode(plans, K, T, q):
    N = plans.shape[0]
    data = np.zeros((N, K * (T + 1) * q), dtype=bool)
    for k in range(K):
        for t in range(T + 1):
            base = (k * (T + 1) + t) * q
            data[np.arange(N), base + plans[:, k, t]] = True
    return data


def decode(spins, K, T, q):
    """Bool visible spins (N, K*(T+1)*q) -> plans (N,K,T+1) by argmax per (k,t) block."""
    N = spins.shape[0]
    return spins.reshape(N, K, T + 1, q).astype(np.int8).argmax(-1).astype(np.int64)


# --------------------------------------------------------------------------- #
# data: valid plans from the FFBS teacher
# --------------------------------------------------------------------------- #
def teacher_data(model, key, n_chains=12000, n_sweeps=40, beta=3.0):
    trace = ts.ffbs_gibbs(model, key, n_chains, n_sweeps, beta=beta)
    plans = trace[:, -1]
    valid = mapf.is_valid(model, plans)
    plans = plans[valid]
    uniq = np.unique(plans.reshape(len(plans), -1), axis=0)
    return plans, uniq.shape[0]


# --------------------------------------------------------------------------- #
# RBM
# --------------------------------------------------------------------------- #
def build_rbm(n_vis, n_lat, key):
    vis = [SpinNode() for _ in range(n_vis)]
    lat = [SpinNode() for _ in range(n_lat)]
    alln = vis + lat
    edges = [(vis[i], lat[j]) for i in range(n_vis) for j in range(n_lat)]
    kb, kw = jax.random.split(key)
    biases = jnp.zeros(len(alln))
    weights = 0.01 * jax.random.normal(kw, (len(edges),))
    return dict(vis=vis, lat=lat, alln=alln, edges=edges, biases=biases, weights=weights)


def make_trainer(rbm, lr=0.05, sched=SamplingSchedule(5, 20, 5)):
    alln, edges, vis, lat = rbm["alln"], rbm["edges"], rbm["vis"], rbm["lat"]
    free_blocks = [Block(vis), Block(lat)]
    pos_blocks = [Block(lat)]  # positive phase: clamp visible to data, sample latents
    beta = jnp.array(1.0)

    def step(biases, weights, data_batch, key):
        model = IsingEBM(alln, edges, biases, weights, beta)
        spec = IsingTrainingSpec(model, [Block(vis)], [], pos_blocks, free_blocks, sched, sched)
        kp, kn, kg = jax.random.split(key, 3)
        batch = data_batch.shape[0]
        init_pos = hinton_init(kp, model, pos_blocks, (1, batch))
        init_neg = hinton_init(kn, model, free_blocks, (batch,))
        gw, gb, _, _ = estimate_kl_grad(kg, spec, alln, edges, [data_batch], [], init_pos, init_neg)
        with jax.numpy_dtype_promotion("standard"):
            biases = biases - lr * gb
            weights = weights - lr * gw
        return biases, weights

    return jax.jit(step)


def sample_rbm(rbm, biases, weights, key, n_chains, sched=SamplingSchedule(200, 1, 10)):
    alln, edges, vis, lat = rbm["alln"], rbm["edges"], rbm["vis"], rbm["lat"]
    model = IsingEBM(alln, edges, biases, weights, jnp.array(1.0))
    free_blocks = [Block(vis), Block(lat)]
    prog = IsingSamplingProgram(model, free_blocks, [])
    k_init, k_run = jax.random.split(key)
    init = hinton_init(k_init, model, free_blocks, (n_chains,))

    def one(init_c, k):
        return sample_states(k, prog, sched, init_c, [], [Block(vis)])

    states = jax.jit(jax.vmap(one))(init, jax.random.split(k_run, n_chains))
    return np.asarray(states[0][:, -1, :])  # (n_chains, n_vis) bool


# --------------------------------------------------------------------------- #
# eval: how often does the model emit valid plans?
# --------------------------------------------------------------------------- #
def eval_validity(model_light, rbm, biases, weights, key, K, T, q, n=4000):
    spins = sample_rbm(rbm, biases, weights, key, n)
    plans = decode(spins, K, T, q)
    valid = mapf.is_valid(model_light, plans)
    viol = mapf.n_vertex_conflicts(model_light, plans) + mapf.n_illegal_moves(model_light, plans)
    n_distinct_valid = 0
    if valid.any():
        n_distinct_valid = np.unique(plans[valid].reshape(valid.sum(), -1), axis=0).shape[0]
    # also: how often is start/goal reproduced correctly
    ok_ends = np.ones(len(plans), bool)
    for k in range(K):
        ok_ends &= (plans[:, k, 0] == model_light["start_cells"][k])
        ok_ends &= (plans[:, k, T] == model_light["goal_cells"][k])
    return dict(valid_frac=float(valid.mean()), mean_viol=float(viol.mean()),
                n_distinct_valid=int(n_distinct_valid), endpoint_frac=float(ok_ends.mean()))


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    rows, starts, goals, T = ["...", "...", "..."], [(0, 0), (2, 0)], [(2, 2), (0, 2)], 4
    model = pt.light_model(rows, starts, goals, T, lam_swap=1.0)
    K, q = model["K"], model["q"]
    n_vis = K * (T + 1) * q
    print(f"instance: K={K} T={T} q={q}  ->  n_vis={n_vis} one-hot spins")

    plans, n_uniq = teacher_data(model, jax.random.key(0))
    print(f"teacher: {len(plans)} valid plans ({n_uniq} distinct) from FFBS")
    data = encode(plans, K, T, q)

    key = jax.random.key(1)
    rbm = build_rbm(n_vis, n_lat=128, key=key)
    biases, weights = rbm["biases"], rbm["weights"]

    before = eval_validity(model, rbm, biases, weights, jax.random.key(2), K, T, q)
    print(f"untrained RBM: valid={before['valid_frac']:.3f}  mean_viol={before['mean_viol']:.2f}  "
          f"endpoints={before['endpoint_frac']:.3f}")

    trainer = make_trainer(rbm, lr=0.05)
    rng = np.random.default_rng(0)
    n_steps, batch = 600, 256
    for s in range(n_steps):
        idx = rng.integers(0, len(data), batch)
        key, ks = jax.random.split(key)
        biases, weights = trainer(biases, weights, jnp.asarray(data[idx]), ks)
        if (s + 1) % 150 == 0:
            ev = eval_validity(model, rbm, biases, weights, jax.random.key(3), K, T, q)
            print(f"  step {s+1:4d}: valid={ev['valid_frac']:.3f}  mean_viol={ev['mean_viol']:.2f}  "
                  f"distinct_valid={ev['n_distinct_valid']}  endpoints={ev['endpoint_frac']:.3f}")

    after = eval_validity(model, rbm, biases, weights, jax.random.key(4), K, T, q)
    print(f"\ntrained RBM: valid={after['valid_frac']:.3f}  mean_viol={after['mean_viol']:.2f}  "
          f"distinct_valid={after['n_distinct_valid']}  endpoints={after['endpoint_frac']:.3f}")
    print(f"validity {before['valid_frac']:.3f} -> {after['valid_frac']:.3f}  "
          f"(teacher had {n_uniq} distinct solutions)")
