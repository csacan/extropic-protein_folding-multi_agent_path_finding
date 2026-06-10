"""Pillar 2 - learn a generative model of folds with THRML's Boltzmann-machine training.

Pillar 1 (sampling/optimization) produces folded conformations. Here we turn each fold
into a binary CONTACT MAP (bit per non-consecutive residue pair: 1 if lattice-adjacent)
and train a spin restricted Boltzmann machine (RBM) on those bit-vectors using THRML's
contrastive KL gradient (estimate_kl_grad) - exactly the spin-EBM training the platform
is built for. A trained RBM is a learned generative model of the folding ensemble: we then
free-sample it and check the generated contact statistics match the data.

This is the "learning" half of the Extropic use case, fed by data from the "sampling" half.

Run:  python ebm_train.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import optax

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, IsingTrainingSpec, estimate_kl_grad, hinton_init

import hp_model as hp
from benchmark import BENCHMARKS


# --------------------------------------------------------------------------- #
# Contact-map featurization
# --------------------------------------------------------------------------- #
def contact_pairs(N):
    return [(i, j) for i in range(N) for j in range(i + 2, N)]


def contact_vectors(meta, pos):
    """Binary contact map per fold: (..., n_pairs) with 1 where a non-consecutive pair touches."""
    r, c = hp._rc(meta, pos)
    bits = []
    for (i, j) in contact_pairs(meta["N"]):
        man = np.abs(r[..., i] - r[..., j]) + np.abs(c[..., i] - c[..., j])
        bits.append(man == 1)
    return np.stack(bits, axis=-1)


# --------------------------------------------------------------------------- #
# Data: fold the sequence and collect valid contact maps
# --------------------------------------------------------------------------- #
def generate_folds(seq, L, beta_data, lam, n_chains, key, n_samples=20, warmup=4000, sps=4):
    run, meta = hp.make_sampler(seq, L, SamplingSchedule(warmup, n_samples, sps), eps=1.0)
    nfree, center = meta["N"] - 1, meta["center"]
    path = hp.saw_init_path(meta)
    init = [jnp.full((n_chains, 1), int(path[i + 1]), dtype=jnp.uint8) for i in range(nfree)]
    states = run(jnp.float32(beta_data), jnp.float32(lam), init, jax.random.split(key, n_chains))
    free_all = jnp.concatenate(states, axis=-1)                       # (n_chains, n_samples, nfree)
    col0 = jnp.full((*free_all.shape[:-1], 1), center, dtype=free_all.dtype)
    pos = np.asarray(jnp.concatenate([col0, free_all], axis=-1))      # (n_chains, n_samples, N)
    valid = hp.is_valid(meta, pos)
    cv = contact_vectors(meta, pos)                                   # (n_chains, n_samples, n_pairs)
    data = cv[valid]                                                  # (M, n_pairs) bool
    return data, meta


# --------------------------------------------------------------------------- #
# RBM (spin Boltzmann machine) trained with THRML's contrastive KL gradient
# --------------------------------------------------------------------------- #
def build_rbm(n_vis, n_lat, key):
    vis = [SpinNode() for _ in range(n_vis)]
    lat = [SpinNode() for _ in range(n_lat)]
    all_nodes = vis + lat
    G = nx.complete_bipartite_graph(n_vis, n_lat)
    G = nx.relabel_nodes(G, {i: vis[i] for i in range(n_vis)} |
                         {n_vis + j: lat[j] for j in range(n_lat)})
    edges = list(G.edges())
    kb, kw = jax.random.split(key)
    biases = 0.01 * jax.random.normal(kb, (len(all_nodes),))
    weights = 0.01 * jax.random.normal(kw, (len(edges),))
    return dict(vis=vis, lat=lat, all_nodes=all_nodes, edges=edges), biases, weights


def train_rbm(rbm, biases, weights, data, key, beta=1.0, steps=800, batch=128, lr=5e-3):
    vis, lat, all_nodes, edges = rbm["vis"], rbm["lat"], rbm["all_nodes"], rbm["edges"]
    free_blocks = [Block(vis), Block(lat)]
    clamped_blocks = [Block(lat)]                  # positive phase: clamp visible(data), sample latent
    schedule = SamplingSchedule(5, 20, 3)
    betaj = jnp.array(beta)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init((biases, weights))

    def step(biases, weights, opt_state, data_batch, key):
        model = IsingEBM(all_nodes, edges, biases, weights, betaj)
        spec = IsingTrainingSpec(model, [Block(vis)], [], clamped_blocks, free_blocks, schedule, schedule)
        k_pos, k_neg, k_grad = jax.random.split(key, 3)
        b = data_batch.shape[0]
        init_pos = hinton_init(k_pos, model, clamped_blocks, (1, b))
        init_neg = hinton_init(k_neg, model, free_blocks, (b,))
        grad_w, grad_b, _, _ = estimate_kl_grad(
            k_grad, spec, all_nodes, edges, [data_batch], [], init_pos, init_neg)
        with jax.numpy_dtype_promotion("standard"):
            updates, opt_state = optimizer.update((grad_b, grad_w), opt_state, (biases, weights))
            biases, weights = optax.apply_updates((biases, weights), updates)
        return biases, weights, opt_state

    step = jax.jit(step)
    M = data.shape[0]
    data = jnp.asarray(data)
    t0 = time.time()
    for s in range(steps):
        key, ks, kb = jax.random.split(key, 3)
        idx = jax.random.randint(kb, (batch,), 0, M)
        biases, weights, opt_state = step(biases, weights, opt_state, data[idx], ks)
        if (s + 1) % 200 == 0:
            print(f"  step {s+1}/{steps}  [{time.time()-t0:.0f}s]")
    return biases, weights


def sample_rbm(rbm, biases, weights, key, n=4000, beta=1.0):
    """Free-sample the trained RBM and return generated visible (contact) bit-vectors."""
    vis, lat, all_nodes, edges = rbm["vis"], rbm["lat"], rbm["all_nodes"], rbm["edges"]
    model = IsingEBM(all_nodes, edges, biases, weights, jnp.array(beta))
    free_blocks = [Block(vis), Block(lat)]
    prog = IsingSamplingProgram(model, free_blocks, [])
    schedule = SamplingSchedule(n_warmup=200, n_samples=1, steps_per_sample=4)
    ki, kr = jax.random.split(key)
    init = hinton_init(ki, model, free_blocks, (n,))
    run = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, schedule, i, [], [Block(vis)])))
    states = run(init, jax.random.split(kr, n))
    return np.asarray(states[0][:, -1, :])   # (n, n_vis) bool, True=+1


def main():
    seq, L = "HHPPHPHPHPHH", 6   # the validated cooperative folder (N=12)
    beta_data, lam = 2.0, 8.0
    print(f"EBM training pillar: seq={seq} (N={len(seq)}) L={L}\n")

    print("generating folds (pillar-1 sampler) ...")
    data, meta = generate_folds(seq, L, beta_data, lam, n_chains=8000, key=jax.random.key(0))
    n_vis = data.shape[1]
    print(f"  collected {data.shape[0]} valid folds, contact-map dim n_vis={n_vis}, "
          f"mean contacts/fold={data.sum(1).mean():.2f}\n")

    print("training spin RBM via contrastive KL (THRML) ...")
    rbm, biases, weights = build_rbm(n_vis, n_lat=32, key=jax.random.key(1))
    biases, weights = train_rbm(rbm, biases, weights, data, jax.random.key(2), steps=800, batch=128)

    print("\nsampling trained RBM ...")
    gen = sample_rbm(rbm, biases, weights, jax.random.key(3), n=8000)

    data_freq = data.mean(0)
    gen_freq = gen.mean(0)
    corr = float(np.corrcoef(data_freq, gen_freq)[0, 1])
    mae = float(np.abs(data_freq - gen_freq).mean())
    print(f"\nper-pair contact frequency: corr(data, RBM)={corr:.3f}  MAE={mae:.3f}")
    print(f"mean contacts/sample: data={data.sum(1).mean():.2f}  RBM={gen.sum(1).mean():.2f}")
    plot(meta, data, gen, data_freq, gen_freq, corr, seq)


def plot(meta, data, gen, data_freq, gen_freq, corr, seq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = meta["N"]
    pairs = contact_pairs(N)

    def to_map(freq):
        m = np.zeros((N, N))
        for (i, j), f in zip(pairs, freq):
            m[i, j] = m[j, i] = f
        return m

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    im0 = axs[0].imshow(to_map(data_freq), cmap="viridis", vmin=0, vmax=max(data_freq.max(), 1e-6))
    axs[0].set_title("Data contact-frequency map\n(folds from THRML sampler)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(to_map(gen_freq), cmap="viridis", vmin=0, vmax=max(data_freq.max(), 1e-6))
    axs[1].set_title("RBM-generated contact-frequency map\n(learned model)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)
    axs[2].scatter(data_freq, gen_freq, s=18, alpha=0.7)
    lim = max(data_freq.max(), gen_freq.max()) * 1.1 + 1e-6
    axs[2].plot([0, lim], [0, lim], "k--", lw=1)
    axs[2].set_xlabel("data contact frequency")
    axs[2].set_ylabel("RBM contact frequency")
    axs[2].set_title(f"per-pair frequency\ncorr = {corr:.3f}")
    fig.suptitle(f"Learned generative model of folds (spin RBM) — seq={seq}", y=1.02)
    fig.tight_layout()
    out = "/home/can/extropic/project/protein/ebm_rbm.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}\n")
    main()
