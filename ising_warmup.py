"""2D Ising warmup on THRML — validates the toolchain against exact physics.

We sample the ferromagnetic 2D Ising model on a periodic L x L lattice
(J = +1, no field) for a range of inverse temperatures beta, using THRML's
block Gibbs sampler on the GPU. THRML's energy is

    E(s) = -beta * ( sum_i b_i s_i + sum_<ij> J_ij s_i s_j ),

so the effective coupling in the Boltzmann weight is K = beta * J. With J = 1
the exact Onsager critical point of the infinite 2D square lattice is

    beta_c = ln(1 + sqrt(2)) / 2 = 0.4406868...

We confirm this two ways:
  * the magnetization |m| onsets and the susceptibility chi peaks near beta_c;
  * the Binder cumulant U = 1 - <m^4> / (3 <m^2>^2) curves for different L all
    cross at a single point that is beta_c (the rigorous, size-independent test).

Run:  python ising_warmup.py            # full run (a few minutes on a 4090)
      python ising_warmup.py --quick    # fast smoke test
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

BETA_C_EXACT = np.log(1 + np.sqrt(2)) / 2  # 0.4406868...


def build_lattice(L: int):
    """Periodic L x L grid of SpinNodes with an exact 2-coloring (checkerboard)."""
    if L % 2 != 0:
        raise ValueError("L must be even so the periodic grid stays bipartite.")
    G = nx.grid_graph(dim=(L, L), periodic=True)
    nx.relabel_nodes(G, {c: SpinNode() for c in G.nodes}, copy=False)
    nodes = list(G.nodes)
    edges = list(G.edges)
    bicol = nx.bipartite.color(G)
    free_blocks = [Block([n for n, c in bicol.items() if c == k]) for k in (0, 1)]
    return nodes, edges, free_blocks


def make_runner(nodes, edges, free_blocks, schedule, n_chains):
    """Return a jitted function beta -> magnetization samples, shape (n_chains*n_samples,)."""
    all_block = Block(nodes)
    N = len(nodes)

    def run(beta, key):
        biases = jnp.zeros(N)
        weights = jnp.ones(len(edges))  # ferromagnetic J = +1
        model = IsingEBM(nodes, edges, biases, weights, beta)
        program = IsingSamplingProgram(model, free_blocks, [])

        k_init, k_run = jax.random.split(key)
        init_free = hinton_init(k_init, model, free_blocks, (n_chains,))
        keys = jax.random.split(k_run, n_chains)

        states = jax.vmap(
            lambda init, k: sample_states(k, program, schedule, init, [], [all_block])
        )(init_free, keys)
        spins = jnp.where(states[0], 1.0, -1.0)  # (n_chains, n_samples, N), True->+1
        m = spins.mean(axis=-1)  # per-chain, per-sample magnetization
        return m.reshape(-1)

    return jax.jit(run)


def observables(m: np.ndarray, beta: float, N: int):
    """Magnetization, susceptibility, and Binder cumulant from magnetization samples."""
    absm = np.abs(m)
    m2 = np.mean(m**2)
    m4 = np.mean(m**4)
    mag = float(np.mean(absm))
    chi = float(beta * N * (m2 - np.mean(absm) ** 2))
    binder = float(1.0 - m4 / (3.0 * m2**2))
    return mag, chi, binder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    args = ap.parse_args()

    if args.quick:
        Ls = [16, 24]
        betas = np.linspace(0.34, 0.55, 12)
        schedule = SamplingSchedule(n_warmup=800, n_samples=40, steps_per_sample=4)
        n_chains = 128
    else:
        Ls = [16, 24, 32]
        betas = np.linspace(0.32, 0.58, 27)
        schedule = SamplingSchedule(n_warmup=3000, n_samples=120, steps_per_sample=6)
        n_chains = 256

    print(f"JAX devices: {jax.devices()}")
    print(f"Exact Onsager beta_c = {BETA_C_EXACT:.6f}\n")

    key = jax.random.key(0)
    results = {}  # L -> dict of arrays

    for L in Ls:
        nodes, edges, free_blocks = build_lattice(L)
        N = len(nodes)
        run = make_runner(nodes, edges, free_blocks, schedule, n_chains)

        mags, chis, binders = [], [], []
        t0 = time.time()
        for beta in betas:
            key, sub = jax.random.split(key)
            m = np.asarray(jax.block_until_ready(run(jnp.array(float(beta)), sub)))
            mag, chi, binder = observables(m, float(beta), N)
            mags.append(mag)
            chis.append(chi)
            binders.append(binder)
        dt = time.time() - t0

        mags, chis, binders = map(np.array, (mags, chis, binders))
        results[L] = dict(mags=mags, chis=chis, binders=binders, N=N)

        beta_peak = betas[int(np.argmax(chis))]
        print(f"L={L:3d} (N={N:4d})  chi-peak beta={beta_peak:.4f}  "
              f"[{dt:.1f}s, {n_chains} chains x {schedule.n_samples} samples/beta]")

    # Binder-cumulant crossing: the size-independent estimate of beta_c.
    beta_cross = estimate_binder_crossing(betas, results, Ls)
    print(f"\nBinder-crossing estimate of beta_c = {beta_cross:.4f}")
    print(f"Exact Onsager           beta_c = {BETA_C_EXACT:.4f}")
    print(f"relative error                 = {abs(beta_cross - BETA_C_EXACT)/BETA_C_EXACT*100:.2f}%")

    plot(betas, results, Ls, beta_cross)


def estimate_binder_crossing(betas, results, Ls):
    """Find where the largest- and smallest-L Binder curves cross (finest pair)."""
    L_lo, L_hi = min(Ls), max(Ls)
    diff = results[L_hi]["binders"] - results[L_lo]["binders"]
    # locate sign change of the difference, linearly interpolate the zero
    sign = np.sign(diff)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return float(betas[np.argmin(np.abs(diff))])
    i = idx[len(idx) // 2]
    b0, b1 = betas[i], betas[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    return float(b0 - d0 * (b1 - b0) / (d1 - d0))


def plot(betas, results, Ls, beta_cross):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for L in Ls:
        r = results[L]
        axs[0].plot(betas, r["mags"], "o-", ms=3, label=f"L={L}")
        axs[1].plot(betas, r["chis"], "o-", ms=3, label=f"L={L}")
        axs[2].plot(betas, r["binders"], "o-", ms=3, label=f"L={L}")

    for ax, title, ylab in zip(
        axs,
        ["Magnetization", "Susceptibility", "Binder cumulant"],
        [r"$\langle|m|\rangle$", r"$\chi$", r"$U = 1 - \langle m^4\rangle/3\langle m^2\rangle^2$"],
    ):
        ax.axvline(BETA_C_EXACT, color="k", ls="--", lw=1, label=r"$\beta_c$ exact")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axs[2].axvline(beta_cross, color="r", ls=":", lw=1.2, label="crossing")
    axs[2].legend(fontsize=8)

    fig.suptitle("2D Ising on THRML vs. exact Onsager solution", y=1.02)
    fig.tight_layout()
    out = "/home/can/extropic/project/ising_warmup.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
