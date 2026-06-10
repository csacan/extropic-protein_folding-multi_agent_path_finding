"""Parallel tempering (replica exchange) for HP folding on THRML.

A fixed temperature ladder beta_1 < ... < beta_R is run simultaneously; each "walker"
owns R replicas (one per temperature). Every round we (a) run a short block-Gibbs burst
at each temperature and (b) attempt swaps between adjacent temperatures.

Because the energy is E(x;beta) = lam*violations(x) - beta*eps*contacts(x) with lam held
constant across the ladder, the violation terms CANCEL in the swap acceptance, leaving

    A = min(1, exp( eps*(beta_{i+1}-beta_i)*(c_i - c_{i+1}) ))

so swaps depend only on contact counts and the temperature gap. Hot replicas explore
(rearrange via bond-break tunneling); cold replicas refine; swaps let a cold chain escape
a trap by importing a better configuration discovered hotter. This is the natural match to
hardware that runs many parallel samplers cheaply.

Gibbs bursts run jitted/vmapped on GPU; swaps are done on host between bursts.

Run:  python pt.py            # PT vs annealing comparison on a few benchmarks
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from thrml import SamplingSchedule, sample_states

import hp_model as hp
import solve
from benchmark import BENCHMARKS


def pt_solve(seq, L, betas, rounds, steps_per_round, n_walkers, key, lam, eps=1.0):
    """Run n_walkers parallel-tempering systems; return (best valid contacts per walker, sweeps_per_run)."""
    betas = np.sort(np.asarray(betas, dtype=float))
    R, N, nfree = len(betas), len(seq), len(seq) - 1
    sch = SamplingSchedule(n_warmup=steps_per_round, n_samples=1, steps_per_sample=1)
    run, meta = hp.make_sampler(seq, L, sch, eps=eps)  # one compile, reused for all temps/rounds
    refm = meta
    betas_j = [jnp.float32(b) for b in betas]
    lam_j = jnp.float32(lam)

    path = hp.saw_init_path(refm)  # valid SAW, shape (N,)
    pos = np.broadcast_to(path[None, None, :], (n_walkers, R, N)).astype(np.uint8).copy()
    best = np.full(n_walkers, -1)
    rng = np.random.default_rng(0)

    for rnd in range(rounds):
        for r in range(R):  # Gibbs burst at each temperature
            init_blocks = [jnp.asarray(pos[:, r, i + 1:i + 2]) for i in range(nfree)]
            key, kk = jax.random.split(key)
            out = run(betas_j[r], lam_j, init_blocks, jax.random.split(kk, n_walkers))
            pos[:, r, 1:] = np.concatenate([np.asarray(o[:, -1, :]) for o in out], axis=1)

        c = hp.n_contacts(refm, pos)          # (W, R)
        v = hp.is_valid(refm, pos)            # (W, R)
        best = np.maximum(best, np.where(v, c, -1).max(axis=1))

        for r in range(rnd % 2, R - 1, 2):    # adjacent-pair swaps, alternating parity
            delta = eps * (betas[r + 1] - betas[r]) * (c[:, r] - c[:, r + 1])
            acc = np.log(rng.random(n_walkers) + 1e-12) < np.minimum(0.0, delta)
            pos[acc, r, :], pos[acc, r + 1, :] = pos[acc, r + 1, :].copy(), pos[acc, r, :].copy()
            c[acc, r], c[acc, r + 1] = c[acc, r + 1].copy(), c[acc, r].copy()

    sweeps_per_run = rounds * steps_per_round * R
    return best, sweeps_per_run


def compare():
    names = ["2d20", "2d24", "2d25", "2d36"]
    L_betas = np.linspace(0.1, 4.0, 12)        # PT ladder
    a_betas = np.linspace(0.1, 4.0, 28)        # annealing ramp
    rounds, spr_steps, W = 160, 5, 2000        # PT: sweeps/run = 160*5*12 = 9600
    lam = 8.0
    print(f"PT ladder R={len(L_betas)}  rounds={rounds}  walkers={W}  sweeps/run={rounds*spr_steps*len(L_betas)}")
    print(f"{'name':>6} {'N':>3} {'E_opt':>5} | {'p_anneal':>9} {'TTS_anneal':>11} | {'p_PT':>8} {'TTS_PT':>11} | {'speedup':>7}")
    rows = []
    key = jax.random.key(7)
    for name in names:
        seq, Eopt = BENCHMARKS[name]
        Lb = solve.box_for(len(seq))

        key, ka = jax.random.split(key)
        ann_best, ann_spr = solve.solve(seq, Lb, a_betas, warmup=300, n_chains=W, key=ka, lam=None)
        pa = float((ann_best == Eopt).mean())
        tts_a = solve.tts99(pa, ann_spr)

        key, kp = jax.random.split(key)
        t = time.time()
        pt_best, pt_spr = pt_solve(seq, Lb, L_betas, rounds, spr_steps, W, kp, lam=lam)
        pp = float((pt_best == Eopt).mean())
        tts_p = solve.tts99(pp, pt_spr)
        spd = (tts_a / tts_p) if (tts_p not in (0, math.inf) and tts_a != math.inf) else float("nan")
        rows.append((len(seq), pa, tts_a, pp, tts_p, spd))
        print(f"{name:>6} {len(seq):>3} {Eopt:>5} | {pa:>9.4f} {tts_a:>11.0f} | {pp:>8.4f} {tts_p:>11.0f} | {spd:>7.1f}x  [{time.time()-t:.0f}s]")
    plot(names, rows)


def plot(names, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = [r[0] for r in rows]
    pa = [r[1] for r in rows]
    pp = [r[3] for r in rows]
    tts_a = [r[2] for r in rows]
    tts_p = [r[4] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(N, pa, "o-", label="annealing", color="gray")
    axs[0].plot(N, pp, "s-", label="parallel tempering", color="crimson")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("chain length N")
    axs[0].set_ylabel("p(reach optimum)")
    axs[0].set_title("Success probability: PT vs annealing")
    axs[0].legend()
    axs[1].plot(N, tts_a, "o-", label="annealing", color="gray")
    axs[1].plot(N, tts_p, "s-", label="parallel tempering", color="crimson")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("chain length N")
    axs[1].set_ylabel("TTS @ 99% (sweeps)")
    axs[1].set_title("Time-to-solution: PT vs annealing")
    axs[1].legend()
    fig.tight_layout()
    out = "/home/can/extropic/project/protein/pt_vs_anneal.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}\n")
    compare()
