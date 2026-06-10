"""Phase B - HP folding as optimization on THRML: success probability and time-to-solution.

A "replica" is one annealing chain run through the cooling ramp. Success = the replica
visits a valid fold whose contact count equals the known optimum (benchmark.py). From the
per-run success probability p we report the standard 99%-confidence time-to-solution

    TTS = sweeps_per_run * ln(1-0.99)/ln(1-p)

i.e. expected total Gibbs sweeps to find the optimum with 99% probability allowing restarts.
Reported in sweeps (hardware-agnostic); Phase C converts to wall-clock via flips/ns.

Two experiments:
  budget : fix one sequence, vary the per-stage warmup -> p(budget), TTS(budget) [TTS has a minimum]
  scaling: fix the budget, sweep the benchmark sequences -> p(N), TTS(N), best/optimum

Run:  python solve.py --budget
      python solve.py --scaling
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import numpy as np

from thrml import SamplingSchedule

import hp_model as hp
import thermo
from benchmark import BENCHMARKS


def box_for(N, cap=16):
    """Box side just large enough to hold a compact fold, capped by the uint8 limit (q<=256)."""
    return min(cap, 2 * math.ceil(math.sqrt(N)) + 2)


def solve(seq, L, betas, warmup, n_chains, key, n_samples=5, sps=2, lam=None):
    """Run n_chains annealing replicas; return (best valid contacts per replica, sweeps_per_run)."""
    sch = SamplingSchedule(n_warmup=warmup, n_samples=n_samples, steps_per_sample=sps)
    sweep, model = thermo.temperature_sweep(seq, L, betas, sch, n_chains, key, eps=1.0, lam=lam)
    best = np.full(n_chains, -1)
    for b in betas:
        pos = sweep[float(b)]
        v = hp.is_valid(model, pos)
        ct = hp.n_contacts(model, pos)
        best = np.maximum(best, np.where(v, ct, -1).max(axis=1))
    sweeps_per_run = len(betas) * (warmup + n_samples * sps)
    return best, sweeps_per_run


def tts99(p, sweeps_per_run):
    """Sweeps to reach the optimum with 99% confidence allowing independent restarts."""
    if p <= 0:
        return math.inf
    if p >= 1:
        return float(sweeps_per_run)
    return sweeps_per_run * math.log(0.01) / math.log(1.0 - p)


def run_budget():
    name = "2d20"
    seq, Eopt = BENCHMARKS[name]
    L = box_for(len(seq))
    betas = np.linspace(0.1, 4.0, 30)
    n_chains = 3000
    warmups = [100, 300, 800, 2000, 5000]
    print(f"budget experiment: seq={name} N={len(seq)} L={L} E_opt={Eopt}\n")
    print(f"{'warmup':>7} {'sweeps/run':>11} {'p_success':>10} {'best':>5} {'TTS(sweeps)':>13}")
    rows = []
    key = jax.random.key(0)
    for w in warmups:
        key, k = jax.random.split(key)
        t = time.time()
        best, spr = solve(seq, L, betas, w, n_chains, k)
        p = float((best == Eopt).mean())
        tts = tts99(p, spr)
        rows.append((spr, p, tts, int(best.max())))
        print(f"{w:>7} {spr:>11} {p:>10.4f} {int(best.max()):>5} {tts:>13.0f}  [{time.time()-t:.0f}s]")
    plot_budget(rows, name, Eopt)


def run_scaling():
    names = ["2d20", "2d24", "2d25", "2d36", "2d48", "2d50"]
    betas = np.linspace(0.1, 4.0, 20)
    warmup, n_chains = 400, 2000  # short budget; large-N p~0 so we only need to show the collapse
    print(f"scaling experiment: warmup={warmup} chains={n_chains} betas={len(betas)}\n")
    print(f"{'name':>6} {'N':>3} {'L':>3} {'E_opt':>5} {'best':>5} {'p_success':>10} {'TTS(sweeps)':>13}")
    rows = []
    key = jax.random.key(1)
    for name in names:
        seq, Eopt = BENCHMARKS[name]
        N = len(seq)
        L = box_for(N)
        key, k = jax.random.split(key)
        t = time.time()
        best, spr = solve(seq, L, betas, warmup, n_chains, k)
        p = float((best == Eopt).mean())
        tts = tts99(p, spr)
        rows.append((N, p, tts, int(best.max()), Eopt))
        print(f"{name:>6} {N:>3} {L:>3} {Eopt:>5} {int(best.max()):>5} {p:>10.5f} {tts:>13.0f}  [{time.time()-t:.0f}s]")
    plot_scaling(rows)


def plot_budget(rows, name, Eopt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spr = [r[0] for r in rows]
    p = [r[1] for r in rows]
    tts = [r[2] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(spr, p, "o-", color="navy")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("sweeps per run (annealing budget)")
    axs[0].set_ylabel("p(reach optimum)")
    axs[0].set_title(f"Success probability vs budget — {name}")
    axs[1].plot(spr, tts, "o-", color="crimson")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("sweeps per run")
    axs[1].set_ylabel("TTS @ 99% (sweeps)")
    axs[1].set_title("Time-to-solution (look for the minimum)")
    fig.tight_layout()
    out = "/home/can/extropic/project/protein/phaseB_budget.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


def plot_scaling(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    tts = np.array([r[2] for r in rows], dtype=float)
    ratio = [r[3] / r[4] for r in rows]
    finite = np.isfinite(tts) & (p > 0)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    axs[0].plot(N[p > 0], p[p > 0], "o-", color="navy")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("chain length N")
    axs[0].set_ylabel("p(reach optimum)")
    axs[0].set_title("Success probability vs N (0 plotted as gap)")
    axs[1].plot(N[finite], tts[finite], "o-", color="crimson")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("chain length N")
    axs[1].set_ylabel("TTS @ 99% (sweeps)")
    axs[1].set_title("Time-to-solution vs N")
    axs[2].plot(N, ratio, "o-", color="seagreen")
    axs[2].axhline(1.0, color="gray", ls=":")
    axs[2].set_ylim(0, 1.1)
    axs[2].set_xlabel("chain length N")
    axs[2].set_ylabel("best found / optimum")
    axs[2].set_title("Solution quality vs N")
    fig.tight_layout()
    out = "/home/can/extropic/project/protein/phaseB_scaling.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    args = ap.parse_args()
    print(f"JAX devices: {jax.devices()}\n")
    if args.budget:
        run_budget()
    if args.scaling:
        run_scaling()
    if not (args.budget or args.scaling):
        run_budget()
        run_scaling()
