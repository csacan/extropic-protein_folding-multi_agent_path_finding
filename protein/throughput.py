"""Phase C - hardware throughput and the Extropic projection.

Everything so far was measured in SWEEPS (hardware-agnostic). Here we measure the GPU's
raw throughput and convert the optimization results to wall-clock time and energy:

  C1  flips/ns vs degrees of freedom (parallel chains) -> the throughput saturation curve.
  C2  wall-clock time-to-solution  = TTS_flips / throughput, with TTS_flips = TTS_sweeps * N.
      (preserves the PT-vs-annealing advantage measured in Phase B / pt.py)
  C3  energy-to-solution = wall-clock TTS * GPU power, then projected to Extropic hardware
      using its reported energy-efficiency advantage.

A "flip" is one single-variable Gibbs update; flips = n_chains * N * n_sweeps (matching the
convention in THRML example 02). The restart TTS already accounts for the ~1/p attempts to
find the optimum; dividing total work by machine throughput gives wall-clock at saturation.

Run:  python throughput.py
"""

from __future__ import annotations

import math
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from thrml import SamplingSchedule

import hp_model as hp
from benchmark import BENCHMARKS

# Time-to-solution in SWEEPS measured in pt.py (per-replica, 99% confidence). inf = never solved.
TTS_SWEEPS = {
    # name: (N, TTS_anneal, TTS_PT)
    "2d20": (20, 39_952_887, 161_092),
    "2d24": (24, 79_925_766, 120_080),
    "2d25": (25, 79_925_766, 510_222),
    "2d36": (36, math.inf, 8_819_803),
}

GPU_POWER_W = 450.0          # RTX 4090 TDP (overwritten by a live nvidia-smi reading if available)
# Extropic reports large energy-efficiency gains vs GPUs for these block-Gibbs workloads
# (arXiv:2510.23972). Treated here as a labeled projection factor, not a measurement.
EXTROPIC_ENERGY_FACTOR = 1e4


def gpu_power_draw():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], timeout=5)
        return float(out.decode().split("\n")[0])
    except Exception:
        return None


def measure_throughput(seq, L, n_chains_list, sweeps=100):
    """flips/ns for a range of parallel-chain counts. Returns list of (dof, flips_per_ns, t)."""
    run, meta = hp.make_sampler(seq, L, SamplingSchedule(sweeps, 1, 1))
    N, nfree, center = meta["N"], meta["N"] - 1, meta["center"]
    path = hp.saw_init_path(meta)
    out = []
    for nch in n_chains_list:
        try:
            init = [jnp.full((nch, 1), int(path[i + 1]), dtype=jnp.uint8) for i in range(nfree)]
            keys = jax.random.split(jax.random.key(0), nch)
            jax.block_until_ready(run(jnp.float32(1.0), jnp.float32(8.0), init, keys))  # compile
            t0 = time.time()
            reps = 3
            for _ in range(reps):
                r = run(jnp.float32(1.0), jnp.float32(8.0), init, keys)
            jax.block_until_ready(r)
            dt = (time.time() - t0) / reps
            flips = nch * N * sweeps
            out.append((nch * N, flips / (dt * 1e9), dt))
        except Exception as e:
            print(f"    n_chains={nch}: skipped ({type(e).__name__})")
    return out, meta


def saturated_throughput(seq, L, batch=16384, sweeps=100):
    res, _ = measure_throughput(seq, L, [batch], sweeps=sweeps)
    return res[0][1] if res else float("nan")  # flips/ns


def main():
    global GPU_POWER_W
    print(f"JAX devices: {jax.devices()}")
    p = gpu_power_draw()
    # the start-of-run reading is usually idle; energy is reported at TDP unless a load reading is high
    GPU_POWER_W = p if (p and p > 150) else 450.0
    print(f"GPU power for energy estimate: {GPU_POWER_W:.0f} W (live reading {p} W; TDP 450 W)\n")

    # ---- C1: throughput saturation curve (small vs large interaction graph) ----
    print("C1  throughput (flips/ns) vs parallel degrees of freedom")
    chains = [256, 1024, 4096, 16384, 65536, 131072]
    curve20, _ = measure_throughput("HPHPPHHPHPPHPHHPPHPH", 12, chains)   # 2d20, small graph
    curve36, _ = measure_throughput(BENCHMARKS["2d36"][0], 14, chains)    # 2d36, larger graph
    F20 = max(f for _, f, _ in curve20)
    F36 = max(f for _, f, _ in curve36)
    print(f"    saturated throughput: 2d20 (N=20,q=144) = {F20:.2f} flips/ns ; "
          f"2d36 (N=36,q=196) = {F36:.2f} flips/ns")
    print(f"    (FPGA reference from THRML docs ~ 60 flips/ns on a sparse Ising model)\n")

    # ---- C2/C3: wall-clock and energy to solution ----
    print("C2/C3  wall-clock and energy to solution (annealing vs parallel tempering)")
    # use a representative saturated throughput per problem size
    def F_for(N):
        return F20 if N <= 30 else F36

    print(f"{'name':>6} {'N':>3} | {'wall_anneal':>12} {'wall_PT':>10} | "
          f"{'E_anneal(J)':>12} {'E_PT(J)':>10} | {'E_PT_extropic(J)':>16}")
    rows = []
    for name, (N, tts_a, tts_p) in TTS_SWEEPS.items():
        F = F_for(N) * 1e9  # flips/s
        def wall(tts):
            return math.inf if not math.isfinite(tts) else tts * N / F
        wa, wp = wall(tts_a), wall(tts_p)
        ea = math.inf if not math.isfinite(wa) else wa * GPU_POWER_W
        ep = wp * GPU_POWER_W
        ep_x = ep / EXTROPIC_ENERGY_FACTOR
        rows.append((N, wa, wp, ea, ep, ep_x))
        fa = "never" if not math.isfinite(wa) else f"{wa*1e3:.2f} ms"
        ea_s = "n/a" if not math.isfinite(ea) else f"{ea:.3f}"
        print(f"{name:>6} {N:>3} | {fa:>12} {wp*1e3:>8.2f}ms | {ea_s:>12} {ep:>10.4f} | {ep_x:>16.2e}")

    plot(chains, curve20, curve36, F20, F36, rows)


def plot(chains, curve20, curve36, F20, F36, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))

    dof20 = [d for d, _, _ in curve20]; f20 = [f for _, f, _ in curve20]
    dof36 = [d for d, _, _ in curve36]; f36 = [f for _, f, _ in curve36]
    axs[0].plot(dof20, f20, "o-", label="2d20 (N=20, q=144)")
    axs[0].plot(dof36, f36, "s-", label="2d36 (N=36, q=196)")
    axs[0].axhline(60, color="gray", ls=":", lw=1, label="FPGA ref (~60, sparse Ising)")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("parallel degrees of freedom (chains x N)")
    axs[0].set_ylabel("throughput (flips / ns)")
    axs[0].set_title("C1: throughput saturation on RTX 4090\n(dense categorical encoding << sparse Ising)")
    axs[0].legend(fontsize=8)

    names = list(TTS_SWEEPS.keys())
    N = [r[0] for r in rows]
    wa = [r[1] * 1e3 for r in rows]  # ms
    wp = [r[2] * 1e3 for r in rows]
    x = np.arange(len(names))
    wa_plot = [v if math.isfinite(v) else 0 for v in wa]
    axs[1].bar(x - 0.2, wa_plot, 0.4, label="annealing", color="gray")
    axs[1].bar(x + 0.2, wp, 0.4, label="parallel tempering", color="crimson")
    for i, v in enumerate(wa):
        if not math.isfinite(v):
            axs[1].text(i - 0.2, max(wp) * 0.5, "never", ha="center", rotation=90, fontsize=8)
    axs[1].set_yscale("log")
    axs[1].set_xticks(x); axs[1].set_xticklabels(names)
    axs[1].set_ylabel("wall-clock TTS (ms)")
    axs[1].set_title("C2: wall-clock time-to-solution")
    axs[1].legend(fontsize=8)

    ep = [r[4] for r in rows]
    ep_x = [r[5] for r in rows]
    axs[2].bar(x - 0.2, ep, 0.4, label="PT on GPU", color="crimson")
    axs[2].bar(x + 0.2, ep_x, 0.4, label=f"PT projected to Extropic (/{EXTROPIC_ENERGY_FACTOR:.0e})",
               color="seagreen")
    axs[2].set_yscale("log")
    axs[2].set_xticks(x); axs[2].set_xticklabels(names)
    axs[2].set_ylabel("energy to solution (J)")
    axs[2].set_title("C3: energy to solution + Extropic projection")
    axs[2].legend(fontsize=8)

    fig.tight_layout()
    out = "/home/can/extropic/project/protein/phaseC_hardware.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
