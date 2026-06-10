"""Phase A - folding thermodynamics of HP lattice proteins on THRML.

Backbone is hard at all temperatures; only H-H contacts are thermal (see hp_model).
We sweep inverse temperature beta with a quasi-static cooling ramp (each stage
warm-started from the slightly hotter one) and measure, per temperature:

  * mean H-H contacts and contact energy  E_c = -eps * contacts
  * specific heat   C(T) = beta^2 * Var(E_c)        -> peak = folding transition
  * native-contact fraction  Q(T) = <contacts> / max_contacts
  * radius of gyration  R_g(T)                       -> collapse/compaction
  * valid (self-avoiding-walk) fraction              -> sampler health check

`calibrate` first proves the sampled thermodynamics matches exact enumeration on a
tiny sequence (the accuracy gate). `run_benchmark` then produces the curves for a
longer sequence, marking the specific-heat peak.

Run:  python thermo.py --calibrate
      python thermo.py                 # benchmark sequence
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, SamplingSchedule, sample_states

import hp_model as hp


def temperature_sweep(seq, L, betas, schedule, n_chains, key, eps=1.0, lam=None):
    """Cooling ramp over ascending betas; returns {beta: positions (n_chains, n_samples, N)}.

    Each temperature is warm-started from the previous (hotter) stage's final state,
    seeded from a valid SAW at the hottest point.
    """
    betas = np.sort(np.asarray(betas, dtype=float))  # ascending beta == cooling
    run, meta = hp.make_sampler(seq, L, schedule, eps=eps)  # compiled once, beta/lam are traced
    center, n_free = meta["center"], len(meta["free_blocks"])
    path = hp.saw_init_path(meta)
    init = [jnp.full((n_chains, 1), int(path[i + 1]), dtype=jnp.uint8) for i in range(n_free)]

    out = {}
    for b in betas:
        lam_b = hp.default_lam(seq, float(b), eps) if lam is None else float(lam)
        key, ks = jax.random.split(key)
        states = run(jnp.float32(b), jnp.float32(lam_b), init, jax.random.split(ks, n_chains))
        init = [s[:, -1, :] for s in states]
        free_all = jnp.concatenate(states, axis=-1)
        center_col = jnp.full((*free_all.shape[:-1], 1), center, dtype=free_all.dtype)
        out[float(b)] = np.asarray(jnp.concatenate([center_col, free_all], axis=-1))
    return out, meta


def observables(model, pos, beta, max_contacts):
    """Per-temperature observables over the valid-SAW ensemble (HP model is defined on SAWs).

    Invalid configs at high T are soft-penalty sampler artifacts; we condition them out
    and report the valid fraction separately as a sampler-health diagnostic.
    """
    valid = hp.is_valid(model, pos).reshape(-1)
    ct_all = hp.n_contacts(model, pos).astype(float).reshape(-1)
    rg_all = hp.radius_of_gyration(model, pos).reshape(-1)
    vfrac = float(valid.mean())
    if valid.sum() < 50:  # too few valid samples to estimate from
        return dict(valid=vfrac, contacts=np.nan, C=np.nan, Q=np.nan, Rg=np.nan)
    ct = ct_all[valid]
    rg = rg_all[valid]
    Ec = -model["eps"] * ct
    return dict(
        valid=vfrac,
        contacts=float(ct.mean()),
        C=float(beta**2 * Ec.var()),
        Q=float(ct.mean() / max_contacts) if max_contacts > 0 else 0.0,
        Rg=float(rg.mean()),
    )


# --------------------------------------------------------------------------- #
# Exact thermodynamics via SAW density of states (the accuracy ground truth)
# --------------------------------------------------------------------------- #
def exact_canonical(dos, eps, betas, max_contacts):
    """Exact canonical <contacts>, C(T), Q(T) over the SAW ensemble from g[c]."""
    cs = np.arange(len(dos))
    rows = []
    for b in betas:
        logw = b * eps * cs + np.log(np.where(dos > 0, dos, 1.0))
        w = np.where(dos > 0, np.exp(logw - logw.max()), 0.0)
        w /= w.sum()
        mc = float((w * cs).sum())
        mc2 = float((w * cs * cs).sum())
        rows.append(dict(beta=float(b), contacts=mc, C=b**2 * eps**2 * (mc2 - mc**2),
                         Q=mc / max_contacts if max_contacts > 0 else 0.0))
    return rows


def calibrate():
    """Accuracy gate: sampled valid-SAW thermodynamics must match exact SAW enumeration."""
    seq, L, lam = "HHPPHPHPHPHH", 6, 8.0  # constant lam keeps bond-break rearrangement alive at all T
    betas = np.linspace(0.1, 4.0, 14)
    gc, deg = hp.saw_ground_state(seq)
    dos = hp.box_saw_density_of_states(seq, L)  # ground truth must use THRML's box+clamp ensemble
    print(f"calibration seq={seq} (N={len(seq)}) L={L} exact max_contacts={gc} deg~{deg}\n")

    schedule = SamplingSchedule(n_warmup=8000, n_samples=250, steps_per_sample=8)
    sweep, model = temperature_sweep(seq, L, betas, schedule, n_chains=6000, key=jax.random.key(0), lam=lam)
    exact = exact_canonical(dos, 1.0, np.sort(betas), gc)

    print(f"{'beta':>6} | {'ct_smp':>7} {'ct_exact':>8} | {'C_smp':>7} {'C_exact':>8} | {'valid':>6}")
    max_ct_err = max_C_err = 0.0
    for b, ex in zip(np.sort(betas), exact):
        o = observables(model, sweep[float(b)], b, gc)
        max_ct_err = max(max_ct_err, abs(o["contacts"] - ex["contacts"]))
        max_C_err = max(max_C_err, abs(o["C"] - ex["C"]))
        print(f"{b:6.3f} | {o['contacts']:7.3f} {ex['contacts']:8.3f} | "
              f"{o['C']:7.3f} {ex['C']:8.3f} | {o['valid']:6.3f}")
    print(f"\nmax |contacts_sampled - exact| = {max_ct_err:.3f}")
    print(f"max |C_sampled - exact|        = {max_C_err:.3f}")
    # contacts (a mean) must match tightly; C is a variance estimate, so it carries
    # more sampling noise at the cold tail where the valid fraction drops.
    ok = max_ct_err < 0.05 and max_C_err < 0.25
    print("CALIBRATION", "PASS" if ok else "NEEDS MORE WARMUP/CHAINS")


def run_benchmark():
    seq, L, lam = "HHPPHPHPHPHH", 6, 8.0  # N=12 cooperative folder: max 5 contacts, ground-state degeneracy 4
    betas = np.linspace(0.05, 4.0, 32)
    gc, deg = hp.saw_ground_state(seq)
    dos = hp.box_saw_density_of_states(seq, L)  # exact ground truth in the same box+clamp ensemble
    print(f"benchmark seq={seq} (N={len(seq)}) L={L}  exact max_contacts={gc} (deg~{deg})")

    schedule = SamplingSchedule(n_warmup=8000, n_samples=300, steps_per_sample=8)
    sweep, model = temperature_sweep(seq, L, betas, schedule, n_chains=6000, key=jax.random.key(1), lam=lam)
    bs = np.sort(betas)
    rows = [observables(model, sweep[float(b)], b, gc) for b in bs]
    exact = exact_canonical(dos, 1.0, bs, gc)

    C_ex = np.array([e["C"] for e in exact])
    b_fold = bs[int(np.argmax(C_ex))]
    best = max(int(hp.n_contacts(model, sweep[float(b)])[hp.is_valid(model, sweep[float(b)])].max())
               for b in bs if hp.is_valid(model, sweep[float(b)]).any())
    print(f"exact specific-heat peak at beta_fold = {b_fold:.3f}  (T_fold = {1/b_fold:.3f})")
    print(f"best contacts found by THRML = {best} / exact ground state {gc}")
    plot(bs, rows, exact, b_fold, gc, seq)


def plot(bs, rows, exact, b_fold, gc, seq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def col(rows, k):
        return [r[k] for r in rows]

    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
    pairs = [
        ("C", "Specific heat  C(T)", r"$C=\beta^2\,\mathrm{Var}(E_c)$"),
        ("Q", "Native-contact fraction  Q(T)", r"$Q=\langle$contacts$\rangle/$max"),
        ("Rg", "Radius of gyration  $R_g$(T)", r"$R_g$"),
    ]
    for ax, (k, title, ylab) in zip(axs[:3], pairs):
        ax.plot(bs, col(rows, k), "o", ms=4, label="THRML (sampled)")
        if k in ("C", "Q"):
            ax.plot(bs, col(exact, k), "-", color="k", lw=1.5, label="exact (SAW enum)")
        if k == "Q":
            ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(title)
        ax.set_ylabel(ylab)
    axs[3].plot(bs, col(rows, "valid"), "o-", ms=3, color="darkorange")
    axs[3].set_ylim(0, 1.05)
    axs[3].set_title("Valid-SAW fraction (sampler health)")
    axs[3].set_ylabel("valid fraction")
    for ax in axs:
        ax.axvline(b_fold, color="r", ls="--", lw=1, label=r"$\beta_{fold}$")
        ax.set_xlabel(r"$\beta = 1/T$")
        ax.legend(fontsize=8)
    fig.suptitle(f"HP folding thermodynamics on THRML vs exact — seq={seq}, max_contacts={gc}", y=1.03)
    fig.tight_layout()
    out = "/home/can/extropic/project/protein/thermo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    print(f"JAX devices: {jax.devices()}\n")
    if args.calibrate:
        calibrate()
    else:
        run_benchmark()
