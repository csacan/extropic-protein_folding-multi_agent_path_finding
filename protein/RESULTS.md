# Protein folding on a probabilistic computer (THRML) — results so far

A probabilistic-computing use case on Extropic's THRML: the HP lattice-protein model
encoded as a Potts energy-based model, sampled by GPU block-Gibbs. Two pillars —
**sampling/optimization** and **energy-based learning** — both validated against exact
ground truth.

## Punchline

THRML is a *correct and useful probabilistic engine for protein folding*, demonstrated end to end:

1. **It samples the right distribution.** Sampled folding thermodynamics matches exact
   enumeration to within statistical noise (contacts to 0.013).
2. **It is a weak optimizer when used naively, and a strong one with the right algorithm.**
   Plain simulated annealing only reaches the optimum for N ≤ 24; **parallel tempering gives
   150–660× time-to-solution speedups and solves instances annealing never does.**
3. **It learns.** A spin Boltzmann machine trained with THRML's contrastive-KL gradient on
   folds produced in (2) reproduces the fold ensemble's contact statistics at corr 0.997.

The optimizer result is the one that maps onto Extropic hardware: the winning method is
**many parallel replicas with cheap temperature swaps**, which is exactly what the hardware
is built to accelerate.

---

## The model (encoding)

Each residue is a categorical variable over an L×L lattice (`CategoricalEBMFactor` +
`CategoricalGibbsConditional`). Energy decoupled into a **thermal** part and an **always-on
backbone constraint**:

    E(x) = lam * (overlaps + broken_bonds)  -  beta * eps * (H-H contacts)

so beta controls only the physical contact thermodynamics; the self-avoiding-walk backbone
is enforced at every temperature. Residue 0 is clamped to the box center. `hp_model.py`.

---

## 1. Correctness gate — `validate_hp.py`

| check | result |
|---|---|
| weight-tensor energy == first-principles energy | 4e-7 |
| THRML sampled histogram vs exact enumerated Boltzmann | total-variation 0.007 |
| annealing reaches exact SAW ground state (small seqs) | PASS (valid fraction 1.0) |

Sampling is provably correct before any science is read off it.

## 2. Folding thermodynamics — `thermo.py` → `thermo.png`

Cooperative folder `HHPPHPHPHPHH` (N=12, unique fold). THRML samples (dots) lie on the
**exact SAW-enumeration curve** (line) for specific heat C(T), native-contact fraction Q(T),
and radius of gyration R_g(T). Folding transition at β_fold ≈ 2.22; contacts match exact to
**0.013**. This is the validated "measuring instrument" used for everything downstream.

*(Three sampler fixes made this work: decouple penalties from temperature; constant λ so
bond-break rearrangement stays possible at all T; ground truth enumerated in the same
box+clamp ensemble THRML samples.)*

## 3. Optimization: benchmark, budget, scaling — `benchmark.py`, `solve.py`

Canonical 2D HP benchmark set (lengths 20–64, published optima 9/9/8/14/23/21/36/42;
cross-checked vs PMC5172541). Reachability of the optimum confirmed in-box for the small
instances.

- **Budget (`phaseB_budget.png`)**: time-to-solution has a **minimum at a short budget
  (~9k sweeps)** — many short restarts beat few long anneals.
- **Scaling (`phaseB_scaling.png`)**: plain annealing reaches the exact optimum only for
  **N ≤ 24**; for larger N the best fold degrades to 71–87% of optimum and success
  probability collapses to 0.

TTS metric: `TTS = sweeps_per_run · ln(0.01)/ln(1−p)` (99% confidence with restarts).

## 4. Parallel tempering — `pt.py` → `pt_vs_anneal.png`  ← headline optimization result

| seq | N | p (annealing) | p (PT) | TTS speedup |
|---|---|---|---|---|
| 2d20 | 20 | 0.0010 | 0.240 | **248×** |
| 2d24 | 24 | 0.0005 | 0.308 | **666×** |
| 2d25 | 25 | 0.0005 | 0.083 | **157×** |
| 2d36 | 36 | 0 (never) | 0.005 | **PT solves it** |

Replica exchange across a temperature ladder, with swaps accepted on contact counts alone
(the constant-λ penalties cancel exactly in the swap criterion). 150–660× speedups, and it
reaches optima annealing cannot.

## 5. Learned generative model of folds — `ebm_train.py` → `ebm_rbm.png`  ← learning pillar

148k folds from the sampler → binary contact maps (55 bits) → spin RBM (55 visible + 32
hidden) trained via THRML `estimate_kl_grad` + optax in ~6 s. The learned model reproduces
the per-pair contact-frequency map at **corr 0.997** (MAE 0.008) and matches the mean
contact count (4.78 data vs 4.85 RBM). THRML doing inference *and* learning on one problem.

---

## 6. Hardware / Extropic projection — `throughput.py` → `phaseC_hardware.png`

Converts the sweep-based results to wall-clock and energy on the RTX 4090.

- **Throughput (C1):** saturates at only **~0.04 flips/ns** (2d20) / ~0.03 (2d36), far below
  the ~60 flips/ns sparse-Ising FPGA reference. Honest finding: the **dense categorical
  encoding** (q up to 256, complete interaction graph ⇒ single-residue sequential block
  updates) is ~10³× less throughput-friendly than sparse binary Ising — a property of the
  problem encoding, and an argument for sparser encodings / native hardware.
- **Wall-clock TTS (C2):** the PT advantage carries straight through — PT solves 2d20/2d24/2d25
  in **72–320 ms** vs **20–50 s** for annealing (~250–666×), and solves 2d36 in ~12 s where
  annealing never does.
- **Energy-to-solution (C3, at 450 W TDP):** PT ≈ 32–5400 J; annealing ≈ 9000–22500 J (and
  ∞ for 2d36). **Projected to Extropic hardware** (÷10⁴, a labeled factor from arXiv:2510.23972,
  to be verified against the paper): ~3.6 mJ–0.5 J.

The defensible claims are the **ratios** (PT vs annealing ≈ 250–666× in wall-clock and energy;
Extropic projection a stated factor); absolute throughput carries the encoding caveat above.

## Figures to present
- `thermo.png` — THRML vs exact folding thermodynamics (correctness + physics)
- `phaseB_budget.png` — time-to-solution minimum
- `phaseB_scaling.png` — annealing degrades with N
- `pt_vs_anneal.png` — **parallel tempering 150–660× speedup** (headline)
- `ebm_rbm.png` — **learned generative model, corr 0.997** (learning pillar)
- `phaseC_hardware.png` — throughput, wall-clock TTS, energy + Extropic projection

## Key engineering notes
- `hp_model.make_sampler`: compile-once sampler with β, λ as traced inputs (eliminated a
  28×-per-temperature recompile; large speedup, used everywhere).
- THRML categorical readouts are uint8 — cast before any distance arithmetic (a silent
  underflow bug cost a debugging session early on).
- `uint8` state ⇒ q = L² ≤ 256 ⇒ L ≤ 16, so in-box benchmarks reach ~N ≤ 50.

## Reproduce (conda env `thrml`, RTX 4090, `XLA_PYTHON_CLIENT_PREALLOCATE=false`)
```
python validate_hp.py            # correctness gate
python thermo.py --calibrate     # accuracy gate vs exact
python thermo.py                 # folding thermodynamics figure
python solve.py --budget         # TTS minimum
python solve.py --scaling        # scaling collapse
python pt.py                     # parallel tempering vs annealing
python ebm_train.py              # learned generative model (RBM)
python throughput.py             # Phase C: throughput, wall-clock TTS, energy + Extropic projection
```

## Status
All phases complete: correctness → thermodynamics (A) → optimization/scaling (B) →
parallel tempering → EBM training → hardware/Extropic projection (C). The two-pillar story
(sample/optimize + learn) stands on validated ground truth throughout.
