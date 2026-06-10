# Lattice protein folding on a probabilistic computer (THRML)

Folding HP lattice proteins as an energy-based model sampled on
[THRML](https://github.com/extropic-ai/thrml), Extropic's GPU simulator of the block-Gibbs
sampling programs that run natively on its probabilistic-computing hardware. The project
treats folding as both a **sampling/optimization** problem and a **learning** problem, and
validates every result against exact ground truth.

> **Punchline.** THRML samples the folding distribution correctly; used naively (simulated
> annealing) it is a weak optimizer, but with **parallel tempering** it gives **150–660×
> time-to-solution speedups** and solves instances annealing never reaches. The winning
> method — many parallel replicas with cheap temperature swaps — is exactly the workload
> Extropic hardware is designed to accelerate. A Boltzmann machine trained on the resulting
> folds (THRML's contrastive-KL training) reproduces the fold ensemble at correlation 0.997.

---

## Background

**Probabilistic computers** sample from probability distributions using massively parallel
hardware randomness, making them efficient at the sampling-heavy workloads behind
energy-based models (EBMs). THRML simulates these programs on GPUs via **block Gibbs
sampling** of EBMs `P(x) ∝ exp(−E(x))`.

**The HP model** (Dill, 1985) is the canonical coarse-grained protein-folding model. Each
residue is **H** (hydrophobic) or **P** (polar); the chain is a self-avoiding walk on a
lattice, and energy rewards **H–H contacts** (H pairs adjacent on the lattice but not
consecutive in the sequence). Finding the minimum-energy fold is NP-hard and is a standard
benchmark for folding algorithms — a natural target for a sampling/optimization machine.

## The problem, concretely

Given an H/P sequence, (1) characterize its **folding thermodynamics** (how it folds as it
cools), and (2) **find its ground-state fold** (maximum H–H contacts). Both reduce to
sampling/optimizing an energy landscape — THRML's native operation.

## Approach

**Encoding (`hp_model.py`).** One categorical variable per residue indexing an L×L lattice
site (`CategoricalEBMFactor` + `CategoricalGibbsConditional`). All constraints and rewards
are pairwise q×q matrices. The energy is **decoupled** into a thermal part and an always-on
backbone constraint:

```
E(x) = lam · (overlaps + broken_bonds)  −  beta · eps · (H–H contacts)
```

so temperature controls only the physical contact thermodynamics while the self-avoiding
backbone is enforced at all T. Residue 0 is **clamped** to the box center (removes
translational symmetry, showcases THRML clamping). Because excluded volume couples every
residue pair, the interaction graph is complete → single-residue color blocks → parallelism
comes from `vmap`-ing thousands of independent chains.

**Samplers.** Plain block Gibbs, simulated annealing (cooling ramp from a valid-SAW init),
and **parallel tempering** (a temperature ladder with replica-exchange swaps; the decoupled
constant-λ design makes the swap criterion depend only on contact counts).

**Validation philosophy.** Nothing is trusted until it matches exact ground truth — exact
state enumeration, exact self-avoiding-walk enumeration / density of states, and the exact
Onsager solution in the Ising warmup. Sampling fails silently, so every stage has an
accuracy gate.

## Results (highlights)

| result | figure | headline number |
|---|---|---|
| Sampling is correct | (validate_hp) | sampled vs exact Boltzmann: TV **0.007** |
| Folding thermodynamics | `thermo.png` | matches exact SAW enumeration; contacts to **0.013** |
| Time-to-solution minimum | `phaseB_budget.png` | short restarts beat long anneals |
| Annealing scaling | `phaseB_scaling.png` | solves only N ≤ 24 |
| **Parallel tempering** | `pt_vs_anneal.png` | **150–660× TTS speedup; solves N=36** |
| **Learned generative model** | `ebm_rbm.png` | contact-map corr **0.997** |

Full breakdown with all numbers in [`RESULTS.md`](RESULTS.md).

## Repository structure

```
hp_model.py     encoding, observables, exact enumeration, compile-once sampler
validate_hp.py  correctness gate (matrix / exact-distribution / ground state)
thermo.py       Phase A: folding thermodynamics vs exact (C(T), Q(T), R_g)
benchmark.py    canonical 2D HP benchmark sequences + published optima
solve.py        Phase B: success probability, time-to-solution, scaling with N
pt.py           parallel tempering vs annealing comparison
ebm_train.py    learned generative model of folds (spin RBM via contrastive KL)
RESULTS.md      detailed results log
*.png           figures
```

## Running

Requires an NVIDIA GPU. Environment (conda):

```bash
conda create -n thrml python=3.12 -y && conda activate thrml
pip install "jax[cuda12]" optax networkx matplotlib scikit-learn
pip install thrml            # or: pip install -e /path/to/thrml
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Then:

```bash
python validate_hp.py         # correctness gate
python thermo.py --calibrate  # accuracy gate vs exact enumeration
python thermo.py              # folding thermodynamics figure
python solve.py --budget      # time-to-solution minimum
python solve.py --scaling     # scaling with chain length
python pt.py                  # parallel tempering vs annealing
python ebm_train.py           # learned generative model (RBM)
python throughput.py          # Phase C: throughput, wall-clock TTS, energy + Extropic projection
```

## Notes / limitations
- Categorical state is `uint8` ⇒ q = L² ≤ 256 ⇒ L ≤ 16, so in-box benchmarks reach ~N ≤ 50.
- Single-site block Gibbs (the hardware-native move) mixes slowly for connected chains;
  this is *the* finding that motivates annealing, valid-SAW initialization, and parallel
  tempering.

## Status / roadmap
- [x] Encoding + correctness validation
- [x] Phase A — folding thermodynamics (validated vs exact)
- [x] Phase B — optimization: benchmark set, time-to-solution, scaling
- [x] Parallel tempering (sampling/optimization pillar)
- [x] EBM training (learning pillar)
- [x] Phase C — hardware / Extropic projection (`throughput.py`, `phaseC_hardware.png`):
      throughput saturation, wall-clock TTS (PT 250–666× faster), energy-to-solution +
      labeled Extropic projection

## References
- THRML: https://github.com/extropic-ai/thrml · docs: https://docs.thrml.ai
- Extropic probabilistic computing paper: https://arxiv.org/abs/2510.23972
- HP benchmark optima cross-checked vs Mann et al., PMC5172541
