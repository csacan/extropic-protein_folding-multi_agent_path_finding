"""B1: HYBRID conditional solver = fixed hard-constraint Ising terms + trainable latents.

The pure-learned B2 has to *learn* the MAPF rules implicitly, so a flat RBM only reaches
~18% per-sample validity on held-out instances. Here we instead bake the constraints in
as FIXED Ising energy and let the latents learn only the coordination prior:

  energy = [ one-hot + motion + vertex penalties ]   (fixed, hand-derived, frozen)
         + [ position-spin <-> latent couplings ]     (trained by estimate_kl_grad)

so correctness comes (softly) from the constraints and the latents learn to improve
*findability*. We compare, on held-out instances:
  * constraints-only (latent weights zeroed) -- the bare EBM's per-sample validity,
  * hybrid (trained latents)                 -- does learning raise it?

(Swap conflicts are order-4 and cannot be an Ising term, so the fixed energy covers
one-hot + motion + vertex; swap is still scored in the validity metric.)

Constraint construction: penalties are written as QUBO on x in {0,1} and converted to
the Ising spin form THRML uses, E_ising = -beta*(b.s + s'.W.s); a constraint that ADDS
energy E_pen(x) maps to biases b=-h, weights W=-J. Verified empirically by checking the
constraints-only model samples valid plans.
"""

from __future__ import annotations

from collections import defaultdict

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
# layout with addressable position spins pos[k][t][c]
# --------------------------------------------------------------------------- #
def build_layout(rows, K, T, q, n_lat, P=2.0, A=2.0, B=2.0):
    m0 = pt.light_model(rows, [(0, 0)] * 1 * 0 + _first_cells(rows, K),
                        _first_cells(rows, K, off=K), T)  # just to grab `allowed`
    allowed = m0["allowed"]

    pos = [[[SpinNode() for _ in range(q)] for _ in range(T + 1)] for _ in range(K)]
    inp = [pos[k][0][c] for k in range(K) for c in range(q)] + \
          [pos[k][T][c] for k in range(K) for c in range(q)]
    out = [pos[k][t][c] for k in range(K) for t in range(1, T) for c in range(q)]
    lat = [SpinNode() for _ in range(n_lat)]
    alln = inp + out + lat
    idx = {n: i for i, n in enumerate(alln)}

    # ----- QUBO penalty accumulation over position spins -----
    a = defaultdict(float)
    b = defaultdict(float)

    def add_pair(ni, nj, val):
        i, j = idx[ni], idx[nj]
        if i > j:
            i, j = j, i
        b[(i, j)] += val

    for k in range(K):              # one-hot per (k,t)
        for t in range(T + 1):
            blk = pos[k][t]
            for c in range(q):
                a[idx[blk[c]]] += -P
                for c2 in range(c + 1, q):
                    add_pair(blk[c], blk[c2], 2 * P)
    for k in range(K):              # motion (illegal transitions)
        for t in range(T):
            for c in range(q):
                for c2 in range(q):
                    if not allowed[c, c2]:
                        add_pair(pos[k][t][c], pos[k][t + 1][c2], A)
    for t in range(T + 1):          # vertex conflict
        for k in range(K):
            for k2 in range(k + 1, K):
                for c in range(q):
                    add_pair(pos[k][t][c], pos[k2][t][c], B)

    # ----- QUBO -> Ising (x=(s+1)/2): h, J ; constraint biases=-h, weights=-J -----
    h = defaultdict(float)
    for i, val in a.items():
        h[i] += val / 2
    J = {}
    for (i, j), val in b.items():
        J[(i, j)] = val / 4
        h[i] += val / 4
        h[j] += val / 4

    n_pos = len(inp) + len(out)
    biases = np.zeros(len(alln), dtype=np.float32)
    for i in range(n_pos):
        biases[i] = -h.get(i, 0.0)                  # fixed constraint bias on position spins

    cons_edges = [(alln[i], alln[j]) for (i, j) in J]
    cons_w = np.array([-J[(i, j)] for (i, j) in J], dtype=np.float32)
    lat_edges = [(v, l) for v in (inp + out) for l in lat]
    edges = cons_edges + lat_edges
    n_cons, n_lat_e = len(cons_edges), len(lat_edges)

    wmask = np.concatenate([np.zeros(n_cons), np.ones(n_lat_e)]).astype(np.float32)   # train latent edges
    bmask = np.array([1.0 if alln[i] in set(lat) else 0.0 for i in range(len(alln))], np.float32)

    return dict(pos=pos, inp=inp, out=out, lat=lat, alln=alln, edges=edges,
                n_cons=n_cons, cons_w=cons_w, biases0=biases, wmask=jnp.asarray(wmask),
                bmask=jnp.asarray(bmask), allowed=allowed,
                K=K, T=T, q=q, n_in=len(inp), n_out=len(out))


def _first_cells(rows, n, off=0):
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    return [tuple(int(x) for x in coords[c]) for c in range(off, off + n)]


# --------------------------------------------------------------------------- #
# encode / decode (same convention as B2)
# --------------------------------------------------------------------------- #
def encode_input(lay, sc, gc):
    K, q = lay["K"], lay["q"]
    v = np.zeros(lay["n_in"], bool)
    for k in range(K):
        v[k * q + sc[k]] = True
        v[K * q + k * q + gc[k]] = True
    return v


def encode_output(lay, plans):
    K, T, q = lay["K"], lay["T"], lay["q"]
    N = plans.shape[0]
    d = np.zeros((N, lay["n_out"]), bool)
    for k in range(K):
        for t in range(1, T):
            base = (k * (T - 1) + (t - 1)) * q
            d[np.arange(N), base + plans[:, k, t]] = True
    return d


def decode_full(lay, out_spins, sc, gc):
    K, T, q = lay["K"], lay["T"], lay["q"]
    N = out_spins.shape[0]
    mid = out_spins.reshape(N, K, T - 1, q).astype(np.int8).argmax(-1)
    plans = np.empty((N, K, T + 1), np.int64)
    for k in range(K):
        plans[:, k, 0] = sc[k]; plans[:, k, T] = gc[k]; plans[:, k, 1:T] = mid[:, k]
    return plans


# --------------------------------------------------------------------------- #
# instances + teacher data
# --------------------------------------------------------------------------- #
def make_instances(rows, T, lay, key, n, min_solutions=40, seed=0):
    coords, q, *_ = mapf.grid_graph(mapf.parse_map(rows))
    K = lay["K"]
    rng = np.random.default_rng(seed)
    inst, tries = [], 0
    while len(inst) < n and tries < n * 12:
        tries += 1
        cells = rng.choice(q, 2 * K, replace=False)
        starts = [tuple(int(x) for x in coords[c]) for c in cells[:K]]
        goals = [tuple(int(x) for x in coords[c]) for c in cells[K:]]
        model = pt.light_model(rows, starts, goals, T, lam_swap=1.0)
        key, kk = jax.random.split(key)
        trace = ts.ffbs_gibbs(model, kk, 6000, 30, beta=3.0)[:, -1]
        valid = mapf.is_valid(model, trace)
        if valid.sum() < min_solutions:
            continue
        inst.append(dict(model=model, sc=model["start_cells"], gc=model["goal_cells"],
                         input_vec=encode_input(lay, model["start_cells"], model["goal_cells"]),
                         out_data=encode_output(lay, trace[valid])))
    return inst


# --------------------------------------------------------------------------- #
# training (update latent params only)
# --------------------------------------------------------------------------- #
def make_trainer(lay, beta=1.0, lr=0.02,
                 sched_pos=SamplingSchedule(5, 20, 5), sched_neg=SamplingSchedule(40, 20, 5)):
    alln, edges, inp, out, lat = lay["alln"], lay["edges"], lay["inp"], lay["out"], lay["lat"]
    data_blocks, cond_blocks = [Block(out)], [Block(inp)]
    pos_sample, neg_sample = [Block(lat)], [Block(out), Block(lat)]
    wmask, bmask = lay["wmask"], lay["bmask"]
    beta = jnp.array(beta)

    def step(biases, weights, out_batch, input_vec, key):
        model = IsingEBM(alln, edges, biases, weights, beta)
        spec = IsingTrainingSpec(model, data_blocks, cond_blocks, pos_sample, neg_sample,
                                 sched_pos, sched_neg)
        kp, kn, kg = jax.random.split(key, 3)
        bs = out_batch.shape[0]
        init_pos = hinton_init(kp, model, pos_sample, (1, bs))
        init_neg = hinton_init(kn, model, neg_sample, (bs,))
        gw, gb, _, _ = estimate_kl_grad(kg, spec, alln, edges, [out_batch], [input_vec],
                                        init_pos, init_neg)
        with jax.numpy_dtype_promotion("standard"):
            biases = biases - lr * gb * bmask          # only latent biases move
            weights = weights - lr * gw * wmask        # only latent edges move
        return biases, weights

    return jax.jit(step)


# --------------------------------------------------------------------------- #
# inference + eval
# --------------------------------------------------------------------------- #
def solve(lay, biases, weights, ins, key, beta=1.0, n_chains=1500,
          sched=SamplingSchedule(120, 1, 10)):
    alln, edges, inp, out, lat = lay["alln"], lay["edges"], lay["inp"], lay["out"], lay["lat"]
    model = IsingEBM(alln, edges, biases, weights, jnp.array(beta))
    prog = IsingSamplingProgram(model, [Block(out), Block(lat)], [Block(inp)])
    clamp = [jnp.asarray(ins["input_vec"])]
    ki, kr = jax.random.split(key)
    init = hinton_init(ki, model, [Block(out), Block(lat)], (n_chains,))
    states = jax.jit(jax.vmap(lambda i, k: sample_states(k, prog, sched, i, clamp, [Block(out)])))(
        init, jax.random.split(kr, n_chains))
    plans = decode_full(lay, np.asarray(states[0][:, -1, :]), ins["sc"], ins["gc"])
    valid = mapf.is_valid(ins["model"], plans)
    return float(valid.mean()), float(valid.any())


def eval_set(lay, biases, weights, instances, key, **kw):
    vfs = []
    solved = []
    for ins in instances:
        key, k = jax.random.split(key)
        vf, sv = solve(lay, biases, weights, ins, k, **kw)
        vfs.append(vf); solved.append(sv)
    return float(np.mean(vfs)), float(np.mean(solved))


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    rows, T, q = ["...", "...", "..."], 4, 9
    BETA = 1.0
    lay = build_layout(rows, K=2, T=T, q=q, n_lat=128, P=2.0, A=2.0, B=2.0)
    print(f"layout: n_in={lay['n_in']} n_out={lay['n_out']} n_lat={len(lay['lat'])}  "
          f"constraint_edges={lay['n_cons']}  latent_edges={len(lay['edges'])-lay['n_cons']}")

    biases = jnp.asarray(lay["biases0"])
    key = jax.random.key(0)
    key, kw = jax.random.split(key)
    weights = jnp.concatenate([jnp.asarray(lay["cons_w"]),
                               0.01 * jax.random.normal(kw, (len(lay["edges"]) - lay["n_cons"],))])

    key, kd = jax.random.split(key)
    allinst = make_instances(rows, T, lay, kd, n=70, min_solutions=40)
    train, test = allinst[:55], allinst[55:]
    print(f"instances: {len(train)} train, {len(test)} held-out")

    # constraints-only baseline (zero the latent params)
    b0 = biases * (1 - lay["bmask"])
    w0 = weights * (1 - lay["wmask"])
    key, ke = jax.random.split(key)
    base = eval_set(lay, b0, w0, test, ke, beta=BETA)
    print(f"constraints-only (no latents): HELD-OUT valid={base[0]:.3f} solved={base[1]:.3f}")

    trainer = make_trainer(lay, beta=BETA, lr=0.02)
    rng = np.random.default_rng(0)
    for s in range(2500):
        ins = train[rng.integers(0, len(train))]
        od = ins["out_data"]; bidx = rng.integers(0, len(od), 128)
        key, ks = jax.random.split(key)
        biases, weights = trainer(biases, weights, jnp.asarray(od[bidx]),
                                  jnp.asarray(ins["input_vec"]), ks)
        if (s + 1) % 500 == 0:
            key, ke = jax.random.split(key)
            te = eval_set(lay, biases, weights, test, ke, beta=BETA)
            print(f"  step {s+1:4d}: HELD-OUT valid={te[0]:.3f} solved={te[1]:.3f}")

    key, ke = jax.random.split(key)
    te = eval_set(lay, biases, weights, test, ke, beta=BETA)
    print(f"\nFINAL HELD-OUT  constraints-only valid={base[0]:.3f} -> hybrid valid={te[0]:.3f}  "
          f"(solved {base[1]:.3f} -> {te[1]:.3f})")
