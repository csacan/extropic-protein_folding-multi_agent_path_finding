"""HP lattice-protein model encoded as a Potts EBM for THRML (2D square lattice).

Each residue i is a categorical variable x_i in {0, ..., q-1} indexing a site on
an L x L lattice (q = L*L). The chain energy is built from pairwise q x q reward
matrices, in THRML's convention where  E = -sum_pairs W[x_i, x_j]  (so a POSITIVE
matrix entry lowers energy / raises probability):

  * connectivity   (consecutive i, i+1):  -lam_conn  where sites are NOT adjacent
  * excluded volume (all i < j)         :  -lam_excl  on the diagonal (same site)
  * H-H contact    (non-consecutive H,H):  +eps       where sites ARE adjacent

beta (inverse temperature) multiplies every matrix, exactly as in example 00.
Residue 0 is clamped to the lattice center to remove translational degeneracy.

This maps onto THRML's CategoricalEBMFactor / CategoricalGibbsConditional. Because
excluded volume couples every residue pair, the interaction graph is complete, so
each residue is its own color block and a Gibbs sweep is a sequential scan over
residues -- the parallelism comes from vmapping many independent chains/replicas.
"""

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from thrml import Block, BlockGibbsSpec, CategoricalNode, FactorSamplingProgram, sample_states
from thrml.models import CategoricalGibbsConditional, SquareCategoricalEBMFactor


# --------------------------------------------------------------------------- #
# Lattice
# --------------------------------------------------------------------------- #
def square_lattice(L: int):
    """Return (coords, q, adj, center) for an L x L square lattice.

    Site index s maps to (row, col) = (s // L, s % L); adj[a, b] is True for
    von Neumann (edge) neighbors; center is the index of the middle site.
    """
    coords = np.array([(s // L, s % L) for s in range(L * L)])
    q = L * L
    d = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)
    adj = d == 1
    center = (L // 2) * L + (L // 2)
    return coords, q, adj, center


def saw_init_path(model):
    """A valid self-avoiding walk of length N anchored at the box center (DFS).

    Used to seed annealing: with the backbone penalty always strong, a random start
    freezes, so every chain begins from this intact walk and explores from there.
    Returns an array of N site indices, path[0] == center.
    """
    L, N, center = model["L"], model["N"], model["center"]
    cr, cc = center // L, center % L
    occ, path = {(cr, cc)}, [(cr, cc)]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def dfs():
        if len(path) == N:
            return True
        r, c = path[-1]
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < L and 0 <= nc < L and (nr, nc) not in occ:
                occ.add((nr, nc))
                path.append((nr, nc))
                if dfs():
                    return True
                path.pop()
                occ.discard((nr, nc))
        return False

    if not dfs():
        raise RuntimeError(f"no SAW of length {N} fits in a {L}x{L} box; increase L")
    return np.array([r * L + c for r, c in path], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Energy matrices
# --------------------------------------------------------------------------- #
def pair_matrices(seq: str, q: int, adj: np.ndarray):
    """Per-pair q x q matrices, split into a thermal contact part and a constraint part.

    Returns (heads, tails, contact, penalty) where for each residue pair p:
      contact[p][a,b] = +1 on lattice-adjacent sites for non-consecutive H-H pairs (else 0)
      penalty[p][a,b] = -(1-A) for consecutive pairs (non-adjacent forbidden) and -I always
                        (excluded volume). These are unit-magnitude; coefficients applied later.
    """
    N = len(seq)
    I = np.eye(q, dtype=np.float32)
    A = adj.astype(np.float32)
    heads, tails, contact, penalty = [], [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            c = np.zeros((q, q), dtype=np.float32)
            p = -I.copy()  # excluded volume on every pair
            if j == i + 1:
                p = p - (1.0 - A)  # consecutive residues must be lattice-adjacent
            elif seq[i] == "H" and seq[j] == "H":
                c = A.copy()  # non-consecutive H-H topological contact (thermal)
            heads.append(i)
            tails.append(j)
            contact.append(c)
            penalty.append(p)
    return np.array(heads), np.array(tails), np.stack(contact), np.stack(penalty)


def default_lam(seq: str, beta: float, eps: float = 1.0):
    """Backbone/excluded-volume penalty coefficient that always dominates the contact energy.

    Held effectively infinite at every temperature so the chain stays a self-avoiding
    walk at all T; only the H-H contacts are thermal. Exceeds beta*eps*(max contacts).
    """
    n_h = sum(c == "H" for c in seq)
    # beta*eps*(n_h+1) already exceeds the whole-chain contact energy; small constant
    # margin keeps the penalty just-dominant so e^{-lam} tunneling (mixing) stays fast.
    return float(beta * eps * (n_h + 1) + 3.0)


# --------------------------------------------------------------------------- #
# THRML model
# --------------------------------------------------------------------------- #
def build_model(seq: str, L: int, beta: float, eps: float = 1.0, lam=None):
    """Assemble the THRML program. Thermal contacts (beta*eps) + constant backbone penalty (lam).

    Decoupling lam from beta keeps the backbone intact at every temperature, so beta
    controls only the H-H contact thermodynamics (the physical HP folding model).
    """
    coords, q, adj, center = square_lattice(L)
    if q > 256:
        raise ValueError(f"q={q} exceeds 256 (categorical state is uint8); use a smaller L.")
    if lam is None:
        lam = default_lam(seq, beta, eps)

    N = len(seq)
    heads, tails, contactM, penaltyM = pair_matrices(seq, q, adj)
    M = beta * eps * contactM + lam * penaltyM  # (n_pairs, q, q)
    weights = jnp.asarray(M, dtype=jnp.float32)

    residues = [CategoricalNode() for _ in range(N)]
    head_nodes = [residues[i] for i in heads]
    tail_nodes = [residues[j] for j in tails]
    factor = SquareCategoricalEBMFactor([Block(head_nodes), Block(tail_nodes)], weights)

    free_blocks = [Block([r]) for r in residues[1:]]  # residue 0 is clamped
    clamped_blocks = [Block([residues[0]])]
    spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    sampler = CategoricalGibbsConditional(q)
    prog = FactorSamplingProgram(spec, [sampler for _ in free_blocks], [factor], [])

    return dict(
        prog=prog, residues=residues, free_blocks=free_blocks, clamped_blocks=clamped_blocks,
        spec=spec, center=center, q=q, L=L, N=N, coords=coords, adj=adj,
        heads=heads, tails=tails, weights=np.asarray(weights), seq=seq,
        eps=eps, lam=lam, beta=beta,
    )


def make_sampler(seq: str, L: int, schedule, eps: float = 1.0):
    """Compile-once sampler: returns (run, meta).

    `run(beta, lam, init_blocks, keys)` takes beta and lam as TRACED scalars, so the
    XLA program is compiled a single time and reused for every temperature/penalty --
    instead of recompiling the ~N^2-pair graph at each beta. The energy is rebuilt
    inside (cheap at trace time): weights = beta*eps*contact + lam*penalty. `eqx.filter_jit`
    treats the nodes/blocks/schedule as static and only the arrays as dynamic.
    """
    coords, q, adj, center = square_lattice(L)
    if q > 256:
        raise ValueError(f"q={q} exceeds 256 (categorical state is uint8); use a smaller L.")
    N = len(seq)
    heads, tails, contactM, penaltyM = pair_matrices(seq, q, adj)
    residues = [CategoricalNode() for _ in range(N)]
    head_nodes = [residues[i] for i in heads]
    tail_nodes = [residues[j] for j in tails]
    free_blocks = [Block([r]) for r in residues[1:]]
    clamped_blocks = [Block([residues[0]])]
    spec = BlockGibbsSpec(free_blocks, clamped_blocks)
    sampler = CategoricalGibbsConditional(q)
    contactJ = jnp.asarray(contactM, dtype=jnp.float32)
    penaltyJ = jnp.asarray(penaltyM, dtype=jnp.float32)
    clamp = [jnp.array([center], dtype=jnp.uint8)]

    @eqx.filter_jit
    def run(beta, lam, init_blocks, keys):
        weights = beta * eps * contactJ + lam * penaltyJ
        factor = SquareCategoricalEBMFactor([Block(head_nodes), Block(tail_nodes)], weights)
        prog = FactorSamplingProgram(spec, [sampler for _ in free_blocks], [factor], [])
        one = lambda ic, k: sample_states(k, prog, schedule, ic, clamp, free_blocks)
        return jax.vmap(one)(init_blocks, keys)  # list[n_free] of (n_chains, n_samples, 1)

    meta = dict(center=center, q=q, L=L, N=N, coords=coords, adj=adj, seq=seq, eps=eps,
                free_blocks=free_blocks, residues=residues, heads=heads, tails=tails)
    return run, meta


def anneal_positions(seq, L, betas, schedule, n_chains, key, eps=1.0, lam=None,
                     init_path=None, return_model=False):
    """Simulated annealing: sweep beta low->high, carrying chain state across stages.

    The backbone penalty is always strong (decoupled from beta), so a random start
    would freeze; initialize every chain from a valid SAW and let high-temperature
    (low beta) stages explore conformations while the backbone stays intact. Each
    stage rebuilds the program at the next beta and re-inits from the previous final
    state. Returns positions (n_chains, n_samples, N) from the final (coldest) stage.
    """
    key, k_init = jax.random.split(key)
    m0 = build_model(seq, L, float(betas[0]), eps, lam)
    q, center, n_free = m0["q"], m0["center"], len(m0["free_blocks"])
    clamp = [jnp.array([center], dtype=jnp.uint8)]
    path = saw_init_path(m0) if init_path is None else np.asarray(init_path)
    init = [jnp.full((n_chains, 1), int(path[i + 1]), dtype=jnp.uint8) for i in range(n_free)]

    model, states = m0, None
    for b in betas:
        model = build_model(seq, L, float(b), eps, lam)
        key, k_stage = jax.random.split(key)
        keys = jax.random.split(k_stage, n_chains)

        def one(init_c, k):
            return sample_states(k, model["prog"], schedule, init_c, clamp, model["free_blocks"])

        states = jax.jit(jax.vmap(one))(init, keys)  # list[n_free] of (n_chains, n_samples, 1)
        init = [s[:, -1, :] for s in states]         # carry final state into next stage

    free_all = jnp.concatenate(states, axis=-1)      # (n_chains, n_samples, n_free), order residues[1:]
    center_col = jnp.full((*free_all.shape[:-1], 1), center, dtype=free_all.dtype)
    pos = np.asarray(jnp.concatenate([center_col, free_all], axis=-1))
    return (pos, model) if return_model else pos


def sample_positions(model, key, n_chains, schedule):
    """Run n_chains parallel Gibbs chains; return positions array (n_chains, n_samples, N)."""
    free_blocks = model["free_blocks"]
    center, q, N = model["center"], model["q"], model["N"]
    readout = Block(model["residues"][1:])  # the free residues, in order 1..N-1
    clamp = [jnp.array([center], dtype=jnp.uint8)]  # constant residue-0 clamp, same for all chains

    k_init, k_run = jax.random.split(key)
    init = [
        jax.random.randint(k, (n_chains, 1), 0, q, dtype=jnp.uint8)
        for k in jax.random.split(k_init, len(free_blocks))
    ]
    keys = jax.random.split(k_run, n_chains)

    def one(init_c, k):
        return sample_states(k, model["prog"], schedule, init_c, clamp, [readout])

    states = jax.jit(jax.vmap(one))(init, keys)
    free_pos = states[0]  # (n_chains, n_samples, N-1)
    center_col = jnp.full((*free_pos.shape[:-1], 1), center, dtype=free_pos.dtype)
    pos = jnp.concatenate([center_col, free_pos], axis=-1)
    return np.asarray(pos)


# --------------------------------------------------------------------------- #
# Observables (numpy, vectorized over arbitrary leading batch axes)
# --------------------------------------------------------------------------- #
def _rc(model, pos):
    L = model["L"]
    pos = np.asarray(pos).astype(np.int64)  # cast: uint8 positions underflow on subtraction
    return pos // L, pos % L  # rows, cols, each same shape as pos (..., N)


def n_overlaps(model, pos):
    """Number of residue pairs sharing a site (excluded-volume violations)."""
    N = model["N"]
    out = np.zeros(pos.shape[:-1], dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            out += (pos[..., i] == pos[..., j]).astype(int)
    return out


def n_broken_bonds(model, pos):
    """Number of consecutive residues that are NOT lattice-adjacent."""
    r, c = _rc(model, pos)
    man = np.abs(np.diff(r, axis=-1)) + np.abs(np.diff(c, axis=-1))
    return (man != 1).sum(-1)


def n_contacts(model, pos):
    """Number of non-consecutive H-H topological contacts."""
    seq, N = model["seq"], model["N"]
    r, c = _rc(model, pos)
    out = np.zeros(pos.shape[:-1], dtype=int)
    for i in range(N):
        for j in range(i + 2, N):
            if seq[i] == "H" and seq[j] == "H":
                man = np.abs(r[..., i] - r[..., j]) + np.abs(c[..., i] - c[..., j])
                out += (man == 1).astype(int)
    return out


def is_valid(model, pos):
    """True where the conformation is a self-avoiding walk (no overlaps, all bonds intact)."""
    return (n_overlaps(model, pos) == 0) & (n_broken_bonds(model, pos) == 0)


def radius_of_gyration(model, pos):
    r, c = _rc(model, pos)
    rc = np.stack([r, c], axis=-1).astype(float)  # (..., N, 2)
    com = rc.mean(axis=-2, keepdims=True)
    return np.sqrt(((rc - com) ** 2).sum(-1).mean(-1))


def energy_fp(model, pos, beta=None):
    """First-principles THRML energy E = lam*(overlaps + broken) - beta*eps*contacts.

    Independent of the weight matrices, so agreement with `energy_from_weights`
    validates the matrix construction. Backbone penalty (lam) is temperature-
    independent; only the contact term carries beta.
    """
    beta = model["beta"] if beta is None else beta
    return (
        model["lam"] * (n_overlaps(model, pos) + n_broken_bonds(model, pos))
        - beta * model["eps"] * n_contacts(model, pos)
    )


def contact_energy(model, pos):
    """The thermal (physical) energy whose moments give the specific heat: E_c = -eps * contacts."""
    return -model["eps"] * n_contacts(model, pos)


def energy_from_weights(model, pos):
    """Energy reconstructed from the THRML weight matrices: E = -sum_p W[p, x_head, x_tail]."""
    w, h, t = model["weights"], model["heads"], model["tails"]
    E = np.zeros(pos.shape[:-1], dtype=np.float64)
    for p in range(len(h)):
        E -= w[p, pos[..., h[p]], pos[..., t[p]]]
    return E


# --------------------------------------------------------------------------- #
# Ground truth for validation
# --------------------------------------------------------------------------- #
def enumerate_states(model):
    """All assignments of residues 1..N-1 over the q sites (residue 0 = center).

    Only tractable for tiny N and q (q^(N-1) states). Returns array (#states, N).
    """
    N, q, center = model["N"], model["q"], model["center"]
    free = np.array(list(itertools.product(range(q), repeat=N - 1)), dtype=np.int64)
    col0 = np.full((free.shape[0], 1), center, dtype=np.int64)
    return np.concatenate([col0, free], axis=1)


def saw_ground_state(seq: str):
    """Exact maximum number of H-H contacts (and its degeneracy) over all SAWs.

    Brute-force DFS on the infinite 2D lattice with residue 0 at the origin and
    the first bond fixed to +x (removes the 4-fold rotation). Tractable for N up
    to ~14. Returns (max_contacts, degeneracy_up_to_fixed_first_bond).
    """
    N = len(seq)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    path = [(0, 0), (1, 0)]
    occ = {(0, 0), (1, 0)}
    best = {"c": -1, "deg": 0}

    def count_contacts():
        c = 0
        for i in range(N):
            for j in range(i + 2, N):
                if seq[i] == "H" and seq[j] == "H":
                    dx = abs(path[i][0] - path[j][0])
                    dy = abs(path[i][1] - path[j][1])
                    if dx + dy == 1:
                        c += 1
        return c

    def dfs(i):
        if i == N:
            c = count_contacts()
            if c > best["c"]:
                best["c"], best["deg"] = c, 1
            elif c == best["c"]:
                best["deg"] += 1
            return
        x, y = path[-1]
        for dx, dy in moves:
            nx_, ny_ = x + dx, y + dy
            if (nx_, ny_) not in occ:
                occ.add((nx_, ny_))
                path.append((nx_, ny_))
                dfs(i + 1)
                path.pop()
                occ.remove((nx_, ny_))

    if N == 1:
        return 0, 1
    dfs(2)
    return best["c"], best["deg"]


def saw_density_of_states(seq: str):
    """Histogram g[c] = number of SAWs (first bond fixed +x) with c H-H contacts.

    Enumerates every self-avoiding walk and bins by contact count, giving the exact
    canonical thermodynamics for any temperature via P(c) proportional to g[c]*exp(beta*eps*c).
    Tractable for N up to ~13. Returns a float array g with g[c] = count.
    """
    from collections import Counter

    N = len(seq)
    H = [c == "H" for c in seq]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    path = [(0, 0), (1, 0)]
    occ = {(0, 0), (1, 0)}
    g = Counter()

    def contacts():
        cc = 0
        for i in range(N):
            if not H[i]:
                continue
            for j in range(i + 2, N):
                if H[j] and abs(path[i][0] - path[j][0]) + abs(path[i][1] - path[j][1]) == 1:
                    cc += 1
        return cc

    def dfs(i):
        if i == N:
            g[contacts()] += 1
            return
        x, y = path[-1]
        for dx, dy in moves:
            p = (x + dx, y + dy)
            if p not in occ:
                occ.add(p)
                path.append(p)
                dfs(i + 1)
                path.pop()
                occ.discard(p)

    dfs(2)
    arr = np.zeros(max(g) + 1, dtype=np.float64)
    for c, n in g.items():
        arr[c] = n
    return arr


def box_saw_density_of_states(seq: str, L: int):
    """g[c] over SAWs that fit the L x L box with residue 0 clamped at the center.

    This is the exact ensemble THRML samples (same box + clamp), so it is the correct
    ground truth for the calibration -- unlike the infinite-lattice DOS, it includes
    the box's exclusion of extended conformations. Contacts are counted incrementally
    for speed. Tractable for small N and L.
    """
    from collections import Counter

    N = len(seq)
    H = [c == "H" for c in seq]
    c0 = L // 2
    occ = {(c0, c0): 0}
    path = [(c0, c0)]
    g = Counter()

    def dfs(i, contacts):
        if i == N:
            g[contacts] += 1
            return
        x, y = path[-1]
        for p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            px, py = p
            if 0 <= px < L and 0 <= py < L and p not in occ:
                add = 0
                if H[i]:
                    for nb in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                        j = occ.get(nb, -1)
                        if 0 <= j <= i - 2 and H[j]:  # placed, non-consecutive, H
                            add += 1
                occ[p] = i
                path.append(p)
                dfs(i + 1, contacts + add)
                path.pop()
                del occ[p]

    dfs(1, 0)
    arr = np.zeros(max(g) + 1, dtype=np.float64)
    for c, n in g.items():
        arr[c] = n
    return arr
