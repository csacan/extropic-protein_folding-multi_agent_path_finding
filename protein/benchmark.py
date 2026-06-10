"""Canonical 2D HP benchmark sequences (square lattice) with published optimal energies.

Standard instances used across the HP-folding literature (Unger-Moult / Dill lineage;
values cross-checked against the branch-and-bound table in Mann et al., PMC5172541,
and review tabulations). E_opt is the optimal number of non-consecutive H-H contacts
(reported as -E* in energy units). We VERIFY the small instances by confirming our
THRML annealer reaches E_opt in the clamped box; large instances rely on the literature
value (with the check that we never EXCEED it).

Sources:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5172541/   (sequences + E*)
"""

from __future__ import annotations

# name -> (sequence, optimal_contacts)
BENCHMARKS = {
    "2d20": ("HPHPPHHPHPPHPHHPPHPH", 9),
    "2d24": ("HHPPHPPHPPHPPHPPHPPHPPHH", 9),
    "2d25": ("PPHPPHHPPPPHHPPPPHHPPPPHH", 8),
    "2d36": ("PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", 14),
    "2d48": ("PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH", 23),
    "2d50": ("PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH", 21),
    "2d60": ("PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP", 36),
    "2d64": ("HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH", 42),
}

# declared lengths, asserted against the strings to catch any transcription error
LENGTHS = {"2d20": 20, "2d24": 24, "2d25": 25, "2d36": 36,
           "2d48": 48, "2d50": 50, "2d60": 60, "2d64": 64}


def check_lengths():
    for name, (seq, _) in BENCHMARKS.items():
        n = len(seq)
        assert n == LENGTHS[name], f"{name}: length {n} != declared {LENGTHS[name]}"
    return True


if __name__ == "__main__":
    check_lengths()
    print(f"{'name':>6} {'N':>3} {'nH':>3} {'E_opt':>5}  sequence")
    for name, (seq, e) in BENCHMARKS.items():
        print(f"{name:>6} {len(seq):>3} {seq.count('H'):>3} {e:>5}  {seq}")
    print("\nall length checks passed.")
