#!/usr/bin/env python3
"""
Test suite for affine gap alignment.

Every property that belongs to the algorithm is asserted against each backend, so the NumPy
reference and the compiled kernels are held to one standard rather than compared after the fact.
A test opts into that axis by naming a `backend` argument; `conftest.py` supplies the values and
skips the ones the machine cannot serve.

The suite leans on three kinds of oracle:

- __Self-consistency__, where the traceback's score must equal the score-only kernel's.
- __Cross-implementation__, where every backend must return what the reference returns.
- __External__, where BioPython and brute-force enumeration answer the same question independently.

The third kind matters most: the first two would agree with each other while both being wrong.
"""

import os
from itertools import combinations, product
from random import choice, randint

import pytest
from Bio import Align
from Bio.Align import substitution_matrices

import numpy as np

import affinegaps
from affinegaps import (
    needleman_wunsch_gotoh_alignment,
    needleman_wunsch_gotoh_score,
    smith_waterman_gotoh_alignment,
    smith_waterman_gotoh_score,
    levenshtein_alignment,
    default_proteins_alphabet,
    AffineGapCosts,
    UniformSubstitutionCosts,
    TabulatedSubstitutionCosts,
)

MODES = ["global", "local"]

# The substitution table is BLOSUM62 scaled by five, so anything compared against it must be
# scaled to match or the comparison is vacuous.
BLOSUM_SCALE = 5


def costs(match: int, mismatch: int, open: int, extend: int) -> dict:
    """The two cost records as keyword arguments, so a test can splat one scoring into any call."""
    return {
        "substitution": UniformSubstitutionCosts(match=match, mismatch=mismatch),
        "gaps": AffineGapCosts(open=open, extend=extend),
    }


# One representative scoring per regime rather than a grid: a cheap gap, an expensive one, and
# unit costs. The grids they replace were overlapping draws from the same family.
SCORINGS = [
    pytest.param(costs(5, -4, -20, -1), id="expensive-gap"),
    pytest.param(costs(2, -1, -2, -1), id="cheap-gap"),
    pytest.param(costs(0, -1, -1, -1), id="unit-cost"),
]
SCORING = SCORINGS[0].values[0]


def aligner_for(mode: str):
    """The alignment entry point for a mode."""
    return needleman_wunsch_gotoh_alignment if mode == "global" else smith_waterman_gotoh_alignment


def scorer_for(mode: str):
    """The score-only entry point for a mode."""
    return needleman_wunsch_gotoh_score if mode == "global" else smith_waterman_gotoh_score


def random_pair(shortest: int = 5, longest: int = 25, alphabet: str = default_proteins_alphabet):
    """Two independent random sequences over the default alphabet."""
    return (
        "".join(choice(alphabet) for _ in range(randint(shortest, longest))),
        "".join(choice(alphabet) for _ in range(randint(shortest, longest))),
    )


def rescore(first: str, second: str, scoring: dict | None = None) -> int:
    """Scores a gapped pair under the affine rule the recurrence claims to optimize.

    Independent of the dynamic programming, so it catches a traceback that returns a path the
    score never took.
    """
    scoring = scoring or SCORING
    total, in_first, in_second = 0, False, False
    for left, right in zip(first, second, strict=True):
        if left == "-" and right == "-":
            continue
        if left == "-":
            total += scoring["gaps"].extend if in_first else scoring["gaps"].open
            in_first, in_second = True, False
        elif right == "-":
            total += scoring["gaps"].extend if in_second else scoring["gaps"].open
            in_first, in_second = False, True
        else:
            total += scoring["substitution"].match if left == right else scoring["substitution"].mismatch
            in_first = in_second = False
    return total


# region Backend Axis

# Keyed by both axes. Naming a key just "gpu" would be ambiguous the moment a second backend grows
# a device path, and Numba already has one in `numba.cuda`.
ALL_BACKENDS = {
    "python-cpu": ("python", "cpu"),
    "numba-cpu": ("numba", "cpu"),
    "mojo-cpu": ("mojo", "cpu"),
    "mojo-gpu": ("mojo", "gpu"),
}


def requested_backends() -> list:
    """Which backends to exercise, narrowed by `AFFINEGAPS_BACKENDS` when it is set.

    An environment variable rather than a command-line option, because `pytest_addoption` is only
    honoured from a `conftest.py` and this suite is one file. Narrowing is rarely needed anyway:
    a backend the machine cannot serve is skipped by the probe below without being asked.
    """
    names = [n.strip() for n in os.environ.get("AFFINEGAPS_BACKENDS", "").split(",") if n.strip()]
    if not names:
        return list(ALL_BACKENDS)
    unknown = set(names) - set(ALL_BACKENDS)
    if unknown:
        raise pytest.UsageError(f"Unknown backend(s): {', '.join(sorted(unknown))}")
    return names


def pytest_generate_tests(metafunc):
    """Supplies the backend axis to any test that names it."""
    for argument, choices in (
        ("backend", requested_backends()),
        ("compiled_backend", [n for n in requested_backends() if not n.startswith("python")]),
    ):
        if argument in metafunc.fixturenames:
            metafunc.parametrize(argument, choices, indirect=True)


def _scoring_keywords(name: str) -> dict:
    """The keywords selecting one backend, skipping when this machine cannot serve it."""
    backend, device = ALL_BACKENDS[name]
    if not affinegaps.available(backend, device):
        pytest.skip(f"the {name} backend does not run here; see the README for how to build it")
    return {"backend": backend, "device": device}


@pytest.fixture
def backend(request):
    """Scoring keywords for one backend, across every implementation."""
    return _scoring_keywords(request.param)


@pytest.fixture
def compiled_backend(request):
    """The same, restricted to the compiled backends."""
    return _scoring_keywords(request.param)


# endregion Backend Axis


# region Algorithm Properties


@pytest.mark.repeat(10)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_score_matches_alignment(backend, mode: str, scoring: dict):
    """The traceback's score must equal what the score-only kernel computes.

    The two are separate entry points in every compiled backend, so this is the invariant a kernel
    can break on its own without any cross-backend comparison noticing.
    """
    first, second = random_pair()
    assert aligner_for(mode)(first, second, **scoring, **backend)[2] == scorer_for(mode)(
        first, second, **scoring, **backend
    )


@pytest.mark.repeat(20)
@pytest.mark.parametrize("mode", MODES)
def test_alignment_achieves_its_score(backend, mode: str):
    """Every returned path must be well formed and must realize the score reported beside it.

    With affine gaps neither is automatic. A walk that reads only the winning operation at each
    cell can leave a gap run and re-enter it, paying a second opening penalty the score never did;
    and a local walk that flushes what it did not trace returns flanking sequence that was never
    part of the alignment.
    """
    first, second = random_pair()
    gapped_first, gapped_second, score = aligner_for(mode)(first, second, **SCORING, **backend)
    assert len(gapped_first) == len(gapped_second)
    assert gapped_first.replace("-", "") in first
    assert gapped_second.replace("-", "") in second
    assert rescore(gapped_first, gapped_second) == score


@pytest.mark.repeat(20)
def test_symmetry(backend):
    """Swapping the arguments must not change the score."""
    first, second = random_pair()
    assert needleman_wunsch_gotoh_score(first, second, **SCORING, **backend) == (
        needleman_wunsch_gotoh_score(second, first, **SCORING, **backend)
    )


@pytest.mark.repeat(20)
def test_against_levenshtein(backend):
    """At unit costs the Gotoh recurrence must reduce to edit distance.

    A second algorithm answering the same question, which pins the recurrence where a cross-backend
    comparison cannot: both backends could agree and both be wrong.
    """
    first, second = random_pair(3, 15)
    distance = levenshtein_alignment(first, second)[2]
    unit = costs(0, -1, -1, -1)
    assert -needleman_wunsch_gotoh_score(first, second, **unit, **backend) == distance


@pytest.mark.repeat(8)
def test_gap_expansions(backend):
    """A gap that costs nothing to extend must cost the same however wide it is.

    The filler is `W`, which the default alphabet carries so the test stays on every backend, and
    the scoring makes opening a gap cheaper than a mismatch so the filler is always gapped rather
    than mismatched. Without that precondition the recurrence may prefer to mismatch the filler and
    the width legitimately changes the score — which is a property of the scoring, not a bug.
    """
    free_extension = costs(5, -10, -1, 0)
    first, second = random_pair(5, 15, alphabet="ACGT")
    cut = len(second) // 2

    widths = {}
    for width in range(1, 6):
        widened = second[:cut] + "W" * width + second[cut:]
        widths[width] = needleman_wunsch_gotoh_score(first, widened, **free_extension, **backend)
    assert len(set(widths.values())) == 1, f"a free-extension gap changed price with its width: {widths}"


@pytest.mark.parametrize("mode", MODES)
def test_scores_match_brute_force_enumeration(backend, mode: str):
    """Checks the recurrence against enumerating every alignment, using no dynamic programming.

    The strongest oracle in the suite, and the only one that can catch a recurrence which is
    self-consistently wrong across every backend at once.
    """

    def best_global(first: str, second: str) -> int:
        rows, columns = len(first), len(second)
        if not rows and not columns:
            return 0
        best = None
        for length in range(max(rows, columns), rows + columns + 1):
            for first_gaps in combinations(range(length), length - rows):
                for second_gaps in combinations(range(length), length - columns):
                    if set(first_gaps) & set(second_gaps):
                        continue
                    top, bottom = [], []
                    taken_first = taken_second = 0
                    for position in range(length):
                        if position in first_gaps:
                            top.append("-")
                        else:
                            top.append(first[taken_first])
                            taken_first += 1
                        if position in second_gaps:
                            bottom.append("-")
                        else:
                            bottom.append(second[taken_second])
                            taken_second += 1
                    candidate = rescore("".join(top), "".join(bottom))
                    if best is None or candidate > best:
                        best = candidate
        return best if best is not None else 0

    def brute_optimal(first: str, second: str) -> int:
        if mode == "global":
            return best_global(first, second)
        best = 0
        for start_first in range(len(first) + 1):
            for stop_first in range(start_first, len(first) + 1):
                for start_second in range(len(second) + 1):
                    for stop_second in range(start_second, len(second) + 1):
                        piece_first, piece_second = first[start_first:stop_first], second[start_second:stop_second]
                        if piece_first and piece_second:
                            best = max(best, best_global(piece_first, piece_second))
        return best

    scorer = scorer_for(mode)
    for length in range(4):
        for first in ["".join(t) for t in product("AR", repeat=length)] or [""]:
            for second_length in range(4):
                for second in ["".join(t) for t in product("AR", repeat=second_length)] or [""]:
                    if len(first) + len(second) <= 5:
                        assert scorer(first, second, **SCORING, **backend) == brute_optimal(first, second)


# endregion Algorithm Properties

# region External Oracles


def biopython_aligner(mode: str, scoring: dict):
    """A BioPython aligner scaled to match our table, so the comparison can actually fail.

    Our default matrix is BLOSUM62 multiplied by five. Handing BioPython the unscaled matrix while
    giving both the same gap penalties makes our score larger by construction, which is how the
    comparison this replaces could never fail.
    """
    aligner = Align.PairwiseAligner(mode=mode)
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62") * BLOSUM_SCALE
    aligner.open_gap_score = scoring["gaps"].open
    aligner.extend_gap_score = scoring["gaps"].extend
    return aligner


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "pair",
    [
        ("GIVEQCCTSICSLYQLENYCN", "HSQGTFTSDYSKYLDSRAEQDFV"),
        ("MSTAVLENPGLGRKLSDFGQETSYIEDNC", "MSTAVLENPGLGRKLSDFGQETSYIEDNS"),
        ("ACGTACGTACGT", "ACGTCGTACGTA"),
        ("W", "W"),
        ("WWWWW", "W"),
        ("ARNDCQEGHILKMFPSTWYV", "VYWTSPFMKLIHGEQCDNRA"),
    ],
)
def test_against_biopython(backend, mode: str, pair: tuple):
    """Our score must equal BioPython's on the same matrix and the same penalties.

    Scaled to match, this is an equality rather than an inequality, so a change that inflates our
    scores now fails here instead of passing silently.
    """
    gaps = {"gaps": AffineGapCosts(open=-20, extend=-1)}
    first, second = pair
    expected = biopython_aligner(mode, gaps).score(first, second)
    assert scorer_for(mode)(first, second, **gaps, **backend) == expected


@pytest.mark.repeat(10)
@pytest.mark.parametrize("mode", MODES)
def test_against_biopython_fuzzy(backend, mode: str):
    """The same equality on random proteins rather than curated pairs."""
    gaps = {"gaps": AffineGapCosts(open=-20, extend=-1)}
    first, second = random_pair(10, 40)
    expected = biopython_aligner(mode, gaps).score(first, second)
    assert scorer_for(mode)(first, second, **gaps, **backend) == expected


# endregion External Oracles

# region Compiled Backends


@pytest.mark.repeat(15)
@pytest.mark.parametrize("mode", MODES)
def test_matches_reference(compiled_backend, mode: str):
    """A compiled backend must return exactly what the reference returns, strings included."""
    first, second = random_pair()
    assert aligner_for(mode)(first, second, **SCORING, **compiled_backend) == aligner_for(mode)(
        first, second, **SCORING, backend="python"
    )


@pytest.mark.repeat(10)
@pytest.mark.parametrize("mode", MODES)
def test_linear_space_matches_stored(compiled_backend, mode: str, monkeypatch):
    """The linear-space traceback must reach the same answer as a stored decision matrix.

    Both are compiled paths, so the budget constant is the seam that forces each; there is no
    public knob and the two are otherwise indistinguishable from outside.
    """
    first, second = random_pair()
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 10**12)
    stored = aligner_for(mode)(first, second, **SCORING, **compiled_backend)
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 0)
    linear = aligner_for(mode)(first, second, **SCORING, **compiled_backend)
    assert linear == stored


def test_linear_space_beats_the_stored_limit(compiled_backend, monkeypatch):
    """Linear space must carry a pair far past what a stored matrix could hold."""
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 0)
    alphabet = default_proteins_alphabet
    first = "".join(choice(alphabet) for _ in range(3000))
    second = "".join(choice(alphabet) for _ in range(3000))
    gapped_first, gapped_second, score = needleman_wunsch_gotoh_alignment(first, second, **SCORING, **compiled_backend)
    assert gapped_first.replace("-", "") == first
    assert gapped_second.replace("-", "") == second
    assert rescore(gapped_first, gapped_second) == score


@pytest.mark.parametrize("mode", MODES)
def test_batch_matches_single_pair(mode: str):
    """The batched entry points must reproduce the reference pair by pair."""
    pairs = [random_pair(5, 40) for _ in range(24)]
    firsts, seconds = [a for a, _ in pairs], [b for _, b in pairs]
    batched = getattr(affinegaps, f"{'needleman_wunsch' if mode == 'global' else 'smith_waterman'}_gotoh_alignments")
    produced = batched(firsts, seconds, **SCORING)
    expected = [aligner_for(mode)(a, b, **SCORING, backend="python") for a, b in pairs]
    assert produced == expected


def test_rejects_what_it_cannot_do():
    """The dispatcher must refuse rather than quietly doing something else."""
    with pytest.raises(ValueError):
        needleman_wunsch_gotoh_score("AR", "RA", backend="python", device="gpu")
    with pytest.raises(ValueError):
        needleman_wunsch_gotoh_score("AR", "RA", backend="CPU")
    if affinegaps.available("mojo", "cpu"):
        with pytest.raises(NotImplementedError):
            needleman_wunsch_gotoh_score(
                "AR",
                "RA",
                substitution=TabulatedSubstitutionCosts("AR", np.zeros((2, 2), dtype=np.int8)),
                backend="mojo",
            )


def test_costs_cannot_express_a_contradiction():
    """The pairing rules that once needed runtime checks are carried by the types."""
    with pytest.raises(TypeError):
        UniformSubstitutionCosts(match=5)  # a uniform cost cannot omit half of itself
    with pytest.raises(ValueError):
        AffineGapCosts(open=-1, extend=-20)  # a gap that costs less to open than to extend


# endregion Compiled Backends
