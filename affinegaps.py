#!/usr/bin/env python3
"""
Affine Gaps Alignment Toolkit

This single-file library and CLI tool provides robust implementations of sequence alignment algorithms,
including Needleman-Wunsch, Smith-Waterman, and Levenshtein, with support for affine gap penalties.
The toolkit is designed for both programmatic use and command-line operation, making it versatile
for bio-informatics, computational biology, and general sequence alignment tasks.

Key Features:
- Global alignment (Needleman-Wunsch with Gotoh extensions)
- Local alignment (Smith-Waterman with Gotoh extensions)
- Edit distance computation (Levenshtein)
- Customizable substitution matrices and gap penalties
- Optimized for performance with optional NumBa acceleration
- CLI for quick alignment and scoring of sequences

Usage:
1. Library:
    Import and use the library functions for programmatic sequence alignment:
    >>> from affinegaps import needleman_wunsch_gotoh_alignment
    >>> align1, align2, score = needleman_wunsch_gotoh_alignment("GATTACA", "GCATGCU")
    >>> print("Alignment 1:", align1)
    >>> print("Alignment 2:", align2)
    >>> print("Score:", score)

2. CLI:
    Use the tool directly from the command line for quick alignment tasks:
    $ affinegaps GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV --local
    Sequence 1: GATTACA
    Sequence 2: GCATGCU

    Alignment 1: G-ATTACA
    Alignment 2: GCAT-GCU
    Score:       3

Dependencies:
- Python 3.12+
- NumPy (required)
- NumBa (optional, for acceleration)
- colorama (optional, for colored CLI output)

Author: Ash Vardanian
License: Apache 2.0
"""

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from functools import cache, lru_cache
from typing import Any, Literal, cast
from collections.abc import Callable

import numpy as np

# NumBa is a heavy dependency, that we may not want to avoid
try:
    import numba as nb

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

__version__ = "0.2.5"


# region Backends

# Above this many cells the traceback switches to the linear-space recursion, which produces
# identical output. Five matrices at seventeen bytes a cell keeps a stored alignment near 100 MB.
_STORED_MATRIX_BUDGET = 6_000_000


class Algorithm(StrEnum):
    """The four Gotoh entry points, named by what they compute rather than by symbol."""

    GLOBAL_SCORE = "global-score"
    GLOBAL_ALIGNMENT = "global-alignment"
    LOCAL_SCORE = "local-score"
    LOCAL_ALIGNMENT = "local-alignment"


class Result(StrEnum):
    """Whether an entry point returns a number or a pair of gapped strings and a number."""

    SCORE = "score"
    ALIGNMENT = "alignment"


class Mode(StrEnum):
    """Which of the two alignment problems the recurrence solves."""

    GLOBAL = "global"
    LOCAL = "local"


@dataclass(frozen=True)
class _Spec:
    """Everything the dispatcher needs to serve one algorithm on any backend.

    `reference` is the public function, so the batch path never looks a name up in module globals.
    """

    reference: Callable
    result: Result
    mode: Mode


# The compiled module exports one entry point per result shape; every other axis is an argument.
_COMPILED_ENTRY = {Result.SCORE: "gotoh_scores", Result.ALIGNMENT: "gotoh_alignments"}


@lru_cache(maxsize=1)
def _mojo_backend():
    """Imports the compiled module, looking in the local build directory as a fallback.

    Keeping this in one place means callers never manipulate `sys.path` themselves, and a project
    checkout behaves the same as an installed package.
    """
    try:
        import affinegaps_mojo
    except ImportError:
        build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        if build not in sys.path:
            sys.path.insert(0, build)
        try:
            import affinegaps_mojo
        except ImportError:
            return None
    return affinegaps_mojo


@cache
def available(backend: str = "mojo", device: str = "cpu") -> bool:
    """Whether a backend and device combination actually runs, tried once and remembered.

    This executes a tiny alignment rather than inferring from an import, because a built extension
    on a machine with an unsupported driver imports cleanly and then fails at every device call.
    """
    if backend == "python":
        return device == "cpu"
    if backend == "numba":
        return device == "cpu" and HAS_NUMBA
    if _mojo_backend() is None:
        return False
    try:
        needleman_wunsch_gotoh_score("AR", "RA", backend=cast(Any, backend), device=cast(Any, device))
    except Exception:
        return False
    return True


def _kernel(function, backend):
    """The kernel body a backend asks for: NumBa's compiled dispatcher, or the Python it wrapped."""
    return function if backend == "numba" else getattr(function, "py_func", function)


def _resolve_for(substitution, backend, device):
    """Resolves the backend for a call, knowing what the caller wants scored.

    An unnamed backend adapts, and a table the compiled kernels do not carry is a reason to
    adapt away from them. A named backend is still honoured or refused, never quietly swapped.
    """
    if backend is None and _custom_scoring(substitution):
        backend = "numba" if HAS_NUMBA else "python"
    return _resolve(backend, device)


def _resolve(backend, device):
    """Validates a backend and device pair, filling in whichever the caller left unspecified.

    Unspecified adapts to the machine; specified is honoured or refused. Falling back silently
    would let a benchmark report the GPU while measuring the reference.
    """
    if backend is None and device is None:
        for candidate in (("mojo", "gpu"), ("mojo", "cpu"), ("numba", "cpu")):
            if available(*candidate):
                return candidate
        return ("python", "cpu")
    if backend is None:
        backend = "mojo"
        if device == "cpu" and not available("mojo", "cpu"):
            backend = "numba" if HAS_NUMBA else "python"
    if device is None:
        device = "gpu" if backend == "mojo" and available("mojo", "gpu") else "cpu"

    if backend in ("python", "numba"):
        if device != "cpu":
            raise ValueError(f"The {backend} backend runs on the CPU only")
        if backend == "numba" and not HAS_NUMBA:
            raise RuntimeError("NumBa is not installed. Install the `numba` extra.")
        return (backend, device)
    if backend != "mojo" or device not in ("cpu", "gpu"):
        raise ValueError(f"Unknown backend {backend!r} or device {device!r}")
    if _mojo_backend() is None:
        raise RuntimeError("The Mojo backend is not built. See the README for how to build it.")
    return (backend, device)


def _custom_scoring(substitution) -> bool:
    """Whether the caller supplied a table the compiled backend does not carry.

    The compiled kernels hold the default matrix and can build a uniform one, so only an explicit
    table is out of reach.
    """
    return isinstance(substitution, TabulatedSubstitutionCosts)


def _reject_custom_scoring(algorithm, backend, options):
    """The compiled backends carry only the default table, plus a uniform match/mismatch pair."""
    if _custom_scoring(options.get("substitution")):
        raise NotImplementedError(f"{algorithm} on the {backend} backend needs match/mismatch or the default matrix")


def _compiled_call(algorithm, backend, device, first, second, options):
    """Routes one pair to the compiled backend, where a single pair is a batch of one."""
    return _compiled_batch(algorithm, backend, device, [first], [second], options)[0]


def _compiled_batch(algorithm, backend, device, firsts, seconds, options):
    """Routes a whole batch, which is what the device is for.

    The compiled side chooses between the stored and the linear traceback from `stored_budget`,
    so the only axes crossing the boundary are the ones the caller actually named.
    """
    _reject_custom_scoring(algorithm, backend, options)
    spec = _ALGORITHMS[algorithm]
    entry = getattr(_mojo_backend(), _COMPILED_ENTRY[spec.result])
    substitution = options.get("substitution")
    gaps = options.get("gaps") or AffineGapCosts()
    if spec.result is Result.SCORE:
        return list(entry(list(firsts), list(seconds), spec.mode, device, substitution, gaps))
    outcome = entry(list(firsts), list(seconds), spec.mode, device, substitution, gaps, _STORED_MATRIX_BUDGET)
    return [tuple(row) for row in outcome]


def _batch(algorithm, backend, device, firsts, seconds, **options):
    """One batch entry point, on whichever backend resolves."""
    backend, device = _resolve_for(options.get("substitution"), backend, device)
    if device == "cpu":
        single = _ALGORITHMS[algorithm].reference
        return [single(a, b, backend=backend, device=device, **options) for a, b in zip(firsts, seconds, strict=True)]
    return _compiled_batch(algorithm, backend, device, firsts, seconds, options)


def needleman_wunsch_gotoh_alignments(firsts, seconds, *, backend=None, device=None, **options):
    """Globally aligns a whole batch, one thread block per pair on the device."""
    return _batch(Algorithm.GLOBAL_ALIGNMENT, backend, device, firsts, seconds, **options)


def smith_waterman_gotoh_alignments(firsts, seconds, *, backend=None, device=None, **options):
    """Locally aligns a whole batch, one thread block per pair on the device."""
    return _batch(Algorithm.LOCAL_ALIGNMENT, backend, device, firsts, seconds, **options)


def needleman_wunsch_gotoh_scores(firsts, seconds, *, backend=None, device=None, **options):
    """Scores a whole batch globally, without reconstructing the alignments."""
    return _batch(Algorithm.GLOBAL_SCORE, backend, device, firsts, seconds, **options)


def smith_waterman_gotoh_scores(firsts, seconds, *, backend=None, device=None, **options):
    """Scores a whole batch locally, without reconstructing the alignments."""
    return _batch(Algorithm.LOCAL_SCORE, backend, device, firsts, seconds, **options)


# endregion Backends


def jit_if_available(*jit_args, **jit_kwargs):
    """Compiles with NumBa when it is installed, and leaves the function untouched when it is not."""

    def decorator(func):
        if HAS_NUMBA:
            # Cached to disk, so only the first process on a machine pays the compile.
            return nb.jit(*jit_args, cache=True, nogil=True, **jit_kwargs)(func)
        return func

    return decorator


# Constants for operation codes
MATCH, INSERT, DELETE, SUBSTITUTE = 0, 1, 2, 3

# By default, we use BLOSUM62 with affine gap penalties
default_proteins_alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX"
# fmt: off
default_proteins_matrix = (
    np.array(
        [
            4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4,
            -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4,
            -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4,
            -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4,
            0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,
            -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4,
            -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
            0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4,
            -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4,
            -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4,
            -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4,
            -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4,
            -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4,
            -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4,
            -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,
            1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4,
            0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4,
            -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4,
            -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,
            0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4,
            -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4,
            -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
            0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4,
            -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1,
        ],
        dtype=np.int8,
    ).reshape(24, 24)
    * 5
)
# fmt: on


@dataclass(frozen=True)
class AffineGapCosts:
    """Gotoh's two-parameter gap model. Both penalties are negative."""

    open: int = -4 * 5
    extend: int = int(-0.2 * 5)

    def __post_init__(self):
        # Checked here rather than per call, so no backend can be handed a non-affine recurrence.
        if self.open > self.extend:
            raise ValueError("Opening a gap must not cost less than extending it.")


@dataclass(frozen=True)
class UniformSubstitutionCosts:
    """One score for equal symbols and one for unequal, over any alphabet."""

    match: int
    mismatch: int


@dataclass(frozen=True)
class TabulatedSubstitutionCosts:
    """A substitution matrix and the alphabet that indexes it."""

    alphabet: str
    matrix: np.ndarray


# A closed pair rather than four loose parameters, so "a match score and a matrix" cannot be said.
SubstitutionCosts = UniformSubstitutionCosts | TabulatedSubstitutionCosts


def _reconstruct_alignment(
    changes: np.ndarray,
    scores: np.ndarray,
    deletes: np.ndarray,
    inserts: np.ndarray,
    seq1: np.ndarray,
    seq2: np.ndarray,
    open: int,
    extend: int,
    code_to_char: Callable,
    should_continue: Callable,
    flush_prefixes: bool = True,
) -> tuple[str, str]:
    """Walks the three layers back, so a gap run is never charged twice.

    Reading only `changes` conflates the best move at a cell with whether a gap run is still open,
    which can split one run in two and pay a second opening penalty. Consulting `deletes` and
    `inserts` keeps the walk in the layer it entered until that layer says the run began.
    """

    align1, align2 = "", ""
    i, j = len(seq1), len(seq2)
    state = MATCH

    # Backtrack to recover the alignment
    while should_continue(i, j):
        if state == DELETE:
            align1 += code_to_char(seq1[i - 1])
            align2 += "-"
            extends = deletes[i - 1, j] + extend > scores[i - 1, j] + open
            i -= 1
            state = DELETE if extends else MATCH
        elif state == INSERT:
            align1 += "-"
            align2 += code_to_char(seq2[j - 1])
            extends = inserts[i, j - 1] + extend > scores[i, j - 1] + open
            j -= 1
            state = INSERT if extends else MATCH
        elif changes[i, j] == DELETE:
            state = DELETE
        elif changes[i, j] == INSERT:
            state = INSERT
        else:  # MATCH or SUBSTITUTE
            align1 += code_to_char(seq1[i - 1])
            align2 += code_to_char(seq2[j - 1])
            i -= 1
            j -= 1

    # A global path must reach the origin, so whatever is left is genuinely aligned against gaps.
    # A local path stops wherever the score falls to zero, and everything before that is outside
    # the alignment entirely.
    if not flush_prefixes:
        return align1[::-1], align2[::-1]

    # Add remaining characters from `seq1` (with gaps in `seq2`)
    while i > 0:
        align1 += code_to_char(seq1[i - 1])
        align2 += "-"
        i -= 1

    # Add remaining characters from `seq2` (with gaps in `seq1`)
    while j > 0:
        align1 += "-"
        align2 += code_to_char(seq2[j - 1])
        j -= 1

    return align1[::-1], align2[::-1]


def _translate_sequence(seq: str, alphabet: str) -> np.ndarray:
    # def map_char(char):
    #     offset = alphabet.find(char)
    #     return offset if offset >= 0 else len(seq) - 1
    assert all(char in alphabet for char in seq), f"Found unknown character in sequence: {seq}"
    return np.array([alphabet.index(char) for char in seq], dtype=np.uint8)


def _validate_gotoh_arguments(
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
) -> tuple[str, np.ndarray, int, int]:
    """Resolves the two cost records into the alphabet, table and penalties a kernel wants.

    There is nothing left to validate. The pairing rules the flat parameters needed checking for
    are carried by the types: a uniform cost cannot omit half of itself, and it cannot also be a
    table.
    """
    gaps = gaps or AffineGapCosts()
    if substitution is None:
        return default_proteins_alphabet, default_proteins_matrix, gaps.open, gaps.extend
    if isinstance(substitution, TabulatedSubstitutionCosts):
        return substitution.alphabet, substitution.matrix, gaps.open, gaps.extend

    width = len(default_proteins_alphabet)
    matrix = np.full((width, width), substitution.mismatch)
    matrix[np.diag_indices(width)] = substitution.match
    return default_proteins_alphabet, matrix, gaps.open, gaps.extend


@jit_if_available(nopython=True)
def _levenshtein_alignment_kernel(seq1: np.ndarray, seq2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns two sequences using Levenshtein's algorithm.
    The returned distance is the minimum number of single-character edits,
    including insertions, deletions, and substitutions, required to change one
    sequence into the other.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Should be called through `levenshtein_alignment`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The matrices for alignment scoring:
        - scores: The primary scoring matrix.
        - changes: The matrix of enums, with cells equal to MATCH, INSERT, DELETE, or SUBSTITUTE.
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    # Initialize the scoring matrix
    scores[0, 0] = 0
    for i in range(1, seq1_len + 1):
        scores[i, 0] = i
        changes[i, 0] = DELETE
    for j in range(1, seq2_len + 1):
        scores[0, j] = j
        changes[0, j] = INSERT

    # Fill the scoring matrix and track operations
    for i in range(1, seq1_len + 1):
        for j in range(1, seq2_len + 1):

            substitution = int(seq1[i - 1] != seq2[j - 1])

            delete = scores[i - 1, j] + 1
            insert = scores[i, j - 1] + 1
            replace = scores[i - 1, j - 1] + substitution
            score = min(replace, delete, insert)
            scores[i, j] = score

            # Determine the minimum cost operation, preserving the operation kind
            if score == replace:
                changes[i, j] = SUBSTITUTE if substitution else MATCH
            elif score == delete:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

    return scores, changes


def levenshtein_alignment(str1: str, str2: str) -> tuple[str, str, int]:
    """
    Aligns two sequences using Levenshtein's algorithm.
    The returned distance is the minimum number of single-character edits,
    including insertions, deletions, and substitutions, required to change one
    sequence into the other.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.

    Returns:
    Tuple[str, str, int]: The optimal alignment of the two sequences and the alignment score.
    """
    seq1 = np.array([ord(c) for c in str1], dtype=np.uint32)
    seq2 = np.array([ord(c) for c in str2], dtype=np.uint32)
    scores, changes = _levenshtein_alignment_kernel(seq1, seq2)
    align1, align2 = _reconstruct_alignment(
        changes, scores, scores, scores, seq1, seq2, 1, 1, chr, lambda i, j: i > 0 and j > 0
    )
    return align1, align2, int(scores[-1, -1])


@jit_if_available(nopython=True)
def _needleman_wunsch_gotoh_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    open: int,
    extend: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `needleman_wunsch_gotoh`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): A substitution matrix for scoring matches/mismatches.
    open (int): The penalty for opening a gap.
    extend (int): The penalty for extending a gap.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The matrices the traceback needs:
        - scores: The best score reaching each cell in any state.
        - changes: The operation that achieved it, one of MATCH, SUBSTITUTE, DELETE or INSERT.
        - deletes: The best score reaching each cell inside a deletion run.
        - inserts: The best score reaching each cell inside an insertion run.

    Example usage:
    >>> seq1 = np.array([1, 2, 3])  # Example sequence
    >>> seq2 = np.array([3, 2, 1])  # Example sequence
    >>> substitution_matrix = np.array([[...], [...], [...]])  # Example substitution matrix
    >>> open = 5
    >>> extend = 2
    >>> scores, changes, deletes, inserts = _needleman_wunsch_gotoh_kernel(seq1, seq2, matrix, opening, extension)
    >>> print("Optimal alignment score matrix:\n", scores)

    Notes:
    The basis and recurrence relations for the matrices are as follows:
    Basis:
    - scores[i, 0] = open + (i - 1) * extend
    - scores[0, j] = open + (j - 1) * extend
    - deletes[i, 0] = open + (i - 1) * extend
    - inserts[0, j] = open + (j - 1) * extend

    Recurrence:
    - deletes[i, j] = max(scores[i - 1, j] + open, deletes[i - 1, j] + extend)
    - inserts[i, j] = max(scores[i, j - 1] + open, inserts[i, j - 1] + extend)
    - match = scores[i - 1, j - 1] + substitution_matrix[(seq1[i - 1], seq2[j - 1])]
    - scores[i, j] = max(match, deletes[i, j], inserts[i, j])
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Initialize the scoring matrix, following the suggestions in the paper.
    # There:
    #
    #   v ~ is gap opening penalty (always non-negative in paper, opposite for us)
    #   u ~ is gap extension penalty (always non-positive in paper, opposite for us)
    #   w(k) = u * k + v
    #
    #   D(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n
    #   P(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with a deletion of at least one residue from A, such that A(m)
    #             is aligned with a gap symbol
    #   Q(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with an insertion of at least one residue from B, such that B(n)
    #             is aligned with a gap symbol
    #
    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    deletes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    inserts = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    scores[0, 0] = 0
    for j in range(1, seq2_len + 1):
        scores[0, j] = open + (j - 1) * extend
        deletes[0, j] = scores[0, j] + open + extend
        changes[0, j] = INSERT

    # Fill the scoring matrix
    for i in range(1, seq1_len + 1):
        scores[i, 0] = open + (i - 1) * extend
        inserts[i, 0] = scores[i, 0] + open + extend
        changes[i, 0] = DELETE

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(
                scores[i - 1, j] + open,
                deletes[i - 1, j] + extend,
            )
            insert = max(
                scores[i, j - 1] + open,
                inserts[i, j - 1] + extend,
            )
            replace = scores[i - 1, j - 1] + substitution
            score = max(replace, delete, insert)
            scores[i, j] = score
            deletes[i, j] = delete
            inserts[i, j] = insert

            # Track changes
            if score == replace:
                changes[i, j] = MATCH if seq1[i - 1] == seq2[j - 1] else SUBSTITUTE
            elif score == delete:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

    return scores, changes, deletes, inserts


def needleman_wunsch_gotoh_alignment(
    str1: str,
    str2: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Literal["python", "numba", "mojo"] | None = None,
    device: Literal["cpu", "gpu"] | None = None,
) -> tuple[str, str, int]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution (Optional[SubstitutionCosts]): Uniform match and mismatch scores, or a table
        with the alphabet indexing it. Defaults to BLOSUM62 scaled by five.
    gaps (Optional[AffineGapCosts]): The penalties for opening and extending a gap.

    Returns:
    Tuple[str, str, int]: The optimal alignment of the two sequences and the alignment score.

    Default values:
    >>> substitution = None  # BLOSUM62 * 5 over "ARNDCQEGHILKMFPSTWYVBZX"
    >>> gaps = AffineGapCosts(open=-20, extend=-1)

    Example usage:
    >>> from affinegaps import needleman_wunsch_gotoh_alignment
    >>> str1 = "GATTACA"
    >>> str2 = "GCATGCU"
    >>> align1, align2, score = needleman_wunsch_gotoh_alignment(str1, str2)
    >>> print("Alignment 1:", align1)
    >>> print("Alignment 2:", align2)
    >>> print("Score:", score)
    """
    backend, device = _resolve_for(substitution, backend, device)
    if backend == "mojo":
        return _compiled_call(
            Algorithm.GLOBAL_ALIGNMENT,
            backend,
            device,
            str1,
            str2,
            dict(substitution=substitution, gaps=gaps),
        )

    substitution_alphabet, substitution_matrix, open, extend = _validate_gotoh_arguments(substitution, gaps)

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, changes, deletes, inserts = _kernel(_needleman_wunsch_gotoh_kernel, backend)(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        open=open,
        extend=extend,
    )

    align1, align2 = _reconstruct_alignment(
        changes,
        scores,
        deletes,
        inserts,
        seq1,
        seq2,
        open,
        extend,
        lambda x: substitution_alphabet[x],
        lambda i, j: i > 0 and j > 0,
    )
    return align1, align2, int(scores[-1, -1])


@jit_if_available(nopython=True)
def _needleman_wunsch_gotoh_score_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    open: int,
    extend: int,
) -> int:
    """
    Measures the alignment score of two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm. Uses less memory than the alignment function.

    The kernel has quadratic complexity in time and linear in space, as it stores
    only two rows of each matrix. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `needleman_wunsch_gotoh_score`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): The substitution matrix for scoring matches/mismatches.
    open (int): The penalty for opening a gap.
    extend (int): The penalty for extending a gap.

    Returns:
    int: The alignment score.
    """

    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    old_scores = np.empty(seq2_len + 1, dtype=np.int32)
    new_scores = np.empty(seq2_len + 1, dtype=np.int32)
    old_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    new_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    old_inserts = np.empty(seq2_len + 1, dtype=np.int32)
    new_inserts = np.empty(seq2_len + 1, dtype=np.int32)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    old_scores[0] = 0
    for j in range(1, seq2_len + 1):
        old_scores[j] = open + (j - 1) * extend
        old_deletes[j] = old_scores[j] + open + extend

    for i in range(1, seq1_len + 1):
        new_scores[0] = open + (i - 1) * extend
        new_inserts[0] = new_scores[0] + open + extend

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(old_scores[j] + open, old_deletes[j] + extend)
            insert = max(new_scores[j - 1] + open, new_inserts[j - 1] + extend)
            replace = old_scores[j - 1] + substitution
            score = max(replace, delete, insert)
            new_scores[j] = score
            new_deletes[j] = delete
            new_inserts[j] = insert

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return old_scores[-1]


def needleman_wunsch_gotoh_score(
    str1: str,
    str2: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Literal["python", "numba", "mojo"] | None = None,
    device: Literal["cpu", "gpu"] | None = None,
) -> int:
    """
    Measures the alignment score of two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm. Uses less memory than the alignment function.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution (Optional[SubstitutionCosts]): Uniform match and mismatch scores, or a table
        with the alphabet indexing it. Defaults to BLOSUM62 scaled by five.
    gaps (Optional[AffineGapCosts]): The penalties for opening and extending a gap.

    Returns:
    int: The alignment score.

    Default values:
    >>> substitution = None  # BLOSUM62 * 5 over "ARNDCQEGHILKMFPSTWYVBZX"
    >>> gaps = AffineGapCosts(open=-20, extend=-1)

    Example usage:
    >>> from affinegaps import needleman_wunsch_gotoh_score
    >>> str1 = "GATTACA"
    >>> str2 = "GCATGCU"
    >>> score = needleman_wunsch_gotoh_score(str1, str2)
    >>> print("Alignment 1:", align1)
    >>> print("Alignment 2:", align2)
    >>> print("Score:", score)
    """
    backend, device = _resolve_for(substitution, backend, device)
    if backend == "mojo":
        return _compiled_call(
            Algorithm.GLOBAL_SCORE,
            backend,
            device,
            str1,
            str2,
            dict(substitution=substitution, gaps=gaps),
        )

    # The inner loop must be the longer one, assuming the latency of calls
    # from Python into the C layer implementation of NumPy, so lets swap
    # the sequences if needed:
    #
    # if (substitution_matrix == substitution_matrix.T).all():
    #     if len(str1) > len(str2):
    #         str1, str2 = str2, str1
    substitution_alphabet, substitution_matrix, open, extend = _validate_gotoh_arguments(substitution, gaps)

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)

    score = _kernel(_needleman_wunsch_gotoh_score_kernel, backend)(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        open=open,
        extend=extend,
    )

    return int(score)


@jit_if_available(nopython=True)
def _smith_waterman_gotoh_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    open: int,
    extend: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Smith-Waterman local alignment algorithm.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `smith_waterman_gotoh`.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): A substitution matrix for scoring matches/mismatches.
    open (int): The penalty for opening a gap.
    extend (int): The penalty for extending a gap.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]: What the traceback needs:
        - scores: The best score reaching each cell in any state.
        - changes: The operation that achieved it, one of MATCH, SUBSTITUTE, DELETE or INSERT.
        - deletes: The best score reaching each cell inside a deletion run.
        - inserts: The best score reaching each cell inside an insertion run.
        - max_pos: The first row-major cell attaining the highest score, where traceback starts.

    Example usage:
    >>> seq1 = np.array([1, 2, 3])  # Example sequence
    >>> seq2 = np.array([3, 2, 1])  # Example sequence
    >>> substitution_matrix = np.array([[...], [...], [...]])  # Example substitution matrix
    >>> open = 5
    >>> extend = 2
    >>> scores, changes, deletes, inserts, best = _smith_waterman_gotoh_kernel(seq1, seq2, matrix, opening, extension)
    >>> print("Optimal alignment score matrix:\n", scores)

    Notes:
    The basis and recurrence relations for the matrices are as follows:
    Basis:
    - scores[i, 0] = open + (i - 1) * extend
    - scores[0, j] = open + (j - 1) * extend
    - deletes[i, 0] = open + (i - 1) * extend
    - inserts[0, j] = open + (j - 1) * extend

    Recurrence:
    - deletes[i, j] = max(scores[i - 1, j] + open, deletes[i - 1, j] + extend)
    - inserts[i, j] = max(scores[i, j - 1] + open, inserts[i, j - 1] + extend)
    - match = scores[i - 1, j - 1] + substitution_matrix[(seq1[i - 1], seq2[j - 1])]
    - scores[i, j] = max(match, deletes[i, j], inserts[i, j], 0)
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Initialize the scoring matrix, following the suggestions in the paper.
    # There:
    #
    #   v ~ is gap opening penalty (always non-negative in paper, opposite for us)
    #   u ~ is gap extension penalty (always non-positive in paper, opposite for us)
    #   w(k) = u * k + v
    #
    #   D(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n
    #   P(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with a deletion of at least one residue from A, such that A(m)
    #             is aligned with a gap symbol
    #   Q(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with an insertion of at least one residue from B, such that B(n)
    #             is aligned with a gap symbol
    #
    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    deletes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    inserts = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.int32)
    changes = np.empty((seq1_len + 1, seq2_len + 1), dtype=np.uint8)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    scores[0, :] = 0
    deletes[0, :] = open + extend
    changes[0, :] = INSERT

    # Unlike Needleman-Wunsch, we also track the position of the maximum score.
    max_score = 0
    max_pos = (0, 0)

    # Fill the scoring matrix
    for i in range(1, seq1_len + 1):
        scores[i, 0] = 0
        inserts[i, 0] = open + extend
        changes[i, 0] = DELETE

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(
                scores[i - 1, j] + open,
                deletes[i - 1, j] + extend,
            )
            insert = max(
                scores[i, j - 1] + open,
                inserts[i, j - 1] + extend,
            )
            replace = scores[i - 1, j - 1] + substitution
            score = max(replace, delete, insert, 0)
            scores[i, j] = score
            deletes[i, j] = delete
            inserts[i, j] = insert

            # Track changes
            if score == replace:
                changes[i, j] = MATCH if seq1[i - 1] == seq2[j - 1] else SUBSTITUTE
            elif score == delete:
                changes[i, j] = DELETE
            else:
                changes[i, j] = INSERT

            # Update max score and position
            if score > max_score:
                max_score = score
                max_pos = (i, j)

    return scores, changes, deletes, inserts, max_pos


def smith_waterman_gotoh_alignment(
    str1: str,
    str2: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Literal["python", "numba", "mojo"] | None = None,
    device: Literal["cpu", "gpu"] | None = None,
) -> tuple[str, str, int]:
    """
    Aligns two sequences using the Smith-Waterman algorithm for local alignment.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution (Optional[SubstitutionCosts]): Uniform match and mismatch scores, or a table
        with the alphabet indexing it. Defaults to BLOSUM62 scaled by five.
    gaps (Optional[AffineGapCosts]): The penalties for opening and extending a gap.

    Returns:
    Tuple[str, str, int]: The optimal local alignment of the two sequences and the alignment score.
    """
    backend, device = _resolve_for(substitution, backend, device)
    if backend == "mojo":
        return _compiled_call(
            Algorithm.LOCAL_ALIGNMENT,
            backend,
            device,
            str1,
            str2,
            dict(substitution=substitution, gaps=gaps),
        )

    substitution_alphabet, substitution_matrix, open, extend = _validate_gotoh_arguments(substitution, gaps)

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)
    scores, changes, deletes, inserts, max_pos = _kernel(_smith_waterman_gotoh_kernel, backend)(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        open=open,
        extend=extend,
    )

    prefix1, prefix2 = max_pos
    align1, align2 = _reconstruct_alignment(
        changes[: prefix1 + 1, : prefix2 + 1],
        scores,
        deletes,
        inserts,
        seq1[:prefix1],
        seq2[:prefix2],
        open,
        extend,
        lambda x: substitution_alphabet[x],
        lambda i, j: i > 0 and j > 0 and scores[i, j] > 0,
        flush_prefixes=False,
    )
    return align1, align2, int(scores[prefix1, prefix2])


@jit_if_available(nopython=True)
def _smith_waterman_gotoh_score_kernel(
    seq1: np.ndarray,
    seq2: np.ndarray,
    substitution_matrix: np.ndarray,
    open: int,
    extend: int,
) -> int:
    """
    Computes the Smith-Waterman alignment score using Gotoh's affine gap penalty extensions.
    Uses only two rows per matrix to reduce memory usage.

    Parameters:
    seq1 (np.ndarray): The first sequence to be aligned.
    seq2 (np.ndarray): The second sequence to be aligned.
    substitution_matrix (np.ndarray): The substitution matrix for scoring matches/mismatches.
    open (int): The penalty for opening a gap.
    extend (int): The penalty for extending a gap.

    Returns:
    int: The highest alignment score.
    """
    seq1_len = len(seq1)
    seq2_len = len(seq2)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    old_scores = np.empty(seq2_len + 1, dtype=np.int32)
    new_scores = np.empty(seq2_len + 1, dtype=np.int32)
    old_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    new_deletes = np.empty(seq2_len + 1, dtype=np.int32)
    old_inserts = np.empty(seq2_len + 1, dtype=np.int32)
    new_inserts = np.empty(seq2_len + 1, dtype=np.int32)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    old_scores[0] = 0
    for j in range(1, seq2_len + 1):
        old_scores[j] = 0
        old_deletes[j] = open + extend

    max_score = 0

    for i in range(1, seq1_len + 1):
        new_scores[0] = 0
        new_inserts[0] = open + extend

        for j in range(1, seq2_len + 1):
            substitution = substitution_matrix[seq1[i - 1], seq2[j - 1]]
            delete = max(old_scores[j] + open, old_deletes[j] + extend)
            insert = max(new_scores[j - 1] + open, new_inserts[j - 1] + extend)
            replace = old_scores[j - 1] + substitution
            score = max(replace, delete, insert, 0)
            new_scores[j] = score
            new_deletes[j] = delete
            new_inserts[j] = insert

            if score > max_score:
                max_score = score

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return max_score


def smith_waterman_gotoh_score(
    str1: str,
    str2: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Literal["python", "numba", "mojo"] | None = None,
    device: Literal["cpu", "gpu"] | None = None,
) -> int:
    """
    Measures the Smith-Waterman local alignment score using Gotoh's affine gap penalty extensions.

    Parameters:
    str1 (str): The first sequence to be aligned.
    str2 (str): The second sequence to be aligned.
    substitution (Optional[SubstitutionCosts]): Uniform match and mismatch scores, or a table
        with the alphabet indexing it. Defaults to BLOSUM62 scaled by five.
    gaps (Optional[AffineGapCosts]): The penalties for opening and extending a gap.

    Returns:
    int: The highest alignment score.

    Example usage:
    >>> from affinegaps import smith_waterman_gotoh_score
    >>> str1 = "GATTACA"
    >>> str2 = "GCATGCU"
    >>> score = smith_waterman_gotoh_score(str1, str2)
    >>> print("Score:", score)
    """
    backend, device = _resolve_for(substitution, backend, device)
    if backend == "mojo":
        return _compiled_call(
            Algorithm.LOCAL_SCORE,
            backend,
            device,
            str1,
            str2,
            dict(substitution=substitution, gaps=gaps),
        )

    substitution_alphabet, substitution_matrix, open, extend = _validate_gotoh_arguments(substitution, gaps)

    seq1 = _translate_sequence(str1, substitution_alphabet)
    seq2 = _translate_sequence(str2, substitution_alphabet)

    score = _kernel(_smith_waterman_gotoh_score_kernel, backend)(
        seq1,
        seq2,
        substitution_matrix=substitution_matrix,
        open=open,
        extend=extend,
    )

    return int(score)


# region Dispatch Table

# Defined here rather than beside the enums so `reference` binds the function object itself.
_ALGORITHMS: dict[Algorithm, _Spec] = {
    Algorithm.GLOBAL_SCORE: _Spec(needleman_wunsch_gotoh_score, Result.SCORE, Mode.GLOBAL),
    Algorithm.LOCAL_SCORE: _Spec(smith_waterman_gotoh_score, Result.SCORE, Mode.LOCAL),
    Algorithm.GLOBAL_ALIGNMENT: _Spec(needleman_wunsch_gotoh_alignment, Result.ALIGNMENT, Mode.GLOBAL),
    Algorithm.LOCAL_ALIGNMENT: _Spec(smith_waterman_gotoh_alignment, Result.ALIGNMENT, Mode.LOCAL),
}

# endregion Dispatch Table


def colorize_alignment(align1: str, align2: str, background: Literal["dark", "light"] = "dark") -> tuple[str, str]:
    """
    Colorizes the alignment strings for visual distinction between matches, mismatches, and gaps.
    Adjusts colors based on the specified background color.

    Parameters:
    align1 (str): The first aligned sequence.
    align2 (str): The second aligned sequence.
    background (str): The background color of the terminal ('dark' or 'light').

    Returns:
    Tuple[str, str]: The colorized alignments.
    """
    if background not in ("dark", "light"):
        raise ValueError("Background must be either 'dark' or 'light'")

    # Define color schemes
    from colorama import Fore, Style

    if background == "dark":
        match_color = Fore.GREEN
        mismatch_color = Fore.RED
        gap_color = Fore.WHITE
    else:
        match_color = Fore.GREEN
        mismatch_color = Fore.RED
        gap_color = Fore.BLACK

    colored_align1 = ""
    colored_align2 = ""

    for a, b in zip(align1, align2, strict=True):
        if a == b and a != "-":
            colored_align1 += match_color + a + Style.RESET_ALL
            colored_align2 += match_color + b + Style.RESET_ALL
        elif a == "-" or b == "-":
            colored_align1 += gap_color + a + Style.RESET_ALL
            colored_align2 += gap_color + b + Style.RESET_ALL
        else:
            colored_align1 += mismatch_color + a + Style.RESET_ALL
            colored_align2 += mismatch_color + b + Style.RESET_ALL

    return colored_align1, colored_align2


def main():
    # Let's parse the input arguments for alignments in CLI
    import argparse

    parser = argparse.ArgumentParser(description="Affine Gaps alignment CLI utility")
    parser.add_argument(
        "seq1",
        type=str,
        help="The first sequence to be aligned, like insulin (GIVEQCCTSICSLYQLENYCN)",
    )
    parser.add_argument(
        "seq2",
        type=str,
        help="The second sequence to be aligned, like glucagon (HSQGTFTSDYSKYLDSRAEQDFV)",
    )
    parser.add_argument(
        "--match",
        type=int,
        default=None,
        help="The score for a match, to compose the substitution matrix; uses scaled BLOSUM62 by default",
    )
    parser.add_argument(
        "--mismatch",
        type=int,
        default=None,
        help="The score for a mismatch, to compose the substitution matrix; uses scaled BLOSUM62 by default",
    )
    parser.add_argument(
        "--open",
        type=int,
        default=None,
        help=f"The penalty for opening a gap; uses {AffineGapCosts().open} by default",
    )
    parser.add_argument(
        "--extend",
        type=int,
        default=None,
        help=f"The penalty for extending a gap; uses {AffineGapCosts().extend} by default",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use the Smith-Waterman algorithm for local alignment instead of Needleman-Wunsch",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Align on the GPU, which needs the compiled backend",
    )
    args = parser.parse_args()

    aligner = smith_waterman_gotoh_alignment if args.local else needleman_wunsch_gotoh_alignment
    placement = {"backend": "mojo", "device": "gpu"} if args.gpu else {}
    try:
        substitution = None
        if args.match is not None or args.mismatch is not None:
            if args.match is None or args.mismatch is None:
                raise ValueError("Both --match and --mismatch must be provided.")
            substitution = UniformSubstitutionCosts(match=args.match, mismatch=args.mismatch)
        defaults = AffineGapCosts()
        gaps = AffineGapCosts(
            open=defaults.open if args.open is None else args.open,
            extend=defaults.extend if args.extend is None else args.extend,
        )
        align1, align2, score = aligner(args.seq1, args.seq2, substitution=substitution, gaps=gaps, **placement)
    except Exception as exc:
        print("Error:", exc)
        exit(1)

    # Colored output may require additional dependencies
    try:
        from colorama import init as _colorama_init

        _colorama_init(autoreset=True)
        colored1, colored2 = colorize_alignment(align1, align2)
    except Exception:
        colored1, colored2 = align1, align2

    print()
    print("Sequence 1:", args.seq1)
    print("Sequence 2:", args.seq2)
    print()
    print("Alignment 1:", colored1)
    print("Alignment 2:", colored2)
    print("Score:      ", score)


if __name__ == "__main__":
    main()
