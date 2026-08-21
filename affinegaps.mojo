"""
Affine Gaps Alignment Kernels for CPU and GPU.

Gotoh affine-gap sequence alignment with the alignment reconstruction itself running on
the GPU, in linear space. The scoring recurrences, the tie-breaking and the border
initialization are transcribed from `affinegaps.py`, which stays the parity oracle.

A row is the wrong sweep axis for a GPU, because the insertion term of a cell reads the
insertion term of its left neighbour. This module sweeps anti-diagonals instead, where every
cell of `d = i + j` reads only `d - 1` and `d - 2` and the whole diagonal is independent.

Traceback is Hirschberg with a Myers-Miller affine join, splitting on rows rather than
anti-diagonals: a substitution step advances `i + j` by two and can skip a diagonal entirely,
while the row index advances by zero or one per step. The recursion bottoms out in a direct
traceback over a stored decision tile.

The traceback walks all three layers — match, deletion and insertion — so every path realizes
the score reported alongside it, which is not automatic for affine gaps.

For score-only work at much higher throughput, see `ashvardanian/StringZilla`, whose
`stringzillas/similarities` kernels compute the same distances and scores. It carries no
traceback, which is what this module exists to provide.

## Usage

```bash
pixi run build      # build/affinegaps_mojo.so, importable from Python
pixi run test       # the differential suite against affinegaps.py
```
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.memory.pointer import AddressSpace
from std.os import abort
from std.python import Python, PythonObject
from std.sys import argv
from std.python.bindings import PythonModuleBuilder
from max.gpu import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.host.info import H100

# region Scoring

comptime SCORE_DTYPE = DType.int32
comptime SYMBOL_DTYPE = DType.uint8
comptime SUBSTITUTION_DTYPE = DType.int8
comptime CHANGE_DTYPE = DType.uint8

comptime DEFAULT_PROTEINS_ALPHABET = "ARNDCQEGHILKMFPSTWYVBZX"
comptime DEFAULT_GAP_OPENING = Int32(-20)
comptime DEFAULT_GAP_EXTENSION = Int32(-1)

comptime THREADS_PER_BLOCK = 256

# Caps the substitution table staged into shared memory. Thirty-two covers the twenty-three
# protein letters with room to spare, and costs one kilobyte per block.
comptime MAX_ALPHABET_SIZE = 32

# How many blocks should stay resident per multiprocessor. This is the knob; the band capacity
# below follows from it and from the device, rather than being guessed. Four is chosen because a
# batch has to carry more than nine hundred pairs before occupancy stops being bound by batch size
# on a hundred-and-thirty-two-multiprocessor device, so a longer reach is usually free.
comptime BLOCKS_PER_MULTIPROCESSOR = 4

# Bytes a block may hold without opting into the dynamic carve-out.
comptime STATIC_SHARED_LIMIT = 48 * 1024
comptime SHARED_PER_BLOCK = min(
    H100.shared_memory_per_multiprocessor // BLOCKS_PER_MULTIPROCESSOR, STATIC_SHARED_LIMIT
)

# Seven bands of scores, deletions and insertions, plus the staged table and the reductions.
comptime BANDS_PER_SWEEP = 7
comptime REDUCTION_BYTES = THREADS_PER_BLOCK * 12
comptime MAX_BAND_LENGTH = (
    SHARED_PER_BLOCK - MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE - REDUCTION_BYTES
) // (BANDS_PER_SWEEP * 4) - 1

# A quarter of the range: far below any reachable score, which a path bounds by
# `(rows + columns) * max_cost`, while leaving headroom for the gap penalties that get added to
# the seed as a reduction proceeds. `Int32.MIN` itself underflows on the first addition.
comptime NEGATIVE_INFINITY = Int32.MIN // 4


struct Operation:
    """Which edit the recurrence chose at a cell."""

    comptime MATCH = UInt8(0)
    comptime INSERT = UInt8(1)
    comptime DELETE = UInt8(2)
    comptime SUBSTITUTE = UInt8(3)


def default_proteins_matrix() -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """BLOSUM62 scaled by five, trimmed to the 23 letters the alphabet actually emits.

    The canonical table is 24 by 24; its last row and column are the `*` stop codon, which
    `translate` never produces.
    """
    var values: List[Int] = [
        20, -5, -10, -10, 0, -5, -5, 0, -10, -5, -5, -5, -5, -10, -5, 5, 0, -15, -10, 0, -10, -5, 0,
        -5, 25, 0, -10, -15, 5, 0, -10, 0, -15, -10, 10, -5, -15, -10, -5, -5, -15, -10, -15, -5, 0, -5,
        -10, 0, 30, 5, -15, 0, 0, 0, 5, -15, -15, 0, -10, -15, -10, 5, 0, -20, -10, -15, 15, 0, -5,
        -10, -10, 5, 30, -15, 0, 10, -5, -5, -15, -20, -5, -15, -15, -5, 0, -5, -20, -15, -15, 20, 5, -5,
        0, -15, -15, -15, 45, -15, -20, -15, -15, -5, -5, -15, -5, -10, -15, -5, -5, -10, -10, -5, -15, -15, -10,
        -5, 5, 0, 0, -15, 25, 10, -10, 0, -15, -10, 5, 0, -15, -5, 0, -5, -10, -5, -10, 0, 15, -5,
        -5, 0, 0, 10, -20, 10, 25, -10, 0, -15, -15, 5, -10, -15, -5, 0, -5, -15, -10, -10, 5, 20, -5,
        0, -10, 0, -5, -15, -10, -10, 30, -10, -20, -20, -10, -15, -15, -10, 0, -10, -10, -15, -15, -5, -10, -5,
        -10, 0, 5, -5, -15, 0, 0, -10, 40, -15, -15, -5, -10, -5, -10, -5, -10, -10, 10, -15, 0, 0, -5,
        -5, -15, -15, -15, -5, -15, -15, -20, -15, 20, 10, -15, 5, 0, -15, -10, -5, -15, -5, 15, -15, -15, -5,
        -5, -10, -15, -20, -5, -10, -15, -20, -15, 10, 20, -10, 10, 0, -15, -10, -5, -10, -5, 5, -20, -15, -5,
        -5, 10, 0, -5, -15, 5, 5, -10, -5, -15, -10, 25, -5, -15, -5, 0, -5, -15, -10, -10, 0, 5, -5,
        -5, -5, -10, -15, -5, 0, -10, -15, -10, 5, 10, -5, 25, 0, -10, -5, -5, -5, -5, 5, -15, -5, -5,
        -10, -15, -15, -15, -10, -15, -15, -15, -5, 0, 0, -15, 0, 30, -20, -10, -10, 5, 15, -5, -15, -15, -5,
        -5, -10, -10, -5, -15, -5, -5, -10, -10, -15, -15, -5, -10, -20, 35, -5, -5, -20, -15, -10, -10, -5, -10,
        5, -5, 5, 0, -5, 0, 0, 0, -5, -10, -10, 0, -5, -10, -5, 20, 5, -15, -10, -10, 0, 0, 0,
        0, -5, 0, -5, -5, -5, -5, -10, -10, -5, -5, -5, -5, -10, -5, 5, 25, -10, -10, 0, -5, -5, 0,
        -15, -15, -20, -20, -10, -10, -15, -10, -10, -15, -10, -15, -5, 5, -20, -15, -10, 55, 10, -15, -20, -15, -10,
        -10, -10, -10, -15, -10, -5, -10, -15, 10, -5, -5, -10, -5, 15, -15, -10, -10, 10, 35, -5, -15, -10, -5,
        0, -15, -15, -15, -5, -10, -10, -15, -15, 15, 5, -10, 5, -5, -10, -10, 0, -15, -5, 20, -15, -10, -5,
        -10, -5, 15, 20, -15, 0, 5, -5, 0, -15, -20, 0, -15, -15, -10, 0, -5, -20, -15, -15, 20, 5, -5,
        -5, 0, 0, 5, -15, 15, 20, -10, 0, -15, -15, 5, -5, -15, -5, 0, -5, -15, -10, -10, 5, 20, -5,
        0, -5, -5, -5, -10, -5, -5, -5, -5, -5, -5, -5, -5, -5, -10, 0, 0, -10, -5, -5, -5, -5, -5,
    ]
    var matrix = List[Scalar[SUBSTITUTION_DTYPE]](capacity=len(values))
    for index in range(len(values)):
        matrix.append(Scalar[SUBSTITUTION_DTYPE](values[index]))
    return matrix^


def uniform_matrix(alphabet_size: Int, match_score: Int, mismatch_score: Int) -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """Diagonal substitution matrix, the `match`/`mismatch` path of `_validate_gotoh_arguments`."""
    var matrix = List[Scalar[SUBSTITUTION_DTYPE]](
        length=alphabet_size * alphabet_size, fill=Scalar[SUBSTITUTION_DTYPE](mismatch_score)
    )
    for index in range(alphabet_size):
        matrix[index * alphabet_size + index] = Scalar[SUBSTITUTION_DTYPE](match_score)
    return matrix^


def translate(text: String, alphabet: String) raises -> List[Scalar[SYMBOL_DTYPE]]:
    """Maps characters to alphabet indices, raising on anything outside the alphabet."""
    var alphabet_bytes = alphabet.as_bytes()
    var text_bytes = text.as_bytes()
    var codes = List[Scalar[SYMBOL_DTYPE]](capacity=len(text_bytes))
    for position in range(len(text_bytes)):
        var found = -1
        for index in range(len(alphabet_bytes)):
            if text_bytes[position] == alphabet_bytes[index]:
                found = index
                break
        if found < 0:
            raise Error(String("Found unknown character in sequence: ", text))
        codes.append(Scalar[SYMBOL_DTYPE](found))
    return codes^


@fieldwise_init
struct AffineScoring(Copyable, Movable):
    """Gotoh's two-parameter gap model. Both penalties are negative."""

    var gap_opening: Int32
    var gap_extension: Int32


@fieldwise_init
struct AlignmentMode(Copyable, Movable, TrivialRegisterPassable):
    """Which of the two alignment problems the recurrence solves."""

    var identifier: Int
    comptime GLOBAL = Self(0)
    comptime LOCAL = Self(1)

    def __eq__(self, other: Self) -> Bool:
        return self.identifier == other.identifier


# endregion Scoring

# region Serial Reference

@fieldwise_init
struct AlignmentResult(Copyable, Movable):
    """A score and the two gapped strings that realize it."""

    var score: Int32
    var first_gapped: String
    var second_gapped: String


def serial_score[
    mode: AlignmentMode
](
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
) -> Int32:
    """Two-row reference, transcribed from the `*_score_kernel` functions of `affinegaps.py`."""
    var rows = len(first)
    var columns = len(second)
    var scores_above = List[Int32](length=columns + 1, fill=Int32(0))
    var scores_row = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_above = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_row = List[Int32](length=columns + 1, fill=Int32(0))
    var inserts_above = List[Int32](length=columns + 1, fill=Int32(0))
    var inserts_row = List[Int32](length=columns + 1, fill=Int32(0))

    scores_above[0] = 0
    for column in range(1, columns + 1):
        if mode == AlignmentMode.GLOBAL:
            scores_above[column] = scoring.gap_opening + Int32(column - 1) * scoring.gap_extension
            deletes_above[column] = scores_above[column] + scoring.gap_opening + scoring.gap_extension
        else:
            scores_above[column] = 0
            deletes_above[column] = scoring.gap_opening + scoring.gap_extension

    var best = Int32(0)
    for row in range(1, rows + 1):
        if mode == AlignmentMode.GLOBAL:
            scores_row[0] = scoring.gap_opening + Int32(row - 1) * scoring.gap_extension
        else:
            scores_row[0] = 0
        inserts_row[0] = scores_row[0] + scoring.gap_opening + scoring.gap_extension

        for column in range(1, columns + 1):
            var substitution = Int32(
                substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])]
            )
            var deletion = max(
                scores_above[column] + scoring.gap_opening,
                deletes_above[column] + scoring.gap_extension,
            )
            var insertion = max(
                scores_row[column - 1] + scoring.gap_opening,
                inserts_row[column - 1] + scoring.gap_extension,
            )
            var replacement = scores_above[column - 1] + substitution
            var cell = max(max(replacement, deletion), insertion)
            if mode == AlignmentMode.LOCAL:
                cell = max(cell, Int32(0))
                best = max(best, cell)
            scores_row[column] = cell
            deletes_row[column] = deletion
            inserts_row[column] = insertion

        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)
        swap(inserts_above, inserts_row)

    if mode == AlignmentMode.LOCAL:
        return best
    return scores_above[columns]


def serial_align[
    mode: AlignmentMode
](
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
    alphabet: String,
) -> AlignmentResult:
    """Full-matrix reference, transcribed from the `_*_kernel` plus `_reconstruct_alignment`."""
    var rows = len(first)
    var columns = len(second)
    var stride = columns + 1
    var cells = (rows + 1) * stride
    var scores = List[Int32](length=cells, fill=Int32(0))
    var deletes = List[Int32](length=cells, fill=Int32(0))
    var inserts = List[Int32](length=cells, fill=Int32(0))
    var changes = List[UInt8](length=cells, fill=UInt8(0))

    scores[0] = 0
    for column in range(1, stride):
        if mode == AlignmentMode.GLOBAL:
            scores[column] = scoring.gap_opening + Int32(column - 1) * scoring.gap_extension
            deletes[column] = scores[column] + scoring.gap_opening + scoring.gap_extension
        else:
            scores[column] = 0
            deletes[column] = scoring.gap_opening + scoring.gap_extension
        changes[column] = Operation.INSERT

    var best = Int32(0)
    var best_row = 0
    var best_column = 0

    for row in range(1, rows + 1):
        var base = row * stride
        var above = base - stride
        if mode == AlignmentMode.GLOBAL:
            scores[base] = scoring.gap_opening + Int32(row - 1) * scoring.gap_extension
            inserts[base] = scores[base] + scoring.gap_opening + scoring.gap_extension
        else:
            scores[base] = 0
            inserts[base] = scoring.gap_opening + scoring.gap_extension
        changes[base] = Operation.DELETE

        for column in range(1, stride):
            var substitution = Int32(
                substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])]
            )
            var deletion = max(
                scores[above + column] + scoring.gap_opening,
                deletes[above + column] + scoring.gap_extension,
            )
            var insertion = max(
                scores[base + column - 1] + scoring.gap_opening,
                inserts[base + column - 1] + scoring.gap_extension,
            )
            var replacement = scores[above + column - 1] + substitution
            var cell = max(max(replacement, deletion), insertion)
            if mode == AlignmentMode.LOCAL:
                cell = max(cell, Int32(0))
            scores[base + column] = cell
            deletes[base + column] = deletion
            inserts[base + column] = insertion

            if cell == replacement:
                changes[base + column] = (
                    Operation.MATCH if first[row - 1] == second[column - 1] else Operation.SUBSTITUTE
                )
            elif cell == deletion:
                changes[base + column] = Operation.DELETE
            else:
                changes[base + column] = Operation.INSERT

            if mode == AlignmentMode.LOCAL and cell > best:
                best = cell
                best_row = row
                best_column = column

    var start_row = rows
    var start_column = columns
    if mode == AlignmentMode.LOCAL:
        start_row = best_row
        start_column = best_column

    var reconstruction = reconstruct(
        scores,
        deletes,
        inserts,
        stride,
        first,
        second,
        substitutions,
        alphabet_size,
        start_row,
        start_column,
        alphabet,
        scoring,
        mode,
    )
    var final_score = scores[start_row * stride + start_column]
    return AlignmentResult(final_score, reconstruction[0], reconstruction[1])


def reconstruct(
    scores: List[Int32],
    deletes: List[Int32],
    inserts: List[Int32],
    stride: Int,
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    start_row: Int,
    start_column: Int,
    alphabet: String,
    scoring: AffineScoring,
    mode: AlignmentMode,
) -> Tuple[String, String]:
    """Three-state backward walk over the match, deletion and insertion layers.

    The Python walks a single layer: after a deletion step it reads the winning operation of the
    next cell instead of asking whether that deletion was opened or extended, so it can split one
    run into two and return a path that does not achieve its own reported score. Carrying the state
    fixes that. Ties resolve towards opening, which leaves the linear-gap case walking exactly as
    the Python does.
    """
    var letters = alphabet.as_bytes()
    var gap = UInt8(45)
    var first_reversed = List[UInt8]()
    var second_reversed = List[UInt8]()
    var row = start_row
    var column = start_column
    var state = Operation.MATCH

    while row > 0 and column > 0:
        if state == Operation.MATCH:
            if mode == AlignmentMode.LOCAL and scores[row * stride + column] <= 0:
                break
            var substitution = Int32(
                substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])]
            )
            var replacement = scores[(row - 1) * stride + column - 1] + substitution
            var cell = scores[row * stride + column]
            if cell == replacement:
                first_reversed.append(letters[Int(first[row - 1])])
                second_reversed.append(letters[Int(second[column - 1])])
                row -= 1
                column -= 1
            elif cell == deletes[row * stride + column]:
                state = Operation.DELETE
            else:
                state = Operation.INSERT
        elif state == Operation.DELETE:
            first_reversed.append(letters[Int(first[row - 1])])
            second_reversed.append(gap)
            var extends = deletes[(row - 1) * stride + column] + scoring.gap_extension > scores[
                (row - 1) * stride + column
            ] + scoring.gap_opening
            row -= 1
            state = Operation.DELETE if extends else Operation.MATCH
        else:
            first_reversed.append(gap)
            second_reversed.append(letters[Int(second[column - 1])])
            var extends = inserts[row * stride + column - 1] + scoring.gap_extension > scores[
                row * stride + column - 1
            ] + scoring.gap_opening
            column -= 1
            state = Operation.INSERT if extends else Operation.MATCH

    # A global path must reach the origin, so what remains really is aligned against gaps. A local
    # path stops wherever the score falls to zero, and everything before that is outside it.
    if mode == AlignmentMode.GLOBAL:
        while row > 0:
            first_reversed.append(letters[Int(first[row - 1])])
            second_reversed.append(gap)
            row -= 1
        while column > 0:
            first_reversed.append(gap)
            second_reversed.append(letters[Int(second[column - 1])])
            column -= 1

    var first_gapped = List[UInt8](capacity=len(first_reversed))
    var second_gapped = List[UInt8](capacity=len(second_reversed))
    for index in range(len(first_reversed) - 1, -1, -1):
        first_gapped.append(first_reversed[index])
        second_gapped.append(second_reversed[index])
    return (String(unsafe_from_utf8=first_gapped), String(unsafe_from_utf8=second_gapped))


@fieldwise_init
struct Boundary(Copyable, Movable, TrivialRegisterPassable):
    """Whether a deletion run touching a subproblem's edge is already open."""

    var identifier: Int
    comptime OPENS = Self(0)
    comptime EXTENDS = Self(1)

    def __eq__(self, other: Self) -> Bool:
        return self.identifier == other.identifier


@fieldwise_init
struct Frame(Copyable, Movable, TrivialRegisterPassable):
    """One pending subproblem of the Hirschberg recursion.

    Rows `[first_from, first_to)` against columns `[second_from, second_to)`, plus whether a
    deletion run already touches each horizontal edge.
    """

    var first_from: Int
    var first_to: Int
    var second_from: Int
    var second_to: Int
    var top: Boundary
    var bottom: Boundary


def sweep_bands(
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    first_from: Int,
    first_to: Int,
    second_from: Int,
    second_to: Int,
    backwards: Bool,
    entry: Boundary,
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
    mut final_scores: List[Int32],
    mut final_deletes: List[Int32],
):
    """Linear-space sweep of one sub-rectangle, leaving the last row's two layers behind.

    `backwards` walks both sequences from their far ends, which is how the reverse half of a
    Hirschberg split is computed without a second recurrence. `entry` carries the open deletion
    run across the boundary: an already-open run makes the first deletion cost only an extension.
    """
    var rows = first_to - first_from
    var columns = second_to - second_from
    var scores_above = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_above = List[Int32](length=columns + 1, fill=Int32(0))
    var scores_row = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_row = List[Int32](length=columns + 1, fill=Int32(0))
    var inserts_row = List[Int32](length=columns + 1, fill=Int32(0))

    scores_above[0] = 0
    deletes_above[0] = scoring.gap_opening + scoring.gap_extension
    for column in range(1, columns + 1):
        scores_above[column] = scoring.gap_opening + Int32(column - 1) * scoring.gap_extension
        if entry == Boundary.EXTENDS:
            deletes_above[column] = scores_above[column]
        else:
            deletes_above[column] = scores_above[column] + scoring.gap_opening + scoring.gap_extension

    for row in range(1, rows + 1):
        if entry == Boundary.EXTENDS:
            scores_row[0] = Int32(row) * scoring.gap_extension
        else:
            scores_row[0] = scoring.gap_opening + Int32(row - 1) * scoring.gap_extension
        deletes_row[0] = scores_row[0]
        inserts_row[0] = scores_row[0] + scoring.gap_opening + scoring.gap_extension

        var first_index = first_to - row if backwards else first_from + row - 1
        for column in range(1, columns + 1):
            var second_index = second_to - column if backwards else second_from + column - 1
            var substitution = Int32(
                substitutions[Int(first[first_index]) * alphabet_size + Int(second[second_index])]
            )
            var deletion = max(
                scores_above[column] + scoring.gap_opening,
                deletes_above[column] + scoring.gap_extension,
            )
            var insertion = max(
                scores_row[column - 1] + scoring.gap_opening,
                inserts_row[column - 1] + scoring.gap_extension,
            )
            scores_row[column] = max(max(scores_above[column - 1] + substitution, deletion), insertion)
            deletes_row[column] = deletion
            inserts_row[column] = insertion

        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)

    for column in range(columns + 1):
        final_scores[column] = scores_above[column]
        final_deletes[column] = deletes_above[column]


def direct_tile(
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    first_from: Int,
    first_to: Int,
    second_from: Int,
    second_to: Int,
    top: Boundary,
    bottom: Boundary,
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
    mut path_columns: List[Int32],
    mut path_entries: List[UInt8],
) -> Int32:
    """Solves one sub-rectangle outright and walks it back through the three layers.

    Writes `path_columns[i]` for `i` in `[first_from, first_to)` and `path_entries[i]` for `i` in
    `(first_from, first_to]`, so sibling subproblems tile the row axis without overlapping.
    Returns the corner score, which for a root-level call is the alignment's own score.
    """
    var rows = first_to - first_from
    var columns = second_to - second_from
    var stride = columns + 1
    var cells = (rows + 1) * stride
    var scores = List[Int32](length=cells, fill=Int32(0))
    var deletes = List[Int32](length=cells, fill=Int32(0))
    var inserts = List[Int32](length=cells, fill=Int32(0))

    scores[0] = 0
    for column in range(1, stride):
        scores[column] = scoring.gap_opening + Int32(column - 1) * scoring.gap_extension
        if top == Boundary.EXTENDS:
            deletes[column] = scores[column]
        else:
            deletes[column] = scores[column] + scoring.gap_opening + scoring.gap_extension

    for row in range(1, rows + 1):
        var base = row * stride
        var above = base - stride
        if top == Boundary.EXTENDS:
            scores[base] = Int32(row) * scoring.gap_extension
        else:
            scores[base] = scoring.gap_opening + Int32(row - 1) * scoring.gap_extension
        deletes[base] = scores[base]
        inserts[base] = scores[base] + scoring.gap_opening + scoring.gap_extension
        for column in range(1, stride):
            var substitution = Int32(
                substitutions[
                    Int(first[first_from + row - 1]) * alphabet_size
                    + Int(second[second_from + column - 1])
                ]
            )
            var deletion = max(
                scores[above + column] + scoring.gap_opening,
                deletes[above + column] + scoring.gap_extension,
            )
            var insertion = max(
                scores[base + column - 1] + scoring.gap_opening,
                inserts[base + column - 1] + scoring.gap_extension,
            )
            scores[base + column] = max(
                max(scores[above + column - 1] + substitution, deletion), insertion
            )
            deletes[base + column] = deletion
            inserts[base + column] = insertion

    var row = rows
    var column = columns
    var state = Operation.MATCH
    if bottom == Boundary.EXTENDS:
        var corner = row * stride + column
        if deletes[corner] + scoring.gap_extension - scoring.gap_opening > scores[corner]:
            state = Operation.DELETE

    while row > 0 and column > 0:
        var here = row * stride + column
        if state == Operation.MATCH:
            var substitution = Int32(
                substitutions[
                    Int(first[first_from + row - 1]) * alphabet_size
                    + Int(second[second_from + column - 1])
                ]
            )
            if scores[here] == scores[here - stride - 1] + substitution:
                path_entries[first_from + row] = Operation.SUBSTITUTE
                path_columns[first_from + row - 1] = Int32(second_from + column - 1)
                row -= 1
                column -= 1
            elif scores[here] == deletes[here]:
                state = Operation.DELETE
            else:
                state = Operation.INSERT
        elif state == Operation.DELETE:
            path_entries[first_from + row] = Operation.DELETE
            path_columns[first_from + row - 1] = Int32(second_from + column)
            var extends = deletes[here - stride] + scoring.gap_extension > scores[
                here - stride
            ] + scoring.gap_opening
            row -= 1
            state = Operation.DELETE if extends else Operation.MATCH
        else:
            var extends = inserts[here - 1] + scoring.gap_extension > scores[
                here - 1
            ] + scoring.gap_opening
            column -= 1
            state = Operation.INSERT if extends else Operation.MATCH

    while row > 0:
        path_entries[first_from + row] = Operation.DELETE
        path_columns[first_from + row - 1] = Int32(second_from + column)
        row -= 1

    return scores[rows * stride + columns]


def hirschberg_window(
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    window_first_from: Int,
    window_first_to: Int,
    window_second_from: Int,
    window_second_to: Int,
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
    tile_cells: Int,
    mut path_columns: List[Int32],
    mut path_entries: List[UInt8],
) raises:
    """Linear-space traceback: split on rows, join the two halves, recurse without recursion.

    A substitution step advances `i + j` by two and can skip an anti-diagonal entirely, while the
    row index advances by exactly zero or one per step, so the cut has to be a row. The two halves
    are joined by Myers-Miller: either the path crosses in the match layer, or a deletion run
    straddles the cut, in which case both halves charged an opening and one is refunded.
    """
    if scoring.gap_opening > scoring.gap_extension:
        raise Error("Gap opening must be at least as expensive as extension for the row split.")

    var columns = window_second_to - window_second_from
    path_columns[window_first_to] = Int32(window_second_to)

    var frames = List[Frame]()
    frames.append(
        Frame(
            window_first_from, window_first_to, window_second_from, window_second_to,
            Boundary.OPENS, Boundary.OPENS,
        )
    )

    var forward_scores = List[Int32](length=columns + 1, fill=Int32(0))
    var forward_deletes = List[Int32](length=columns + 1, fill=Int32(0))
    var reverse_scores = List[Int32](length=columns + 1, fill=Int32(0))
    var reverse_deletes = List[Int32](length=columns + 1, fill=Int32(0))

    while len(frames) > 0:
        var frame = frames.pop()
        var first_from = frame.first_from
        var first_to = frame.first_to
        var second_from = frame.second_from
        var second_to = frame.second_to
        var top = frame.top
        var bottom = frame.bottom

        var height = first_to - first_from
        var width = second_to - second_from
        if height == 0:
            continue
        if width == 0 or height <= 2 or (height + 1) * (width + 1) <= tile_cells:
            _ = direct_tile(
                first, second, first_from, first_to, second_from, second_to, top, bottom,
                substitutions, alphabet_size, scoring, path_columns, path_entries,
            )
            continue

        var split = (first_from + first_to) // 2
        sweep_bands(
            first, second, first_from, split, second_from, second_to, False, top,
            substitutions, alphabet_size, scoring, forward_scores, forward_deletes,
        )
        sweep_bands(
            first, second, split, first_to, second_from, second_to, True, bottom,
            substitutions, alphabet_size, scoring, reverse_scores, reverse_deletes,
        )

        var best_plain = NEGATIVE_INFINITY
        var best_plain_column = 0
        var best_gapped = NEGATIVE_INFINITY
        var refund = scoring.gap_extension - scoring.gap_opening
        for offset in range(width + 1):
            var plain = forward_scores[offset] + reverse_scores[width - offset]
            if plain > best_plain:
                best_plain = plain
                best_plain_column = offset
            var gapped = forward_deletes[offset] + reverse_deletes[width - offset] + refund
            if gapped > best_gapped:
                best_gapped = gapped

        # Take the answer from the branch actually followed. The gapped candidate can overshoot
        # the true optimum on degenerate rectangles, where the delete layers at the extreme
        # columns are still sentinels rather than reachable values.
        if best_gapped > best_plain:
            _ = direct_tile(
                first, second, first_from, first_to, second_from, second_to, top, bottom,
                substitutions, alphabet_size, scoring, path_columns, path_entries,
            )
            continue

        var crossing = second_from + best_plain_column
        # Lower half first, so the upper half pops first and the two row ranges tile in order.
        frames.append(Frame(split, first_to, crossing, second_to, Boundary.OPENS, bottom))
        frames.append(Frame(first_from, split, second_from, crossing, top, Boundary.OPENS))


def score_path(
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    path_columns: List[Int32],
    path_entries: List[UInt8],
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
    rows: Int,
) -> Int32:
    """Scores a reconstructed path under the affine rule, in time linear in the alignment.

    Deriving the score from the path rather than from a second dynamic-programming pass costs
    `O(rows + columns)` instead of `O(rows * columns)`, and makes the reported score consistent
    with the returned strings by construction rather than by coincidence.
    """
    var total = Int32(0)
    var in_first = False
    var in_second = False

    for _ in range(Int(path_columns[0])):
        total += scoring.gap_extension if in_first else scoring.gap_opening
        in_first = True
        in_second = False

    for row in range(1, rows + 1):
        var before = Int(path_columns[row - 1])
        var after = Int(path_columns[row])
        if path_entries[row] == Operation.DELETE:
            total += scoring.gap_extension if in_second else scoring.gap_opening
            in_first = False
            in_second = True
        else:
            total += Int32(
                substitutions[Int(first[row - 1]) * alphabet_size + Int(second[before])]
            )
            before += 1
            in_first = False
            in_second = False
        for _ in range(before, after):
            total += scoring.gap_extension if in_first else scoring.gap_opening
            in_first = True
            in_second = False

    return total


def expand_path(
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    path_columns: List[Int32],
    path_entries: List[UInt8],
    alphabet: String,
    mode: AlignmentMode,
    from_row: Int,
    rows: Int,
) -> Tuple[String, String]:
    """Turns the per-row crossing columns back into the two gapped strings.

    The walk covers rows `(from_row, rows]`. A global alignment spans the whole matrix and opens
    with however many insertions precede its first row; a local one spans only its own core, and
    the sequence outside that core is not part of the alignment.
    """
    var letters = alphabet.as_bytes()
    var gap = UInt8(45)
    var left = List[UInt8]()
    var right = List[UInt8]()

    if mode == AlignmentMode.GLOBAL:
        for column in range(Int(path_columns[0])):
            left.append(gap)
            right.append(letters[Int(second[column])])
    for row in range(from_row + 1, rows + 1):
        var before = Int(path_columns[row - 1])
        var after = Int(path_columns[row])
        if path_entries[row] == Operation.DELETE:
            left.append(letters[Int(first[row - 1])])
            right.append(gap)
        else:
            left.append(letters[Int(first[row - 1])])
            right.append(letters[Int(second[before])])
            before += 1
        for column in range(before, after):
            left.append(gap)
            right.append(letters[Int(second[column])])
    return (String(unsafe_from_utf8=left), String(unsafe_from_utf8=right))


def local_extremum(
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    first_to: Int,
    second_to: Int,
    backwards: Bool,
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
) -> Tuple[Int, Int, Int32]:
    """Linear-space local sweep returning the first row-major maximum and its value.

    Strict `>` keeps the earliest maximum in row-major order, which is the cell the Python's scan
    settles on. Run backwards over the same prefixes, it instead reports how far the best local
    alignment reaches back, which is where the alignment starts.
    """
    var scores_above = List[Int32](length=second_to + 1, fill=Int32(0))
    var deletes_above = List[Int32](length=second_to + 1, fill=Int32(0))
    var scores_row = List[Int32](length=second_to + 1, fill=Int32(0))
    var deletes_row = List[Int32](length=second_to + 1, fill=Int32(0))
    var inserts_row = List[Int32](length=second_to + 1, fill=Int32(0))

    for column in range(1, second_to + 1):
        deletes_above[column] = scoring.gap_opening + scoring.gap_extension

    var best = Int32(0)
    var best_row = 0
    var best_column = 0

    for row in range(1, first_to + 1):
        scores_row[0] = 0
        inserts_row[0] = scoring.gap_opening + scoring.gap_extension
        var first_index = first_to - row if backwards else row - 1
        for column in range(1, second_to + 1):
            var second_index = second_to - column if backwards else column - 1
            var substitution = Int32(
                substitutions[Int(first[first_index]) * alphabet_size + Int(second[second_index])]
            )
            var deletion = max(
                scores_above[column] + scoring.gap_opening,
                deletes_above[column] + scoring.gap_extension,
            )
            var insertion = max(
                scores_row[column - 1] + scoring.gap_opening,
                inserts_row[column - 1] + scoring.gap_extension,
            )
            var cell = max(
                max(max(scores_above[column - 1] + substitution, deletion), insertion), Int32(0)
            )
            scores_row[column] = cell
            deletes_row[column] = deletion
            inserts_row[column] = insertion
            if cell > best:
                best = cell
                best_row = row
                best_column = column
        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)

    return (best_row, best_column, best)

# endregion Serial Reference


# region GPU Wavefront


def wavefront_kernel[
    mode: AlignmentMode
](
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    offsets: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    results: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    alphabet_size: Int32,
    gap_opening: Int32,
    gap_extension: Int32,
):
    """One block per pair, one thread per cell of the live anti-diagonal.

    Bands are indexed by the row, so cell `(i, d - i)` reads rows `i - 1` and `i` of the two
    preceding diagonals. The band written each diagonal is never one read that diagonal, so a
    single barrier per diagonal suffices.
    """
    var pair = Int(block_idx.x)
    var first_start = Int(offsets[unsafe_offset=2 * pair])
    var second_start = Int(offsets[unsafe_offset=2 * pair + 1])
    var rows = second_start - first_start
    var columns = Int(offsets[unsafe_offset=2 * pair + 2]) - second_start
    var width = Int(alphabet_size)

    var bands = stack_allocation[
        7 * (MAX_BAND_LENGTH + 1), Scalar[SCORE_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var reduction = stack_allocation[
        THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space = AddressSpace.SHARED
    ]()

    var stride = MAX_BAND_LENGTH + 1
    var scores_two_back = bands
    var scores_one_back = bands.unsafe_offset(stride)
    var scores_current = bands.unsafe_offset(2 * stride)
    var deletes_one_back = bands.unsafe_offset(3 * stride)
    var deletes_current = bands.unsafe_offset(4 * stride)
    var inserts_one_back = bands.unsafe_offset(5 * stride)
    var inserts_current = bands.unsafe_offset(6 * stride)

    # The substitution table is read once per cell, so it is staged into shared memory rather
    # than fetched from global on every comparison. It is at most a kilobyte.
    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space = AddressSpace.SHARED,
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        scores_table[unsafe_offset=index] = substitutions[unsafe_offset=index]
    barrier()

    var best = Int32(0)
    var answer = Int32(0)
    var last_diagonal = rows + columns

    for diagonal in range(0, last_diagonal + 1):
        var row_low = max(0, diagonal - columns)
        var row_high = min(rows, diagonal)

        for row in range(row_low + Int(thread_idx.x), row_high + 1, THREADS_PER_BLOCK):
            var column = diagonal - row
            var cell = Int32(0)
            var deletion = Int32(0)
            var insertion = Int32(0)

            if row == 0 and column == 0:
                cell = 0
                deletion = cell + gap_opening + gap_extension
                insertion = deletion
            elif row == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = gap_opening + Int32(column - 1) * gap_extension
                    deletion = cell + gap_opening + gap_extension
                else:
                    cell = 0
                    deletion = gap_opening + gap_extension
                insertion = deletion
            elif column == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = gap_opening + Int32(row - 1) * gap_extension
                    insertion = cell + gap_opening + gap_extension
                else:
                    cell = 0
                    insertion = gap_opening + gap_extension
                deletion = insertion
            else:
                var left_symbol = Int(sequences[unsafe_offset=first_start + row - 1])
                var right_symbol = Int(sequences[unsafe_offset=second_start + column - 1])
                var substitution = Int32(
                    scores_table[unsafe_offset=left_symbol * width + right_symbol]
                )
                deletion = max(
                    scores_one_back[unsafe_offset=row - 1] + gap_opening,
                    deletes_one_back[unsafe_offset=row - 1] + gap_extension,
                )
                insertion = max(
                    scores_one_back[unsafe_offset=row] + gap_opening,
                    inserts_one_back[unsafe_offset=row] + gap_extension,
                )
                var replacement = scores_two_back[unsafe_offset=row - 1] + substitution
                cell = max(max(replacement, deletion), insertion)
                if mode == AlignmentMode.LOCAL:
                    cell = max(cell, Int32(0))
                    best = max(best, cell)

            scores_current[unsafe_offset=row] = cell
            deletes_current[unsafe_offset=row] = deletion
            inserts_current[unsafe_offset=row] = insertion
            if diagonal == last_diagonal and row == rows:
                answer = cell

        barrier()

        var recycled = scores_two_back
        scores_two_back = scores_one_back
        scores_one_back = scores_current
        scores_current = recycled
        swap(deletes_one_back, deletes_current)
        swap(inserts_one_back, inserts_current)

    if mode == AlignmentMode.LOCAL:
        reduction[unsafe_offset=Int(thread_idx.x)] = best
        barrier()
        var span = THREADS_PER_BLOCK // 2
        while span > 0:
            if Int(thread_idx.x) < span:
                reduction[unsafe_offset=Int(thread_idx.x)] = max(
                    reduction[unsafe_offset=Int(thread_idx.x)],
                    reduction[unsafe_offset=Int(thread_idx.x) + span],
                )
            barrier()
            span //= 2
        if thread_idx.x == 0:
            results[unsafe_offset=pair] = reduction[unsafe_offset=0]
    else:
        # Only the thread that owned the bottom-right cell carries the global answer, and the
        # host zeroed the buffer so a genuine zero needs no write.
        if answer != 0:
            results[unsafe_offset=pair] = answer


def wavefront_scores[
    mode: AlignmentMode
](
    ctx: DeviceContext,
    sequences: List[Scalar[SYMBOL_DTYPE]],
    offsets: List[Scalar[SCORE_DTYPE]],
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
) raises -> List[Int32]:
    """Scores every pair in the batch, one thread block each."""
    var pairs = (len(offsets) - 1) // 2
    if alphabet_size > MAX_ALPHABET_SIZE:
        raise Error("Alphabet exceeds the shared-memory substitution table capacity.")
    var sequences_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(len(sequences), 1))
    var offsets_buffer = ctx.enqueue_create_buffer[SCORE_DTYPE](len(offsets))
    var substitutions_buffer = ctx.enqueue_create_buffer[SUBSTITUTION_DTYPE](len(substitutions))
    var results_buffer = ctx.enqueue_create_buffer[SCORE_DTYPE](max(pairs, 1))

    with sequences_buffer.map_to_host() as host:
        for index in range(len(sequences)):
            host[index] = sequences[index]
    with offsets_buffer.map_to_host() as host:
        for index in range(len(offsets)):
            host[index] = offsets[index]
    with substitutions_buffer.map_to_host() as host:
        for index in range(len(substitutions)):
            host[index] = substitutions[index]
    with results_buffer.map_to_host() as host:
        for index in range(max(pairs, 1)):
            host[index] = 0

    ctx.enqueue_function[wavefront_kernel[mode]](
        sequences_buffer.unsafe_ptr(),
        offsets_buffer.unsafe_ptr(),
        substitutions_buffer.unsafe_ptr(),
        results_buffer.unsafe_ptr(),
        Int32(alphabet_size),
        scoring.gap_opening,
        scoring.gap_extension,
        grid_dim=pairs,
        block_dim=THREADS_PER_BLOCK,
    )
    ctx.synchronize()

    var results = List[Int32](capacity=pairs)
    with results_buffer.map_to_host() as host:
        for index in range(pairs):
            results.append(host[index])
    return results^


comptime OPERATION_MASK = UInt8(0x03)
comptime DELETE_EXTENDS_FLAG = UInt8(0x04)
comptime INSERT_EXTENDS_FLAG = UInt8(0x08)
comptime ZERO_CELL_FLAG = UInt8(0x80)


def direct_align_kernel[
    mode: AlignmentMode
](
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    offsets: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    letters: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    changes: Pointer[Scalar[CHANGE_DTYPE], MutAnyOrigin],
    change_offsets: Pointer[Scalar[DType.int64], MutAnyOrigin],
    results: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    first_gapped: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    second_gapped: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    gapped_lengths: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    gapped_stride: Int32,
    alphabet_size: Int32,
    gap_opening: Int32,
    gap_extension: Int32,
):
    """Anti-diagonal sweep that records every decision, then walks it back on the device.

    The decision byte carries the operation in its low bits and, for local alignment, a flag
    marking a cell whose score is zero. That flag stands in for the `scores[i, j] > 0` test the
    Python traceback performs, so the score matrix itself never has to be kept.
    """
    var pair = Int(block_idx.x)
    var first_start = Int(offsets[unsafe_offset=2 * pair])
    var second_start = Int(offsets[unsafe_offset=2 * pair + 1])
    var rows = second_start - first_start
    var columns = Int(offsets[unsafe_offset=2 * pair + 2]) - second_start
    var width = Int(alphabet_size)
    var stride = columns + 1
    var tile = Int(change_offsets[unsafe_offset=pair])

    var bands = stack_allocation[
        7 * (MAX_BAND_LENGTH + 1), Scalar[SCORE_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var best_scores = stack_allocation[
        THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var best_places = stack_allocation[
        THREADS_PER_BLOCK, Scalar[DType.int64], address_space = AddressSpace.SHARED
    ]()

    var band = MAX_BAND_LENGTH + 1
    var scores_two_back = bands
    var scores_one_back = bands.unsafe_offset(band)
    var scores_current = bands.unsafe_offset(2 * band)
    var deletes_one_back = bands.unsafe_offset(3 * band)
    var deletes_current = bands.unsafe_offset(4 * band)
    var inserts_one_back = bands.unsafe_offset(5 * band)
    var inserts_current = bands.unsafe_offset(6 * band)

    # The substitution table is read once per cell, so it is staged into shared memory rather
    # than fetched from global on every comparison. It is at most a kilobyte.
    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space = AddressSpace.SHARED,
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        scores_table[unsafe_offset=index] = substitutions[unsafe_offset=index]
    barrier()

    var best = Int32(0)
    var best_place = Int64(0)
    var answer = Int32(0)
    var last_diagonal = rows + columns

    for diagonal in range(0, last_diagonal + 1):
        var row_low = max(0, diagonal - columns)
        var row_high = min(rows, diagonal)

        for row in range(row_low + Int(thread_idx.x), row_high + 1, THREADS_PER_BLOCK):
            var column = diagonal - row
            var cell = Int32(0)
            var deletion = Int32(0)
            var insertion = Int32(0)
            var change = Operation.INSERT

            if row == 0 and column == 0:
                cell = 0
                deletion = cell + gap_opening + gap_extension
                insertion = deletion
            elif row == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = gap_opening + Int32(column - 1) * gap_extension
                    deletion = cell + gap_opening + gap_extension
                else:
                    cell = 0
                    deletion = gap_opening + gap_extension
                insertion = deletion
                change = Operation.INSERT
            elif column == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = gap_opening + Int32(row - 1) * gap_extension
                    insertion = cell + gap_opening + gap_extension
                else:
                    cell = 0
                    insertion = gap_opening + gap_extension
                deletion = insertion
                change = Operation.DELETE
            else:
                var left_symbol = Int(sequences[unsafe_offset=first_start + row - 1])
                var right_symbol = Int(sequences[unsafe_offset=second_start + column - 1])
                var substitution = Int32(
                    scores_table[unsafe_offset=left_symbol * width + right_symbol]
                )
                deletion = max(
                    scores_one_back[unsafe_offset=row - 1] + gap_opening,
                    deletes_one_back[unsafe_offset=row - 1] + gap_extension,
                )
                insertion = max(
                    scores_one_back[unsafe_offset=row] + gap_opening,
                    inserts_one_back[unsafe_offset=row] + gap_extension,
                )
                var replacement = scores_two_back[unsafe_offset=row - 1] + substitution
                cell = max(max(replacement, deletion), insertion)
                if mode == AlignmentMode.LOCAL:
                    cell = max(cell, Int32(0))

                if cell == replacement:
                    change = Operation.MATCH if left_symbol == right_symbol else Operation.SUBSTITUTE
                elif cell == deletion:
                    change = Operation.DELETE
                else:
                    change = Operation.INSERT
                if (
                    deletes_one_back[unsafe_offset=row - 1] + gap_extension
                    > scores_one_back[unsafe_offset=row - 1] + gap_opening
                ):
                    change |= DELETE_EXTENDS_FLAG
                if (
                    inserts_one_back[unsafe_offset=row] + gap_extension
                    > scores_one_back[unsafe_offset=row] + gap_opening
                ):
                    change |= INSERT_EXTENDS_FLAG

                if mode == AlignmentMode.LOCAL:
                    var place = Int64(row) * Int64(stride) + Int64(column)
                    if cell > best or (cell == best and cell != 0 and place < best_place):
                        if cell > best:
                            best = cell
                            best_place = place
                        elif place < best_place:
                            best_place = place

            scores_current[unsafe_offset=row] = cell
            deletes_current[unsafe_offset=row] = deletion
            inserts_current[unsafe_offset=row] = insertion
            if mode == AlignmentMode.LOCAL and cell == 0:
                change |= ZERO_CELL_FLAG
            changes[unsafe_offset=tile + row * stride + column] = change
            if diagonal == last_diagonal and row == rows:
                answer = cell

        barrier()

        var recycled = scores_two_back
        scores_two_back = scores_one_back
        scores_one_back = scores_current
        scores_current = recycled
        swap(deletes_one_back, deletes_current)
        swap(inserts_one_back, inserts_current)

    # Pick the first row-major maximum, matching the strict `>` of the Python scan.
    best_scores[unsafe_offset=Int(thread_idx.x)] = best
    best_places[unsafe_offset=Int(thread_idx.x)] = best_place
    barrier()
    var span = THREADS_PER_BLOCK // 2
    while span > 0:
        if Int(thread_idx.x) < span:
            var here = Int(thread_idx.x)
            var there = here + span
            var mine = best_scores[unsafe_offset=here]
            var theirs = best_scores[unsafe_offset=there]
            if theirs > mine or (
                theirs == mine
                and theirs != 0
                and best_places[unsafe_offset=there] < best_places[unsafe_offset=here]
            ):
                best_scores[unsafe_offset=here] = theirs
                best_places[unsafe_offset=here] = best_places[unsafe_offset=there]
        barrier()
        span //= 2

    if thread_idx.x != 0:
        return

    var start_row = rows
    var start_column = columns
    var final_score = answer
    if mode == AlignmentMode.LOCAL:
        var place = best_places[unsafe_offset=0]
        final_score = best_scores[unsafe_offset=0]
        start_row = Int(place // Int64(stride))
        start_column = Int(place % Int64(stride))
        if final_score == 0:
            start_row = 0
            start_column = 0

    var row = start_row
    var column = start_column
    var produced = 0
    var base = pair * Int(gapped_stride)
    var state = Operation.MATCH

    while row > 0 and column > 0:
        var change = changes[unsafe_offset=tile + row * stride + column]
        if state == Operation.MATCH:
            if mode == AlignmentMode.LOCAL and (change & ZERO_CELL_FLAG) != 0:
                break
            var operation = change & OPERATION_MASK
            if operation == Operation.DELETE:
                state = Operation.DELETE
            elif operation == Operation.INSERT:
                state = Operation.INSERT
            else:
                first_gapped[unsafe_offset=base + produced] = letters[
                    unsafe_offset=Int(sequences[unsafe_offset=first_start + row - 1])
                ]
                second_gapped[unsafe_offset=base + produced] = letters[
                    unsafe_offset=Int(sequences[unsafe_offset=second_start + column - 1])
                ]
                row -= 1
                column -= 1
                produced += 1
        elif state == Operation.DELETE:
            first_gapped[unsafe_offset=base + produced] = letters[
                unsafe_offset=Int(sequences[unsafe_offset=first_start + row - 1])
            ]
            second_gapped[unsafe_offset=base + produced] = Scalar[SYMBOL_DTYPE](45)
            row -= 1
            produced += 1
            state = Operation.DELETE if (change & DELETE_EXTENDS_FLAG) != 0 else Operation.MATCH
        else:
            first_gapped[unsafe_offset=base + produced] = Scalar[SYMBOL_DTYPE](45)
            second_gapped[unsafe_offset=base + produced] = letters[
                unsafe_offset=Int(sequences[unsafe_offset=second_start + column - 1])
            ]
            column -= 1
            produced += 1
            state = Operation.INSERT if (change & INSERT_EXTENDS_FLAG) != 0 else Operation.MATCH

    # Only a global path is required to reach the origin; see the host reconstruction.
    if mode == AlignmentMode.GLOBAL:
        while row > 0:
            first_gapped[unsafe_offset=base + produced] = letters[
                unsafe_offset=Int(sequences[unsafe_offset=first_start + row - 1])
            ]
            second_gapped[unsafe_offset=base + produced] = Scalar[SYMBOL_DTYPE](45)
            row -= 1
            produced += 1
        while column > 0:
            first_gapped[unsafe_offset=base + produced] = Scalar[SYMBOL_DTYPE](45)
            second_gapped[unsafe_offset=base + produced] = letters[
                unsafe_offset=Int(sequences[unsafe_offset=second_start + column - 1])
            ]
            column -= 1
            produced += 1

    results[unsafe_offset=pair] = final_score
    gapped_lengths[unsafe_offset=pair] = Int32(produced)


def direct_alignments[
    mode: AlignmentMode
](
    ctx: DeviceContext,
    sequences: List[Scalar[SYMBOL_DTYPE]],
    offsets: List[Scalar[SCORE_DTYPE]],
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet: String,
    scoring: AffineScoring,
) raises -> List[AlignmentResult]:
    """Aligns a batch on the GPU by recording every decision, one thread block per pair."""
    var alphabet_bytes = alphabet.as_bytes()
    var alphabet_size = len(alphabet_bytes)
    if alphabet_size > MAX_ALPHABET_SIZE:
        raise Error("Alphabet exceeds the shared-memory substitution table capacity.")
    var pairs = (len(offsets) - 1) // 2

    var change_offsets = List[Int64](capacity=pairs + 1)
    var running = Int64(0)
    var widest = 1
    for pair in range(pairs):
        var rows = Int(offsets[2 * pair + 1]) - Int(offsets[2 * pair])
        var columns = Int(offsets[2 * pair + 2]) - Int(offsets[2 * pair + 1])
        change_offsets.append(running)
        running += Int64(rows + 1) * Int64(columns + 1)
        widest = max(widest, rows + columns)
    change_offsets.append(running)

    var sequences_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(len(sequences), 1))
    var offsets_buffer = ctx.enqueue_create_buffer[SCORE_DTYPE](len(offsets))
    var substitutions_buffer = ctx.enqueue_create_buffer[SUBSTITUTION_DTYPE](len(substitutions))
    var letters_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](alphabet_size)
    var changes_buffer = ctx.enqueue_create_buffer[CHANGE_DTYPE](Int(max(running, Int64(1))))
    var change_offsets_buffer = ctx.enqueue_create_buffer[DType.int64](pairs + 1)
    var results_buffer = ctx.enqueue_create_buffer[SCORE_DTYPE](max(pairs, 1))
    var first_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(pairs * widest, 1))
    var second_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(pairs * widest, 1))
    var lengths_buffer = ctx.enqueue_create_buffer[SCORE_DTYPE](max(pairs, 1))

    with sequences_buffer.map_to_host() as host:
        for index in range(len(sequences)):
            host[index] = sequences[index]
    with offsets_buffer.map_to_host() as host:
        for index in range(len(offsets)):
            host[index] = offsets[index]
    with substitutions_buffer.map_to_host() as host:
        for index in range(len(substitutions)):
            host[index] = substitutions[index]
    with letters_buffer.map_to_host() as host:
        for index in range(alphabet_size):
            host[index] = Scalar[SYMBOL_DTYPE](alphabet_bytes[index])
    with change_offsets_buffer.map_to_host() as host:
        for index in range(pairs + 1):
            host[index] = change_offsets[index]
    with results_buffer.map_to_host() as host:
        for index in range(max(pairs, 1)):
            host[index] = 0
    with lengths_buffer.map_to_host() as host:
        for index in range(max(pairs, 1)):
            host[index] = 0

    ctx.enqueue_function[direct_align_kernel[mode]](
        sequences_buffer.unsafe_ptr(),
        offsets_buffer.unsafe_ptr(),
        substitutions_buffer.unsafe_ptr(),
        letters_buffer.unsafe_ptr(),
        changes_buffer.unsafe_ptr(),
        change_offsets_buffer.unsafe_ptr(),
        results_buffer.unsafe_ptr(),
        first_buffer.unsafe_ptr(),
        second_buffer.unsafe_ptr(),
        lengths_buffer.unsafe_ptr(),
        Int32(widest),
        Int32(alphabet_size),
        scoring.gap_opening,
        scoring.gap_extension,
        grid_dim=pairs,
        block_dim=THREADS_PER_BLOCK,
    )
    ctx.synchronize()

    var aligned = List[AlignmentResult](capacity=pairs)
    with results_buffer.map_to_host() as scores_host, lengths_buffer.map_to_host() as lengths_host, first_buffer.map_to_host() as first_host, second_buffer.map_to_host() as second_host:
        for pair in range(pairs):
            var produced = Int(lengths_host[pair])
            var base = pair * widest
            var left = List[UInt8](capacity=produced + 1)
            var right = List[UInt8](capacity=produced + 1)
            for index in range(produced - 1, -1, -1):
                left.append(UInt8(first_host[base + index]))
                right.append(UInt8(second_host[base + index]))
            aligned.append(
                AlignmentResult(
                    scores_host[pair],
                    String(unsafe_from_utf8=left),
                    String(unsafe_from_utf8=right),
                )
            )
    return aligned^


def hirschberg_path_gpu(
    ctx: DeviceContext,
    first: List[Scalar[SYMBOL_DTYPE]],
    second: List[Scalar[SYMBOL_DTYPE]],
    window_first_from: Int,
    window_first_to: Int,
    window_second_from: Int,
    window_second_to: Int,
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
    tile_cells: Int,
    mut path_columns: List[Int32],
    mut path_entries: List[UInt8],
) raises:
    """Hirschberg with every sweep on the device.

    The recursion itself is a few hundred bookkeeping steps and stays on the host; the sweeps carry
    the whole quadratic cost and run as kernels over global-memory bands, so nothing here is bounded
    by shared memory. Each launch is its own barrier, which is what stands in for the grid-wide sync
    Mojo does not expose.
    """
    if scoring.gap_opening > scoring.gap_extension:
        raise Error("Gap opening must be at least as expensive as extension for the row split.")

    var rows = len(first)
    var columns = len(second)
    path_columns[window_first_to] = Int32(window_second_to)

    var sequences = List[Scalar[SYMBOL_DTYPE]](capacity=rows + columns)
    for index in range(rows):
        sequences.append(first[index])
    for index in range(columns):
        sequences.append(second[index])

    var tile_rows_count = (rows + TILE_SIDE - 1) // TILE_SIDE + 2
    var tile_columns_count = (columns + TILE_SIDE - 1) // TILE_SIDE + 2
    var buffers = SweepBuffers(
        ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(rows + columns, 1)),
        ctx.enqueue_create_buffer[SUBSTITUTION_DTYPE](len(substitutions)),
        ctx.enqueue_create_buffer[SCORE_DTYPE](columns + 2),
        ctx.enqueue_create_buffer[SCORE_DTYPE](columns + 2),
        ctx.enqueue_create_buffer[SCORE_DTYPE](columns + 2),
        ctx.enqueue_create_buffer[SCORE_DTYPE](columns + 2),
        ctx.enqueue_create_buffer[SCORE_DTYPE](4),
        ctx.enqueue_create_buffer[SCORE_DTYPE](rows + 2),
        ctx.enqueue_create_buffer[SCORE_DTYPE](rows + 2),
        ctx.enqueue_create_buffer[SCORE_DTYPE](tile_rows_count * tile_columns_count),
    )

    with buffers.sequences.map_to_host() as host:
        for index in range(len(sequences)):
            host[index] = sequences[index]
    with buffers.substitutions.map_to_host() as host:
        for index in range(len(substitutions)):
            host[index] = substitutions[index]

    var frames = List[Frame]()
    frames.append(
        Frame(
            window_first_from, window_first_to, window_second_from, window_second_to,
            Boundary.OPENS, Boundary.OPENS,
        )
    )

    while len(frames) > 0:
        var frame = frames.pop()
        var first_from = frame.first_from
        var first_to = frame.first_to
        var second_from = frame.second_from
        var second_to = frame.second_to
        var top = frame.top
        var bottom = frame.bottom

        var height = first_to - first_from
        var width = second_to - second_from
        if height == 0:
            continue
        if width == 0 or height <= 2 or (height + 1) * (width + 1) <= tile_cells:
            _ = direct_tile(
                first, second, first_from, first_to, second_from, second_to, top, bottom,
                substitutions, alphabet_size, scoring, path_columns, path_entries,
            )
            continue

        var split = (first_from + first_to) // 2
        # The second sequence is stored after the first, so its indices carry that offset.
        tiled_sweep(
            ctx, buffers, split - first_from, second_to - second_from,
            first_from, rows + second_from, False, top,
            alphabet_size, scoring, False,
        )
        tiled_sweep(
            ctx, buffers, first_to - split, second_to - second_from,
            split, rows + second_from, True, bottom,
            alphabet_size, scoring, True,
        )
        ctx.enqueue_function[crossing_kernel](
            buffers.top_scores.unsafe_ptr(),
            buffers.top_deletes.unsafe_ptr(),
            buffers.reverse_scores.unsafe_ptr(),
            buffers.reverse_deletes.unsafe_ptr(),
            buffers.crossing.unsafe_ptr(),
            Int32(width),
            scoring.gap_extension - scoring.gap_opening,
            grid_dim=1,
            block_dim=1,
        )
        ctx.synchronize()

        var best_plain = NEGATIVE_INFINITY
        var best_plain_column = 0
        var best_gapped = NEGATIVE_INFINITY
        with buffers.crossing.map_to_host() as host:
            best_plain = host[0]
            best_plain_column = Int(host[1])
            best_gapped = host[2]

        # Take the answer from the branch actually followed. The gapped candidate can overshoot
        # the true optimum on degenerate rectangles, where the delete layers at the extreme
        # columns are still sentinels rather than reachable values.
        if best_gapped > best_plain:
            _ = direct_tile(
                first, second, first_from, first_to, second_from, second_to, top, bottom,
                substitutions, alphabet_size, scoring, path_columns, path_entries,
            )
            continue

        var crossing = second_from + best_plain_column
        # Lower half first, so the upper half pops first and the two row ranges tile in order.
        frames.append(Frame(split, first_to, crossing, second_to, Boundary.OPENS, bottom))
        frames.append(Frame(first_from, split, second_from, crossing, top, Boundary.OPENS))


def local_extremum_kernel(
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    bands: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    outcome: Pointer[Scalar[DType.int64], MutAnyOrigin],
    first_from: Int32,
    first_to: Int32,
    second_from: Int32,
    second_to: Int32,
    backwards: Int32,
    alphabet_size: Int32,
    gap_opening: Int32,
    gap_extension: Int32,
):
    """Local sweep reporting the first row-major maximum, with its bands in global memory.

    Local alignment has unknown endpoints, so this runs twice: forwards it says where the best
    alignment ends, and backwards over those prefixes it says how far back the same alignment
    reaches. Ties resolve to the earliest cell in row-major order, matching the Python's scan.
    """
    var rows = Int(first_to - first_from)
    var columns = Int(second_to - second_from)
    var width = Int(alphabet_size)
    var stride = rows + 1
    var reversed_order = backwards != 0

    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space = AddressSpace.SHARED,
    ]()
    var best_scores = stack_allocation[
        THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var best_places = stack_allocation[
        THREADS_PER_BLOCK, Scalar[DType.int64], address_space = AddressSpace.SHARED
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        scores_table[unsafe_offset=index] = substitutions[unsafe_offset=index]
    barrier()

    var scores_two_back = bands
    var scores_one_back = bands.unsafe_offset(stride)
    var scores_current = bands.unsafe_offset(2 * stride)
    var deletes_one_back = bands.unsafe_offset(3 * stride)
    var deletes_current = bands.unsafe_offset(4 * stride)
    var inserts_one_back = bands.unsafe_offset(5 * stride)
    var inserts_current = bands.unsafe_offset(6 * stride)

    var best = Int32(0)
    var best_place = Int64(0)
    var last_diagonal = rows + columns

    for diagonal in range(0, last_diagonal + 1):
        var row_low = max(0, diagonal - columns)
        var row_high = min(rows, diagonal)

        for row in range(row_low + Int(thread_idx.x), row_high + 1, Int(block_dim.x)):
            var column = diagonal - row
            var cell = Int32(0)
            var deletion = gap_opening + gap_extension
            var insertion = gap_opening + gap_extension

            if row > 0 and column > 0:
                var left_index = (
                    Int(first_to) - row if reversed_order else Int(first_from) + row - 1
                )
                var right_index = (
                    Int(second_to) - column if reversed_order else Int(second_from) + column - 1
                )
                var left_symbol = Int(sequences[unsafe_offset=left_index])
                var right_symbol = Int(sequences[unsafe_offset=right_index])
                var substitution = Int32(
                    scores_table[unsafe_offset=left_symbol * width + right_symbol]
                )
                deletion = max(
                    scores_one_back[unsafe_offset=row - 1] + gap_opening,
                    deletes_one_back[unsafe_offset=row - 1] + gap_extension,
                )
                insertion = max(
                    scores_one_back[unsafe_offset=row] + gap_opening,
                    inserts_one_back[unsafe_offset=row] + gap_extension,
                )
                cell = max(
                    max(max(scores_two_back[unsafe_offset=row - 1] + substitution, deletion), insertion),
                    Int32(0),
                )
                var place = Int64(row) * Int64(columns + 1) + Int64(column)
                if cell > best or (cell == best and cell != 0 and place < best_place):
                    if cell > best:
                        best = cell
                        best_place = place
                    elif place < best_place:
                        best_place = place

            scores_current[unsafe_offset=row] = cell
            deletes_current[unsafe_offset=row] = deletion
            inserts_current[unsafe_offset=row] = insertion

        barrier()

        var recycled = scores_two_back
        scores_two_back = scores_one_back
        scores_one_back = scores_current
        scores_current = recycled
        swap(deletes_one_back, deletes_current)
        swap(inserts_one_back, inserts_current)

    best_scores[unsafe_offset=Int(thread_idx.x)] = best
    best_places[unsafe_offset=Int(thread_idx.x)] = best_place
    barrier()
    var span = THREADS_PER_BLOCK // 2
    while span > 0:
        if Int(thread_idx.x) < span:
            var here = Int(thread_idx.x)
            var there = here + span
            var mine = best_scores[unsafe_offset=here]
            var theirs = best_scores[unsafe_offset=there]
            if theirs > mine or (
                theirs == mine
                and theirs != 0
                and best_places[unsafe_offset=there] < best_places[unsafe_offset=here]
            ):
                best_scores[unsafe_offset=here] = theirs
                best_places[unsafe_offset=here] = best_places[unsafe_offset=there]
        barrier()
        span //= 2

    if thread_idx.x == 0:
        var place = best_places[unsafe_offset=0]
        outcome[unsafe_offset=0] = place // Int64(columns + 1)
        outcome[unsafe_offset=1] = place % Int64(columns + 1)
        outcome[unsafe_offset=2] = Int64(best_scores[unsafe_offset=0])


# Subproblems at or below this many cells are solved outright rather than split. Small enough
# that a leaf stays cheap, large enough that the recursion does not dominate.
comptime DEFAULT_TILE_CELLS = 4096

comptime TILE_SIDE = 256


def tiled_sweep_kernel(
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    top_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    top_deletes: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    left_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    left_inserts: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    corner_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    tile_diagonal: Int32,
    tile_row_low: Int32,
    tile_columns_count: Int32,
    rows: Int32,
    columns: Int32,
    first_from: Int32,
    second_from: Int32,
    backwards: Int32,
    entry_extends: Int32,
    alphabet_size: Int32,
    gap_opening: Int32,
    gap_extension: Int32,
):
    """One tile of the matrix per block, one tile-anti-diagonal per launch.

    Tiles sharing an anti-diagonal are independent, so a launch boundary supplies the barrier that
    Mojo 1.0 has no grid-wide primitive for. The state crossing a tile edge is small: a downward
    move carries the deletion layer, a rightward move carries the insertion layer, and the
    diagonal predecessor of a tile's own corner is kept per tile so nothing aliases.
    """
    var tile_row = Int(tile_row_low) + Int(block_idx.x)
    var tile_column = Int(tile_diagonal) - tile_row
    var row_begin = tile_row * TILE_SIDE
    var column_begin = tile_column * TILE_SIDE
    var height = min(TILE_SIDE, Int(rows) - row_begin)
    var width_span = min(TILE_SIDE, Int(columns) - column_begin)
    if height <= 0 or width_span <= 0:
        return

    var width = Int(alphabet_size)
    var stride = TILE_SIDE + 1
    var reversed_order = backwards != 0
    var extends = entry_extends != 0

    var bands = stack_allocation[
        7 * (TILE_SIDE + 1), Scalar[SCORE_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space = AddressSpace.SHARED,
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        scores_table[unsafe_offset=index] = substitutions[unsafe_offset=index]
    barrier()

    var scores_two_back = bands
    var scores_one_back = bands.unsafe_offset(stride)
    var scores_current = bands.unsafe_offset(2 * stride)
    var deletes_one_back = bands.unsafe_offset(3 * stride)
    var deletes_current = bands.unsafe_offset(4 * stride)
    var inserts_one_back = bands.unsafe_offset(5 * stride)
    var inserts_current = bands.unsafe_offset(6 * stride)

    var last_diagonal = height + width_span
    for diagonal in range(0, last_diagonal + 1):
        var row_low = max(0, diagonal - width_span)
        var row_high = min(height, diagonal)

        for local_row in range(row_low + Int(thread_idx.x), row_high + 1, Int(block_dim.x)):
            var local_column = diagonal - local_row
            var row = row_begin + local_row
            var column = column_begin + local_column
            var cell = Int32(0)
            var deletion = gap_opening + gap_extension
            var insertion = gap_opening + gap_extension

            if local_row == 0 or local_column == 0:
                # A tile edge. Outside the matrix it is the affine ramp; inside it is whatever the
                # neighbouring tile left behind.
                if row == 0 and column == 0:
                    cell = 0
                    deletion = gap_opening + gap_extension
                    insertion = deletion
                elif row == 0:
                    cell = gap_opening + Int32(column - 1) * gap_extension
                    deletion = cell if extends else cell + gap_opening + gap_extension
                    insertion = deletion
                elif column == 0:
                    cell = Int32(row) * gap_extension if extends else gap_opening + Int32(
                        row - 1
                    ) * gap_extension
                    deletion = cell
                    insertion = cell + gap_opening + gap_extension
                elif local_row == 0 and local_column == 0:
                    cell = corner_scores[
                        unsafe_offset=tile_row * Int(tile_columns_count) + tile_column
                    ]
                    deletion = cell + gap_opening + gap_extension
                    insertion = deletion
                elif local_row == 0:
                    cell = top_scores[unsafe_offset=column]
                    deletion = top_deletes[unsafe_offset=column]
                    insertion = cell + gap_opening + gap_extension
                else:
                    cell = left_scores[unsafe_offset=row]
                    insertion = left_inserts[unsafe_offset=row]
                    deletion = cell + gap_opening + gap_extension
            else:
                var left_index = (
                    Int(first_from) + Int(rows) - row if reversed_order else Int(first_from) + row - 1
                )
                var right_index = (
                    Int(second_from) + Int(columns) - column
                    if reversed_order
                    else Int(second_from) + column - 1
                )
                var left_symbol = Int(sequences[unsafe_offset=left_index])
                var right_symbol = Int(sequences[unsafe_offset=right_index])
                var substitution = Int32(
                    scores_table[unsafe_offset=left_symbol * width + right_symbol]
                )
                deletion = max(
                    scores_one_back[unsafe_offset=local_row - 1] + gap_opening,
                    deletes_one_back[unsafe_offset=local_row - 1] + gap_extension,
                )
                insertion = max(
                    scores_one_back[unsafe_offset=local_row] + gap_opening,
                    inserts_one_back[unsafe_offset=local_row] + gap_extension,
                )
                cell = max(
                    max(scores_two_back[unsafe_offset=local_row - 1] + substitution, deletion),
                    insertion,
                )

            scores_current[unsafe_offset=local_row] = cell
            deletes_current[unsafe_offset=local_row] = deletion
            inserts_current[unsafe_offset=local_row] = insertion

            # Hand the tile's own edges to the neighbours, but only for cells this tile actually
            # computed. Its own left edge carries a fabricated deletion and its own top edge a
            # fabricated insertion; publishing either clobbers the true value the neighbouring
            # tile wrote, and races with it, since the two share a tile-anti-diagonal. The matrix
            # borders are genuine and do publish.
            if local_row == height and (local_column > 0 or column == 0):
                top_scores[unsafe_offset=column] = cell
                top_deletes[unsafe_offset=column] = deletion
            if local_column == width_span and (local_row > 0 or row == 0):
                left_scores[unsafe_offset=row] = cell
                left_inserts[unsafe_offset=row] = insertion
            if local_row == height and local_column == width_span:
                corner_scores[
                    unsafe_offset=(tile_row + 1) * Int(tile_columns_count) + tile_column + 1
                ] = cell

        barrier()

        var recycled = scores_two_back
        scores_two_back = scores_one_back
        scores_one_back = scores_current
        scores_current = recycled
        swap(deletes_one_back, deletes_current)
        swap(inserts_one_back, inserts_current)


@fieldwise_init
struct SweepBuffers(Movable):
    """Device scratch reused across every sweep of one alignment."""

    var sequences: DeviceBuffer[SYMBOL_DTYPE]
    var substitutions: DeviceBuffer[SUBSTITUTION_DTYPE]
    var top_scores: DeviceBuffer[SCORE_DTYPE]
    var top_deletes: DeviceBuffer[SCORE_DTYPE]
    var reverse_scores: DeviceBuffer[SCORE_DTYPE]
    var reverse_deletes: DeviceBuffer[SCORE_DTYPE]
    var crossing: DeviceBuffer[SCORE_DTYPE]
    var left_scores: DeviceBuffer[SCORE_DTYPE]
    var left_inserts: DeviceBuffer[SCORE_DTYPE]
    var corner_scores: DeviceBuffer[SCORE_DTYPE]


def crossing_kernel(
    forward_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    forward_deletes: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    reverse_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    reverse_deletes: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    crossing: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    width: Int32,
    refund: Int32,
):
    """Picks where the optimal path crosses the cut, without shipping two bands to the host.

    Reading four frontier arrays back per split costs more than the sweeps themselves once the
    recursion is thousands of nodes deep, so the reduction happens here and three numbers travel.
    """
    var best_plain = NEGATIVE_INFINITY
    var best_plain_column = Int32(0)
    var best_gapped = NEGATIVE_INFINITY
    var span = Int(width)
    for offset in range(span + 1):
        var plain = forward_scores[unsafe_offset=offset] + reverse_scores[unsafe_offset=span - offset]
        if plain > best_plain:
            best_plain = plain
            best_plain_column = Int32(offset)
        var gapped = (
            forward_deletes[unsafe_offset=offset]
            + reverse_deletes[unsafe_offset=span - offset]
            + refund
        )
        if gapped > best_gapped:
            best_gapped = gapped
    crossing[unsafe_offset=0] = best_plain
    crossing[unsafe_offset=1] = best_plain_column
    crossing[unsafe_offset=2] = best_gapped


def tiled_sweep(
    ctx: DeviceContext,
    mut buffers: SweepBuffers,
    rows: Int,
    columns: Int,
    first_from: Int,
    second_from: Int,
    backwards: Bool,
    entry: Boundary,
    alphabet_size: Int,
    scoring: AffineScoring,
    into_reverse: Bool,
) raises:
    """Sweeps one sub-rectangle as a wavefront of tiles, one launch per tile-anti-diagonal.

    The last tile row leaves the matrix's final row in the top frontier, which is exactly the pair
    of layers a Hirschberg join consumes, so no separate capture is needed.
    """
    var tile_rows_count = (rows + TILE_SIDE - 1) // TILE_SIDE
    var tile_columns_count = (columns + TILE_SIDE - 1) // TILE_SIDE
    if tile_rows_count == 0:
        tile_rows_count = 1
    if tile_columns_count == 0:
        tile_columns_count = 1

    for tile_diagonal in range(tile_rows_count + tile_columns_count - 1):
        var tile_row_low = max(0, tile_diagonal - (tile_columns_count - 1))
        var tile_row_high = min(tile_diagonal, tile_rows_count - 1)
        var tiles = tile_row_high - tile_row_low + 1
        ctx.enqueue_function[tiled_sweep_kernel](
            buffers.sequences.unsafe_ptr(),
            buffers.substitutions.unsafe_ptr(),
            buffers.reverse_scores.unsafe_ptr() if into_reverse else buffers.top_scores.unsafe_ptr(),
            buffers.reverse_deletes.unsafe_ptr() if into_reverse else buffers.top_deletes.unsafe_ptr(),
            buffers.left_scores.unsafe_ptr(),
            buffers.left_inserts.unsafe_ptr(),
            buffers.corner_scores.unsafe_ptr(),
            Int32(tile_diagonal),
            Int32(tile_row_low),
            Int32(tile_columns_count + 1),
            Int32(rows),
            Int32(columns),
            Int32(first_from),
            Int32(second_from),
            Int32(1) if backwards else Int32(0),
            Int32(entry.identifier),
            Int32(alphabet_size),
            scoring.gap_opening,
            scoring.gap_extension,
            grid_dim=tiles,
            block_dim=THREADS_PER_BLOCK,
        )

# endregion GPU Wavefront


# region Presentation

def colorize(first_gapped: String, second_gapped: String) raises -> Tuple[String, String]:
    """Green for a match, red for a mismatch, dim for a gap, mirroring `colorize_alignment`."""
    comptime green = "\x1b[32m"
    comptime red = "\x1b[31m"
    comptime white = "\x1b[37m"
    comptime reset = "\x1b[0m"
    var left = String()
    var right = String()
    var top = first_gapped.as_bytes()
    var bottom = second_gapped.as_bytes()
    if len(top) != len(bottom):
        raise Error("Both aligned strings must have the same length.")
    for index in range(len(top)):
        var color = red
        if top[index] == bottom[index] and top[index] != UInt8(45):
            color = green
        elif top[index] == UInt8(45) or bottom[index] == UInt8(45):
            color = white
        var top_byte: List[UInt8] = [top[index]]
        var bottom_byte: List[UInt8] = [bottom[index]]
        left += String(color, String(unsafe_from_utf8=top_byte), reset)
        right += String(color, String(unsafe_from_utf8=bottom_byte), reset)
    return (left, right)


# endregion Presentation

# region Python Bindings

def optional_int(value: PythonObject) -> Optional[Int]:
    """Reads a Python argument that may be `None`, without asking Python what `None` is."""
    try:
        var present: Optional[Int] = Int(String(value))
        return present
    except:
        return None


def scoring_from(gap_opening: PythonObject, gap_extension: PythonObject) -> AffineScoring:
    """Applies the defaults of `_validate_gotoh_arguments` to the two gap penalties."""
    var opening = optional_int(gap_opening).or_else(Int(DEFAULT_GAP_OPENING))
    var extension = optional_int(gap_extension).or_else(Int(DEFAULT_GAP_EXTENSION))
    return AffineScoring(Int32(opening), Int32(extension))


def matrix_from(
    match_score: PythonObject, mismatch_score: PythonObject, alphabet_size: Int
) raises -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """BLOSUM62 unless both `match` and `mismatch` are supplied, mirroring the Python."""
    var given_match = optional_int(match_score)
    var given_mismatch = optional_int(mismatch_score)
    if Bool(given_match) != Bool(given_mismatch):
        raise Error("Both match and mismatch must be provided.")
    if not given_match:
        return default_proteins_matrix()
    return uniform_matrix(alphabet_size, given_match.value(), given_mismatch.value())


def gotoh_score[
    mode: AlignmentMode
](
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)
    return PythonObject(Int(serial_score[mode](left, right, substitutions, alphabet_size, scoring)))


def gotoh_alignment[
    mode: AlignmentMode
](
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)
    var result = serial_align[mode](left, right, substitutions, alphabet_size, scoring, alphabet)
    var triple = Python().list()
    triple.append(PythonObject(result.first_gapped))
    triple.append(PythonObject(result.second_gapped))
    triple.append(PythonObject(Int(result.score)))
    return triple


def needleman_wunsch_gotoh_score(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_score[AlignmentMode.GLOBAL](
        first, second, gap_opening, gap_extension, match_score, mismatch_score
    )


def smith_waterman_gotoh_score(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_score[AlignmentMode.LOCAL](
        first, second, gap_opening, gap_extension, match_score, mismatch_score
    )


def needleman_wunsch_gotoh_alignment(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_alignment[AlignmentMode.GLOBAL](
        first, second, gap_opening, gap_extension, match_score, mismatch_score
    )


def smith_waterman_gotoh_alignment(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_alignment[AlignmentMode.LOCAL](
        first, second, gap_opening, gap_extension, match_score, mismatch_score
    )


def python_length(value: PythonObject) raises -> Int:
    return Int(String(value.__len__()))


def gotoh_scores_batch[
    mode: AlignmentMode
](
    firsts: PythonObject,
    seconds: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    """Scores a whole batch on the GPU, one thread block per pair."""
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)

    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise Error("Both sides of the batch must have the same length.")
    if pairs == 0:
        return Python().list()

    var sequences = List[Scalar[SYMBOL_DTYPE]]()
    var offsets = List[Scalar[SCORE_DTYPE]]()
    offsets.append(0)
    for index in range(pairs):
        var left = translate(String(firsts[index]), alphabet)
        if len(left) > MAX_BAND_LENGTH:
            raise Error("First sequence exceeds the shared-memory band capacity.")
        for symbol in range(len(left)):
            sequences.append(left[symbol])
        offsets.append(Scalar[SCORE_DTYPE](len(sequences)))
        var right = translate(String(seconds[index]), alphabet)
        for symbol in range(len(right)):
            sequences.append(right[symbol])
        offsets.append(Scalar[SCORE_DTYPE](len(sequences)))

    var ctx = DeviceContext()
    var scores = wavefront_scores[mode](
        ctx, sequences, offsets, substitutions, alphabet_size, scoring
    )
    var output = Python().list()
    for index in range(len(scores)):
        output.append(PythonObject(Int(scores[index])))
    return output


def needleman_wunsch_gotoh_scores_gpu(
    firsts: PythonObject,
    seconds: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_scores_batch[AlignmentMode.GLOBAL](
        firsts, seconds, gap_opening, gap_extension, match_score, mismatch_score
    )


def smith_waterman_gotoh_scores_gpu(
    firsts: PythonObject,
    seconds: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_scores_batch[AlignmentMode.LOCAL](
        firsts, seconds, gap_opening, gap_extension, match_score, mismatch_score
    )


def gotoh_alignments_batch[
    mode: AlignmentMode
](
    firsts: PythonObject,
    seconds: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    """Aligns a whole batch on the GPU, returning `[first_gapped, second_gapped, score]` triples."""
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)

    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise Error("Both sides of the batch must have the same length.")
    if pairs == 0:
        return Python().list()

    var sequences = List[Scalar[SYMBOL_DTYPE]]()
    var offsets = List[Scalar[SCORE_DTYPE]]()
    offsets.append(0)
    for index in range(pairs):
        var left = translate(String(firsts[index]), alphabet)
        if len(left) > MAX_BAND_LENGTH:
            raise Error("First sequence exceeds the shared-memory band capacity.")
        for symbol in range(len(left)):
            sequences.append(left[symbol])
        offsets.append(Scalar[SCORE_DTYPE](len(sequences)))
        var right = translate(String(seconds[index]), alphabet)
        for symbol in range(len(right)):
            sequences.append(right[symbol])
        offsets.append(Scalar[SCORE_DTYPE](len(sequences)))

    var ctx = DeviceContext()
    var aligned = direct_alignments[mode](
        ctx, sequences, offsets, substitutions, alphabet, scoring
    )
    var output = Python().list()
    for index in range(len(aligned)):
        var triple = Python().list()
        triple.append(PythonObject(aligned[index].first_gapped))
        triple.append(PythonObject(aligned[index].second_gapped))
        triple.append(PythonObject(Int(aligned[index].score)))
        output.append(triple)
    return output


def needleman_wunsch_gotoh_alignments_gpu(
    firsts: PythonObject,
    seconds: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_alignments_batch[AlignmentMode.GLOBAL](
        firsts, seconds, gap_opening, gap_extension, match_score, mismatch_score
    )


def smith_waterman_gotoh_alignments_gpu(
    firsts: PythonObject,
    seconds: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    return gotoh_alignments_batch[AlignmentMode.LOCAL](
        firsts, seconds, gap_opening, gap_extension, match_score, mismatch_score
    )


def combined_alphabet(first: String, second: String) -> String:
    """The distinct characters of both strings, so unit-cost alignment needs no fixed alphabet."""
    var seen = List[Bool](length=256, fill=False)
    var letters = List[UInt8]()
    var first_bytes = first.as_bytes()
    var second_bytes = second.as_bytes()
    for index in range(len(first_bytes)):
        if not seen[Int(first_bytes[index])]:
            seen[Int(first_bytes[index])] = True
            letters.append(first_bytes[index])
    for index in range(len(second_bytes)):
        if not seen[Int(second_bytes[index])]:
            seen[Int(second_bytes[index])] = True
            letters.append(second_bytes[index])
    if len(letters) == 0:
        letters.append(UInt8(65))
    return String(unsafe_from_utf8=letters)


def levenshtein_alignment(first: PythonObject, second: PythonObject) raises -> PythonObject:
    """Unit-cost edit distance, expressed as the global recurrence with linear gaps.

    Setting both penalties to minus one and the substitution scores to zero and minus one turns
    the maximizing Gotoh recurrence into the negated Levenshtein minimization, tie-break chain
    included, so this needs no kernel of its own.
    """
    var left = String(first)
    var right = String(second)
    for byte in left.as_bytes():
        if byte >= 0x80:
            raise Error("Unit-cost alignment handles ASCII only; the oracle compares code points.")
    for byte in right.as_bytes():
        if byte >= 0x80:
            raise Error("Unit-cost alignment handles ASCII only; the oracle compares code points.")
    var alphabet = combined_alphabet(left, right)
    var alphabet_size = alphabet.byte_length()
    var substitutions = uniform_matrix(alphabet_size, 0, -1)
    var scoring = AffineScoring(Int32(-1), Int32(-1))
    var encoded_left = translate(left, alphabet)
    var encoded_right = translate(right, alphabet)
    var result = serial_align[AlignmentMode.GLOBAL](
        encoded_left, encoded_right, substitutions, alphabet_size, scoring, alphabet
    )
    var triple = Python().list()
    triple.append(PythonObject(result.first_gapped))
    triple.append(PythonObject(result.second_gapped))
    triple.append(PythonObject(-Int(result.score)))
    return triple


def colorize_alignment(first_gapped: PythonObject, second_gapped: PythonObject) raises -> PythonObject:
    """Wraps each column in an ANSI colour: green for a match, red for a mismatch, dim for a gap."""
    var painted = colorize(String(first_gapped), String(second_gapped))
    var pair = Python().list()
    pair.append(PythonObject(painted[0]))
    pair.append(PythonObject(painted[1]))
    return pair


def needleman_wunsch_gotoh_alignment_linear(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    """Global alignment in linear space, splitting rows and joining halves Myers-Miller style.

    `tile_cells` is the only knob: subproblems at or below it are solved outright, above it they
    are split. Raising it past the whole matrix collapses to a single direct traceback, which is
    what makes the two paths comparable.
    """
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[UInt8](length=len(left) + 1, fill=Operation.SUBSTITUTE)
    hirschberg_window(
        left, right, 0, len(left), 0, len(right),
        substitutions, alphabet_size, scoring, cells, path_columns, path_entries,
    )
    var score = score_path(
        left, right, path_columns, path_entries, substitutions, alphabet_size, scoring, len(left)
    )
    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.GLOBAL, 0, len(left))
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


def needleman_wunsch_gotoh_alignment_linear_gpu(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    """Global alignment in linear space with every sweep running on the device."""
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[UInt8](length=len(left) + 1, fill=Operation.SUBSTITUTE)
    var ctx = DeviceContext()
    hirschberg_path_gpu(
        ctx, left, right, 0, len(left), 0, len(right),
        substitutions, alphabet_size, scoring, cells, path_columns, path_entries,
    )
    var score = score_path(
        left, right, path_columns, path_entries, substitutions, alphabet_size, scoring, len(left)
    )
    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.GLOBAL, 0, len(left))
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


def smith_waterman_gotoh_alignment_linear(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    """Local alignment in linear space, by reduction to the global problem.

    A forward local sweep finds where the best alignment ends, a backward sweep over those
    prefixes finds where it starts, and the global recursion then runs on that rectangle alone.
    The untrimmed flanks the Python leaves in front of a local result are filled in afterwards.

    Hirschberg can in fact be aimed at a local matrix directly, at twice the cell count and the
    same linear space — the local optimum is a maximum over sub-rectangles, so the join gains
    cases for an optimum lying wholly above or wholly below the cut. The reduction is used anyway
    because it keeps one well-understood global recursion instead of four subproblem modes, which
    is what Myers-Miller's own authors prescribe.
    """
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var ending = local_extremum(
        left, right, len(left), len(right), False, substitutions, alphabet_size, scoring
    )
    var last_row = ending[0]
    var last_column = ending[1]
    var score = ending[2]

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[UInt8](length=len(left) + 1, fill=Operation.SUBSTITUTE)

    var first_row = last_row
    if score > 0:
        var beginning = local_extremum(
            left, right, last_row, last_column, True, substitutions, alphabet_size, scoring
        )
        first_row = last_row - beginning[0]
        var first_column = last_column - beginning[1]

        path_columns[last_row] = Int32(last_column)
        var core_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
        var core_entries = List[UInt8](length=len(left) + 1, fill=Operation.SUBSTITUTE)
        for index in range(len(path_columns)):
            core_columns[index] = path_columns[index]
            core_entries[index] = path_entries[index]
        hirschberg_window(
            left, right, first_row, last_row, first_column, last_column,
            substitutions, alphabet_size, scoring, cells, core_columns, core_entries,
        )
        for index in range(first_row, last_row + 1):
            path_columns[index] = core_columns[index]
            path_entries[index] = core_entries[index]

    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.LOCAL, first_row, last_row)
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


def device_local_extremum(
    ctx: DeviceContext,
    sequences: List[Scalar[SYMBOL_DTYPE]],
    rows: Int,
    first_to: Int,
    second_to: Int,
    backwards: Bool,
    substitutions: List[Scalar[SUBSTITUTION_DTYPE]],
    alphabet_size: Int,
    scoring: AffineScoring,
) raises -> Tuple[Int, Int, Int32]:
    """Runs one local sweep on the device and reads back where its maximum sits."""
    var sequences_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(len(sequences), 1))
    var substitutions_buffer = ctx.enqueue_create_buffer[SUBSTITUTION_DTYPE](len(substitutions))
    var bands_buffer = ctx.enqueue_create_buffer[SCORE_DTYPE](7 * (first_to + 1))
    var outcome_buffer = ctx.enqueue_create_buffer[DType.int64](3)

    with sequences_buffer.map_to_host() as host:
        for index in range(len(sequences)):
            host[index] = sequences[index]
    with substitutions_buffer.map_to_host() as host:
        for index in range(len(substitutions)):
            host[index] = substitutions[index]
    with outcome_buffer.map_to_host() as host:
        for index in range(3):
            host[index] = 0

    ctx.enqueue_function[local_extremum_kernel](
        sequences_buffer.unsafe_ptr(),
        substitutions_buffer.unsafe_ptr(),
        bands_buffer.unsafe_ptr(),
        outcome_buffer.unsafe_ptr(),
        Int32(0),
        Int32(first_to),
        Int32(rows),
        Int32(rows + second_to),
        Int32(1) if backwards else Int32(0),
        Int32(alphabet_size),
        scoring.gap_opening,
        scoring.gap_extension,
        grid_dim=1,
        block_dim=THREADS_PER_BLOCK,
    )
    ctx.synchronize()

    var best_row = 0
    var best_column = 0
    var best = Int32(0)
    with outcome_buffer.map_to_host() as host:
        best_row = Int(host[0])
        best_column = Int(host[1])
        best = Int32(Int(host[2]))
    return (best_row, best_column, best)


def smith_waterman_gotoh_alignment_linear_gpu(
    first: PythonObject,
    second: PythonObject,
    gap_opening: PythonObject,
    gap_extension: PythonObject,
    match_score: PythonObject,
    mismatch_score: PythonObject,
) raises -> PythonObject:
    """Local alignment in linear space with every sweep running on the device."""
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = scoring_from(gap_opening, gap_extension)
    var substitutions = matrix_from(match_score, mismatch_score, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var sequences = List[Scalar[SYMBOL_DTYPE]](capacity=len(left) + len(right))
    for index in range(len(left)):
        sequences.append(left[index])
    for index in range(len(right)):
        sequences.append(right[index])

    var ctx = DeviceContext()
    var ending = device_local_extremum(
        ctx, sequences, len(left), len(left), len(right), False,
        substitutions, alphabet_size, scoring,
    )
    var last_row = ending[0]
    var last_column = ending[1]
    var score = ending[2]

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[UInt8](length=len(left) + 1, fill=Operation.SUBSTITUTE)

    var first_row = last_row
    if score > 0:
        var beginning = device_local_extremum(
            ctx, sequences, len(left), last_row, last_column, True,
            substitutions, alphabet_size, scoring,
        )
        first_row = last_row - beginning[0]
        var first_column = last_column - beginning[1]

        path_columns[last_row] = Int32(last_column)
        hirschberg_path_gpu(
            ctx, left, right, first_row, last_row, first_column, last_column,
            substitutions, alphabet_size, scoring, cells, path_columns, path_entries,
        )

    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.LOCAL, first_row, last_row)
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


@export
def PyInit_affinegaps_mojo() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("affinegaps_mojo")
        builder.def_function[needleman_wunsch_gotoh_score]("needleman_wunsch_gotoh_score")
        builder.def_function[smith_waterman_gotoh_score]("smith_waterman_gotoh_score")
        builder.def_function[needleman_wunsch_gotoh_alignment]("needleman_wunsch_gotoh_alignment")
        builder.def_function[smith_waterman_gotoh_alignment]("smith_waterman_gotoh_alignment")
        builder.def_function[needleman_wunsch_gotoh_scores_gpu]("needleman_wunsch_gotoh_scores_gpu")
        builder.def_function[smith_waterman_gotoh_scores_gpu]("smith_waterman_gotoh_scores_gpu")
        builder.def_function[needleman_wunsch_gotoh_alignments_gpu](
            "needleman_wunsch_gotoh_alignments_gpu"
        )
        builder.def_function[smith_waterman_gotoh_alignments_gpu](
            "smith_waterman_gotoh_alignments_gpu"
        )
        builder.def_function[levenshtein_alignment]("levenshtein_alignment")
        builder.def_function[colorize_alignment]("colorize_alignment")
        builder.def_function[needleman_wunsch_gotoh_alignment_linear](
            "needleman_wunsch_gotoh_alignment_linear"
        )
        builder.def_function[needleman_wunsch_gotoh_alignment_linear_gpu](
            "needleman_wunsch_gotoh_alignment_linear_gpu"
        )
        builder.def_function[smith_waterman_gotoh_alignment_linear](
            "smith_waterman_gotoh_alignment_linear"
        )
        builder.def_function[smith_waterman_gotoh_alignment_linear_gpu](
            "smith_waterman_gotoh_alignment_linear_gpu"
        )
        return builder.finalize()
    except error:
        abort(String("Failed to initialize affinegaps_mojo: ", error))

# endregion Python Bindings

# region Command Line

def parse_int(text: String) -> Optional[Int]:
    """Reads a command-line integer, returning nothing when the text is not one."""
    try:
        var parsed: Optional[Int] = Int(text)
        return parsed
    except:
        return None


def main() raises:
    """Aligns one pair and prints the report, so the binary needs nothing from Python.

    A shared library may not define `main`, so the extension is built from a copy of this file with
    the region below removed. See `scripts/build_extension.py`.
    """
    var arguments = argv()
    var count = len(arguments)
    if count < 3 or String(arguments[1]) == "--help" or String(arguments[1]) == "-h":
        print("Usage: affinegaps FIRST SECOND [--local] [--linear] [--gpu]")
        print("                  [--match N] [--mismatch N]")
        print("                  [--gap-opening N] [--gap-extension N]")
        print()
        print("Gotoh affine-gap alignment. Reconstructs the alignment, not just the score.")
        return

    var first = String(arguments[1])
    var second = String(arguments[2])
    var local = False
    var linear = False
    var opening = Int(DEFAULT_GAP_OPENING)
    var extension = Int(DEFAULT_GAP_EXTENSION)
    var match_score = Optional[Int]()
    var mismatch_score = Optional[Int]()

    var index = 3
    while index < count:
        var flag = String(arguments[index])
        if flag == "--local":
            local = True
            index += 1
        elif flag == "--linear":
            linear = True
            index += 1
        elif index + 1 < count:
            var value = parse_int(String(arguments[index + 1]))
            if flag == "--gap-opening":
                opening = value.or_else(opening)
            elif flag == "--gap-extension":
                extension = value.or_else(extension)
            elif flag == "--match":
                match_score = value
            elif flag == "--mismatch":
                mismatch_score = value
            else:
                raise Error(String("Unknown option: ", flag))
            index += 2
        else:
            raise Error(String("Option needs a value: ", flag))

    if Bool(match_score) != Bool(mismatch_score):
        raise Error("Both match and mismatch must be provided.")

    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = AffineScoring(Int32(opening), Int32(extension))
    var substitutions = default_proteins_matrix() if not match_score else uniform_matrix(
        alphabet_size, match_score.value(), mismatch_score.value()
    )
    var left = translate(first, alphabet)
    var right = translate(second, alphabet)

    var result: AlignmentResult
    if linear:
        var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
        var path_entries = List[UInt8](length=len(left) + 1, fill=Operation.SUBSTITUTE)
        var ctx = DeviceContext()
        hirschberg_path_gpu(
            ctx, left, right, 0, len(left), 0, len(right),
            substitutions, alphabet_size, scoring, DEFAULT_TILE_CELLS, path_columns, path_entries,
        )
        var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.GLOBAL, 0, len(left))
        var score = score_path(
            left, right, path_columns, path_entries, substitutions, alphabet_size, scoring, len(left)
        )
        result = AlignmentResult(score, expanded[0], expanded[1])
    elif local:
        result = serial_align[AlignmentMode.LOCAL](
            left, right, substitutions, alphabet_size, scoring, alphabet
        )
    else:
        result = serial_align[AlignmentMode.GLOBAL](
            left, right, substitutions, alphabet_size, scoring, alphabet
        )

    var painted = colorize(result.first_gapped, result.second_gapped)
    print()
    print("Sequence 1:", first)
    print("Sequence 2:", second)
    print()
    print("Alignment 1:", painted[0])
    print("Alignment 2:", painted[1])
    print("Score:      ", result.score)

# endregion Command Line
