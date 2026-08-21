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
from max.gpu.primitives import block
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.memory import external_memory
from std.gpu.host.info import GPUInfo
from std.sys.info import _accelerator_arch, has_accelerator

# region Scoring

comptime SCORE_DTYPE = DType.int32
comptime SYMBOL_DTYPE = DType.uint8
comptime SUBSTITUTION_DTYPE = DType.int8
comptime CHANGE_DTYPE = DType.uint8
comptime OFFSET_DTYPE = DType.uint32

comptime ALL_MODES = (AlignmentMode.GLOBAL, AlignmentMode.LOCAL)

comptime DEFAULT_PROTEINS_ALPHABET = "ARNDCQEGHILKMFPSTWYVBZX"
comptime DEFAULT_GAP_OPENING = Int32(-20)
comptime DEFAULT_GAP_EXTENSION = Int32(-1)

# The character a gapped alignment prints where a sequence has nothing, and the letter an
# empty alphabet falls back to so unit-cost alignment always has one symbol to work with.
comptime GAP_BYTE = Byte(ord("-"))
comptime FALLBACK_LETTER = Byte(ord("A"))

# No alphabet reaches 255 symbols, so it doubles as the "not in this alphabet" marker.
comptime UNKNOWN_SYMBOL = UInt8(255)

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


def target_shared_per_multiprocessor() -> Int:
    """Shared memory per multiprocessor on whatever this is being compiled for.

    A `comptime if` elides the untaken branch where a ternary would instantiate both, which is
    what lets the no-accelerator case avoid naming a device at all: it falls back to the static
    carve-out limit, and nothing sized from it can run on such a host anyway.
    """
    comptime if has_accelerator():
        return GPUInfo.from_name[_accelerator_arch()]().shared_memory_per_multiprocessor
    return BLOCKS_PER_MULTIPROCESSOR * STATIC_SHARED_LIMIT


comptime SHARED_PER_BLOCK = min(target_shared_per_multiprocessor() // BLOCKS_PER_MULTIPROCESSOR, STATIC_SHARED_LIMIT)

# Seven bands of scores, deletions and insertions. The staged table and the reductions stay in
# static shared memory; only the bands are carved out dynamically, so their length follows the
# batch rather than a constant.
comptime BANDS_PER_SWEEP = 7
comptime REDUCTION_BYTES = THREADS_PER_BLOCK * 12
comptime STATIC_SHARED_USED = MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE + REDUCTION_BYTES

# A block may opt into all of a multiprocessor's shared memory but the kilobyte the driver keeps.
# Measured on this target: 227 kibibytes of dynamic shared memory launches, 228 does not.
comptime SHARED_RESERVED = 1024
comptime MAX_DYNAMIC_SHARED = target_shared_per_multiprocessor() - SHARED_RESERVED
comptime MAX_BAND_LENGTH = (MAX_DYNAMIC_SHARED - STATIC_SHARED_USED) // (BANDS_PER_SWEEP * 4) - 1


def band_bytes(band_stride: Int) -> Int:
    """Dynamic shared memory one block needs to hold its seven bands."""
    return BANDS_PER_SWEEP * band_stride * 4


# A quarter of the range: far below any reachable score, which a path bounds by
# `(rows + columns) * max_cost`, while leaving headroom for the gap penalties that get added to
# the seed as a reduction proceeds. `Int32.MIN` itself underflows on the first addition.
comptime NEGATIVE_INFINITY = Int32.MIN // 4


@fieldwise_init
struct Layer(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which of Gotoh's three layers a score came from, and which one a traceback is inside.

    The walk never needs to know whether an aligning step matched or substituted, so the two
    collapse into one layer here; only the emitted letters differ, and those come from the
    sequences.
    """

    var identifier: UInt8
    comptime ALIGNING = Self(0)
    comptime DELETING = Self(1)
    comptime INSERTING = Self(2)


@fieldwise_init
struct GapRun(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a gap run arriving at a cell was already open one step earlier.

    Ties resolve towards opening, because both extension tests use a strict `>`.
    """

    var identifier: UInt8
    comptime OPENS = Self(0)
    comptime EXTENDS = Self(1)


@fieldwise_init
struct PathReach(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a local path passes through a cell, or its score falls to zero there."""

    var identifier: UInt8
    comptime CONTINUES_PAST = Self(0)
    comptime ENDS_HERE = Self(1)


@fieldwise_init
struct Step(ImplicitlyCopyable, TrivialRegisterPassable):
    """One backward move: how far to walk on each axis, and the layer it lands in.

    A zero move on an axis is a gap on that sequence, so the emitted pair follows from the move.
    """

    var row_step: Int
    var column_step: Int
    var lands_in: Layer


# BLOSUM62 scaled by five, trimmed to the 23 letters the alphabet emits. The canonical table
# is 24 by 24; its last row and column are the `*` stop codon, which `translate` never produces.
# fmt: off
comptime BLOSUM62_SCALED: Array[Scalar[SUBSTITUTION_DTYPE], 529] = [
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
# fmt: on


def default_proteins_matrix() -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """BLOSUM62 scaled by five, trimmed to the 23 letters the alphabet actually emits.

    The canonical table is 24 by 24; its last row and column are the `*` stop codon, which
    `translate` never produces.
    """
    var table = materialize[BLOSUM62_SCALED]()
    var matrix = List[Scalar[SUBSTITUTION_DTYPE]](capacity=529)
    for index in range(529):
        matrix.append(table[index])
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
    var codes_by_byte = Array[UInt8, 256](fill=UNKNOWN_SYMBOL)
    for index in range(len(alphabet_bytes)):
        codes_by_byte[Int(alphabet_bytes[index])] = UInt8(index)

    var codes = List[Scalar[SYMBOL_DTYPE]](capacity=len(text_bytes))
    for position in range(len(text_bytes)):
        var code = codes_by_byte[Int(text_bytes[position])]
        if code == UNKNOWN_SYMBOL:
            raise Error(String("Found unknown character in sequence: ", text))
        codes.append(Scalar[SYMBOL_DTYPE](code))
    return codes^


@fieldwise_init
struct AffineGapCosts(ImplicitlyCopyable, TrivialRegisterPassable):
    """Gotoh's two-parameter gap model. Both penalties are negative."""

    var open: Int32
    var extend: Int32


@fieldwise_init
struct Cell(ImplicitlyCopyable, TrivialRegisterPassable):
    """The three layers of one dynamic-programming cell."""

    var score: Int32
    var deletion: Int32
    var insertion: Int32


@fieldwise_init
struct AlignmentMode(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which of the two alignment problems the recurrence solves."""

    var identifier: UInt8
    comptime GLOBAL = Self(0)
    comptime LOCAL = Self(1)


@always_inline
def gotoh_cell[
    mode: AlignmentMode
](
    diagonal: Int32,
    above: Int32,
    above_delete: Int32,
    left: Int32,
    left_insert: Int32,
    substitution: Int32,
    scoring: AffineGapCosts,
) -> Cell:
    """One interior cell of the Gotoh recurrence, with the local clamp folded in at comptime.

    This is the single transcription of the recurrence that `affinegaps.py` holds as the oracle;
    every sweep on the host and on the device goes through it.
    """
    var deletion = max(above + scoring.open, above_delete + scoring.extend)
    var insertion = max(left + scoring.open, left_insert + scoring.extend)
    var score = max(max(diagonal + substitution, deletion), insertion)
    comptime if mode == AlignmentMode.LOCAL:
        score = max(score, Int32(0))
    return Cell(score, deletion, insertion)


@fieldwise_init
struct Wavefront[address_space: AddressSpace](ImplicitlyCopyable, TrivialRegisterPassable):
    """The seven live bands of an anti-diagonal sweep, rotated one diagonal at a time.

    Scores reach two diagonals back because a substitution step skips one; the two gap layers
    reach only one. The band written on a diagonal is never one read on that diagonal, which is
    why a single barrier per diagonal suffices.
    """

    var scores_two_back: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space]
    var scores_one_back: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space]
    var scores_current: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space]
    var deletes_one_back: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space]
    var deletes_current: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space]
    var inserts_one_back: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space]
    var inserts_current: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space]

    @staticmethod
    @always_inline
    def over(
        bands: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=Self.address_space],
        stride: Int,
    ) -> Self:
        """Carves `BANDS_PER_SWEEP` equal bands out of one allocation."""
        return Self(
            bands,
            bands.unsafe_offset(stride),
            bands.unsafe_offset(2 * stride),
            bands.unsafe_offset(3 * stride),
            bands.unsafe_offset(4 * stride),
            bands.unsafe_offset(5 * stride),
            bands.unsafe_offset(6 * stride),
        )

    @always_inline
    def rotate(mut self):
        """Retires the oldest score band and hands it back as the next diagonal's target."""
        var recycled = self.scores_two_back
        self.scores_two_back = self.scores_one_back
        self.scores_one_back = self.scores_current
        self.scores_current = recycled
        swap(self.deletes_one_back, self.deletes_current)
        swap(self.inserts_one_back, self.inserts_current)


@always_inline
def source_layer(cell: Cell, replacement: Int32) -> Layer:
    """Which layer the score came from; ties resolve to aligning, then deleting."""
    if cell.score == replacement:
        return Layer.ALIGNING
    if cell.score == cell.deletion:
        return Layer.DELETING
    return Layer.INSERTING


@fieldwise_init
struct CellDecision(ImplicitlyCopyable, TrivialRegisterPassable):
    """One stored byte per cell: the layer the score came from, and what both gap runs did.

    Each field answers a question asked from a different walk state, so they are three
    independent facts about one cell rather than one composite state. Five of the eight bits
    are used.
    """

    var bits: UInt8

    @staticmethod
    @always_inline
    def recording(source: Layer, deletion: GapRun, insertion: GapRun, reach: PathReach) -> Self:
        return Self(
            source.identifier | (deletion.identifier << 2) | (insertion.identifier << 3) | (reach.identifier << 7)
        )

    @always_inline
    def source(self) -> Layer:
        return Layer(self.bits & 0x03)

    @always_inline
    def deletion(self) -> GapRun:
        return GapRun((self.bits >> 2) & 0x01)

    @always_inline
    def insertion(self) -> GapRun:
        return GapRun((self.bits >> 3) & 0x01)

    @always_inline
    def reach(self) -> PathReach:
        return PathReach(self.bits >> 7)


@always_inline
def decide(cell: Cell, replacement: Int32, above: Cell, left: Cell, scoring: AffineGapCosts) -> CellDecision:
    """The four answers a kernel packs, re-derived from the three score layers."""
    var deletion_run = GapRun.EXTENDS if above.deletion + scoring.extend > above.score + scoring.open else GapRun.OPENS
    var insertion_run = GapRun.EXTENDS if left.insertion + scoring.extend > left.score + scoring.open else GapRun.OPENS
    var reach = PathReach.ENDS_HERE if cell.score <= 0 else PathReach.CONTINUES_PAST
    return CellDecision.recording(source_layer(cell, replacement), deletion_run, insertion_run, reach)


@always_inline
def advance(state: Layer, decision: CellDecision) -> Step:
    """The traceback's transition function, shared by every walk in this file.

    Entering a gap run from the aligning layer moves in the same step rather than re-reading the
    cell, which the layered walk is free to do because that entry never moves on its own.
    """
    var layer = decision.source() if state == Layer.ALIGNING else state
    if layer == Layer.DELETING:
        var next_layer = Layer.DELETING if decision.deletion() == GapRun.EXTENDS else Layer.ALIGNING
        return Step(-1, 0, next_layer)
    if layer == Layer.INSERTING:
        var next_layer = Layer.INSERTING if decision.insertion() == GapRun.EXTENDS else Layer.ALIGNING
        return Step(0, -1, next_layer)
    return Step(-1, -1, Layer.ALIGNING)


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
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
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
            scores_above[column] = scoring.open + Int32(column - 1) * scoring.extend
            deletes_above[column] = scores_above[column] + scoring.open + scoring.extend
        else:
            scores_above[column] = 0
            deletes_above[column] = scoring.open + scoring.extend

    var best = Int32(0)
    for row in range(1, rows + 1):
        if mode == AlignmentMode.GLOBAL:
            scores_row[0] = scoring.open + Int32(row - 1) * scoring.extend
        else:
            scores_row[0] = 0
        inserts_row[0] = scores_row[0] + scoring.open + scoring.extend

        for column in range(1, columns + 1):
            var substitution = Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])])
            var cell = gotoh_cell[mode](
                scores_above[column - 1],
                scores_above[column],
                deletes_above[column],
                scores_row[column - 1],
                inserts_row[column - 1],
                substitution,
                scoring,
            )
            comptime if mode == AlignmentMode.LOCAL:
                best = max(best, cell.score)
            scores_row[column] = cell.score
            deletes_row[column] = cell.deletion
            inserts_row[column] = cell.insertion

        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)
        swap(inserts_above, inserts_row)

    if mode == AlignmentMode.LOCAL:
        return best
    return scores_above[columns]


def serial_align[
    mode: AlignmentMode
](
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
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

    scores[0] = 0
    for column in range(1, stride):
        if mode == AlignmentMode.GLOBAL:
            scores[column] = scoring.open + Int32(column - 1) * scoring.extend
            deletes[column] = scores[column] + scoring.open + scoring.extend
        else:
            scores[column] = 0
            deletes[column] = scoring.open + scoring.extend

    var best = Int32(0)
    var best_row = 0
    var best_column = 0

    for row in range(1, rows + 1):
        var base = row * stride
        var above = base - stride
        if mode == AlignmentMode.GLOBAL:
            scores[base] = scoring.open + Int32(row - 1) * scoring.extend
            inserts[base] = scores[base] + scoring.open + scoring.extend
        else:
            scores[base] = 0
            inserts[base] = scoring.open + scoring.extend

        for column in range(1, stride):
            var substitution = Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])])
            var cell = gotoh_cell[mode](
                scores[above + column - 1],
                scores[above + column],
                deletes[above + column],
                scores[base + column - 1],
                inserts[base + column - 1],
                substitution,
                scoring,
            )
            scores[base + column] = cell.score
            deletes[base + column] = cell.deletion
            inserts[base + column] = cell.insertion

            comptime if mode == AlignmentMode.LOCAL:
                if cell.score > best:
                    best = cell.score
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
    scores: ImmSpan[Int32, _],
    deletes: ImmSpan[Int32, _],
    inserts: ImmSpan[Int32, _],
    stride: Int,
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    start_row: Int,
    start_column: Int,
    alphabet: String,
    scoring: AffineGapCosts,
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
    var first_reversed = List[UInt8]()
    var second_reversed = List[UInt8]()
    var row = start_row
    var column = start_column
    var state = Layer.ALIGNING

    while row > 0 and column > 0:
        var here = row * stride + column
        var substitution = Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])])
        var decision = decide(
            Cell(scores[here], deletes[here], inserts[here]),
            scores[here - stride - 1] + substitution,
            Cell(scores[here - stride], deletes[here - stride], inserts[here - stride]),
            Cell(scores[here - 1], deletes[here - 1], inserts[here - 1]),
            scoring,
        )
        if mode == AlignmentMode.LOCAL and state == Layer.ALIGNING:
            if decision.reach() == PathReach.ENDS_HERE:
                break
        var step = advance(state, decision)
        first_reversed.append(letters[Int(first[row - 1])] if step.row_step != 0 else GAP_BYTE)
        second_reversed.append(letters[Int(second[column - 1])] if step.column_step != 0 else GAP_BYTE)
        row += step.row_step
        column += step.column_step
        state = step.lands_in

    # A global path must reach the origin, so what remains really is aligned against gaps. A local
    # path stops wherever the score falls to zero, and everything before that is outside it.
    if mode == AlignmentMode.GLOBAL:
        while row > 0:
            first_reversed.append(letters[Int(first[row - 1])])
            second_reversed.append(GAP_BYTE)
            row -= 1
        while column > 0:
            first_reversed.append(GAP_BYTE)
            second_reversed.append(letters[Int(second[column - 1])])
            column -= 1

    first_reversed.reverse()
    second_reversed.reverse()
    return (
        String(unsafe_from_utf8=first_reversed),
        String(unsafe_from_utf8=second_reversed),
    )


@fieldwise_init
struct SweepHalf(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which half of a Hirschberg split a sweep is computing.

    The reverse half walks both sequences from their far ends, which is how it is computed
    without a second recurrence, and it leaves its frontier in the reverse band pair.
    """

    var identifier: UInt8
    comptime FORWARD = Self(0)
    comptime REVERSE = Self(1)


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
    var top: GapRun
    var bottom: GapRun


def sweep_bands[
    half: SweepHalf
](
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    first_from: Int,
    first_to: Int,
    second_from: Int,
    second_to: Int,
    entry: GapRun,
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    final_scores: MutSpan[Int32, _],
    final_deletes: MutSpan[Int32, _],
):
    """Linear-space sweep of one sub-rectangle, leaving the last row's two layers behind.

    `entry` carries the open deletion run across the boundary: an already-open run makes the
    first deletion cost only an extension.
    """
    var rows = first_to - first_from
    var columns = second_to - second_from
    var scores_above = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_above = List[Int32](length=columns + 1, fill=Int32(0))
    var scores_row = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_row = List[Int32](length=columns + 1, fill=Int32(0))
    var inserts_row = List[Int32](length=columns + 1, fill=Int32(0))

    scores_above[0] = 0
    deletes_above[0] = scoring.open + scoring.extend
    # An open run arrives at the top-left corner only. Reaching any other cell of the top row
    # means the run already ended, so a deletion from there pays a fresh opening.
    for column in range(1, columns + 1):
        scores_above[column] = scoring.open + Int32(column - 1) * scoring.extend
        deletes_above[column] = scores_above[column] + scoring.open + scoring.extend

    for row in range(1, rows + 1):
        if entry == GapRun.EXTENDS:
            scores_row[0] = Int32(row) * scoring.extend
        else:
            scores_row[0] = scoring.open + Int32(row - 1) * scoring.extend
        deletes_row[0] = scores_row[0]
        inserts_row[0] = scores_row[0] + scoring.open + scoring.extend

        comptime reversed_order = half == SweepHalf.REVERSE
        var first_index = first_to - row if reversed_order else first_from + row - 1
        for column in range(1, columns + 1):
            var second_index = second_to - column if reversed_order else second_from + column - 1
            var substitution = Int32(substitutions[Int(first[first_index]) * alphabet_size + Int(second[second_index])])
            var cell = gotoh_cell[AlignmentMode.GLOBAL](
                scores_above[column - 1],
                scores_above[column],
                deletes_above[column],
                scores_row[column - 1],
                inserts_row[column - 1],
                substitution,
                scoring,
            )
            scores_row[column] = cell.score
            deletes_row[column] = cell.deletion
            inserts_row[column] = cell.insertion

        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)

    for column in range(columns + 1):
        final_scores[column] = scores_above[column]
        final_deletes[column] = deletes_above[column]


def direct_tile(
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    first_from: Int,
    first_to: Int,
    second_from: Int,
    second_to: Int,
    top: GapRun,
    bottom: GapRun,
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    path_columns: MutSpan[Int32, _],
    path_entries: MutSpan[Layer, _],
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
    # An open run arrives at the top-left corner only. Reaching any other cell of the top row
    # means the run already ended, so a deletion from there pays a fresh opening.
    for column in range(1, stride):
        scores[column] = scoring.open + Int32(column - 1) * scoring.extend
        deletes[column] = scores[column] + scoring.open + scoring.extend

    for row in range(1, rows + 1):
        var base = row * stride
        var above = base - stride
        if top == GapRun.EXTENDS:
            scores[base] = Int32(row) * scoring.extend
        else:
            scores[base] = scoring.open + Int32(row - 1) * scoring.extend
        deletes[base] = scores[base]
        inserts[base] = scores[base] + scoring.open + scoring.extend
        for column in range(1, stride):
            var substitution = Int32(
                substitutions[Int(first[first_from + row - 1]) * alphabet_size + Int(second[second_from + column - 1])]
            )
            var cell = gotoh_cell[AlignmentMode.GLOBAL](
                scores[above + column - 1],
                scores[above + column],
                deletes[above + column],
                scores[base + column - 1],
                inserts[base + column - 1],
                substitution,
                scoring,
            )
            scores[base + column] = cell.score
            deletes[base + column] = cell.deletion
            inserts[base + column] = cell.insertion

    var row = rows
    var column = columns
    var state = Layer.ALIGNING
    # An open run at the bottom edge is a fact the join established, not a candidate to weigh:
    # the two halves were scored on the assumption that this walk leaves in the deleting layer.
    if bottom == GapRun.EXTENDS:
        state = Layer.DELETING

    # Only a row-consuming step records anything, so sibling subproblems tile the row axis.
    while row > 0 and column > 0:
        var here = row * stride + column
        var substitution = Int32(
            substitutions[Int(first[first_from + row - 1]) * alphabet_size + Int(second[second_from + column - 1])]
        )
        var decision = decide(
            Cell(scores[here], deletes[here], inserts[here]),
            scores[here - stride - 1] + substitution,
            Cell(scores[here - stride], deletes[here - stride], inserts[here - stride]),
            Cell(scores[here - 1], deletes[here - 1], inserts[here - 1]),
            scoring,
        )
        var step = advance(state, decision)
        if step.row_step != 0:
            path_entries[first_from + row] = Layer.ALIGNING if step.column_step != 0 else Layer.DELETING
            path_columns[first_from + row - 1] = Int32(second_from + column + step.column_step)
        row += step.row_step
        column += step.column_step
        state = step.lands_in

    while row > 0:
        path_entries[first_from + row] = Layer.DELETING
        path_columns[first_from + row - 1] = Int32(second_from + column)
        row -= 1

    return scores[rows * stride + columns]


@fieldwise_init
struct Crossing(ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a Myers-Miller join puts the cut, and what each of its two candidates scored."""

    var plain: Int32
    var plain_column: Int
    var gapped: Int32
    var gapped_column: Int


def tile_frame(
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    frame: Frame,
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    path_columns: MutSpan[Int32, _],
    path_entries: MutSpan[Layer, _],
):
    """Solves one pending subproblem outright, which is how every branch of the recursion ends."""
    _ = direct_tile(
        first,
        second,
        frame.first_from,
        frame.first_to,
        frame.second_from,
        frame.second_to,
        frame.top,
        frame.bottom,
        substitutions,
        alphabet_size,
        scoring,
        path_columns,
        path_entries,
    )


def best_crossing(
    forward_scores: ImmSpan[Int32, _],
    forward_deletes: ImmSpan[Int32, _],
    reverse_scores: ImmSpan[Int32, _],
    reverse_deletes: ImmSpan[Int32, _],
    width: Int,
    scoring: AffineGapCosts,
) -> Crossing:
    """Scans the two frontiers for the best cut, in the match layer and across a straddling run."""
    var best = Crossing(NEGATIVE_INFINITY, 0, NEGATIVE_INFINITY, 0)
    var refund = scoring.extend - scoring.open
    for offset in range(width + 1):
        var plain = forward_scores[offset] + reverse_scores[width - offset]
        if plain > best.plain:
            best.plain = plain
            best.plain_column = offset
        var gapped = forward_deletes[offset] + reverse_deletes[width - offset] + refund
        if gapped > best.gapped:
            best.gapped = gapped
            best.gapped_column = offset
    return best


def hirschberg_window(
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    window_first_from: Int,
    window_first_to: Int,
    window_second_from: Int,
    window_second_to: Int,
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    tile_cells: Int,
    path_columns: MutSpan[Int32, _],
    path_entries: MutSpan[Layer, _],
) raises:
    """Linear-space traceback: split on rows, join the two halves, recurse without recursion.

    A substitution step advances `i + j` by two and can skip an anti-diagonal entirely, while the
    row index advances by exactly zero or one per step, so the cut has to be a row. The two halves
    are joined by Myers-Miller: either the path crosses in the match layer, or a deletion run
    straddles the cut, in which case both halves charged an opening and one is refunded.
    """
    if scoring.open > scoring.extend:
        raise Error("Gap opening must be at least as expensive as extension for the row split.")

    var columns = window_second_to - window_second_from
    path_columns[window_first_to] = Int32(window_second_to)

    var frames = List[Frame]()
    frames.append(
        Frame(
            window_first_from,
            window_first_to,
            window_second_from,
            window_second_to,
            GapRun.OPENS,
            GapRun.OPENS,
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
            tile_frame(
                first,
                second,
                frame,
                substitutions,
                alphabet_size,
                scoring,
                path_columns,
                path_entries,
            )
            continue

        var split = (first_from + first_to) // 2
        sweep_bands[SweepHalf.FORWARD](
            first,
            second,
            first_from,
            split,
            second_from,
            second_to,
            top,
            substitutions,
            alphabet_size,
            scoring,
            forward_scores,
            forward_deletes,
        )
        sweep_bands[SweepHalf.REVERSE](
            first,
            second,
            split,
            first_to,
            second_from,
            second_to,
            bottom,
            substitutions,
            alphabet_size,
            scoring,
            reverse_scores,
            reverse_deletes,
        )

        var join = best_crossing(forward_scores, forward_deletes, reverse_scores, reverse_deletes, width, scoring)
        var best_plain = join.plain
        var best_plain_column = join.plain_column
        var best_gapped = join.gapped
        var best_gapped_column = join.gapped_column

        # Either the path crosses the cut in the aligning layer, or it crosses inside a deletion
        # run. Both are ordinary, and the second is what the boundary flags exist to carry: the
        # two halves each charged an opening for their part of the run, the refund removes one,
        # and both children are told the run is already open at the edge they share.
        var crosses_in_a_gap = best_gapped > best_plain
        var crossing = second_from + (
            best_gapped_column if crosses_in_a_gap else best_plain_column
        )
        var shared_edge = GapRun.EXTENDS if crosses_in_a_gap else GapRun.OPENS
        # Lower half first, so the upper half pops first and the two row ranges tile in order.
        frames.append(Frame(split, first_to, crossing, second_to, shared_edge, bottom))
        frames.append(Frame(first_from, split, second_from, crossing, top, shared_edge))


def score_path(
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    path_columns: ImmSpan[Int32, _],
    path_entries: ImmSpan[Layer, _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
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
        total += scoring.extend if in_first else scoring.open
        in_first = True
        in_second = False

    for row in range(1, rows + 1):
        var before = Int(path_columns[row - 1])
        var after = Int(path_columns[row])
        if path_entries[row] == Layer.DELETING:
            total += scoring.extend if in_second else scoring.open
            in_first = False
            in_second = True
        else:
            total += Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[before])])
            before += 1
            in_first = False
            in_second = False
        for _ in range(before, after):
            total += scoring.extend if in_first else scoring.open
            in_first = True
            in_second = False

    return total


def expand_path(
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    path_columns: ImmSpan[Int32, _],
    path_entries: ImmSpan[Layer, _],
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
    var left = List[UInt8]()
    var right = List[UInt8]()

    if mode == AlignmentMode.GLOBAL:
        for column in range(Int(path_columns[0])):
            left.append(GAP_BYTE)
            right.append(letters[Int(second[column])])
    for row in range(from_row + 1, rows + 1):
        var before = Int(path_columns[row - 1])
        var after = Int(path_columns[row])
        if path_entries[row] == Layer.DELETING:
            left.append(letters[Int(first[row - 1])])
            right.append(GAP_BYTE)
        else:
            left.append(letters[Int(first[row - 1])])
            right.append(letters[Int(second[before])])
            before += 1
        for column in range(before, after):
            left.append(GAP_BYTE)
            right.append(letters[Int(second[column])])
    return (String(unsafe_from_utf8=left), String(unsafe_from_utf8=right))


def local_extremum[
    half: SweepHalf
](
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    first_to: Int,
    second_to: Int,
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
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
        deletes_above[column] = scoring.open + scoring.extend

    var best = Int32(0)
    var best_row = 0
    var best_column = 0

    for row in range(1, first_to + 1):
        scores_row[0] = 0
        inserts_row[0] = scoring.open + scoring.extend
        comptime reversed_order = half == SweepHalf.REVERSE
        var first_index = first_to - row if reversed_order else row - 1
        for column in range(1, second_to + 1):
            var second_index = second_to - column if reversed_order else column - 1
            var substitution = Int32(substitutions[Int(first[first_index]) * alphabet_size + Int(second[second_index])])
            var cell = gotoh_cell[AlignmentMode.LOCAL](
                scores_above[column - 1],
                scores_above[column],
                deletes_above[column],
                scores_row[column - 1],
                inserts_row[column - 1],
                substitution,
                scoring,
            )
            scores_row[column] = cell.score
            deletes_row[column] = cell.deletion
            inserts_row[column] = cell.insertion
            if cell.score > best:
                best = cell.score
                best_row = row
                best_column = column
        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)

    return (best_row, best_column, best)


# endregion Serial Reference


# region GPU Wavefront


@always_inline
def block_argmax(
    scores: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    places: Pointer[Scalar[DType.int64], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    best: Int32,
    best_place: Int64,
):
    """Block-wide maximum keeping the earliest cell in row-major order on a tie.

    The stdlib block reductions carry a scalar, and this one has to carry the place alongside the
    score to break ties the way the Python scan's strict `>` does, so the tree stays here.
    Thread zero holds the winner afterwards.
    """
    scores[unsafe_offset=Int(thread_idx.x)] = best
    places[unsafe_offset=Int(thread_idx.x)] = best_place
    barrier()
    var span = THREADS_PER_BLOCK // 2
    while span > 0:
        if Int(thread_idx.x) < span:
            var here = Int(thread_idx.x)
            var there = here + span
            var mine = scores[unsafe_offset=here]
            var theirs = scores[unsafe_offset=there]
            if theirs > mine or (
                theirs == mine and theirs != 0 and places[unsafe_offset=there] < places[unsafe_offset=here]
            ):
                scores[unsafe_offset=here] = theirs
                places[unsafe_offset=here] = places[unsafe_offset=there]
        barrier()
        span //= 2


def wavefront_kernel[
    mode: AlignmentMode
](
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    offsets: Pointer[Scalar[OFFSET_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    results: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    band_stride: Int32,
    alphabet_size: Int32,
    open: Int32,
    extend: Int32,
):
    """One block per pair, one thread per cell of the live anti-diagonal.

    Bands are indexed by the row, so cell `(i, d - i)` reads rows `i - 1` and `i` of the two
    preceding diagonals. The band written each diagonal is never one read that diagonal, so a
    single barrier per diagonal suffices.
    """
    var pair = Int(block_idx.x)
    var first_start = Int(offsets[unsafe_offset=2 * pair])
    var scoring = AffineGapCosts(open, extend)
    var second_start = Int(offsets[unsafe_offset=2 * pair + 1])
    var rows = second_start - first_start
    var columns = Int(offsets[unsafe_offset=2 * pair + 2]) - second_start
    var width = Int(alphabet_size)

    var bands = external_memory[Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED, alignment=16, name="bands"]()

    var stride = Int(band_stride)
    var wavefront = Wavefront.over(bands, stride)

    # The substitution table is read once per cell, so it is staged into shared memory rather
    # than fetched from global on every comparison. It is at most a kilobyte.
    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space=AddressSpace.SHARED,
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        scores_table[unsafe_offset=index] = substitutions[unsafe_offset=index]
    barrier()

    var best = Int32(0)
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
                deletion = cell + open + extend
                insertion = deletion
            elif row == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = open + Int32(column - 1) * extend
                    deletion = cell + open + extend
                else:
                    cell = 0
                    deletion = open + extend
                insertion = deletion
            elif column == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = open + Int32(row - 1) * extend
                    insertion = cell + open + extend
                else:
                    cell = 0
                    insertion = open + extend
                deletion = insertion
            else:
                var left_symbol = Int(sequences[unsafe_offset=first_start + row - 1])
                var right_symbol = Int(sequences[unsafe_offset=second_start + column - 1])
                var substitution = Int32(scores_table[unsafe_offset=left_symbol * width + right_symbol])
                var computed = gotoh_cell[mode](
                    wavefront.scores_two_back[unsafe_offset=row - 1],
                    wavefront.scores_one_back[unsafe_offset=row - 1],
                    wavefront.deletes_one_back[unsafe_offset=row - 1],
                    wavefront.scores_one_back[unsafe_offset=row],
                    wavefront.inserts_one_back[unsafe_offset=row],
                    substitution,
                    scoring,
                )
                cell = computed.score
                deletion = computed.deletion
                insertion = computed.insertion
                comptime if mode == AlignmentMode.LOCAL:
                    best = max(best, cell)

            wavefront.scores_current[unsafe_offset=row] = cell
            wavefront.deletes_current[unsafe_offset=row] = deletion
            wavefront.inserts_current[unsafe_offset=row] = insertion
            # The thread owning the bottom-right cell reports the global answer itself.
            comptime if mode == AlignmentMode.GLOBAL:
                if diagonal == last_diagonal and row == rows:
                    results[unsafe_offset=pair] = cell

        barrier()

        wavefront.rotate()

    if mode == AlignmentMode.LOCAL:
        var highest = block.max[block_size=THREADS_PER_BLOCK](best)
        if thread_idx.x == 0:
            results[unsafe_offset=pair] = highest


def upload[dtype: DType](ctx: DeviceContext, values: ImmSpan[Scalar[dtype], _]) raises -> DeviceBuffer[dtype]:
    """Stages values onto the device, keeping a one-element floor so an empty batch is not a case."""
    var buffer = ctx.enqueue_create_buffer[dtype](max(len(values), 1))
    if len(values) > 0:
        ctx.enqueue_copy(buffer, values)
    return buffer^


def zeroed[dtype: DType](ctx: DeviceContext, count: Int) raises -> DeviceBuffer[dtype]:
    """A device buffer the caller can read before any kernel has written it."""
    var buffer = ctx.enqueue_create_buffer[dtype](max(count, 1))
    ctx.enqueue_memset(buffer, Scalar[dtype](0))
    return buffer^


def wavefront_scores[
    mode: AlignmentMode
](
    ctx: DeviceContext,
    sequences: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    offsets: List[Scalar[OFFSET_DTYPE]],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
) raises -> List[Int32]:
    """Scores every pair in the batch, one thread block each."""
    var pairs = (len(offsets) - 1) // 2
    if alphabet_size > MAX_ALPHABET_SIZE:
        raise Error("Alphabet exceeds the shared-memory substitution table capacity.")

    # The bands are indexed by row, so one launch only needs the longest first sequence it carries.
    var longest_first = 0
    for pair in range(pairs):
        longest_first = max(longest_first, Int(offsets[2 * pair + 1]) - Int(offsets[2 * pair]))
    var band_stride = longest_first + 1
    var dynamic_bytes = band_bytes(band_stride)

    var sequences_buffer = upload(ctx, sequences)
    var offsets_buffer = upload(ctx, offsets)
    var substitutions_buffer = upload(ctx, substitutions)
    var results_buffer = zeroed[SCORE_DTYPE](ctx, pairs)

    ctx.enqueue_function[wavefront_kernel[mode]](
        sequences_buffer.unsafe_ptr(),
        offsets_buffer.unsafe_ptr(),
        substitutions_buffer.unsafe_ptr(),
        results_buffer.unsafe_ptr(),
        Int32(band_stride),
        Int32(alphabet_size),
        scoring.open,
        scoring.extend,
        grid_dim=pairs,
        block_dim=THREADS_PER_BLOCK,
        shared_mem_bytes=dynamic_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(dynamic_bytes)),
    )
    ctx.synchronize()

    var results = List[Int32](capacity=pairs)
    with results_buffer.map_to_host() as host:
        for index in range(pairs):
            results.append(host[index])
    return results^


def direct_align_kernel[
    mode: AlignmentMode
](
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    offsets: Pointer[Scalar[OFFSET_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    letters: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    changes: Pointer[Scalar[CHANGE_DTYPE], MutAnyOrigin],
    change_offsets: Pointer[Scalar[DType.int64], MutAnyOrigin],
    results: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    first_gapped: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    second_gapped: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    gapped_lengths: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    gapped_stride: Int32,
    band_stride: Int32,
    alphabet_size: Int32,
    open: Int32,
    extend: Int32,
):
    """Anti-diagonal sweep that records every decision, then walks it back on the device.

    The decision byte carries the operation in its low bits and, for local alignment, a flag
    marking a cell whose score is zero. That flag stands in for the `scores[i, j] > 0` test the
    Python traceback performs, so the score matrix itself never has to be kept.
    """
    var pair = Int(block_idx.x)
    var first_start = Int(offsets[unsafe_offset=2 * pair])
    var scoring = AffineGapCosts(open, extend)
    var second_start = Int(offsets[unsafe_offset=2 * pair + 1])
    var rows = second_start - first_start
    var columns = Int(offsets[unsafe_offset=2 * pair + 2]) - second_start
    var width = Int(alphabet_size)
    var stride = columns + 1
    var tile = Int(change_offsets[unsafe_offset=pair])

    var bands = external_memory[Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED, alignment=16, name="bands"]()
    var best_scores = stack_allocation[THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED]()
    var best_places = stack_allocation[THREADS_PER_BLOCK, Scalar[DType.int64], address_space=AddressSpace.SHARED]()

    var band = Int(band_stride)
    var wavefront = Wavefront.over(bands, band)

    # The substitution table is read once per cell, so it is staged into shared memory rather
    # than fetched from global on every comparison. It is at most a kilobyte.
    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space=AddressSpace.SHARED,
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
            # The walk is guarded by `row > 0 and column > 0`, so no border decision is ever read.
            var decision = CellDecision(0)

            if row == 0 and column == 0:
                cell = 0
                deletion = cell + open + extend
                insertion = deletion
            elif row == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = open + Int32(column - 1) * extend
                    deletion = cell + open + extend
                else:
                    cell = 0
                    deletion = open + extend
                insertion = deletion
            elif column == 0:
                if mode == AlignmentMode.GLOBAL:
                    cell = open + Int32(row - 1) * extend
                    insertion = cell + open + extend
                else:
                    cell = 0
                    insertion = open + extend
                deletion = insertion
            else:
                var left_symbol = Int(sequences[unsafe_offset=first_start + row - 1])
                var right_symbol = Int(sequences[unsafe_offset=second_start + column - 1])
                var substitution = Int32(scores_table[unsafe_offset=left_symbol * width + right_symbol])
                var replacement = wavefront.scores_two_back[unsafe_offset=row - 1] + substitution
                var computed = gotoh_cell[mode](
                    wavefront.scores_two_back[unsafe_offset=row - 1],
                    wavefront.scores_one_back[unsafe_offset=row - 1],
                    wavefront.deletes_one_back[unsafe_offset=row - 1],
                    wavefront.scores_one_back[unsafe_offset=row],
                    wavefront.inserts_one_back[unsafe_offset=row],
                    substitution,
                    scoring,
                )
                cell = computed.score
                deletion = computed.deletion
                insertion = computed.insertion
                var deletion_run = (
                    GapRun.EXTENDS if wavefront.deletes_one_back[unsafe_offset=row - 1] + extend
                    > wavefront.scores_one_back[unsafe_offset=row - 1] + open else GapRun.OPENS
                )
                var insertion_run = (
                    GapRun.EXTENDS if wavefront.inserts_one_back[unsafe_offset=row] + extend
                    > wavefront.scores_one_back[unsafe_offset=row] + open else GapRun.OPENS
                )
                var reach = PathReach.CONTINUES_PAST
                comptime if mode == AlignmentMode.LOCAL:
                    reach = PathReach.ENDS_HERE if cell == 0 else PathReach.CONTINUES_PAST
                decision = CellDecision.recording(
                    source_layer(computed, replacement), deletion_run, insertion_run, reach
                )

                if mode == AlignmentMode.LOCAL:
                    var place = Int64(row) * Int64(stride) + Int64(column)
                    if cell > best:
                        best = cell
                        best_place = place
                    elif cell == best and cell != 0 and place < best_place:
                        best_place = place

            wavefront.scores_current[unsafe_offset=row] = cell
            wavefront.deletes_current[unsafe_offset=row] = deletion
            wavefront.inserts_current[unsafe_offset=row] = insertion
            changes[unsafe_offset=tile + row * stride + column] = decision.bits
            if diagonal == last_diagonal and row == rows:
                answer = cell

        barrier()

        wavefront.rotate()

    # Pick the first row-major maximum, matching the strict `>` of the Python scan.
    block_argmax(best_scores, best_places, best, best_place)

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
    var state = Layer.ALIGNING

    while row > 0 and column > 0:
        var decision = CellDecision(changes[unsafe_offset=tile + row * stride + column])
        if mode == AlignmentMode.LOCAL and state == Layer.ALIGNING:
            if decision.reach() == PathReach.ENDS_HERE:
                break
        var step = advance(state, decision)
        var gap = Scalar[SYMBOL_DTYPE](GAP_BYTE)
        first_gapped[unsafe_offset=base + produced] = (
            letters[unsafe_offset=Int(sequences[unsafe_offset=first_start + row - 1])] if step.row_step != 0 else gap
        )
        second_gapped[unsafe_offset=base + produced] = (
            letters[unsafe_offset=Int(sequences[unsafe_offset=second_start + column - 1])] if step.column_step
            != 0 else gap
        )
        row += step.row_step
        column += step.column_step
        produced += 1
        state = step.lands_in

    # Only a global path is required to reach the origin; see the host reconstruction.
    if mode == AlignmentMode.GLOBAL:
        while row > 0:
            first_gapped[unsafe_offset=base + produced] = letters[
                unsafe_offset=Int(sequences[unsafe_offset=first_start + row - 1])
            ]
            second_gapped[unsafe_offset=base + produced] = Scalar[SYMBOL_DTYPE](GAP_BYTE)
            row -= 1
            produced += 1
        while column > 0:
            first_gapped[unsafe_offset=base + produced] = Scalar[SYMBOL_DTYPE](GAP_BYTE)
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
    sequences: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    offsets: List[Scalar[OFFSET_DTYPE]],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet: String,
    scoring: AffineGapCosts,
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
    var longest_first = 0
    for pair in range(pairs):
        var rows = Int(offsets[2 * pair + 1]) - Int(offsets[2 * pair])
        longest_first = max(longest_first, rows)
        var columns = Int(offsets[2 * pair + 2]) - Int(offsets[2 * pair + 1])
        change_offsets.append(running)
        running += Int64(rows + 1) * Int64(columns + 1)
        widest = max(widest, rows + columns)
    change_offsets.append(running)

    var sequences_buffer = upload(ctx, sequences)
    var offsets_buffer = upload(ctx, offsets)
    var substitutions_buffer = upload(ctx, substitutions)
    var letters = List[Scalar[SYMBOL_DTYPE]](capacity=alphabet_size)
    for index in range(alphabet_size):
        letters.append(Scalar[SYMBOL_DTYPE](alphabet_bytes[index]))
    var letters_buffer = upload(ctx, letters)
    var changes_buffer = ctx.enqueue_create_buffer[CHANGE_DTYPE](Int(max(running, Int64(1))))
    var change_offsets_buffer = upload(ctx, change_offsets)
    var results_buffer = zeroed[SCORE_DTYPE](ctx, pairs)
    var first_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(pairs * widest, 1))
    var second_buffer = ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(pairs * widest, 1))
    var lengths_buffer = zeroed[SCORE_DTYPE](ctx, pairs)

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
        Int32(longest_first + 1),
        Int32(alphabet_size),
        scoring.open,
        scoring.extend,
        grid_dim=pairs,
        block_dim=THREADS_PER_BLOCK,
        shared_mem_bytes=band_bytes(longest_first + 1),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(band_bytes(longest_first + 1))),
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
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    window_first_from: Int,
    window_first_to: Int,
    window_second_from: Int,
    window_second_to: Int,
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    tile_cells: Int,
    path_columns: MutSpan[Int32, _],
    path_entries: MutSpan[Layer, _],
) raises:
    """Hirschberg with every sweep on the device.

    The recursion itself is a few hundred bookkeeping steps and stays on the host; the sweeps carry
    the whole quadratic cost and run as kernels over global-memory bands, so nothing here is bounded
    by shared memory. Each launch is its own barrier, which is what stands in for the grid-wide sync
    Mojo does not expose.
    """
    if scoring.open > scoring.extend:
        raise Error("Gap opening must be at least as expensive as extension for the row split.")

    var rows = len(first)
    var columns = len(second)
    path_columns[window_first_to] = Int32(window_second_to)

    var sequences = List[Scalar[SYMBOL_DTYPE]](capacity=rows + columns)
    for index in range(rows):
        sequences.append(first[index])
    for index in range(columns):
        sequences.append(second[index])

    # Every sweep of a level claims its own slice, and a deep level is many tiny sweeps, so each
    # array carries the whole level: the real extent plus a constant per sweep, and a level never
    # holds more than two sweeps per row.
    var frontier_span = columns + 4 * rows + 32
    var left_span = 5 * rows + 32
    var tile_rows_count = (rows + TILE_SIDE - 1) // TILE_SIDE + 2
    var tile_columns_count = (columns + TILE_SIDE - 1) // TILE_SIDE + 2
    var corner_span = tile_rows_count * tile_columns_count + 6 * rows + 32
    var buffers = SweepBuffers(
        ctx.enqueue_create_buffer[SYMBOL_DTYPE](max(rows + columns, 1)),
        ctx.enqueue_create_buffer[SUBSTITUTION_DTYPE](len(substitutions)),
        ctx.enqueue_create_buffer[SCORE_DTYPE](frontier_span),
        ctx.enqueue_create_buffer[SCORE_DTYPE](frontier_span),
        ctx.enqueue_create_buffer[SCORE_DTYPE](frontier_span),
        ctx.enqueue_create_buffer[SCORE_DTYPE](frontier_span),
        ctx.enqueue_create_buffer[SCORE_DTYPE](4 * max(rows, 1)),
        ctx.enqueue_create_buffer[SCORE_DTYPE](left_span),
        ctx.enqueue_create_buffer[SCORE_DTYPE](left_span),
        ctx.enqueue_create_buffer[SCORE_DTYPE](corner_span),
        corner_span,
        left_span,
        frontier_span,
    )

    ctx.enqueue_copy(buffers.sequences, Span(sequences))
    ctx.enqueue_copy(buffers.substitutions, Span(substitutions))

    var frames = List[Frame]()
    frames.append(
        Frame(
            window_first_from,
            window_first_to,
            window_second_from,
            window_second_to,
            GapRun.OPENS,
            GapRun.OPENS,
        )
    )

    # Level-synchronous rather than depth-first: every frame of a level is independent, so they
    # sweep together instead of taking turns. Frames at a level partition both axes, so they share
    # the frontier arrays without touching.
    while len(frames) > 0:
        var splitting = List[Frame]()
        for index in range(len(frames)):
            var frame = frames[index]
            var height = frame.first_to - frame.first_from
            var width = frame.second_to - frame.second_from
            if height == 0:
                continue
            if width == 0 or height <= 2 or (height + 1) * (width + 1) <= tile_cells:
                tile_frame(
                    first, second, frame, substitutions, alphabet_size, scoring,
                    path_columns, path_entries,
                )
            else:
                splitting.append(frame)

        if len(splitting) == 0:
            break

        var sweeps = List[Sweep]()
        var joins = List[Scalar[SCORE_DTYPE]](length=len(splitting) * 3, fill=0)
        for index in range(len(splitting)):
            var frame = splitting[index]
            var split = (frame.first_from + frame.first_to) // 2
            var width = frame.second_to - frame.second_from
            # The second sequence is stored after the first, so its indices carry that offset.
            sweeps.append(
                Sweep(
                    split - frame.first_from, width, frame.first_from,
                    rows + frame.second_from, frame.first_from, frame.second_from,
                    frame.top, SweepHalf.FORWARD,
                )
            )
            sweeps.append(
                Sweep(
                    frame.first_to - split, width, split,
                    rows + frame.second_from, split, frame.second_from,
                    frame.bottom, SweepHalf.REVERSE,
                )
            )
            joins[index * 3 + 2] = Scalar[SCORE_DTYPE](width)

        sweep_level(ctx, buffers, sweeps, alphabet_size, scoring)

        for index in range(len(splitting)):
            joins[index * 3] = Scalar[SCORE_DTYPE](sweeps[index * 2].column_base)
            joins[index * 3 + 1] = Scalar[SCORE_DTYPE](sweeps[index * 2 + 1].column_base)
        var joins_buffer = upload(ctx, Span(joins))
        ctx.enqueue_function[crossing_kernel](
            buffers.top_scores.unsafe_ptr(),
            buffers.top_deletes.unsafe_ptr(),
            buffers.reverse_scores.unsafe_ptr(),
            buffers.reverse_deletes.unsafe_ptr(),
            joins_buffer.unsafe_ptr(),
            buffers.crossing.unsafe_ptr(),
            scoring.extend - scoring.open,
            grid_dim=len(splitting),
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()

        var children = List[Frame]()
        with buffers.crossing.map_to_host() as host:
            for index in range(len(splitting)):
                var frame = splitting[index]
                var split = (frame.first_from + frame.first_to) // 2
                var best_plain = host[index * 4]
                var best_plain_column = Int(host[index * 4 + 1])
                var best_gapped = host[index * 4 + 2]
                var best_gapped_column = Int(host[index * 4 + 3])
                # Either the path crosses in the aligning layer, or inside a deletion run. Both
                # are ordinary; the second is what the boundary flags carry, since the two halves
                # each charged an opening and the refund removes one.
                var crosses_in_a_gap = best_gapped > best_plain
                var crossing = frame.second_from + (
                    best_gapped_column if crosses_in_a_gap else best_plain_column
                )
                var shared_edge = GapRun.EXTENDS if crosses_in_a_gap else GapRun.OPENS
                children.append(
                    Frame(frame.first_from, split, frame.second_from, crossing, frame.top, shared_edge)
                )
                children.append(
                    Frame(split, frame.first_to, crossing, frame.second_to, shared_edge, frame.bottom)
                )
        frames = children^


def local_extremum_kernel[
    half: SweepHalf
](
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    bands: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin],
    outcome: Pointer[Scalar[DType.int64], MutAnyOrigin],
    first_from: Int32,
    first_to: Int32,
    second_from: Int32,
    second_to: Int32,
    alphabet_size: Int32,
    open: Int32,
    extend: Int32,
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
    comptime reversed_order = half == SweepHalf.REVERSE
    var scoring = AffineGapCosts(open, extend)

    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space=AddressSpace.SHARED,
    ]()
    var best_scores = stack_allocation[THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED]()
    var best_places = stack_allocation[THREADS_PER_BLOCK, Scalar[DType.int64], address_space=AddressSpace.SHARED]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        scores_table[unsafe_offset=index] = substitutions[unsafe_offset=index]
    barrier()

    var wavefront = Wavefront.over(bands, stride)

    var best = Int32(0)
    var best_place = Int64(0)
    var last_diagonal = rows + columns

    for diagonal in range(0, last_diagonal + 1):
        var row_low = max(0, diagonal - columns)
        var row_high = min(rows, diagonal)

        for row in range(row_low + Int(thread_idx.x), row_high + 1, Int(block_dim.x)):
            var column = diagonal - row
            var cell = Int32(0)
            var deletion = open + extend
            var insertion = open + extend

            if row > 0 and column > 0:
                var left_index = Int(first_to) - row if reversed_order else Int(first_from) + row - 1
                var right_index = Int(second_to) - column if reversed_order else Int(second_from) + column - 1
                var left_symbol = Int(sequences[unsafe_offset=left_index])
                var right_symbol = Int(sequences[unsafe_offset=right_index])
                var substitution = Int32(scores_table[unsafe_offset=left_symbol * width + right_symbol])
                var computed = gotoh_cell[AlignmentMode.LOCAL](
                    wavefront.scores_two_back[unsafe_offset=row - 1],
                    wavefront.scores_one_back[unsafe_offset=row - 1],
                    wavefront.deletes_one_back[unsafe_offset=row - 1],
                    wavefront.scores_one_back[unsafe_offset=row],
                    wavefront.inserts_one_back[unsafe_offset=row],
                    substitution,
                    scoring,
                )
                cell = computed.score
                deletion = computed.deletion
                insertion = computed.insertion
                var place = Int64(row) * Int64(columns + 1) + Int64(column)
                if cell > best:
                    best = cell
                    best_place = place
                elif cell == best and cell != 0 and place < best_place:
                    best_place = place

            wavefront.scores_current[unsafe_offset=row] = cell
            wavefront.deletes_current[unsafe_offset=row] = deletion
            wavefront.inserts_current[unsafe_offset=row] = insertion

        barrier()

        wavefront.rotate()

    block_argmax(best_scores, best_places, best, best_place)

    if thread_idx.x == 0:
        var place = best_places[unsafe_offset=0]
        outcome[unsafe_offset=0] = place // Int64(columns + 1)
        outcome[unsafe_offset=1] = place % Int64(columns + 1)
        outcome[unsafe_offset=2] = Int64(best_scores[unsafe_offset=0])


# Subproblems at or below this many cells are solved outright rather than split. Small enough
# that a leaf stays cheap, large enough that the recursion does not dominate.
comptime DEFAULT_TILE_CELLS = 4096

comptime TILE_SIDE = 256

# rows, columns, first_from, second_from, row_base, column_base, corner_base, entry, half,
# tile_rows_count, tile_columns_count
comptime SWEEP_PLAN_FIELDS = 11


def tiled_sweep_kernel(
    sequences: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    plans: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    forward_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    forward_deletes: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    reverse_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    reverse_deletes: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    left_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    left_inserts: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    corner_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    tile_diagonal: Int32,
    alphabet_size: Int32,
    open: Int32,
    extend: Int32,
):
    """One tile per block, one tile-anti-diagonal per launch, every sweep of a level at once.

    Tiles sharing an anti-diagonal are independent, so a launch boundary supplies the barrier that
    Mojo 1.0 has no grid-wide primitive for. `block_idx.y` selects one sweep out of the level's
    plan, which is what lets independent subproblems share a launch instead of taking turns.
    """
    var plan = Int(block_idx.y) * SWEEP_PLAN_FIELDS
    var rows = Int(plans[unsafe_offset=plan + 0])
    var columns = Int(plans[unsafe_offset=plan + 1])
    var first_from = Int(plans[unsafe_offset=plan + 2])
    var second_from = Int(plans[unsafe_offset=plan + 3])
    var row_base = Int(plans[unsafe_offset=plan + 4])
    var column_base = Int(plans[unsafe_offset=plan + 5])
    var corner_base = Int(plans[unsafe_offset=plan + 6])
    var extends = Int(plans[unsafe_offset=plan + 7]) != 0
    var reversed_order = Int(plans[unsafe_offset=plan + 8]) != 0
    var tile_rows_count = Int(plans[unsafe_offset=plan + 9])
    var tile_columns_count = Int(plans[unsafe_offset=plan + 10])
    var corner_stride = tile_columns_count + 1

    var tile_row_low = Int(tile_diagonal) - min(Int(tile_diagonal), tile_columns_count - 1)
    var tile_row_high = min(Int(tile_diagonal), tile_rows_count - 1)
    var tile_row = tile_row_low + Int(block_idx.x)
    if tile_row > tile_row_high:
        return
    var tile_column = Int(tile_diagonal) - tile_row
    var row_begin = tile_row * TILE_SIDE
    var column_begin = tile_column * TILE_SIDE
    var height = min(TILE_SIDE, rows - row_begin)
    var width_span = min(TILE_SIDE, columns - column_begin)
    if height <= 0 or width_span <= 0:
        return

    var top_scores = reverse_scores if reversed_order else forward_scores
    var top_deletes = reverse_deletes if reversed_order else forward_deletes
    var width = Int(alphabet_size)
    var stride = TILE_SIDE + 1
    var scoring = AffineGapCosts(open, extend)

    var bands = stack_allocation[
        BANDS_PER_SWEEP * (TILE_SIDE + 1), Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED
    ]()
    var scores_table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SUBSTITUTION_DTYPE],
        address_space=AddressSpace.SHARED,
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        scores_table[unsafe_offset=index] = substitutions[unsafe_offset=index]
    barrier()

    var wavefront = Wavefront.over(bands, stride)

    var last_diagonal = height + width_span
    for diagonal in range(0, last_diagonal + 1):
        var row_low = max(0, diagonal - width_span)
        var row_high = min(height, diagonal)

        for local_row in range(row_low + Int(thread_idx.x), row_high + 1, Int(block_dim.x)):
            var local_column = diagonal - local_row
            var row = row_begin + local_row
            var column = column_begin + local_column
            var cell = Int32(0)
            var deletion = open + extend
            var insertion = open + extend

            if local_row == 0 or local_column == 0:
                # A tile edge. Outside the matrix it is the affine ramp; inside it is whatever the
                # neighbouring tile left behind.
                if row == 0 and column == 0:
                    cell = 0
                    deletion = open + extend
                    insertion = deletion
                elif row == 0:
                    # The open run arrives at the corner only; elsewhere on the top row it ended.
                    cell = open + Int32(column - 1) * extend
                    deletion = cell + open + extend
                    insertion = deletion
                elif column == 0:
                    cell = Int32(row) * extend if extends else open + Int32(row - 1) * extend
                    deletion = cell
                    insertion = cell + open + extend
                elif local_row == 0 and local_column == 0:
                    cell = corner_scores[
                        unsafe_offset=corner_base + tile_row * corner_stride + tile_column
                    ]
                    deletion = cell + open + extend
                    insertion = deletion
                elif local_row == 0:
                    cell = top_scores[unsafe_offset=column_base + column]
                    deletion = top_deletes[unsafe_offset=column_base + column]
                    insertion = cell + open + extend
                else:
                    cell = left_scores[unsafe_offset=row_base + row]
                    insertion = left_inserts[unsafe_offset=row_base + row]
                    deletion = cell + open + extend
            else:
                var left_index = first_from + rows - row if reversed_order else first_from + row - 1
                var right_index = (
                    second_from + columns - column if reversed_order else second_from + column - 1
                )
                var left_symbol = Int(sequences[unsafe_offset=left_index])
                var right_symbol = Int(sequences[unsafe_offset=right_index])
                var substitution = Int32(scores_table[unsafe_offset=left_symbol * width + right_symbol])
                var computed = gotoh_cell[AlignmentMode.GLOBAL](
                    wavefront.scores_two_back[unsafe_offset=local_row - 1],
                    wavefront.scores_one_back[unsafe_offset=local_row - 1],
                    wavefront.deletes_one_back[unsafe_offset=local_row - 1],
                    wavefront.scores_one_back[unsafe_offset=local_row],
                    wavefront.inserts_one_back[unsafe_offset=local_row],
                    substitution,
                    scoring,
                )
                cell = computed.score
                deletion = computed.deletion
                insertion = computed.insertion

            wavefront.scores_current[unsafe_offset=local_row] = cell
            wavefront.deletes_current[unsafe_offset=local_row] = deletion
            wavefront.inserts_current[unsafe_offset=local_row] = insertion

            # Hand the tile's own edges to the neighbours, but only for cells this tile actually
            # computed. Its own left edge carries a fabricated deletion and its own top edge a
            # fabricated insertion; publishing either clobbers the true value the neighbouring
            # tile wrote, and races with it, since the two share a tile-anti-diagonal. The matrix
            # borders are genuine and do publish.
            if local_row == height and (local_column > 0 or column == 0):
                top_scores[unsafe_offset=column_base + column] = cell
                top_deletes[unsafe_offset=column_base + column] = deletion
            if local_column == width_span and (local_row > 0 or row == 0):
                left_scores[unsafe_offset=row_base + row] = cell
                left_inserts[unsafe_offset=row_base + row] = insertion
            if local_row == height and local_column == width_span:
                corner_scores[
                    unsafe_offset=corner_base + (tile_row + 1) * corner_stride + tile_column + 1
                ] = cell

        barrier()

        wavefront.rotate()


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
    # How much of each shared array a whole recursion level may claim.
    var corner_span: Int
    var left_span: Int
    var frontier_span: Int


def crossing_kernel(
    forward_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    forward_deletes: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    reverse_scores: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    reverse_deletes: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    joins: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    crossing: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    refund: Int32,
):
    """Picks where each split of a level crosses its cut, one block per split.

    Reading the frontiers back per split costs more than the sweeps once the recursion is
    thousands of nodes deep, so the reduction happens here and four numbers per split travel.
    """
    var split = Int(block_idx.x)
    var forward_base = Int(joins[unsafe_offset=split * 3])
    var reverse_base = Int(joins[unsafe_offset=split * 3 + 1])
    var span = Int(joins[unsafe_offset=split * 3 + 2])

    var plain_scores = stack_allocation[
        THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED
    ]()
    var plain_places = stack_allocation[
        THREADS_PER_BLOCK, Scalar[DType.int64], address_space=AddressSpace.SHARED
    ]()
    var gapped_scores = stack_allocation[
        THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED
    ]()
    var gapped_places = stack_allocation[
        THREADS_PER_BLOCK, Scalar[DType.int64], address_space=AddressSpace.SHARED
    ]()

    var best_plain = NEGATIVE_INFINITY
    var best_plain_column = Int64(0)
    var best_gapped = NEGATIVE_INFINITY
    var best_gapped_column = Int64(0)
    for offset in range(Int(thread_idx.x), span + 1, Int(block_dim.x)):
        var near = forward_base + offset
        var far = reverse_base + span - offset
        var plain = forward_scores[unsafe_offset=near] + reverse_scores[unsafe_offset=far]
        if plain > best_plain:
            best_plain = plain
            best_plain_column = Int64(offset)
        var gapped = (
            forward_deletes[unsafe_offset=near] + reverse_deletes[unsafe_offset=far] + refund
        )
        if gapped > best_gapped:
            best_gapped = gapped
            best_gapped_column = Int64(offset)

    block_argmax(plain_scores, plain_places, best_plain, best_plain_column)
    block_argmax(gapped_scores, gapped_places, best_gapped, best_gapped_column)
    if thread_idx.x == 0:
        crossing[unsafe_offset=split * 4] = plain_scores[unsafe_offset=0]
        crossing[unsafe_offset=split * 4 + 1] = Scalar[SCORE_DTYPE](plain_places[unsafe_offset=0])
        crossing[unsafe_offset=split * 4 + 2] = gapped_scores[unsafe_offset=0]
        crossing[unsafe_offset=split * 4 + 3] = Scalar[SCORE_DTYPE](gapped_places[unsafe_offset=0])


@fieldwise_init
struct Sweep(ImplicitlyCopyable, TrivialRegisterPassable):
    """One independent sub-rectangle sweep, as the plan-driven kernel needs to see it."""

    var rows: Int
    var columns: Int
    var first_from: Int
    var second_from: Int
    var row_base: Int
    var column_base: Int
    var entry: GapRun
    var half: SweepHalf

    def tile_rows(self) -> Int:
        return max((self.rows + TILE_SIDE - 1) // TILE_SIDE, 1)

    def tile_columns(self) -> Int:
        return max((self.columns + TILE_SIDE - 1) // TILE_SIDE, 1)

    def tile_diagonals(self) -> Int:
        return self.tile_rows() + self.tile_columns() - 1


def sweep_level(
    ctx: DeviceContext,
    mut buffers: SweepBuffers,
    mut sweeps: List[Sweep],
    alphabet_size: Int,
    scoring: AffineGapCosts,
) raises:
    """Sweeps every independent sub-rectangle of one recursion level together.

    Depth-first recursion offers the parallelism of a single frame, which halves as the recursion
    deepens while its work halves too, so utilization falls as fast as the work does. Frames at a
    level partition both axes, so they share the frontier arrays without touching, and the grid
    carries them all: `block_idx.x` is the tile, `block_idx.y` is the sweep.
    """
    if len(sweeps) == 0:
        return

    var plan = List[Scalar[SCORE_DTYPE]](length=len(sweeps) * SWEEP_PLAN_FIELDS, fill=0)
    var widest_tiles = 1
    var deepest = 0
    var corner_base = 0
    var top_base = 0
    var left_base = 0
    for index in range(len(sweeps)):
        var sweep = sweeps[index]
        var tile_rows = sweep.tile_rows()
        var tile_columns = sweep.tile_columns()
        var base = index * SWEEP_PLAN_FIELDS
        plan[base + 0] = Scalar[SCORE_DTYPE](sweep.rows)
        plan[base + 1] = Scalar[SCORE_DTYPE](sweep.columns)
        plan[base + 2] = Scalar[SCORE_DTYPE](sweep.first_from)
        plan[base + 3] = Scalar[SCORE_DTYPE](sweep.second_from)
        plan[base + 4] = Scalar[SCORE_DTYPE](left_base)
        plan[base + 5] = Scalar[SCORE_DTYPE](top_base)
        plan[base + 6] = Scalar[SCORE_DTYPE](corner_base)
        plan[base + 7] = Scalar[SCORE_DTYPE](Int(sweep.entry == GapRun.EXTENDS))
        plan[base + 8] = Scalar[SCORE_DTYPE](Int(sweep.half == SweepHalf.REVERSE))
        plan[base + 9] = Scalar[SCORE_DTYPE](tile_rows)
        plan[base + 10] = Scalar[SCORE_DTYPE](tile_columns)
        corner_base += (tile_rows + 1) * (tile_columns + 1)
        sweeps[index].row_base = left_base
        sweeps[index].column_base = top_base
        left_base += sweep.rows + 2
        top_base += sweep.columns + 2
        widest_tiles = max(widest_tiles, min(tile_rows, tile_columns))
        deepest = max(deepest, sweep.tile_diagonals())

    if corner_base > buffers.corner_span or left_base > buffers.left_span or top_base > buffers.frontier_span:
        raise Error("Sweep frontier scratch is too small for this recursion level.")

    var plan_buffer = upload(ctx, Span(plan))
    for tile_diagonal in range(deepest):
        ctx.enqueue_function[tiled_sweep_kernel](
            buffers.sequences.unsafe_ptr(),
            buffers.substitutions.unsafe_ptr(),
            plan_buffer.unsafe_ptr(),
            buffers.top_scores.unsafe_ptr(),
            buffers.top_deletes.unsafe_ptr(),
            buffers.reverse_scores.unsafe_ptr(),
            buffers.reverse_deletes.unsafe_ptr(),
            buffers.left_scores.unsafe_ptr(),
            buffers.left_inserts.unsafe_ptr(),
            buffers.corner_scores.unsafe_ptr(),
            Int32(tile_diagonal),
            Int32(alphabet_size),
            scoring.open,
            scoring.extend,
            grid_dim=(widest_tiles, len(sweeps)),
            block_dim=THREADS_PER_BLOCK,
        )
    ctx.synchronize()


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
        if top[index] == bottom[index] and top[index] != GAP_BYTE:
            color = green
        elif top[index] == GAP_BYTE or bottom[index] == GAP_BYTE:
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


def scoring_from(gaps: PythonObject) raises -> AffineGapCosts:
    """Reads the two gap penalties off a caller's gap-cost record by name."""
    var opening = optional_int(gaps.open).or_else(Int(DEFAULT_GAP_OPENING))
    var extension = optional_int(gaps.extend).or_else(Int(DEFAULT_GAP_EXTENSION))
    return AffineGapCosts(Int32(opening), Int32(extension))


def matrix_from(substitution: PythonObject, alphabet_size: Int) raises -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """BLOSUM62 unless the caller passed a uniform-cost record.

    No record at all means the default table. A tabulated record never reaches here, because the
    caller refuses it before the boundary; these kernels carry only the default.
    """
    var given_match: Optional[Int] = None
    var given_mismatch: Optional[Int] = None
    try:
        given_match = optional_int(substitution.match)
        given_mismatch = optional_int(substitution.mismatch)
    except:
        return default_proteins_matrix()
    if not given_match:
        return default_proteins_matrix()
    return uniform_matrix(alphabet_size, given_match.value(), given_mismatch.value())


def protein_defaults(gaps: PythonObject) raises -> Tuple[String, Int, AffineGapCosts]:
    """The alphabet, its size and the gap costs a binding call resolves to.

    The substitution table stays out of this because a `List` cannot be copied out of a tuple,
    and it is the one part that needs the alphabet size first anyway.
    """
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    return (alphabet, alphabet.byte_length(), scoring_from(gaps))


def gotoh_score[
    mode: AlignmentMode
](first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject,) raises -> PythonObject:
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)
    return PythonObject(Int(serial_score[mode](left, right, substitutions, alphabet_size, scoring)))


def gotoh_alignment[
    mode: AlignmentMode
](first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject,) raises -> PythonObject:
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)
    var result = serial_align[mode](left, right, substitutions, alphabet_size, scoring, alphabet)
    var triple = Python().list()
    triple.append(PythonObject(result.first_gapped))
    triple.append(PythonObject(result.second_gapped))
    triple.append(PythonObject(Int(result.score)))
    return triple


def python_length(value: PythonObject) raises -> Int:
    return Int(String(value.__len__()))


def gotoh_scores_batch[
    mode: AlignmentMode
](firsts: PythonObject, seconds: PythonObject, substitution: PythonObject, gaps: PythonObject,) raises -> PythonObject:
    """Scores a whole batch on the GPU, one thread block per pair.

    Both sides are packed into one Arrow-like tape — every sequence concatenated into a flat
    symbol buffer, with an offset array carrying two entries per pair — so a batch reaches the
    device as two buffers rather than as one transfer per sequence.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)

    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise Error("Both sides of the batch must have the same length.")
    if pairs == 0:
        return Python().list()

    # The tape: symbols concatenated end to end, offsets marking where each sequence starts.
    var sequences = List[Scalar[SYMBOL_DTYPE]]()
    var offsets = List[Scalar[OFFSET_DTYPE]]()
    offsets.append(0)
    for index in range(pairs):
        var left = translate(String(firsts[index]), alphabet)
        if len(left) > MAX_BAND_LENGTH:
            raise Error("First sequence exceeds the shared-memory band capacity.")
        sequences.extend(left^)
        offsets.append(Scalar[OFFSET_DTYPE](len(sequences)))
        var right = translate(String(seconds[index]), alphabet)
        sequences.extend(right^)
        offsets.append(Scalar[OFFSET_DTYPE](len(sequences)))

    var ctx = DeviceContext()
    var scores = wavefront_scores[mode](ctx, sequences, offsets, substitutions, alphabet_size, scoring)
    var output = Python().list()
    for index in range(len(scores)):
        output.append(PythonObject(Int(scores[index])))
    return output


def gotoh_alignments_batch[
    mode: AlignmentMode
](firsts: PythonObject, seconds: PythonObject, substitution: PythonObject, gaps: PythonObject,) raises -> PythonObject:
    """Aligns a whole batch on the GPU, returning `[first_gapped, second_gapped, score]` triples.

    Packed into the same Arrow-like tape the scoring batch uses.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)

    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise Error("Both sides of the batch must have the same length.")
    if pairs == 0:
        return Python().list()

    # The tape: symbols concatenated end to end, offsets marking where each sequence starts.
    var sequences = List[Scalar[SYMBOL_DTYPE]]()
    var offsets = List[Scalar[OFFSET_DTYPE]]()
    offsets.append(0)
    for index in range(pairs):
        var left = translate(String(firsts[index]), alphabet)
        if len(left) > MAX_BAND_LENGTH:
            raise Error("First sequence exceeds the shared-memory band capacity.")
        sequences.extend(left^)
        offsets.append(Scalar[OFFSET_DTYPE](len(sequences)))
        var right = translate(String(seconds[index]), alphabet)
        sequences.extend(right^)
        offsets.append(Scalar[OFFSET_DTYPE](len(sequences)))

    var ctx = DeviceContext()
    var aligned = direct_alignments[mode](ctx, sequences, offsets, substitutions, alphabet, scoring)
    var output = Python().list()
    for index in range(len(aligned)):
        var triple = Python().list()
        triple.append(PythonObject(aligned[index].first_gapped))
        triple.append(PythonObject(aligned[index].second_gapped))
        triple.append(PythonObject(Int(aligned[index].score)))
        output.append(triple)
    return output


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
        letters.append(FALLBACK_LETTER)
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
    var scoring = AffineGapCosts(Int32(-1), Int32(-1))
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
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """Global alignment in linear space, splitting rows and joining halves Myers-Miller style.

    `tile_cells` is the only knob: subproblems at or below it are solved outright, above it they
    are split. Raising it past the whole matrix collapses to a single direct traceback, which is
    what makes the two paths comparable.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)
    hirschberg_window(
        left,
        right,
        0,
        len(left),
        0,
        len(right),
        substitutions,
        alphabet_size,
        scoring,
        cells,
        path_columns,
        path_entries,
    )
    var score = score_path(left, right, path_columns, path_entries, substitutions, alphabet_size, scoring, len(left))
    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.GLOBAL, 0, len(left))
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


def needleman_wunsch_gotoh_alignment_linear_gpu(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """Global alignment in linear space with every sweep running on the device."""
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)
    var ctx = DeviceContext()
    hirschberg_path_gpu(
        ctx,
        left,
        right,
        0,
        len(left),
        0,
        len(right),
        substitutions,
        alphabet_size,
        scoring,
        cells,
        path_columns,
        path_entries,
    )
    var score = score_path(left, right, path_columns, path_entries, substitutions, alphabet_size, scoring, len(left))
    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.GLOBAL, 0, len(left))
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


def smith_waterman_gotoh_alignment_linear(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
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
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var last_row, last_column, score = local_extremum[SweepHalf.FORWARD](
        left, right, len(left), len(right), substitutions, alphabet_size, scoring
    )

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)

    var first_row = last_row
    if score > 0:
        var back_rows, back_columns, _ = local_extremum[SweepHalf.REVERSE](
            left, right, last_row, last_column, substitutions, alphabet_size, scoring
        )
        first_row = last_row - back_rows
        var first_column = last_column - back_columns

        path_columns[last_row] = Int32(last_column)
        var core_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
        var core_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)
        for index in range(len(path_columns)):
            core_columns[index] = path_columns[index]
            core_entries[index] = path_entries[index]
        hirschberg_window(
            left,
            right,
            first_row,
            last_row,
            first_column,
            last_column,
            substitutions,
            alphabet_size,
            scoring,
            cells,
            core_columns,
            core_entries,
        )
        for index in range(first_row, last_row + 1):
            path_columns[index] = core_columns[index]
            path_entries[index] = core_entries[index]

    var expanded = expand_path(
        left, right, path_columns, path_entries, alphabet, AlignmentMode.LOCAL, first_row, last_row
    )
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


def device_local_extremum[
    half: SweepHalf
](
    ctx: DeviceContext,
    sequences: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    rows: Int,
    first_to: Int,
    second_to: Int,
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
) raises -> Tuple[Int, Int, Int32]:
    """Runs one local sweep on the device and reads back where its maximum sits."""
    var sequences_buffer = upload(ctx, sequences)
    var substitutions_buffer = upload(ctx, substitutions)
    var bands_buffer = ctx.enqueue_create_buffer[SCORE_DTYPE](BANDS_PER_SWEEP * (first_to + 1))
    var outcome_buffer = zeroed[DType.int64](ctx, 3)

    ctx.enqueue_function[local_extremum_kernel[half]](
        sequences_buffer.unsafe_ptr(),
        substitutions_buffer.unsafe_ptr(),
        bands_buffer.unsafe_ptr(),
        outcome_buffer.unsafe_ptr(),
        Int32(0),
        Int32(first_to),
        Int32(rows),
        Int32(rows + second_to),
        Int32(alphabet_size),
        scoring.open,
        scoring.extend,
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
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """Local alignment in linear space with every sweep running on the device."""
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var sequences = List[Scalar[SYMBOL_DTYPE]](capacity=len(left) + len(right))
    sequences.extend(Span(left))
    sequences.extend(Span(right))

    var ctx = DeviceContext()
    var last_row, last_column, score = device_local_extremum[SweepHalf.FORWARD](
        ctx,
        sequences,
        len(left),
        len(left),
        len(right),
        substitutions,
        alphabet_size,
        scoring,
    )

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)

    var first_row = last_row
    if score > 0:
        var back_rows, back_columns, _ = device_local_extremum[SweepHalf.REVERSE](
            ctx,
            sequences,
            len(left),
            last_row,
            last_column,
            substitutions,
            alphabet_size,
            scoring,
        )
        first_row = last_row - back_rows
        var first_column = last_column - back_columns

        path_columns[last_row] = Int32(last_column)
        hirschberg_path_gpu(
            ctx,
            left,
            right,
            first_row,
            last_row,
            first_column,
            last_column,
            substitutions,
            alphabet_size,
            scoring,
            cells,
            path_columns,
            path_entries,
        )

    var expanded = expand_path(
        left, right, path_columns, path_entries, alphabet, AlignmentMode.LOCAL, first_row, last_row
    )
    var triple = Python().list()
    triple.append(PythonObject(expanded[0]))
    triple.append(PythonObject(expanded[1]))
    triple.append(PythonObject(Int(score)))
    return triple


@fieldwise_init
struct Executor(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a sweep runs."""

    var identifier: UInt8
    comptime HOST = Self(0)
    comptime DEVICE = Self(1)


def mode_from(value: PythonObject) raises -> AlignmentMode:
    """Reads the alignment mode a caller named."""
    var name = String(value)
    if name == "global":
        return AlignmentMode.GLOBAL
    if name == "local":
        return AlignmentMode.LOCAL
    raise Error(String("Unknown alignment mode: ", name))


def executor_from(value: PythonObject) raises -> Executor:
    """Reads the device a caller named."""
    var name = String(value)
    if name == "cpu":
        return Executor.HOST
    if name == "gpu":
        return Executor.DEVICE
    raise Error(String("Unknown device: ", name))


def paired_length(firsts: PythonObject, seconds: PythonObject) raises -> Int:
    """The number of pairs, refusing two sides that do not line up."""
    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise Error("Both sides of the batch must have the same length.")
    return pairs


def gotoh_scores(
    firsts: PythonObject,
    seconds: PythonObject,
    mode: PythonObject,
    device: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """Scores every pair. The score kernels are two-row, so linear space is the only space."""
    var requested = mode_from(mode)
    if executor_from(device) == Executor.DEVICE:
        comptime for index in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[index]
            if requested == candidate:
                return gotoh_scores_batch[candidate](firsts, seconds, substitution, gaps)

    var results = Python().list()
    for index in range(paired_length(firsts, seconds)):
        comptime for choice in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[choice]
            if requested == candidate:
                results.append(gotoh_score[candidate](firsts[index], seconds[index], substitution, gaps))
    return results


def align_pair(
    first: PythonObject,
    second: PythonObject,
    mode: AlignmentMode,
    executor: Executor,
    stored_budget: Int,
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """One pair, on the path its matrix can afford."""
    var cells = python_length(first) * python_length(second)
    if cells > stored_budget:
        if executor == Executor.DEVICE:
            if mode == AlignmentMode.LOCAL:
                return smith_waterman_gotoh_alignment_linear_gpu(first, second, substitution, gaps)
            return needleman_wunsch_gotoh_alignment_linear_gpu(first, second, substitution, gaps)
        if mode == AlignmentMode.LOCAL:
            return smith_waterman_gotoh_alignment_linear(first, second, substitution, gaps)
        return needleman_wunsch_gotoh_alignment_linear(first, second, substitution, gaps)

    if executor == Executor.DEVICE:
        # No single-pair device entry exists for the stored traceback, so this is a batch of one.
        var lefts = Python().list()
        var rights = Python().list()
        lefts.append(first)
        rights.append(second)
        comptime for index in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[index]
            if mode == candidate:
                return gotoh_alignments_batch[candidate](lefts, rights, substitution, gaps)[0]

    comptime for index in range(len(ALL_MODES)):
        comptime candidate = ALL_MODES[index]
        if mode == candidate:
            return gotoh_alignment[candidate](first, second, substitution, gaps)
    raise Error("Unhandled alignment mode.")


def gotoh_alignments(
    firsts: PythonObject,
    seconds: PythonObject,
    mode: PythonObject,
    device: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    stored_budget: PythonObject,
) raises -> PythonObject:
    """Aligns every pair, taking the linear-space traceback once a matrix outgrows the budget.

    A device batch whose every matrix fits goes out as one launch; anything else walks pair by
    pair, because the two tracebacks cannot share a launch.
    """
    var requested = mode_from(mode)
    var executor = executor_from(device)
    var budget = Int(String(stored_budget))
    var pairs = paired_length(firsts, seconds)

    var widest = 0
    for index in range(pairs):
        var cells = python_length(firsts[index]) * python_length(seconds[index])
        if cells > widest:
            widest = cells

    if executor == Executor.DEVICE and widest <= budget and pairs > 0:
        comptime for index in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[index]
            if requested == candidate:
                return gotoh_alignments_batch[candidate](firsts, seconds, substitution, gaps)

    var results = Python().list()
    for index in range(pairs):
        results.append(
            align_pair(
                firsts[index],
                seconds[index],
                requested,
                executor,
                budget,
                substitution,
                gaps,
            )
        )
    return results


@export
def PyInit_affinegaps_mojo() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("affinegaps_mojo")
        builder.def_function[gotoh_scores]("gotoh_scores")
        builder.def_function[gotoh_alignments]("gotoh_alignments")
        builder.def_function[levenshtein_alignment]("levenshtein_alignment")
        builder.def_function[colorize_alignment]("colorize_alignment")
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
        print(
            """Usage: affinegaps FIRST SECOND [--local] [--gpu]
                  [--match N] [--mismatch N]
                  [--open N] [--extend N] [--help]

Gotoh affine-gap alignment. Reconstructs the alignment, not just the score."""
        )
        return

    var first = String(arguments[1])
    var second = String(arguments[2])
    var local = False
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
            index += 1
        elif index + 1 < count:
            var value = parse_int(String(arguments[index + 1]))
            if flag == "--open":
                opening = value.or_else(opening)
            elif flag == "--extend":
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
    var scoring = AffineGapCosts(Int32(opening), Int32(extension))
    var substitutions = default_proteins_matrix() if not match_score else uniform_matrix(
        alphabet_size, match_score.value(), mismatch_score.value()
    )
    var left = translate(first, alphabet)
    var right = translate(second, alphabet)

    var result: AlignmentResult
    if local:
        result = serial_align[AlignmentMode.LOCAL](left, right, substitutions, alphabet_size, scoring, alphabet)
    else:
        result = serial_align[AlignmentMode.GLOBAL](left, right, substitutions, alphabet_size, scoring, alphabet)

    var painted = colorize(result.first_gapped, result.second_gapped)
    print(
        """
Sequence 1:  {}
Sequence 2:  {}

Alignment 1: {}
Alignment 2: {}
Score:       {}""".format(
            first, second, painted[0], painted[1], result.score
        )
    )


# endregion Command Line
