# AffineGaps

![Affine Gaps Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/AffineGaps.jpg?raw=true)

__Affine Gaps__ is a __less-wrong__ single-file Numba-accelerated Python implementation of Osamu Gotoh affine gap penalty extensions 1982 [paper](https://doc.aporc.org/attach/Course001Papers/gotoh1982.pdf) for the Needleman-Wunsch and Smith-Waterman algorithms often used for global and local sequence alignment in Bioinformatics.
Thanks to the Numba JIT compiler, it's also competitive in terms of performance.
But if you want to go even faster and need more hardware-accelerated string operations, check out [StringZilla](https://github.com/ashvardanian/stringzilla) 🦖

## Less Wrong

As reported in the "Are all global alignment algorithms and implementations correct?" [paper](https://www.biorxiv.org/content/10.1101/031500v1.full.pdf) by Tomas Flouri, Kassian Kobert, Torbjørn Rognes, and Alexandros Stamatakis:

> In 1982 Gotoh presented an improved algorithm with lower time complexity. 
> Gotoh’s algorithm is frequently cited...
> While implementing the algorithm, we discovered two mathematical mistakes in Gotoh’s paper that induce sub-optimal sequence alignments.
> First, there are minor indexing mistakes in the dynamic programming algorithm which become apparent immediately when implementing the procedure.
> Hence, we report on these for the sake of completeness.
> Second, there is a more profound problem with the dynamic programming matrix initialization.
> This initialization issue can easily be missed and find its way into actual implementations.
> This error is also present in standard text books.
> Namely, the widely used books by Gusfield and Waterman.
> To obtain an initial estimate of the extent to which this error has been propagated, we scrutinized freely available undergraduate lecture slides.
> We found that 8 out of 31 lecture slides contained the mistake, while 16 out of 31 simply omit parts of the initialization, thus giving an incomplete description of the algorithm.
> Finally, by inspecting ten source codes and running respective tests, we found that five implementations were incorrect.

During my exploration of exiting implementations, I've noticed several bugs:

- several libraries initialize the header row/columns of penalty matrices with ±∞, causing overflows on the first iteration.
- initialize matrices to zeros, ignoring the first gap opening cost.
- combining opening and expansion costs where only the opening cost should be applied.
- even the most correct `needle` from EMBOSS uses `float` representation, which would obviously be numerically unstable on very long sequences.

## Installation

Numba is optional.
Installing without it gives a pure-Python baseline; installing with the `numba` extra enables JIT acceleration when a compatible Numba is available.

```sh
uv pip install affinegaps          # minimal
uv pip install 'affinegaps[numba]' # with JIT
```

Even without installing Python or touching PyPi, you can just use `uv` to get the latest version of the library:

```bash
$ uv tool install git+https://github.com/unum-bio/AffineGaps.git
$ affinegaps --help
```

## Using the Library

To obtain the alignment of two sequences, use the `needleman_wunsch_gotoh_alignment` function.

```python
from affinegaps import needleman_wunsch_gotoh_alignment

insulin = "GIVEQCCTSICSLYQLENYCN"
glucagon = "HSQGTFTSDYSKYLDSRAEQDFV"
aligned_insulin, aligned_glucagon, aligned_score = needleman_wunsch_gotoh_alignment(insulin, glucagon)

print("Alignment 1:", aligned_insulin)  # GI-V---EQCC-TSICSLY---QL-ENYCN-
print("Alignment 2:", aligned_glucagon) # --D-FVHSQGTFTSDYSKYLDSRAEQDF--V
print("Score:", aligned_score)          # 41
```

If you only need the alignment score, you can use the `needleman_wunsch_gotoh_score` function, which uses less memory and works faster.

```python
from affinegaps import needleman_wunsch_gotoh_score

score = needleman_wunsch_gotoh_score(insulin, glucagon)

print("Score:", score)
```

By default, a BLOSUM62 substitution matrix is used.
You can specify a different substitution matrix by passing it as an argument.

```python
from numpy import np

alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
substitutions = np.zeros((len(alphabet), len(alphabet)), dtype=np.int8)
substitutions.fill(-1)
np.fill_diagonal(substitutions, 1)

aligned_insulin, aligned_glucagon, aligned_score = needleman_wunsch_gotoh_alignment(
    insulin, glucagon,
    substitution_alphabet=alphabet,
    substitution_matrix=substitutions,
    gap_opening=-2,
    gap_extension=-1,
)
```

That is similar to the following usage example of BioPython:

```python
from Bio import Align
from Bio.Align import substitution_matrices

aligner = Align.PairwiseAligner(mode="global")
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = open_gap_score
aligner.extend_gap_score = extend_gap_score
```

## Using the Command Line Interface

To compute the optimal global alignment of insulin and glucagon sequences with (5x-scaled) BLOSUM62 substitution matrix through CLI:

```bash
$ affinegaps GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV
>
> Sequence 1: GIVEQCCTSICSLYQLENYCN
> Sequence 2: HSQGTFTSDYSKYLDSRAEQDFV
>
> Alignment 1: GIVEQCCTSICSLY---QL-ENYCN-
> Alignment 2: GTF----TSDYSKYLDSRAEQDF--V
> Score:       22
```

To compute the local alignment of insulin and glucagon sequences through CLI:

```bash

$ affinegaps GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV --local
> 
> Sequence 1: GIVEQCCTSICSLYQLENYCN
> Sequence 2: HSQGTFTSDYSKYLDSRAEQDFV
> 
> Alignment 1: TSICSLYQLEN
> Alignment 2: TSDYSKY-LDS
> Score:       80
```

## Testing & Development

To test, install the development dependencies and run the tests.

```bash
pip install -e ".[dev,jit]"
pytest test.py -s -x
```

### Symmetry Test for Needleman-Wunsch

First, verify that the Needleman-Wunsch algorithm is symmetric with respect to the argument order, assuming the substitution matrix is symmetric.

```bash
pytest test.py -s -x -k symmetry
```

### Needleman-Wunsch and Levenshtein Score Equivalence

The Needleman-Wunsch alignment score should be equal to the negated Levenshtein distance for specific match/mismatch costs.

```bash
pytest test.py -s -x -k levenshtein
```

### Alignment vs Scoring Consistency

Check that the alignment score is consistent with the scoring function for specific sequences and scoring parameters.

```bash
pytest test.py -s -x -k scoring_vs_alignment
```

### Gap Expansion Test

Check the effect of gap expansions on alignment scores. This test ensures that increasing the width of gaps in alignments with zero gap extension penalties does not change the alignment score.

```bash
pytest test.py -s -x -k gap_expansions
```

### Comparison with BioPython Examples

Compare the affine gap alignment scores with BioPython for specific sequence pairs and scoring parameters. This test ensures that the Needleman-Wunsch-Gotoh alignment scores are at least as good as BioPython's PairwiseAligner scores.

```bash
pytest test.py -s -x -k biopython_examples
```

### Fuzzy Comparison with BioPython

Perform a fuzzy comparison of affine gap alignment scores with BioPython for randomly generated sequences. This test verifies that the Needleman-Wunsch-Gotoh alignment scores are at least as good as BioPython's PairwiseAligner scores for various gap penalties.

```bash
pytest test.py -s -x -k biopython_fuzzy
```

### EMBOSS and Other Tools

Seemingly the only correct known open-source implementation is located in `nucleus/embaln.c` file in the EMBOSS package in the `embAlignPathCalcWithEndGapPenalties` and `embAlignGetScoreNWMatrix` functions.
That program was originally [implemented in 1999 by Alan Bleasby](https://www.bioinformatics.nl/cgi-bin/emboss/help/needle) and tweaked in 2000 for better scoring.
That implementation has no SIMD optimizations, branchless-computing tricks, or other modern optimizations, but it's still widely recommended.
If you want to compare the results, you can download the EMBOSS source code and compile it with following commands:

```bash
wget -m 'ftp://emboss.open-bio.org/pub/EMBOSS/'
cd emboss.open-bio.org/pub/EMBOSS/
gunzip EMBOSS-latest.tar.gz
tar xf EMBOSS-latest.tar
cd EMBOSS-latest
./configure
```

Or if you simply want to explore the source:

```bash
cat emboss.open-bio.org/pub/EMBOSS/EMBOSS-6.6.0/nucleus/embaln.c
```
