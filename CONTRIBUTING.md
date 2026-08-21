# Contributing to Affine Gaps

To test, install the development dependencies and run the tests.

```bash
pip install -e . --group test
pytest test.py
```

Alternatively, consider using `uv`:

```sh
uv venv --python 3.12           # Or your preferred Python version
source .venv/bin/activate       # To activate the virtual environment
uv pip install --group test .   # To install the package and its test dependencies
uv run pytest test.py           # To run the tests
```

### Testing the Mojo Backend

The GPU kernels are optional. Build them with `pixi run build` and the same suite picks them up; without a build, every Mojo test skips and the pure-Python suite still runs.

```sh
pixi run test                                  # builds the extension, then runs the suite against it
AFFINEGAPS_BACKENDS=python-cpu pixi run pytest # the reference alone, which is what CI runs
```

Every property test runs against each backend, named `python-cpu`, `numba-cpu`, `mojo-cpu` and `mojo-gpu`.
A backend the machine cannot serve is skipped rather than failed, and the skip reason carries the real cause — a missing build and an unsupported driver are different problems and say so.

The Mojo kernels and the Python reference are held to the same recurrence, the same border initialization and the same tie-breaking, so the suite compares them exhaustively rather than by sampling — every pair of sequences up to length five over a three-letter alphabet, both global and local, on the host and on the device.

### Every Path Must Achieve Its Own Score

The invariant worth knowing about before touching a traceback.
With affine gaps it is not automatic: a walk that reads only the winning operation at each cell can leave a gap run and re-enter it, paying a second opening penalty the score never did, and the returned strings then score less than the number returned beside them.
Both implementations walk the match, deletion and insertion layers, and `test_reference_alignment_achieves_its_score` re-scores every returned path to enforce it.

Local alignment has a second version of the same trap.
The traceback stops at the first non-positive cell, so the untraced prefixes are outside the alignment and must not be flushed into the result.

Scores are separately checked against brute-force enumeration of every possible alignment for short inputs, which shares no code with the dynamic programming and so catches a recurrence that is self-consistently wrong.

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
