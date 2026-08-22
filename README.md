# AffineGaps

![Affine Gaps Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/AffineGaps.jpg?raw=true)

__Affine Gaps__ is a __less-wrong__ implementation of Osamu Gotoh affine gap penalty extensions 1982 [paper](https://doc.aporc.org/attach/Course001Papers/gotoh1982.pdf) for the Needleman-Wunsch and Smith-Waterman algorithms often used for global and local sequence alignment in Bioinformatics.
Unlike the fast aligners, it doesn't stop at the score — it reconstructs the alignment itself, __on the GPU, in linear memory__.
A NumPy reference implementation ships beside the Mojo kernels and serves as the parity oracle.

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

During my exploration of existing implementations, I've noticed several bugs:

- several libraries initialize the header row/columns of penalty matrices with ±∞, causing overflows on the first iteration.
- initialize matrices to zeros, ignoring the first gap opening cost.
- combining opening and expansion costs where only the opening cost should be applied.
- even the most correct `needle` from EMBOSS uses `float` representation, which would obviously be numerically unstable on very long sequences.

## Performance on GPU

One __NVIDIA H100 80GB HBM3__ against __WFA2__ on one CPU core, both reconstructing the alignment rather than only scoring it.
DNA pairs at 10 percent divergence, the range long-read work lives in, with every score checked to agree exactly before timing.

```
                   wfa2 faster  ←│→  affinegaps faster
    1 kbp                       ▍│                          1.1x
   10 kbp                        │████▋                     2.2x
  100 kbp                        │█████████████████        17.2x
  300 kbp                        │██████████████████████   39.8x
    1 Mbp                        │██████████████████████  out of memory
```

The last row is not a timing.
WFA2 reconstructs exactly, and its traceback grows with the square of the alignment score, so a megabase pair at this divergence exhausts tens of gigabytes before finishing, where the linear-space recursion holds a few frontiers and finishes in seconds.

That same growth is what decides every other row, and it makes the crossover a property of the data rather than of the sequence length.
On a 30 kilobase pair, mutated harder each row:

```
                   wfa2 faster  ←│→  affinegaps faster
     0.1%     ███████████████████│                         55.5x
       1%            ███████████▊│                         12.0x
       5%                        │█▊                        1.5x
      10%                        │███████▌                  5.0x
      25%                        │███████████████▊         28.3x
      50%                        │████████████████████▋    79.7x
    99.9%                        │██████████████████████  104.6x
```

A full sweep never notices how different the sequences are, so reach for WFA2 on near-identical inputs and for this on divergent or very long ones.

## Installation

There is no package-registry release; the repository is the distribution.
Where Mojo ships a toolchain — Linux on x86-64 or ARM, and macOS on Apple silicon — the compiled kernels are built during the install and travel with the package.
Everywhere else you get the NumPy reference alone, from the same command.

```bash
uv pip install git+https://github.com/unum-science/AffineGaps.git
uv pip install 'affinegaps[numba] @ git+https://github.com/unum-science/AffineGaps.git'
```

Pin a tag or a commit when the build has to be reproducible:

```bash
uv pip install 'affinegaps @ git+https://github.com/unum-science/AffineGaps.git@v0.2.5'
```

Two optional extras, neither of them required: `numba` accelerates the NumPy reference, and `color` paints the command-line output.

### The Command-Line Tool

`uv tool install` puts `affinegaps` on your path without touching the current environment:

```bash
$ uv tool install git+https://github.com/unum-science/AffineGaps.git
$ affinegaps GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV --local
```

The same tool is also built natively from a checkout, with no Python involved at run time:

```bash
git clone https://github.com/unum-science/AffineGaps.git && cd AffineGaps
pixi run install      # `affinegaps` on the PATH, compiled kernels included
pixi run build-cli    # or build/affinegaps, a standalone binary
```

Both accept the same flags and print the same thing, so install whichever suits you rather than both.

### As a Pixi Dependency

A downstream [pixi](https://pixi.sh) project takes the same git dependency, pinned in `pixi.lock` beside everything else:

```bash
pixi add --pypi 'affinegaps @ git+https://github.com/unum-science/AffineGaps.git'
```

To run the tool once without installing anything, `uvx` fetches, builds and runs it in a throwaway environment:

```bash
uvx --from git+https://github.com/unum-science/AffineGaps.git affinegaps GATTACA GACTATA
```

### Requirements for the GPU

The compiled kernels run on the CPU anywhere Mojo builds them.
Reaching the GPU additionally needs an NVIDIA device with driver 580 or newer, and `available("mojo", "gpu")` reports whether this machine has one — it tries a real alignment rather than assuming.

## Using the Library

To obtain the alignment of two sequences, use the `needleman_wunsch_gotoh_alignment` function.

```python
from affinegaps import needleman_wunsch_gotoh_alignment

insulin = "GIVEQCCTSICSLYQLENYCN"
glucagon = "HSQGTFTSDYSKYLDSRAEQDFV"
aligned_insulin, aligned_glucagon, score = needleman_wunsch_gotoh_alignment(insulin, glucagon)

print("Alignment 1:", aligned_insulin)  # ---GIVEQCCTSICSLY---QL-ENYCN-
print("Alignment 2:", aligned_glucagon) # HSQGTF----TSDYSKYLDSRAEQDF--V
print("Score:", score)                  # 22
```

If you only need the alignment score, `needleman_wunsch_gotoh_score` uses less memory and works faster.

The same call runs elsewhere by naming where the work goes.
Two axes: `backend` chooses which implementation, `device` chooses which hardware.
Naming neither picks the fastest the machine offers.

```python
needleman_wunsch_gotoh_alignment(insulin, glucagon)                     # fastest available
needleman_wunsch_gotoh_alignment(insulin, glucagon, backend="python")   # the reference
needleman_wunsch_gotoh_alignment(insulin, glucagon, backend="mojo")     # compiled, host
needleman_wunsch_gotoh_alignment(insulin, glucagon, backend="mojo", device="gpu")
```

Unspecified adapts; specified is honoured or refused.
Asking for a backend that is not built raises rather than quietly running something else, so a measurement can never report the GPU while timing the reference.

There is no knob for how the traceback stores its state.
Below a size threshold it keeps a decision per cell, above it recurses in linear space, and the two produce identical output — so the choice is cost, never correctness.

Batches are where the device earns its keep, one thread block per pair:

```python
from affinegaps import needleman_wunsch_gotoh_alignments, available

if available("mojo", "gpu"):
    results = needleman_wunsch_gotoh_alignments(firsts, seconds, backend="mojo", device="gpu")
```

By default a BLOSUM62 substitution matrix scaled by five is used.
Costs come in two records: the gap model, and either a uniform pair of scores or a table.

```python
import numpy as np
from affinegaps import AffineGapCosts, UniformSubstitutionCosts, TabulatedSubstitutionCosts

aligned_insulin, aligned_glucagon, score = needleman_wunsch_gotoh_alignment(
    insulin, glucagon,
    substitution=UniformSubstitutionCosts(match=1, mismatch=-1),
    gaps=AffineGapCosts(open=-2, extend=-1),
)

alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
substitutions = np.full((len(alphabet), len(alphabet)), -1, dtype=np.int8)
np.fill_diagonal(substitutions, 1)

aligned_insulin, aligned_glucagon, score = needleman_wunsch_gotoh_alignment(
    insulin, glucagon,
    substitution=TabulatedSubstitutionCosts(alphabet, substitutions),
    gaps=AffineGapCosts(open=-2, extend=-1),
)
```

A uniform cost carries its match and its mismatch together, and a table carries its matrix and
the alphabet indexing it, so neither can be half-specified and the two cannot be combined.

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

To compute the optimal global alignment of insulin and glucagon sequences with the BLOSUM62 substitution matrix scaled by five through CLI:

```bash
$ affinegaps GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV
>
> Sequence 1: GIVEQCCTSICSLYQLENYCN
> Sequence 2: HSQGTFTSDYSKYLDSRAEQDFV
>
> Alignment 1: ---GIVEQCCTSICSLY---QL-ENYCN-
> Alignment 2: HSQGTF----TSDYSKYLDSRAEQDF--V
> Score:       22
```

To compute the local alignment of the same sequences, pass `--local`.
The result is right-trimmed but not left-trimmed, which is what the recurrence's own traceback produces:

```bash
$ affinegaps GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV --local
>
> Alignment 1: ------GIVEQCCTSICSLYQLEN
> Alignment 2: HSQGTF-------TSDYSKY-LDS
> Score:       80
```

`--gpu` runs the same alignment on the device.

## EMBOSS and Other Tools

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
