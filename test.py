#!/usr/bin/env python3
"""
Test suite for affine gap alignment functions.

This suite validates the correctness and consistency of several alignment functions,
including Needleman-Wunsch, Smith-Waterman, and Levenshtein alignments. Tests ensure:

- Symmetry of alignment scores.
- Consistency with external tools like EMBOSS `Needle` and BioPython `PairwiseAligner`.
- Proper handling of gap expansions and penalty configurations.
- Verification of alignment results against known examples and random inputs.
"""

import os
import re
import math
import tempfile
import subprocess
from random import choice, randint, getrandbits
from typing import Literal

import pytest
from Bio import Align
from Bio.Align import substitution_matrices

from affinegaps import (
    needleman_wunsch_gotoh_alignment,
    needleman_wunsch_gotoh_score,
    smith_waterman_gotoh_alignment,
    smith_waterman_gotoh_score,
    levenshtein_alignment,
    colorize_alignment,
    default_proteins_alphabet,
    default_proteins_matrix,
)


@pytest.mark.repeat(30)
@pytest.mark.parametrize("min_length", [5, 10])
@pytest.mark.parametrize("max_length", [15, 25])
def test_symmetry(min_length: int, max_length: int):
    """
    Verify that Needleman-Wunsch alignment is symmetric.
    """
    alphabet = "ACGT"
    str1 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))
    str2 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))

    # Compute alignment scores
    score1 = needleman_wunsch_gotoh_score(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=-1,
        gap_extension=-1,
        match=0,
        mismatch=-1,
    )

    score2 = needleman_wunsch_gotoh_score(
        str2,
        str1,
        substitution_alphabet=alphabet,
        gap_opening=-1,
        gap_extension=-1,
        match=0,
        mismatch=-1,
    )

    assert (
        score1 == score2
    ), f"Alignment score symmetry failed for {str1} <-> {str2}. Score1: {score1}, Score2: {score2}"


def run_emboss(
    seq1: str,
    seq2: str,
    match: int = 0,
    mismatch: int = -1,
    gap_opening: int = -1,
    gap_extension: int = -1,
):
    """
    Run EMBOSS CLI tool for sequence alignment and extract results.
    """
    matrix_content = f"""
A  C  G  T
A  {match}  {mismatch}  {mismatch}  {mismatch}
C  {mismatch}  {match}  {mismatch}  {mismatch}
G  {mismatch}  {mismatch}  {match}  {mismatch}
T  {mismatch}  {mismatch}  {mismatch}  {match}
    """

    # Create temporary files for sequences and matrix
    with (
        tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fa") as tmp1,
        tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fa") as tmp2,
        tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".mat") as tmp_matrix,
        tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_output,
    ):

        tmp1.write(f">seq1\n{seq1}\n")
        tmp2.write(f">seq2\n{seq2}\n")
        tmp_matrix.write(matrix_content)

        tmp1_name = tmp1.name
        tmp2_name = tmp2.name
        matrix_name = tmp_matrix.name
        output_name = tmp_output.name

    try:
        os.chmod(tmp1_name, 0o644)
        os.chmod(tmp2_name, 0o644)
        os.chmod(matrix_name, 0o644)
        os.chmod(output_name, 0o644)

        command = [
            "needle",
            "-asequence",
            tmp1_name,
            "-bsequence",
            tmp2_name,
            "-gapopen",
            str(-gap_opening),
            "-gapextend",
            str(-gap_extension),
            "-datafile",
            matrix_name,
            "-outfile",
            output_name,
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        # Check if the subprocess call was successful
        if result.returncode != 0:
            raise Exception(f"Needle failed with return code {result.returncode} and error message: {result.stderr}")

        # Read the output file
        with open(output_name) as file:
            output_content = file.read()

        # Extract relevant information using regex
        score_match = re.search(r"Score:\s+([0-9.]+)", output_content)
        alignment_match = re.findall(r"(seq\d\s+\d+\s+([A-Za-z-]+)\s+\d+)", output_content)

        assert score_match, f"Score not found in {output_content}"
        assert alignment_match, f"Alignments not found in {output_content}"

        score = math.floor(float(score_match.group(1)))
        alignments = [x[1] for x in alignment_match]
        assert len(alignments) == 2, f"Expected 2 alignments, got {len(alignments)}"

        return alignments[0], alignments[1], score

    finally:
        os.remove(tmp1_name)
        os.remove(tmp2_name)
        os.remove(matrix_name)
        os.remove(output_name)


def replace_single_dashes(text, replacement):
    """
    Replace single dashes in a sequence with a specified replacement string.
    """
    within_line = r"[^-](-)[^-]"
    before_line = r"^(-)[^-]"
    after_line = r"[^-](-)$"
    result = text
    result = re.sub(within_line, replacement, result)
    result = re.sub(before_line, replacement, result)
    result = re.sub(after_line, replacement, result)
    return result


@pytest.mark.repeat(30)
@pytest.mark.parametrize("min_length", [3, 7])
@pytest.mark.parametrize("max_length", [7, 15])
def test_against_levenshtein(min_length: int, max_length: int):
    """
    Test Levenshtein and Needleman-Wunsch-Gotoh alignment consistency.

    Ensures that Levenshtein and Needleman-Wunsch-Gotoh alignments produce the same scores and
    alignments for sequences of varying lengths within a specified alphabet.
    """
    alphabet = "ACGT"
    str1 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))
    str2 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))

    # A subprocess may take a while to evaluate
    # emboss1, emboss2, emboss_score = run_emboss(str1, str2)
    lev1, lev2, lev_score = levenshtein_alignment(str1, str2)
    nw1, nw2, nw_score = needleman_wunsch_gotoh_alignment(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=-1,
        gap_extension=-1,
        match=0,
        mismatch=-1,
    )
    only_score = needleman_wunsch_gotoh_score(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=-1,
        gap_extension=-1,
        match=0,
        mismatch=-1,
    )

    lev1, lev2 = colorize_alignment(lev1, lev2)
    nw1, nw2 = colorize_alignment(nw1, nw2)
    assert lev_score == -nw_score and lev_score == -only_score, f"""
    Levenshtein and Needleman-Wunsch should return the same score.
    Levenshtein scored {lev_score}:
        {lev1}
        {lev2}
    Needleman-Wunsch scored {nw_score}:
        {nw1}
        {nw2}
    """


@pytest.mark.repeat(30)
@pytest.mark.parametrize("min_length", [3, 7])
@pytest.mark.parametrize("max_length", [7, 15])
@pytest.mark.parametrize("match_score", [1, 2, 3])
@pytest.mark.parametrize("mismatch_score", [-4, -2])
@pytest.mark.parametrize("gap_opening", [-5, -1])
@pytest.mark.parametrize("mode", ["global", "local"])
def test_scoring_vs_alignment(
    min_length: int,
    max_length: int,
    match_score: int,
    mismatch_score: int,
    gap_opening: int,
    mode: str,
):
    """
    Test that alignment and (just) scoring functions return the same scores for global and local alignments.

    Ensures that Needleman-Wunsch-Gotoh and Smith-Waterman-Gotoh alignment functions return the same scores
    as their scoring-only counterparts for sequences of varying lengths within a specified alphabet.
    """

    alphabet = "ACGT"
    str1 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))
    str2 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))

    # A subprocess may take a while to evaluate
    scoring = needleman_wunsch_gotoh_score if mode == "global" else smith_waterman_gotoh_score
    alignment = needleman_wunsch_gotoh_alignment if mode == "global" else smith_waterman_gotoh_alignment
    aligned1, aligned2, aligned_score = alignment(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=gap_opening,
        gap_extension=-1,
        match=match_score,
        mismatch=mismatch_score,
    )
    only_score = scoring(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=gap_opening,
        gap_extension=-1,
        match=match_score,
        mismatch=mismatch_score,
    )

    colored1, colored2 = colorize_alignment(aligned1, aligned2)
    assert aligned_score == only_score, f"""
    Alignment ({aligned_score}) and pure scoring ({only_score}) functions must return identical results for:
        {colored1}
        {colored2}
    """


@pytest.mark.parametrize("min_length", [5, 10])
@pytest.mark.parametrize("max_length", [15, 25])
@pytest.mark.parametrize("match_score", [1, 2, 3])
@pytest.mark.parametrize("mismatch_score", [-4, -2])
@pytest.mark.parametrize("gap_opening", [-5, -1])
def test_gap_expansions(
    min_length: int,
    max_length: int,
    match_score: int,
    mismatch_score: int,
    gap_opening: int,
):
    """
    Test the effect of gap expansions on alignment scores.

    Verifies that increasing the width of gaps in alignments with zero gap extension penalties
    does not change the alignment score, ensuring proper handling of gap costs.
    """

    alphabet = "ACGT"
    str1 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))
    str2 = "".join(choice(alphabet) for _ in range(randint(min_length, max_length)))

    aligned1, aligned2, _score = needleman_wunsch_gotoh_alignment(
        str1,
        str2,
        substitution_alphabet=alphabet,
        gap_opening=gap_opening,
        gap_extension=0,
        match=match_score,
        mismatch=mismatch_score,
    )

    # If there is a gap in any of the strings, we can expand that gap
    # infinitely, and if the `gap_extension` cost is set to zero, no
    # penalty will be incurred.
    present_gaps = aligned1.count("-") + aligned2.count("-")
    if present_gaps == 0:
        return

    def expand_any_one(seq1, seq2, gap_width: int = 1):
        if bool(getrandbits(1)):
            return replace_single_dashes(seq1, "?" * gap_width).replace("-", ""), seq2.replace("-", "")
        else:
            return seq1.replace("-", ""), replace_single_dashes(seq2, "?" * gap_width).replace("-", "")

    # Let's now precompute the baseline for our air-gapped strings
    air_gapped1, air_gapped2 = expand_any_one(aligned1, aligned2)
    air_gapped_aligned1, air_gapped_aligned2, aig_gapped_score = needleman_wunsch_gotoh_alignment(
        air_gapped1,
        air_gapped2,
        substitution_alphabet=alphabet + "?",
        gap_opening=gap_opening,
        gap_extension=0,
        match=match_score,
        mismatch=mismatch_score,
    )

    # Keep growing those gaps and make sure the score remains the same
    wide_gapped1, wide_gapped2 = air_gapped1, air_gapped2
    for gap_width in range(2, 5):
        wide_gapped1, wide_gapped2 = expand_any_one(wide_gapped1, wide_gapped2, gap_width)
        wide_gapped_aligned1, wide_gapped_aligned2, wide_gapped_score = needleman_wunsch_gotoh_alignment(
            wide_gapped1,
            wide_gapped2,
            substitution_alphabet=alphabet + "?",
            gap_opening=gap_opening,
            gap_extension=0,
            match=match_score,
            mismatch=mismatch_score,
        )
        assert wide_gapped_score == aig_gapped_score, f"""
        Expected score:     {aig_gapped_score}
        Expected alignment: {colorize_alignment(air_gapped_aligned1, air_gapped_aligned2)[0]}
                            {colorize_alignment(air_gapped_aligned1, air_gapped_aligned2)[1]}

        Gap width:          {gap_width}
        Actual score:       {wide_gapped_score}
        Final alignment:    {colorize_alignment(wide_gapped_aligned1, wide_gapped_aligned2)[0]}
                            {colorize_alignment(wide_gapped_aligned1, wide_gapped_aligned2)[1]}
        """


@pytest.mark.parametrize(
    "pair",
    [
        ("GGTGTGA", "TCGCGT"),  # presumably fails NW-align
        ("AAAGGG", "TTAAAAGGGGTT"),  # presumably fails Bio++
        ("CGCCTTAC", "AAATTTGC"),  # presumably fails Bio++
        ("TAAATTTGC", "TCGCCTTAC"),  # presumably fails T-Coffee
        ("AAATTTGC", "CGCCTTAC"),  # presumably fails FOGSAA
        ("AGAT", "CTCT"),  # presumably fails HUSAR, MatLab, and BioPyhton
    ],
)
@pytest.mark.parametrize(
    "scores",
    [
        (0, -1, -5, -1),  # match, mismatch, gap_opening, gap_extension
        (10, -30, -40, -1),  # match, mismatch, gap_opening, gap_extension
        (10, -30, -25, -1),  # match, mismatch, gap_opening, gap_extension
    ],
)
@pytest.mark.parametrize("mode", ["global", "local"])
def test_against_biopython_examples(
    pair: tuple[str, str],
    scores: tuple[int, int, int, int],
    mode: Literal["global", "local"],
):
    """
    Compare affine gap alignment scores with BioPython for specific examples.

    Ensures that the Needleman-Wunsch-Gotoh alignment scores are at least as good as
    BioPython's PairwiseAligner scores for a set of sequence pairs and scoring parameters.
    """
    a, b = pair
    match, mismatch, open_gap_score, extend_gap_score = scores

    affinegaps_func = needleman_wunsch_gotoh_alignment if mode == "global" else smith_waterman_gotoh_alignment
    _, _, affinegaps_score = affinegaps_func(
        a,
        b,
        match=match,
        mismatch=mismatch,
        gap_opening=open_gap_score,
        gap_extension=extend_gap_score,
        substitution_alphabet="ACGT",
    )

    # Compute BioPython score using PairwiseAligner
    aligner = Align.PairwiseAligner(mode=mode)
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    biopython_score = int(aligner.score(a, b))

    assert affinegaps_score >= biopython_score, "Affine Gaps alignments should be at least as good as BioPython"

    if affinegaps_score != biopython_score:
        pytest.warns(
            UserWarning,
            match=f"Affine Gaps score is not equal to BioPython score for {a} and {b}",
        )


@pytest.mark.repeat(30)
@pytest.mark.parametrize("first_length", [20, 100])
@pytest.mark.parametrize("second_length", [20, 100])
@pytest.mark.parametrize(
    "gap_scores",
    [
        (-2, -2),
        (-2, -1),
        (-2, 0),
        (2, 3),
        (12, 13),
        (-10, -1),
    ],
)
@pytest.mark.parametrize("mode", ["global", "local"])
def test_against_biopython_fuzzy(
    first_length: int,
    second_length: int,
    gap_scores: tuple[int, int],
    mode: Literal["global", "local"],
):
    """
    Compare affine gap alignment scores with BioPython for random sequences.

    Verifies that the Needleman-Wunsch-Gotoh alignment scores are at least as good as
    BioPython's PairwiseAligner scores for randomly generated sequences with various gap penalties.
    """
    open_gap_score, extend_gap_score = gap_scores

    # Make sure we generate different strings each time
    a = "".join(choice(default_proteins_alphabet) for _ in range(first_length))
    b = "".join(choice(default_proteins_alphabet) for _ in range(second_length))

    # The aligner picks a different algorithm based on settings.
    # The Needleman-Wunsch algorithm is used if `open_gap_score` and `extend_gap_score` are equal.
    # If they are different, the Gotoh algorithm is used.
    # https://github.com/biopython/biopython/blob/abf5a3b077d2b4af08aed390cbe0af48bdb75f97/Bio/Align/_pairwisealigner.c#L3743C1-L3754C37
    aligner = Align.PairwiseAligner(mode=mode)
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    biopython_score = int(aligner.score(a, b))

    # Remove the stop codon from the alphabet before using it
    #
    #   alphabet = str(aligner.substitution_matrix.alphabet).replace("*", "")
    #   matrix = np.array(aligner.substitution_matrix).astype(np.int8)
    #   matrix = matrix[: len(alphabet), : len(alphabet)]

    affinegaps_func = needleman_wunsch_gotoh_alignment if mode == "global" else smith_waterman_gotoh_alignment
    _, _, affinegaps_score = affinegaps_func(
        a,
        b,
        gap_opening=open_gap_score,
        gap_extension=extend_gap_score,
        substitution_alphabet=default_proteins_alphabet,
        substitution_matrix=default_proteins_matrix,
    )

    assert affinegaps_score >= biopython_score, "Affine Gaps alignments should be at least as good as BioPython"

    if affinegaps_score != biopython_score:
        pytest.warns(
            UserWarning,
            match=f"Affine Gaps score is not equal to BioPython score for {a} and {b}",
        )


@pytest.mark.repeat(10)
@pytest.mark.parametrize("mode", ["global", "local"])
def test_against_biopython_long(mode: Literal["global", "local"]):

    # Make sure we generate different strings each time
    alphabet = "AC"
    first_length, second_length = 1200, 1300
    match, mismatch, open_gap_score, extend_gap_score = 1, -1, -1, 0
    a = "".join(choice(alphabet) for _ in range(first_length))
    b = "".join(choice(alphabet) for _ in range(second_length))

    affinegaps_func = needleman_wunsch_gotoh_alignment if mode == "global" else smith_waterman_gotoh_alignment
    _, _, affinegaps_score = affinegaps_func(
        a,
        b,
        match=match,
        mismatch=mismatch,
        gap_opening=open_gap_score,
        gap_extension=extend_gap_score,
        substitution_alphabet=alphabet,
    )

    # Compute BioPython score using PairwiseAligner
    aligner = Align.PairwiseAligner(mode=mode)
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    biopython_score = int(aligner.score(a, b))

    assert affinegaps_score >= biopython_score, "Affine Gaps alignments should be at least as good as BioPython"

    if affinegaps_score != biopython_score:
        pytest.warns(
            UserWarning,
            match=f"Affine Gaps score is not equal to BioPython score for {a} and {b}",
        )
