import jax.numpy as jnp
import pytest

from proteinsmc.utils import constants
from proteinsmc.utils.constants import (
    ALPHAFOLD_TO_ESM_AA_MAP_JAX,
    ESM_SEQUENCE_VOCAB,
    ESM_UNK_ID,
    PROTEINMPNN_RESTYPES,
    PROTEINMPNN_TO_ESM_AA_MAP_JAX,
    restypes,
)


def test_proteinmpnn_to_esm_map_matches_literal_derivation():
    """Before/after parity anchor for the alphex-based `_build_esm_token_map` (Step 3).

    Run against BOTH the pre-alphex per-character-lookup implementation and the
    alphex-`perm()`-based replacement; must pass bit-for-bit against both.
    """
    expected = [ESM_SEQUENCE_VOCAB.index(c) for c in PROTEINMPNN_RESTYPES]
    actual = [int(PROTEINMPNN_TO_ESM_AA_MAP_JAX[i]) for i in range(20)]
    assert actual == expected
    assert int(PROTEINMPNN_TO_ESM_AA_MAP_JAX[20]) == ESM_UNK_ID
    assert int(PROTEINMPNN_TO_ESM_AA_MAP_JAX[21]) == ESM_UNK_ID


def test_alphafold_to_esm_map_matches_literal_derivation():
    """Before/after parity anchor for the alphex-based `_build_esm_token_map` (Step 3)."""
    expected = [ESM_SEQUENCE_VOCAB.index(c) for c in "".join(restypes)]
    actual = [int(ALPHAFOLD_TO_ESM_AA_MAP_JAX[i]) for i in range(20)]
    assert actual == expected
    assert int(ALPHAFOLD_TO_ESM_AA_MAP_JAX[20]) == ESM_UNK_ID
    assert int(ALPHAFOLD_TO_ESM_AA_MAP_JAX[21]) == ESM_UNK_ID


def test_codon_int_to_res_int_jax_shape_and_dtype():
    """Tests the shape and dtype of CODON_INT_TO_RES_INT_JAX."""
    assert constants.CODON_INT_TO_RES_INT_JAX.shape == (4, 4, 4)
    assert constants.CODON_INT_TO_RES_INT_JAX.dtype == jnp.int8


def test_codon_int_to_res_int_jax_values():
    """Tests a few key values in CODON_INT_TO_RES_INT_JAX."""
    # Test a few codons
    # TTT -> F (Phenylalanine)
    assert (
        constants.CODON_INT_TO_RES_INT_JAX[
            constants.NUCLEOTIDES_INT_MAP["T"],
            constants.NUCLEOTIDES_INT_MAP["T"],
            constants.NUCLEOTIDES_INT_MAP["T"],
        ]
        == constants.AA_CHAR_TO_INT_MAP["F"]
    )
    # TAC -> Y (Tyrosine)
    assert (
        constants.CODON_INT_TO_RES_INT_JAX[
            constants.NUCLEOTIDES_INT_MAP["T"],
            constants.NUCLEOTIDES_INT_MAP["A"],
            constants.NUCLEOTIDES_INT_MAP["C"],
        ]
        == constants.AA_CHAR_TO_INT_MAP["Y"]
    )
    # TAA -> X (Stop)
    assert (
        constants.CODON_INT_TO_RES_INT_JAX[
            constants.NUCLEOTIDES_INT_MAP["T"],
            constants.NUCLEOTIDES_INT_MAP["A"],
            constants.NUCLEOTIDES_INT_MAP["A"],
        ]
        == constants.STOP_INT
    )


@pytest.mark.skip(reason="Snapshot testing requires syrupy package - not a critical test")
def test_codon_int_to_res_int_jax_snapshot(snapshot):  # noqa: ANN001
    """Snapshot test for CODON_INT_TO_RES_INT_JAX."""
    snapshot.assert_match(
        str(constants.CODON_INT_TO_RES_INT_JAX), "codon_int_to_res_int_jax.snap"
    )
