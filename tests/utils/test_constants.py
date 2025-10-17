import jax.numpy as jnp
import pytest
from proteinsmc.utils import constants


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
def test_codon_int_to_res_int_jax_snapshot(snapshot):  # noqa: ANN001, ARG001
    """Snapshot test for CODON_INT_TO_RES_INT_JAX."""
    snapshot.assert_match(
        str(constants.CODON_INT_TO_RES_INT_JAX), "codon_int_to_res_int_jax.snap"
    )