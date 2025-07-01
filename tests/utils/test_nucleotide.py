import jax.numpy as jnp
import pytest

from src.utils.constants import COLABDESIGN_X_INT, NUCLEOTIDES_INT_MAP
from src.utils.nucleotide import cai_score, translate


@pytest.fixture
def nuc_seq_valid():
    # Codons: AAA (0,0,0) -> K, CCC (1,1,1) -> P, GGG (2,2,2) -> G
    return jnp.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=jnp.int32)


@pytest.fixture
def nuc_seq_with_stop():
    # Codons: AAA (0,0,0) -> K, TAA (3,0,0) -> X (stop), GGG (2,2,2) -> G
    return jnp.array([0, 0, 0, 3, 0, 0, 2, 2, 2], dtype=jnp.int32)


@pytest.fixture
def nuc_seq_invalid_codon():
    # Codons: AAA (0,0,0) -> K, TTT (3,3,3) -> F, GGG (2,2,2) -> G
    # This sequence is valid, but we can use it to test CAI with different codons
    return jnp.array([0, 0, 0, 3, 3, 3, 2, 2, 2], dtype=jnp.int32)


@pytest.fixture
def aa_seq_valid():
    # Corresponding to nuc_seq_valid: K, P, G (ColabDesign ints)
    # K=11, P=13, G=7
    return jnp.array([11, 13, 7], dtype=jnp.int32)


@pytest.fixture
def aa_seq_with_x():
    # Corresponding to nuc_seq_with_stop: K, X, G
    # K=11, X=21, G=7
    return jnp.array([11, COLABDESIGN_X_INT, 7], dtype=jnp.int32)


def test_translate_valid_sequence(nuc_seq_valid, aa_seq_valid):
    translated_aa, has_x = translate(nuc_seq_valid)
    assert jnp.array_equal(translated_aa, aa_seq_valid)
    assert not has_x


def test_translate_sequence_with_stop_codon(nuc_seq_with_stop, aa_seq_with_x):
    translated_aa, has_x = translate(nuc_seq_with_stop)
    assert jnp.array_equal(translated_aa, aa_seq_with_x)
    assert has_x


def test_cai_score_valid_sequence(nuc_seq_valid, aa_seq_valid):
    # Expected CAI for AAA, CCC, GGG based on E.coli frequencies
    # AAA (K): 33.2, CCC (P): 6.4, GGG (G): 6.6
    # Max freqs: K=33.2, P=26.7, G=38.5
    # wi: 33.2/33.2=1, 6.4/26.7=0.2397, 6.6/38.5=0.1714
    # log(wi): 0, -1.428, -1.763
    # sum(log(wi)): -3.191
    # cai = exp(-3.191 / 3) = exp(-1.063) = 0.345
    cai = cai_score(nuc_seq_valid, aa_seq_valid)
    expected_cai = jnp.exp(
        (jnp.log(1.0) + jnp.log(6.4 / 26.7) + jnp.log(6.6 / 38.5)) / 3
    )
    assert jnp.isclose(cai, expected_cai, atol=1e-6)


def test_cai_score_sequence_with_x_codon(nuc_seq_with_stop, aa_seq_with_x):
    # Codon TAA (stop) maps to X. It should be masked out for CAI calculation.
    # Only AAA (K) and GGG (G) should contribute.
    # AAA (K): 33.2, GGG (G): 6.6
    # Max freqs: K=33.2, G=38.5
    # wi: 33.2/33.2=1, 6.6/38.5=0.1714
    # log(wi): 0, -1.763
    # sum(log(wi)): -1.763
    # cai = exp(-1.763 / 2) = exp(-0.8815) = 0.414
    cai = cai_score(nuc_seq_with_stop, aa_seq_with_x)
    expected_cai = jnp.exp((jnp.log(1.0) + jnp.log(6.6 / 38.5)) / 2)
    assert jnp.isclose(cai, expected_cai, atol=1e-6)


def test_cai_score_all_x_codons():
    # Sequence where all codons translate to 'X'
    nuc_seq_all_x = jnp.array([3, 0, 0, 3, 0, 0, 3, 0, 0], dtype=jnp.int32)  # TAA TAA TAA
    aa_seq_all_x = jnp.array(
        [COLABDESIGN_X_INT, COLABDESIGN_X_INT, COLABDESIGN_X_INT],
        dtype=jnp.int32,
    )
    cai = cai_score(nuc_seq_all_x, aa_seq_all_x)
    assert jnp.isclose(cai, 0.0)  # No valid codons, CAI should be 0


def test_cai_score_empty_sequence():
    nuc_seq_empty = jnp.array([], dtype=jnp.int32)
    aa_seq_empty = jnp.array([], dtype=jnp.int32)
    cai = cai_score(nuc_seq_empty, aa_seq_empty)
    assert jnp.isclose(cai, 0.0)
