import pytest
from hypothesis import given, strategies as st
import jax.numpy as jnp

from proteinsmc.utils.constants import COLABDESIGN_X_INT
from proteinsmc.utils.translation import aa_to_nucleotide, nucleotide_to_aa

# Hypothesis strategy for generating a valid protein sequence
protein_sequence_strategy = st.lists(st.integers(min_value=0, max_value=19), min_size=0, max_size=50)

# Hypothesis strategy for generating a valid nucleotide sequence (multiples of 3)
nucleotide_triplets_strategy = st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=50).map(
    lambda x: x[:(len(x) // 3) * 3]
).filter(lambda x: len(x) > 0)

@given(protein_sequence=protein_sequence_strategy)
def test_aa_to_nucleotide_to_aa_roundtrip(protein_sequence):
    """Tests that converting a protein sequence to nucleotides and back is reversible."""
    protein_seq_arr = jnp.array(protein_sequence, dtype=jnp.int8)
    
    nuc_seq, is_valid_forward = aa_to_nucleotide(protein_seq_arr)
    assert is_valid_forward

    round_trip_protein_seq, is_valid_reverse = nucleotide_to_aa(nuc_seq)
    assert is_valid_reverse
    
    assert jnp.array_equal(protein_seq_arr, round_trip_protein_seq)

@given(nucleotide_sequence=nucleotide_triplets_strategy)
def test_nucleotide_to_aa_is_valid(nucleotide_sequence):
    """Tests the is_valid_translation flag from nucleotide_to_aa."""
    nuc_seq_arr = jnp.array(nucleotide_sequence, dtype=jnp.int8)
    
    aa_seq, is_valid = nucleotide_to_aa(nuc_seq_arr)
    
    # The generated sequence should be valid unless it contains a stop codon.
    # We check this by seeing if any amino acid is 'X'.
    assert is_valid == (not jnp.any(aa_seq == COLABDESIGN_X_INT))

def test_nucleotide_to_aa_empty_sequence():
    """Tests nucleotide_to_aa with an empty sequence."""
    nuc_seq = jnp.array([], dtype=jnp.int8)
    aa_seq, is_valid = nucleotide_to_aa(nuc_seq)
    assert aa_seq.shape[0] == 0
    assert is_valid

def test_nucleotide_to_aa_invalid_length():
    """Tests nucleotide_to_aa with a sequence whose length is not a multiple of 3."""
    nuc_seq = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    with pytest.raises(TypeError, match="Nucleotide sequence length must be a multiple of 3."):
        nucleotide_to_aa(nuc_seq)

def test_nucleotide_to_aa_stop_codon():
    """Tests nucleotide_to_aa with a sequence containing a stop codon."""
    # TAA is a stop codon, which should result in an 'X' amino acid
    nuc_seq = jnp.array([3, 0, 0], dtype=jnp.int8)  # T = 3, A = 0
    aa_seq, is_valid = nucleotide_to_aa(nuc_seq)
    assert not is_valid
    assert aa_seq[0] == COLABDESIGN_X_INT

def test_aa_to_nucleotide_empty_sequence():
    """Tests aa_to_nucleotide with an empty sequence."""
    aa_seq = jnp.array([], dtype=jnp.int8)
    nuc_seq, is_valid = aa_to_nucleotide(aa_seq)
    assert nuc_seq.shape[0] == 0
    assert is_valid

def test_aa_to_nucleotide_invalid_aa():
    """Tests aa_to_nucleotide with a sequence containing an invalid amino acid ('X')."""
    aa_seq = jnp.array([0, 1, COLABDESIGN_X_INT], dtype=jnp.int8)  # A, C, X
    nuc_seq, is_valid = aa_to_nucleotide(aa_seq)
    assert not is_valid