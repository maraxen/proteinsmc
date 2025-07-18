
import jax.numpy as jnp
import chex
import pytest

from proteinsmc.utils.constants import COLABDESIGN_X_INT
from proteinsmc.utils.translation import aa_to_nucelotide, nucleotide_to_aa


@pytest.mark.parametrize(
  "nuc_seq, expected_aa_seq, expected_validity",
  [
    (
      jnp.array([3, 3, 1, 0, 0, 0]),  # TTC, AAA -> F, K
      jnp.array([13, 11]),
      True,
    ),
    (
      jnp.array([3, 0, 0, 0, 1, 2]),  # TAA (Stop), ACG -> X, T
      jnp.array([21, 16]),  # X is 21, T is 16
      False,
    ),
    (jnp.array([0, 1, 2]), jnp.array([16]), True),  # ACG -> T
    (jnp.array([]), jnp.array([]), True),  # Empty sequence
  ],
)
def test_translate(nuc_seq, expected_aa_seq, expected_validity):
  aa_seq, is_valid = nucleotide_to_aa(nuc_seq)
  chex.assert_trees_all_equal(aa_seq, expected_aa_seq)
  chex.assert_equal(is_valid, expected_validity)


def test_translate_invalid_length():
  with pytest.raises(TypeError):
    nucleotide_to_aa(jnp.array([0, 1, 2, 3]))


def test_reverse_translate():
  """Tests the reverse translation logic for both valid and invalid amino acids."""

  aa_seq_valid = jnp.array([13, 11])
  nuc_seq_valid, is_valid_flag = aa_to_nucelotide(aa_seq_valid)
  chex.assert_equal(is_valid_flag, True)
  retranslated_aa, _ = nucleotide_to_aa(nuc_seq_valid)
  chex.assert_trees_all_equal(retranslated_aa, aa_seq_valid)
  aa_seq_invalid = jnp.array([COLABDESIGN_X_INT])
  nuc_seq_invalid, is_valid_flag_invalid = aa_to_nucelotide(aa_seq_invalid)
  chex.assert_equal(is_valid_flag_invalid, False)
  chex.assert_shape(nuc_seq_invalid, (3,))
  retranslated_stop, _ = nucleotide_to_aa(nuc_seq_invalid)
  chex.assert_equal(retranslated_stop[0], COLABDESIGN_X_INT)
