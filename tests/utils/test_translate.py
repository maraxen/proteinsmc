import jax.numpy as jnp
import pytest

from proteinsmc.utils.constants import COLABDESIGN_X_INT
from proteinsmc.utils.translation import reverse_translate, translate


@pytest.mark.parametrize(
  "nuc_seq, expected_aa_seq, expected_validity",
  [
    (
      jnp.array([3, 3, 1, 0, 0, 0]),  # TTC, AAA -> F, K
      jnp.array([13, 11]),  # Corrected integer representation
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
  aa_seq, is_valid = translate(nuc_seq)
  assert jnp.array_equal(aa_seq, expected_aa_seq)
  assert is_valid == expected_validity


def test_translate_invalid_length():
  with pytest.raises(TypeError):
    translate(jnp.array([0, 1, 2, 3]))


def test_reverse_translate():
  """Tests the reverse translation logic for both valid and invalid amino acids."""

  aa_seq_valid = jnp.array([13, 11])  # Example: F, K
  nuc_seq_valid, is_valid_flag = reverse_translate(aa_seq_valid)
  assert is_valid_flag
  retranslated_aa, _ = translate(nuc_seq_valid)
  assert jnp.array_equal(retranslated_aa, aa_seq_valid)
  aa_seq_invalid = jnp.array([COLABDESIGN_X_INT])
  nuc_seq_invalid, is_valid_flag_invalid = reverse_translate(aa_seq_invalid)
  assert not is_valid_flag_invalid
  assert nuc_seq_invalid.shape == (3,)
  retranslated_stop, _ = translate(nuc_seq_invalid)
  assert retranslated_stop[0] == COLABDESIGN_X_INT
