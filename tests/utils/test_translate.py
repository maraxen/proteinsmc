import jax.numpy as jnp
import pytest

from src.utils.translate import translate


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
