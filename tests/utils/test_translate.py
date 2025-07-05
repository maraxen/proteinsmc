import jax.numpy as jnp
import pytest

from src.utils.translate import translate


@pytest.mark.parametrize(
  "nuc_seq, expected_aa_seq, expected_validity",
  [
    (
      jnp.array([3, 3, 1, 0, 0, 0]),  # TTC, AAA -> F, K
      jnp.array([6, 9]),  # Corresponds to F, K in colabdesign order
      True,
    ),
    (
      jnp.array([3, 0, 0, 0, 1, 2]),  # TAA (Stop), ACG -> X, T
      jnp.array([21, 15]),  # X is 21, T is 15
      False,
    ),
    (jnp.array([0, 1, 2]), jnp.array([15]), True),  # ACG -> T
    (jnp.array([]), jnp.array([]), True),  # Empty sequence
  ],
)
def test_translate(nuc_seq, expected_aa_seq, expected_validity):
  aa_seq, is_valid = translate(nuc_seq)
  assert jnp.array_equal(aa_seq, expected_aa_seq)
  assert is_valid == expected_validity


def test_translate_invalid_length():
  # Sequence length is not a multiple of 3
  with pytest.raises(TypeError):
    translate(jnp.array([0, 1, 2, 3]))
