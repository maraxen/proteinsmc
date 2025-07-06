import jax.numpy as jnp
import pytest

from proteinsmc.scoring.cai import cai_score
from proteinsmc.utils.constants import AA_CHAR_TO_INT_MAP, COLABDESIGN_X_INT, NUCLEOTIDES_INT_MAP


def nuc_to_int(seq_str):
  """Helper function to convert nucleotide string to integer array."""
  return jnp.array([NUCLEOTIDES_INT_MAP[c] for c in seq_str], dtype=jnp.int8)


def aa_to_int(seq_str):
  """Helper function to convert amino acid string to integer array."""
  return jnp.array([AA_CHAR_TO_INT_MAP.get(c, COLABDESIGN_X_INT) for c in seq_str], dtype=jnp.int8)


@pytest.mark.parametrize(
  "nuc_seq_str, aa_seq_str, expected_cai",
  [
    (
      "ACGTTT",
      "TF",
      jnp.sqrt(11.5 / 22.8),
    ),
    (
      "ACGTAA",
      "TX",
      11.5 / 22.8,
    ),
    (
      "",
      "",
      0.0,
    ),
    (
      "TAATAG",
      "XX",
      0.0,
    ),
  ],
)
def test_cai_score(nuc_seq_str, aa_seq_str, expected_cai):
  nuc_seq = nuc_to_int(nuc_seq_str)
  aa_seq = aa_to_int(aa_seq_str)
  score = cai_score(nuc_seq, aa_seq)
  assert jnp.allclose(score, expected_cai, atol=1e-5)
