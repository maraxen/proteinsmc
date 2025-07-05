import jax.numpy as jnp
import pytest

from proteinsmc.scoring.cai import cai_score
from proteinsmc.utils.constants import AA_CHAR_TO_INT_MAP, NUCLEOTIDES_INT_MAP


def nuc_to_int(seq_str):
  """Helper function to convert nucleotide string to integer array."""
  return jnp.array([NUCLEOTIDES_INT_MAP[c] for c in seq_str])


def aa_to_int(seq_str):
  """Helper function to convert amino acid string to integer array."""
  return jnp.array([AA_CHAR_TO_INT_MAP[c] for c in seq_str])


@pytest.mark.parametrize(
  "nuc_seq_str, aa_seq_str, expected_cai",
  [
    (
      "ACGTTT",  # T, F
      "TF",
      # w_ACG = 11.5 / 22.8, w_TTT = 19.7 / 19.7 = 1.0
      # cai = exp((log(11.5/22.8) + log(1.0)) / 2) = sqrt(0.50438)
      jnp.sqrt(11.5 / 22.8),
    ),
    (
      "ACGTAA",  # T, X (Stop)
      "TX",
      # Stop codon is ignored, so only w_ACG is considered.
      # cai = exp(log(11.5/22.8) / 1) = 11.5 / 22.8
      11.5 / 22.8,
    ),
    (
      "",  # Empty sequence
      "",
      0.0,
    ),
    (
      "TAATAG",  # Two stop codons
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
