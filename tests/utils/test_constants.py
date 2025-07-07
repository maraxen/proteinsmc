import jax.numpy as jnp

from proteinsmc.utils.constants import (
  AA_CHAR_TO_INT_MAP,
  CODON_INT_TO_RES_INT_JAX,
  CODON_TO_RES_CHAR,
  ECOLI_CODON_FREQ_CHAR,
  ECOLI_CODON_FREQ_JAX,
  INT_TO_AA_CHAR_MAP,
  INT_TO_NUCLEOTIDES_CHAR_MAP,
  NUCLEOTIDES_CHAR,
  NUCLEOTIDES_INT_MAP,
)


def test_nucleotide_constants():
  """
  Tests the integrity of nucleotide constants.

  This test verifies that:
  - The NUCLEOTIDES_CHAR collection contains exactly four elements.
  - The mapping from nucleotide character "A" to its integer representation is correct.
  - The reverse mapping from integer 0 to nucleotide character "A" is correct.
  """
  assert len(NUCLEOTIDES_CHAR) == 4
  assert NUCLEOTIDES_INT_MAP["A"] == 0
  assert INT_TO_NUCLEOTIDES_CHAR_MAP[0] == "A"


def test_codon_to_res_char():
  assert CODON_TO_RES_CHAR["TTC"] == "F"
  assert CODON_TO_RES_CHAR["TAA"] == "X"


def test_aa_char_to_int_map():
  assert AA_CHAR_TO_INT_MAP["A"] == 0
  assert INT_TO_AA_CHAR_MAP[0] == "A"


def test_codon_int_to_res_int_jax():
  """
  Test the shape and specific mapping of the CODON_INT_TO_RES_INT_JAX array.

  This test verifies that:
  - The CODON_INT_TO_RES_INT_JAX array has the expected shape of (4, 4, 4), corresponding to all possible codon combinations.
  - The codon 'TTC' (represented by its integer mapping) correctly maps to the amino acid 'F' (Phenylalanine) in the AA_CHAR_TO_INT_MAP.

  Assertions:
    - CODON_INT_TO_RES_INT_JAX.shape == (4, 4, 4)
    - CODON_INT_TO_RES_INT_JAX[NUCLEOTIDES_INT_MAP["T"], NUCLEOTIDES_INT_MAP["T"], NUCLEOTIDES_INT_MAP["C"]] == AA_CHAR_TO_INT_MAP["F"]
  """
  assert CODON_INT_TO_RES_INT_JAX.shape == (4, 4, 4)
  # Check for TTC -> F
  n1, n2, n3 = (
    NUCLEOTIDES_INT_MAP["T"],
    NUCLEOTIDES_INT_MAP["T"],
    NUCLEOTIDES_INT_MAP["C"],
  )
  assert CODON_INT_TO_RES_INT_JAX[n1, n2, n3] == AA_CHAR_TO_INT_MAP["F"]


def test_ecoli_codon_freq():
  assert len(ECOLI_CODON_FREQ_CHAR) == 64
  assert ECOLI_CODON_FREQ_JAX.shape == (4, 4, 4)
  print(jnp.sum(ECOLI_CODON_FREQ_JAX))
  assert jnp.isclose(jnp.sum(ECOLI_CODON_FREQ_JAX), 1000.0, atol=10.0)
