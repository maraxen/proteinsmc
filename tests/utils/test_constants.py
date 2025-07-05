import jax.numpy as jnp

from src.utils.constants import (
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
  assert len(NUCLEOTIDES_CHAR) == 4
  assert NUCLEOTIDES_INT_MAP["A"] == 0
  assert INT_TO_NUCLEOTIDES_CHAR_MAP[0] == "A"


def test_codon_to_res_char():
  assert CODON_TO_RES_CHAR["TTC"] == "F"
  assert CODON_TO_RES_CHAR["TAA"] == "X"  # Stop codon


def test_aa_char_to_int_map():
  assert AA_CHAR_TO_INT_MAP["A"] == 0
  assert INT_TO_AA_CHAR_MAP[0] == "A"


def test_codon_int_to_res_int_jax():
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
