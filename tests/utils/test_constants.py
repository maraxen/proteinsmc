import chex

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
  """
  chex.assert_equal(len(NUCLEOTIDES_CHAR), 4)
  chex.assert_equal(NUCLEOTIDES_INT_MAP["A"], 0)
  chex.assert_equal(INT_TO_NUCLEOTIDES_CHAR_MAP[0], "A")


def test_codon_to_res_char():
  chex.assert_equal(CODON_TO_RES_CHAR["TTC"], "F")
  chex.assert_equal(CODON_TO_RES_CHAR["TAA"], "X")


def test_aa_char_to_int_map():
  chex.assert_equal(AA_CHAR_TO_INT_MAP["A"], 0)
  chex.assert_equal(INT_TO_AA_CHAR_MAP[0], "A")


def test_codon_int_to_res_int_jax():
  """
  Test the shape and specific mapping of the CODON_INT_TO_RES_INT_JAX array.
  """
  chex.assert_shape(CODON_INT_TO_RES_INT_JAX, (4, 4, 4))
  n1, n2, n3 = (
    NUCLEOTIDES_INT_MAP["T"],
    NUCLEOTIDES_INT_MAP["T"],
    NUCLEOTIDES_INT_MAP["C"],
  )
  chex.assert_equal(CODON_INT_TO_RES_INT_JAX[n1, n2, n3], AA_CHAR_TO_INT_MAP["F"])


def test_ecoli_codon_freq():
  chex.assert_equal(len(ECOLI_CODON_FREQ_CHAR), 64)
  chex.assert_shape(ECOLI_CODON_FREQ_JAX, (4, 4, 4))
  total = jnp.sum(ECOLI_CODON_FREQ_JAX)
  chex.assert_trees_all_close(total, 1000.0, atol=10.0)
