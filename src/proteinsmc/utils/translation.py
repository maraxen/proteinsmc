"""Utilities for translating between nucleotide and protein sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit, vmap

from .constants import (
  CODON_INT_TO_RES_INT_JAX,
  COLABDESIGN_X_INT,
)

if TYPE_CHECKING:
  from jaxtyping import Bool

  from proteinsmc.utils.types import NucleotideSequence, ProteinSequence


@jit
def translate(nuc_seq: NucleotideSequence) -> tuple[ProteinSequence, Bool]:
  """Translate a nucleotide sequence to an amino acid sequence.

  Uses ColabDesign's AA integers.

  Args:
      nuc_seq: JAX array of nucleotide sequence (integer encoded).

  Returns:
      aa_seq: JAX array of amino acid sequence (integer encoded in
              ColabDesign's scheme).
      has_x_residue: Boolean indicating if the sequence contains 'X' residues.

  """
  if nuc_seq.shape[0] == 0:
    return jnp.array([], dtype=jnp.int8), jnp.array(1, dtype=jnp.bool_)
  protein_len = nuc_seq.shape[0] // 3
  if nuc_seq.shape[0] % 3 != 0:
    msg = "Nucleotide sequence length must be a multiple of 3."
    raise TypeError(msg)
  codons_int = nuc_seq[: protein_len * 3].reshape((protein_len, 3))
  aa_seq = CODON_INT_TO_RES_INT_JAX[codons_int[:, 0], codons_int[:, 1], codons_int[:, 2]]
  is_valid_translation = jnp.all(aa_seq != COLABDESIGN_X_INT)
  return aa_seq, is_valid_translation


@jit
def reverse_translate(
  aa_seq: ProteinSequence,
) -> tuple[NucleotideSequence, Bool]:
  """Reverses the translation of an amino acid sequence to a nucleotide sequence."""
  if aa_seq.shape[0] == 0:
    return jnp.array([], dtype=jnp.int8), jnp.array(1, dtype=jnp.bool_)

  def find_first_codon(aa: jnp.ndarray) -> jnp.ndarray:
    """Find the first codon that translates to the given amino acid."""
    match_indices = jnp.argwhere(aa == CODON_INT_TO_RES_INT_JAX, size=6, fill_value=-1)
    return match_indices[0]

  codons = vmap(find_first_codon)(aa_seq)
  nuc_seq = codons.flatten().astype(jnp.int8)

  is_valid_input = jnp.all(aa_seq != COLABDESIGN_X_INT)
  is_successful_lookup = jnp.all(nuc_seq != -1)
  is_valid_translation = is_valid_input & is_successful_lookup

  return nuc_seq, is_valid_translation
