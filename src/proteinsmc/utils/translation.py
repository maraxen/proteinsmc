"""Utilities for translating between nucleotide and protein sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit, vmap

from .constants import (
  CODON_INT_TO_RES_INT_JAX,
  PROTEINMPNN_X_INT,
)

if TYPE_CHECKING:
  from jaxtyping import Array, Bool, PRNGKeyArray

  from proteinsmc.models.types import NucleotideSequence, ProteinSequence


@jit
def nucleotide_to_aa(
  sequence: NucleotideSequence,
  _key: PRNGKeyArray | None = None,
  _context: Array | None = None,
) -> tuple[ProteinSequence, Bool]:
  """Translate a nucleotide sequence to an amino acid sequence.

  Uses ColabDesign's AA integers.

  Args:
      sequence: JAX array of nucleotide sequence (integer encoded).

  Returns:
      aa_seq: JAX array of amino acid sequence (integer encoded in
              ColabDesign's scheme).
      is_valid_translation: Boolean indicating if the sequence contains 'X' residues.

  """
  if sequence.shape[0] == 0:
    return jnp.array([], dtype=jnp.int8), jnp.array(1, dtype=jnp.bool_)
  protein_len = sequence.shape[0] // 3
  if sequence.shape[0] % 3 != 0:
    msg = "Nucleotide sequence length must be a multiple of 3."
    raise TypeError(msg)
  codons_int = sequence[: protein_len * 3].reshape((protein_len, 3))
  aa_seq = CODON_INT_TO_RES_INT_JAX[codons_int[:, 0], codons_int[:, 1], codons_int[:, 2]]
  is_valid_translation = jnp.all(aa_seq != PROTEINMPNN_X_INT)
  return aa_seq, is_valid_translation


@jit
def aa_to_nucleotide(
  sequence: ProteinSequence,
  _key: PRNGKeyArray | None = None,
  _context: Array | None = None,
) -> tuple[NucleotideSequence, Bool]:
  """Reverses the translation of an amino acid sequence to a nucleotide sequence."""
  if sequence.shape[0] == 0:
    return jnp.array([], dtype=jnp.int8), jnp.array(1, dtype=jnp.bool_)

  def find_first_codon(aa: jnp.ndarray) -> jnp.ndarray:
    """Find the first codon that translates to the given amino acid."""
    match_indices = jnp.argwhere(aa == CODON_INT_TO_RES_INT_JAX, size=6, fill_value=-1)
    return match_indices[0]

  codons = vmap(find_first_codon)(sequence)
  nuc_seq = codons.flatten().astype(jnp.int8)

  is_valid_input = jnp.all(sequence != PROTEINMPNN_X_INT)
  is_successful_lookup = jnp.all(nuc_seq != -1)
  is_valid_translation = is_valid_input & is_successful_lookup

  return nuc_seq, is_valid_translation
