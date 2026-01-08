"""Utilities for translating between nucleotide and protein sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jax import jit, vmap

from .constants import (
  AA_CHAR_TO_INT_MAP,
  CODON_INT_TO_RES_INT_JAX,
  NUCLEOTIDES_INT_MAP,
  PROTEINMPNN_X_INT,
)

if TYPE_CHECKING:
  from jaxtyping import Array, Bool

  from proteinsmc.types import ArrayLike, NucleotideSequence, PRNGKey, ProteinSequence


def string_to_int_sequence(
  sequence: str,
  conversion_map: dict[str, int] | None = None,
  sequence_type: Literal["protein", "nucleotide"] | None = None,
) -> ArrayLike:
  """Convert a string sequence to a JAX integer array using ColabDesign's AA mapping.

  Args:
      sequence: Amino acid sequence as a string.
      conversion_map: Optional dictionary mapping characters to integers. If None, defaults to
                      AA_CHAR_TO_INT_MAP or NUCLEOTIDES_INT_MAP based on sequence_type.
      sequence_type: Type of the sequence, either "protein" or "nucleotide".

  Returns:
      JAX array of integer-encoded amino acid sequence.

  """
  conversion_map = conversion_map if conversion_map is not None else AA_CHAR_TO_INT_MAP
  conversion_map = NUCLEOTIDES_INT_MAP if sequence_type == "nucleotide" else AA_CHAR_TO_INT_MAP
  int_list = [conversion_map[res] for res in sequence]
  return jnp.array(int_list, dtype=jnp.int8)


@jit
def nucleotide_to_aa(
  sequence: NucleotideSequence,
  _key: PRNGKey | None = None,
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
  _key: PRNGKey | None = None,
  _context: Array | None = None,
) -> tuple[NucleotideSequence, Bool]:
  """Reverses the translation of an amino acid sequence to a nucleotide sequence."""
  if sequence.shape[0] == 0:
    return jnp.array([], dtype=jnp.int8), jnp.array(1, dtype=jnp.bool_)

  def find_first_codon(aa: ArrayLike) -> ArrayLike:
    """Find the first codon that translates to the given amino acid."""
    match_indices = jnp.argwhere(aa == CODON_INT_TO_RES_INT_JAX, size=6, fill_value=-1)
    return match_indices[0]

  codons = vmap(find_first_codon)(sequence)
  nuc_seq = codons.flatten().astype(jnp.int8)

  is_valid_input = jnp.all(sequence != PROTEINMPNN_X_INT)
  is_successful_lookup = jnp.all(nuc_seq != -1)
  is_valid_translation = is_valid_input & is_successful_lookup

  return nuc_seq, is_valid_translation
