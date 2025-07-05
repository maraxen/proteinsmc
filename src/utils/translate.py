import jax.numpy as jnp
from jax import jit

from .constants import (
  CODON_INT_TO_RES_INT_JAX,
  COLABDESIGN_X_INT,
)
from .types import NucleotideSequence, ProteinSequence, ScalarBool


@jit
def translate(nuc_seq: NucleotideSequence) -> tuple[ProteinSequence, ScalarBool]:
  """Translates a nucleotide sequence to an amino acid sequence.

  Uses ColabDesign's AA integers.
  Args:
      nuc_seq: JAX array of nucleotide sequence (integer encoded).
  Returns:
      aa_seq: JAX array of amino acid sequence (integer encoded in
              ColabDesign's scheme).
      has_x_residue: Boolean indicating if the sequence contains 'X' residues.
  """
  if nuc_seq.shape[0] == 0:
    return jnp.array([], dtype=jnp.int8), jnp.array(True, dtype=jnp.bool_)
  protein_len = nuc_seq.shape[0] // 3
  if nuc_seq.shape[0] % 3 != 0:
    raise TypeError("Nucleotide sequence length must be a multiple of 3.")
  codons_int = nuc_seq[: protein_len * 3].reshape((protein_len, 3))
  aa_seq = CODON_INT_TO_RES_INT_JAX[codons_int[:, 0], codons_int[:, 1], codons_int[:, 2]]
  is_valid_translation = jnp.all(aa_seq != COLABDESIGN_X_INT)
  return aa_seq, is_valid_translation
