"""Calculates the Codon Adaptation Index (CAI) for a given nucleotide and amino acid sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from proteinsmc.utils.constants import (
  COLABDESIGN_X_INT,
  ECOLI_CODON_FREQ_JAX,
  ECOLI_MAX_FREQS_JAX,
)

if TYPE_CHECKING:
  from jaxtyping import Float

  from proteinsmc.utils.types import NucleotideSequence, ProteinSequence


def cai_score(nuc_seq: NucleotideSequence, aa_seq: ProteinSequence) -> Float:
  """Calculate Codon Adaptation Index (CAI).

  `aa_seq` uses ColabDesign's AA integers.

  Args:
      nuc_seq: JAX array of nucleotide sequence (integer encoded).
      aa_seq: JAX array of amino acid sequence (integer encoded in
              ColabDesign's scheme).

  Returns:
      cai: JAX array of CAI values.

  """
  protein_len = nuc_seq.shape[0] // 3
  codons_int = nuc_seq[: protein_len * 3].reshape((protein_len, 3))
  codon_frequencies = ECOLI_CODON_FREQ_JAX[codons_int[:, 0], codons_int[:, 1], codons_int[:, 2]]
  max_aa_frequencies = ECOLI_MAX_FREQS_JAX[aa_seq]
  wi = codon_frequencies / jnp.maximum(max_aa_frequencies, 1e-9)
  valid_codon_mask = aa_seq != COLABDESIGN_X_INT
  log_wi = jnp.log(jnp.maximum(wi, 1e-12))
  sum_log_wi = jnp.sum(log_wi * valid_codon_mask)
  num_valid_codons = jnp.sum(valid_codon_mask)
  cai = jnp.exp(sum_log_wi / jnp.maximum(num_valid_codons, 1.0))
  return jnp.where(num_valid_codons > 0, cai, 0.0)
