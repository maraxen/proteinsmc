"""Utilities for remapping amino acid sequences from ColabDesign to ESM format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Bool

  from proteinsmc.models.types import ProteinSequence


from .constants import (
  COLABDESIGN_TO_ESM_AA_MAP_JAX,
  ESM_BOS_ID,
  ESM_EOS_ID,
)


@jax.jit
def remap_sequences(
  sequence: ProteinSequence,
) -> tuple[ProteinSequence, Bool]:
  """Remap amino acid integer IDs from ColabDesign's scheme to ESM's token scheme.

  Also add BOS/EOS tokens and pad the sequence.

  Args:
      sequence: A ProteinSequence containing amino acid integer IDs.

  Returns:
      A tuple of (esm_amino_acid_ids, attention_mask) as JAX arrays.

  """
  esm_aa_ints_raw = COLABDESIGN_TO_ESM_AA_MAP_JAX[sequence]

  esm_aa_ids_with_special = jnp.concatenate(
    [
      jnp.array([ESM_BOS_ID], dtype=jnp.int32),
      esm_aa_ints_raw,
      jnp.array([ESM_EOS_ID], dtype=jnp.int32),
    ],
  )
  attention_mask = jnp.zeros((sequence.shape[0] + 2,), dtype=jnp.int32)

  attention_mask = jax.lax.dynamic_update_slice(
    attention_mask,
    jnp.ones_like(esm_aa_ids_with_special),
    (0,),
  )

  return esm_aa_ids_with_special, attention_mask
