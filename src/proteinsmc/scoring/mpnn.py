"""MPNN scoring functions."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import IO, TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.decoding_order import (
  DecodingOrder,
  random_decoding_order,
  single_decoding_order,
)

if TYPE_CHECKING:
  from pathlib import Path

  from prxteinmpnn.utils.types import (
    Array,
    Float,
    ModelParameters,
    PRNGKeyArray,
  )

  from proteinsmc.models.fitness import FitnessFn
  from proteinsmc.models.types import ProteinSequence

from prxteinmpnn.io.loaders import create_protein_dataset

DecodingSettings = Literal["random", "same_random", "sequential", "full_ar"]


def sequential_decode_order(
  _key: PRNGKeyArray,
  num_residues: int,
) -> tuple[DecodingOrder, PRNGKeyArray]:
  """Generate a sequential decoding order."""
  return jnp.arange(num_residues, dtype=jnp.int32), _key


def make_mpnn_score(
  mpnn_model_params: ModelParameters,
  inputs: str | Path | Sequence[str | Path | IO[str]],
  decoding_settings: DecodingSettings,
) -> FitnessFn:
  """Create a scoring function for the MPNN model.

  Args:
      mpnn_model_params: Parameters of the MPNN model.
      inputs: Input structure(s) for scoring.
      decoding_settings: Decoding strategy to use.

  Returns:
      A function that scores a protein sequence using the MPNN model.

  """
  dataset = create_protein_dataset(
    inputs,
    batch_size=len(inputs) if isinstance(inputs, Sequence) else 1,
  )
  processed_inputs = next(iter(dataset))

  decoding_order_fn = (
    random_decoding_order
    if decoding_settings == "random"
    else sequential_decode_order
    if decoding_settings == "sequential"
    else single_decoding_order
    if decoding_settings == "same_random"
    else random_decoding_order
  )

  ar_mask = (
    1
    - np.eye(
      processed_inputs.residue_index.shape[0],
      dtype=np.float32,
    )
    if decoding_settings == "full_ar"
    else None
  )

  score_sequence = make_score_sequence(
    model_parameters=mpnn_model_params,
    decoding_order_fn=decoding_order_fn,
  )

  score_fn = partial(score_sequence)(
    structure_coordinates=processed_inputs.coordinates,  # pyright: ignore[reportCallIssue]
    mask=processed_inputs.mask,
    residue_index=processed_inputs.residue_index,
    chain_index=processed_inputs.chain_index,
    ar_mask=ar_mask,
  )

  @jax.jit
  def mpnn_score(
    key: PRNGKeyArray,
    protein_sequence: ProteinSequence,
    _context: Array | None = None,
  ) -> Float:
    """Scores a protein sequence using the MPNN model."""
    return jnp.mean(
      score_fn(
        key,
        protein_sequence,
      )[:, 0],
      axis=0,
    )

  return mpnn_score
