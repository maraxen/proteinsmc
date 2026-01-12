"""MPNN scoring functions."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from functools import partial
from typing import IO, TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

PRXTEINMPNN_AVAILABLE = importlib.util.find_spec("prxteinmpnn") is not None

if PRXTEINMPNN_AVAILABLE:
  from proxide.ops.dataset import create_protein_dataset
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

DecodingSettings = Literal["random", "same_random", "sequential", "full_ar"]


if TYPE_CHECKING:

  def sequential_decode_order(
    _key: PRNGKeyArray,
    num_residues: int,
    tie_group_map: jnp.ndarray | None = None,
    num_groups: int | None = None,
  ) -> tuple[DecodingOrder, PRNGKeyArray]:
    """Generate a sequential decoding order."""
    ...
else:

  def sequential_decode_order(
    _key: PRNGKeyArray,
    num_residues: int,
    tie_group_map: jnp.ndarray | None = None,
    num_groups: int | None = None,
  ) -> tuple[DecodingOrder, PRNGKeyArray]:
    """Generate a sequential decoding order."""
    del tie_group_map, num_groups
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
  if not PRXTEINMPNN_AVAILABLE:
    msg = "prxteinmpnn is not installed. Please install it to use MPNN scoring."
    raise ImportError(msg)

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
    model=mpnn_model_params,  # type: ignore[arg-type]
    decoding_order_fn=decoding_order_fn,  # type: ignore[arg-type]
  )

  score_fn = partial(
    score_sequence,
    structure_coordinates=processed_inputs.coordinates,
    mask=processed_inputs.mask,
    residue_index=processed_inputs.residue_index,
    chain_index=processed_inputs.chain_index,
    ar_mask=ar_mask,
  )

  if TYPE_CHECKING:

    def mpnn_score(
      key: PRNGKeyArray | None,
      protein_sequence: ProteinSequence,
      _context: Array | None = None,
    ) -> Float:
      """Scores a protein sequence using the MPNN model."""
      ...
  else:

    @jax.jit
    def mpnn_score(
      key: PRNGKeyArray | None,
      protein_sequence: ProteinSequence,
      _context: Array | None = None,
    ) -> Float:
      """Scores a protein sequence using the MPNN model."""
      score, _logits, _order = score_fn(
        key,
        protein_sequence,
      )
      return score

  return mpnn_score
