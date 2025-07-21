"""MPNN scoring functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax import jit

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.fitness import FitnessFn
  from proteinsmc.models.types import MPNNModel, ProteinSequence


def make_mpnn_score(
  mpnn_model: MPNNModel,
) -> FitnessFn:
  """Create a scoring function for the MPNN model.

  Args:
      mpnn_model: MPNN model instance.

  Returns:
      A function that scores a protein sequence using the MPNN model.

  """

  @jit
  def mpnn_score(
    _key: PRNGKeyArray,
    protein_sequence: ProteinSequence,
    _context: Array | None = None,
  ) -> Float:
    """Scores a protein sequence using the MPNN model."""
    return mpnn_model.score(seq_numeric=protein_sequence, key=_key)

  return mpnn_score
