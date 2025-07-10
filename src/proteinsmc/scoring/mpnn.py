"""MPNN scoring functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from jax import jit

if TYPE_CHECKING:
  from jaxtyping import Float, PRNGKeyArray

  from proteinsmc.utils.types import MPNNModel, ProteinSequence


def make_mpnn_score(
  mpnn_model: MPNNModel,
) -> Callable[[PRNGKeyArray, ProteinSequence], Float]:
  """Create a scoring function for the MPNN model.

  Args:
      mpnn_model: MPNN model instance.

  Returns:
      A function that scores a protein sequence using the MPNN model.

  """

  @jit
  def mpnn_score(
    key: PRNGKeyArray,
    protein_sequence: ProteinSequence,
  ) -> Float:
    """Scores a protein sequence using the MPNN model."""
    return mpnn_model.score(seq_numeric=protein_sequence, key=key)

  return mpnn_score
