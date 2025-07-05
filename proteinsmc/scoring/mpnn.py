from functools import partial

from jax import jit
from jaxtyping import PRNGKeyArray

from ..utils.types import MPNNModel, ProteinSequence, ScalarFloat


@partial(jit, static_argnames=("mpnn_model",))
def mpnn_score(
  key: PRNGKeyArray,
  protein_sequence: ProteinSequence,
  mpnn_model: MPNNModel,
) -> ScalarFloat:
  """Scores a protein sequence using the MPNN model.

  Args:
      key: JAX PRNG key.
      protein_sequence: JAX array of protein sequence (integer encoded).
      mpnn_model: MPNN model instance.

  Returns:
      MPNN score for the protein sequence.
  """
  return mpnn_model.score(seq_numeric=protein_sequence, key=key)
