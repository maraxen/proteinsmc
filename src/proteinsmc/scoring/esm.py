"""ESM-based protein sequence scoring function."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from proteinsmc.utils.esm import load_model, remap_sequences

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.fitness import FitnessFn
  from proteinsmc.models.types import ProteinSequence

EPSILON = 1e-8


def make_esm_score(
  model_name: Literal["esmc_300m", "esmc_600m"],
  seed: int = 0,
) -> FitnessFn:
  """Create an ESM-based protein sequence scoring function.

  This function loads a specified ESM-J model and returns a scoring function
  that calculates the pseudo-log-likelihood (PLL) of a protein sequence.

  Args:
      model_name: The name of the ESM model to load.
      seed: Random seed for model initialization.

  Returns:
      A FitnessFn that takes a protein sequence and returns its PLL score.

  """
  eqx_model = load_model(
    model_name=model_name,
    key=jax.random.PRNGKey(seed),
  )
  eqx_model = eqx.filter_jit(eqx_model)

  @jax.jit
  def score(
    sequence: ProteinSequence,
    _key: PRNGKeyArray | None = None,
    _context: Array | None = None,
  ) -> Float:
    """Score a protein sequence using the ESM model.

    Args:
        sequence: Protein sequence as an array.
        _key: JAX PRNG key (not used in this function).
        _context: Additional context (not used in this function).

    Returns:
        The PLL score of the sequence.

    """
    sequence = remap_sequences(sequence)
    sequence = sequence[None, :]  # Add batch dimension: (1, seq_len)
    output = eqx_model(sequence)
    log_probs = jax.nn.log_softmax(output.logits, axis=-1)  # (1, seq_len, vocab_size)
    # Index log_probs with ESM token IDs for each position in the sequence
    seq_indices = sequence[..., None]  # (1, seq_len, 1)
    seq_log_probs = jnp.take_along_axis(
      log_probs,
      seq_indices,
      axis=-1,
    ).squeeze(-1)  # (1, seq_len)

    pll_score = jnp.sum(seq_log_probs) / (sequence.shape[1] + EPSILON)
    return pll_score.astype(jnp.float32)

  return score
