"""Utility functions for calculating metrics related to populations of sequences.

Includes log evidence increments and Shannon entropy.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit, vmap

if TYPE_CHECKING:
  from proteinsmc.utils.types import PopulationSequenceFloats, PopulationSequences, ScalarFloat

logger = getLogger(__name__)


@jit
def calculate_logZ_increment(  # noqa: N802
  log_weights: PopulationSequenceFloats,
  population_size: int,
) -> ScalarFloat:
  """Calculate log evidence increment from log weights.

  Args:
    log_weights: Log weights for the population
    population_size: Size of the population

  Returns:
    Log evidence increment for this step

  """
  if log_weights.shape[0] == 0:
    return jnp.array(-jnp.inf)

  valid_log_weights = jnp.where(jnp.isneginf(log_weights), -jnp.inf, log_weights)
  max_l_w = jnp.max(valid_log_weights)
  safe_max_l_w = jnp.where(jnp.isneginf(max_l_w), 0.0, max_l_w)
  log_sum_exp_weights = safe_max_l_w + jnp.log(jnp.sum(jnp.exp(valid_log_weights - safe_max_l_w)))

  current_logZ_increment = log_sum_exp_weights - jnp.log(jnp.maximum(population_size, 1.0))  # noqa: N806

  return jnp.where(
    jnp.isneginf(log_sum_exp_weights) | (population_size == 0),
    -jnp.inf,
    current_logZ_increment,
  )


@jit
def calculate_position_entropy(pos_seqs: PopulationSequences) -> ScalarFloat:
  """Calculate the Shannon entropy for a single position in sequences.

  Args:
      pos_seqs: JAX array of sequences for a single position (integer encoded).

  Returns:
    Scalar representing the Shannon entropy for the position.

  """
  _, counts = jnp.unique(pos_seqs, return_counts=True, size=pos_seqs.shape[0])
  probs = counts / counts.sum()
  return -jnp.sum(probs * jnp.log(probs + 1e-9))


@jit
def shannon_entropy(seqs: PopulationSequences) -> ScalarFloat:
  """Calculate average per-position Shannon entropy of sequences.

  Args:
      seqs: JAX array of nucleotide or amino sequences (integer encoded).

  Returns:
    Scalar representing the Shannon entropy of the sequences.

  """
  if not seqs.size:
    return jnp.array(0.0, dtype=jnp.float32)
  if isinstance(seqs, jnp.ndarray):
    n_total = seqs.shape[0]
    if n_total == 0:
      return jnp.array(0.0, dtype=jnp.float32)
    seq_length = seqs.shape[1]
    position_entropies = vmap(calculate_position_entropy)(seqs.T)
    entropy = jnp.sum(position_entropies)
    return entropy / seq_length
  logger.warning("Warning: shannon_entropy received unexpected type %s", type(seqs))
  return jnp.array(0.0, dtype=jnp.float32)
