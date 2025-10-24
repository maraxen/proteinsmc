"""Utility functions for calculating metrics related to populations of sequences.

Includes log evidence increments and Shannon entropy.
"""

from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit, vmap

if TYPE_CHECKING:
  from jaxtyping import Float

  from proteinsmc.models.types import EvoSequence


from jaxtyping import Array

logger = getLogger(__name__)


def safe_weighted_mean(
  metric: Array,
  weights: Array,
  valid_mask: Array,
  sum_valid_w: Float,
) -> Array:
  """Compute weighted mean safely, handling edge cases.

  Args:
    metric: Values to compute weighted mean of
    weights: Weights for each value
    valid_mask: Boolean mask for valid values
    sum_valid_w: Sum of valid weights

  Returns:
    Weighted mean, or NaN if no valid values

  """
  if not isinstance(metric, Array):
    msg = f"Expected metric to be a JAX array, got {type(metric)}"
    raise TypeError(msg)
  if not isinstance(weights, Array):
    msg = f"Expected weights to be a JAX array, got {type(weights)}"
    raise TypeError(msg)

  eps = 1e-9
  output = jnp.where(
    sum_valid_w > eps,
    jnp.sum(jnp.where(valid_mask, metric * weights, 0.0)) / sum_valid_w,
    jnp.nan,
  )

  if not isinstance(output, Array):
    msg = f"Expected output to be a JAX array, got {type(output)}"
    raise TypeError(msg)

  return output


@jit
def calculate_logZ_increment(  # noqa: N802
  log_weights: Float,
  population_size: int,
) -> Float:
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
  if not isinstance(valid_log_weights, Array):
    msg = f"Expected valid_log_weights to be a JAX array, got {type(valid_log_weights)}"
    raise TypeError(msg)
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
def calculate_position_entropy(seqs: EvoSequence) -> Float:
  """Calculate the Shannon entropy for a single position in sequences.

  Args:
      seqs: JAX array of sequences for a single position (integer encoded).

  Returns:
    Scalar representing the Shannon entropy for the position.

  """
  _, counts = jnp.unique(seqs, return_counts=True, size=seqs.shape[0])
  probs = counts / counts.sum()
  return -jnp.sum(probs * jnp.log(probs + 1e-9))


@jit
def shannon_entropy(seqs: EvoSequence) -> Float:
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
    return jnp.divide(entropy, seq_length)
  return jnp.array(0.0, dtype=jnp.float32)


def kl_divergence(p: Array, q: Array) -> Array:
  """Calculate Kullback-Leibler Divergence between two distributions.

  This measures how one probability distribution diverges from a second, expected probability distribution.

  Args:
      p: First probability distribution.
      q: Second probability distribution.

  Returns:
      KL Divergence value. Returns 0.0 if both distributions are empty/zero.

  """
  # Handle edge case where distributions are all zeros (e.g., empty sequences)
  p_sum = jnp.sum(p)
  q_sum = jnp.sum(q)
  
  # If both are zero, they're identical (no divergence)
  if jnp.allclose(p_sum, 0.0) and jnp.allclose(q_sum, 0.0):
    return jnp.array(0.0)
  
  # If only one is zero, divergence is infinite (return large value)
  if jnp.allclose(p_sum, 0.0) or jnp.allclose(q_sum, 0.0):
    return jnp.array(jnp.inf)
  
  p_norm = p / p_sum
  q_norm = q / q_sum

  epsilon = 1e-10
  p_norm = jnp.clip(p_norm, epsilon, 1.0)
  q_norm = jnp.clip(q_norm, epsilon, 1.0)

  return jnp.sum(p_norm * (jnp.log(p_norm) - jnp.log(q_norm)))


def jeffreys_divergence(p: Array, q: Array) -> Array:
  """Calculate Jeffrey's Divergence between two distributions.

  This effectively quantifies the magnitude of evolutionary change.

  Args:
      p: First probability distribution.
      q: Second probability distribution.

  Returns:
      Jeffrey's Divergence value.

  """
  return kl_divergence(p, q) + kl_divergence(q, p)


def jensen_shannon_divergence(p: Array, q: Array) -> Array:
  """Calculate Jensen-Shannon Divergence between two distributions.

  This provides a symmetric and smoothed measure of divergence.

  Args:
      p: First probability distribution.
      q: Second probability distribution.

  Returns:
      Jensen-Shannon Divergence value.

  """
  m = 0.5 * (p + q)
  return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def calculate_barrier_crossing_frequency(
  fitness_history: Array, fitness_threshold: Float = None
) -> Float:
  """Calculate frequency of fitness barrier crossings.

  Args:
      fitness_history: Array of fitness values over generations.
      fitness_threshold: Optional threshold defining a barrier.

  Returns:
      Frequency of barrier crossings.

  """
  if fitness_threshold is None:
    # Auto-determine threshold as median fitness
    fitness_threshold = jnp.median(fitness_history)

  # Count sign changes in (fitness - threshold)
  diff = fitness_history - fitness_threshold
  crossings = jnp.sum(jnp.diff(jnp.sign(diff)) != 0)

  # Return frequency (crossings per generation)
  return crossings / (len(fitness_history) - 1)


def outcome_shannon_entropy(outcomes: Array) -> Float:
  """Calculate Shannon entropy of evolutionary outcomes.

  Args:
      outcomes: Array of observed evolutionary outcomes (e.g., fitness values).

  Returns:
      Shannon entropy value.

  """
  # Calculate probabilities of unique outcomes
  _, counts = jnp.unique(outcomes, return_counts=True)
  probabilities = counts / jnp.sum(counts)

  # Calculate Shannon entropy
  return -jnp.sum(probabilities * jnp.log2(probabilities + 1e-10))
