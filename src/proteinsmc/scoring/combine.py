"""Registry for fitness functions and their configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.fitness import CombineFn


def make_sum_combine(**_kwargs: Any) -> CombineFn:
  """Make combine function that sums input along axis 0.

  Returns:
    A function that sums input along axis 0.

  """

  def sum_combine(
    fitness_scores: Float,
    _key: PRNGKeyArray | None = None,
    _context: Array | None = None,
  ) -> Float:
    """Combine function that sums input along axis 0."""
    return jnp.sum(fitness_scores, axis=0)

  return sum_combine


def make_weighted_combine(
  fitness_weights: Float | None = None,
) -> CombineFn:
  """Make combine function that combines fitness scores with optional weights.

  Args:
      fitness_weights: Optional weights for combining fitness scores.

  Returns:
      A function that combines fitness scores using the provided weights.

  """

  def weighted_combine(
    fitness_components: Float,
    _key: PRNGKeyArray | None = None,
    _context: Array | None = None,
  ) -> Float:
    """Combine individual fitness scores into a single score using weights.

    Args:
        fitness_components: Dictionary of individual fitness scores.
        fitness_weights: Optional weights for combining fitness scores.

    Returns:
        Combined fitness score.

    """
    if fitness_weights is not None:
      combined_fitness = jnp.tensordot(
        fitness_weights,
        fitness_components,
        axes=1,
      )
    else:
      combined_fitness = jnp.sum(fitness_components, axis=0)

    return combined_fitness

  return weighted_combine
