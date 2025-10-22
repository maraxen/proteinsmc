"""Registry for fitness functions and their configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray, PyTree

  from proteinsmc.models.fitness import CombineFn


def make_sum_combine(**_kwargs: Any) -> CombineFn:  # noqa: ANN401
  """Make combine function that sums input along axis 0.

  Returns:
    A function that sums input along axis 0.

  """

  def sum_combine(
    fitness_scores: Float,
    _key: PRNGKeyArray | None = None,
    _context: PyTree | Array | None = None,
  ) -> Float:
    """Combine function that sums input along axis 0.

    If context is provided, it scales the combined result.
    """
    combined = jnp.sum(fitness_scores, axis=0)
    if _context is not None:
      combined = combined * _context
    return combined

  return sum_combine


def make_weighted_combine(
  fitness_weights: Float | list[float] | None = None,
  **_kwargs: Any,  # noqa: ANN401
) -> CombineFn:
  """Make combine function that combines fitness scores with optional weights.

  Args:
      fitness_weights: Optional weights for combining fitness scores.

  Returns:
      A function that combines fitness scores using the provided weights.

  """
  weights_array = jnp.array(fitness_weights) if fitness_weights is not None else None

  def weighted_combine(
    fitness_components: Float,
    _key: PRNGKeyArray | None = None,
    _context: PyTree | Array | None = None,
  ) -> Float:
    """Combine individual fitness scores into a single score using weights.

    If context is provided, it scales the combined result.

    Args:
        fitness_components: Dictionary of individual fitness scores.

    Returns:
        Combined fitness score.

    """
    if weights_array is not None:
      combined_fitness = jnp.tensordot(
        weights_array,
        fitness_components,
        axes=1,
      )
    else:
      combined_fitness = jnp.sum(fitness_components, axis=0)

    if _context is not None:
      combined_fitness = combined_fitness * _context

    return combined_fitness

  return weighted_combine
