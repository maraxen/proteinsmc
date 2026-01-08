"""Utility functions for integrating Blackjax with custom fitness functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Float, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitness, StackedFitnessFn
  from proteinsmc.models.types import EvoSequence


def make_blackjax_log_prob_fn(
  fitness_fn: StackedFitnessFn,
  key: PRNGKeyArray,
) -> Callable[[EvoSequence], Float]:
  """Create a log-probability function for Blackjax.

  Blackjax's `log_density_fn` can return a tuple `(log_density, auxiliary_data)`.
  This function wraps the `fitness_fn` to provide this signature, where
  `log_density` is the combined fitness and `auxiliary_data` is a dictionary
  containing the components fitness under the key "components_fitness".

  Args:
      fitness_fn: The original fitness function that returns
                  (combined_fitness, components_fitness).
      key: JAX PRNG key for random number generation.

  Returns:
      A function that takes a sequence and returns a tuple:
      (combined_fitness, {"components_fitness": components_fitness}) suitable
      for Blackjax's `log_density_fn`.

  """

  def blackjax_log_prob_fn(sequence: EvoSequence) -> StackedFitness:
    return fitness_fn(
      key,
      sequence,
      None,
    )

  return blackjax_log_prob_fn
