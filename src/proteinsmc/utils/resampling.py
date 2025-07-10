"""Performs systematic resampling on a population."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, random

if TYPE_CHECKING:
  from jaxtyping import Float, PRNGKeyArray

  from proteinsmc.utils.types import PopulationSequenceFloats, PopulationSequences


@jit
def resample(
  key: PRNGKeyArray,
  population: PopulationSequences,
  log_weights: PopulationSequenceFloats,
) -> tuple[PopulationSequences, Float, PopulationSequenceFloats]:
  """Perform systematic resampling on a population.

  Args:
      key: JAX PRNG key.
      population: JAX array of population (shape: (population_size, seq_len)).
      log_weights: JAX array of log weights for each individual.

  Returns:
        - Resampled population.
        - Effective Sample Size (ESS).
        - Normalized weights.

  """
  n_population = population.shape[0]
  log_weights_safe = jnp.where(jnp.isneginf(log_weights), -1e9, log_weights)
  normalized_weights = jax.nn.softmax(log_weights_safe)
  ess = 1.0 / jnp.sum(jnp.square(normalized_weights))
  u = random.uniform(key, (n_population,))
  cumulative_weights = jnp.cumsum(normalized_weights)
  indices = jnp.searchsorted(cumulative_weights, u)
  resampled_population = population[indices]
  resampled_population = resampled_population.astype(jnp.int8)
  return resampled_population, ess, normalized_weights
