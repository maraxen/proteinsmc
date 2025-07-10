"""Core SMC step logic with memory-efficient chunked processing."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, random

from proteinsmc.sampling.smc.data_structures import SMCCarryState, SMCConfig
from proteinsmc.utils.fitness import chunked_calculate_population_fitness


@partial(jit, static_argnames=("config",))
def smc_step(state: SMCCarryState, config: SMCConfig) -> tuple[SMCCarryState, dict]:
  """Perform one step of the SMC algorithm with memory-efficient processing.

  Args:
    state: Current SMC state
    config: SMC configuration

  Returns:
    Tuple of (next_state, metrics)

  """
  from proteinsmc.utils import (
    calculate_logZ_increment,
    resample,
    shannon_entropy,
    suggest_chunk_size_heuristic,
  )

  key_mutate, key_fitness, key_resample, key_next = random.split(state.key, 4)

  # Determine chunk size based on configuration
  if config.memory_config.enable_chunked_vmap:
    if config.memory_config.auto_tuning_config.enable_auto_tuning:
      # Use heuristic for now (benchmarking would be too slow per step)
      chunk_size = suggest_chunk_size_heuristic(
        state.population.shape[0],
        state.population.shape[1] if len(state.population.shape) > 1 else 1,
        config.memory_config.auto_tuning_config,
      )
    else:
      chunk_size = config.memory_config.population_chunk_size
  else:
    chunk_size = state.population.shape[0]  # No chunking

  if not state.population.shape:
    msg = "Population must have at least one dimension (batch size)"
    raise ValueError(msg)

  if config.memory_config.enable_chunked_vmap and state.population.shape[0] > chunk_size:
    mutated_population = chunked_mutation_step(
      key_mutate,
      state.population,
      config.mutation_rate,
      config.sequence_type,
      chunk_size,
    )
  else:
    from proteinsmc.utils import dispatch_mutation

    mutated_population = dispatch_mutation(
      key_mutate,
      state.population,
      config.mutation_rate,
      config.sequence_type,
    )

  if config.memory_config.enable_chunked_vmap and state.population.shape[0] > chunk_size:
    fitness_values, fitness_components = chunked_calculate_population_fitness(
      key_fitness,
      mutated_population,
      config.fitness_evaluator,
      config.sequence_type,
      chunk_size,
    )
  else:
    from proteinsmc.utils import calculate_population_fitness

    fitness_values, fitness_components = calculate_population_fitness(
      key_fitness,
      mutated_population,
      config.sequence_type,
      config.fitness_evaluator,
    )

  # Continue with standard SMC logic
  log_weights = jnp.where(jnp.isneginf(fitness_values), -jnp.inf, state.beta * fitness_values)

  logZ_increment = calculate_logZ_increment(log_weights, state.population.shape[0])  # noqa: N806
  resampled_population, ess, normalized_weights = resample(
    key_resample,
    mutated_population,
    log_weights,
  )

  if not isinstance(normalized_weights, Array):
    msg = f"Expected normalized_weights to be a JAX array, got {type(normalized_weights)}"
    raise TypeError(msg)

  valid_fitness_mask = jnp.isfinite(fitness_values)
  sum_valid_weights = jnp.sum(jnp.where(valid_fitness_mask, normalized_weights, 0.0))

  mean_combined_fitness = safe_weighted_mean(
    fitness_values,
    normalized_weights,
    valid_fitness_mask,
    sum_valid_weights,
  )

  if not isinstance(fitness_values, Array):
    msg = f"Expected fitness_values to be a JAX array, got {type(fitness_values)}"
    raise TypeError(msg)

  max_combined_fitness = jnp.max(jnp.where(valid_fitness_mask, fitness_values, -jnp.inf))
  entropy = shannon_entropy(mutated_population)

  metrics = {
    "mean_combined_fitness": mean_combined_fitness,
    "max_combined_fitness": max_combined_fitness,
    "fitness_components": fitness_components,
    "ess": ess,
    "entropy": entropy,
    "beta": state.beta,
  }

  next_state = SMCCarryState(
    key=key_next,
    population=resampled_population,
    logZ_estimate=state.logZ_estimate + logZ_increment,
    beta=state.beta,
    step=state.step + 1,
  )

  return next_state, metrics
