"""Core SMC step logic with memory-efficient chunked processing."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jax import Array, jit, random

from proteinsmc.sampling.smc.data_structures import SMCCarryState, SMCConfig

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.utils import FitnessEvaluator, PopulationSequences, ScalarFloat
  from proteinsmc.utils.types import EvoSequence


def safe_weighted_mean(
  metric: Array,
  weights: Array,
  valid_mask: Array,
  sum_valid_w: ScalarFloat,
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


def chunked_mutation_step(
  key: PRNGKeyArray,
  population: PopulationSequences,
  mutation_rate: float,
  sequence_type: Literal["protein", "nucleotide"],
  chunk_size: int,
) -> PopulationSequences:
  """Apply mutation to population using chunked processing.

  Args:
    key: PRNG key for mutation
    population: Population to mutate
    mutation_rate: Rate of mutation
    sequence_type: Type of sequences ("protein" or "nucleotide")
    chunk_size: Size of chunks for processing

  Returns:
    Mutated population

  """
  from proteinsmc.utils import chunked_vmap
  from proteinsmc.utils.mutation import dispatch_mutation_single

  def mutate_single(data_tuple: tuple[PRNGKeyArray, EvoSequence]) -> EvoSequence:
    k, seq = data_tuple
    result = dispatch_mutation_single(k, seq, mutation_rate, sequence_type)
    return result.astype(jnp.int8)

  mutation_keys = random.split(key, population.shape[0])
  return chunked_vmap(
    mutate_single,
    (mutation_keys, population),
    chunk_size,
  )


def chunked_fitness_evaluation(
  key: PRNGKeyArray,
  population: PopulationSequences,
  sequence_type: Literal["protein", "nucleotide"],
  fitness_evaluator: FitnessEvaluator,
  chunk_size: int,
) -> tuple[Array, Array]:
  """Evaluate fitness for population using chunked processing.

  Args:
    key: PRNG key for fitness evaluation
    population: Population to evaluate
    sequence_type: Type of sequences
    fitness_evaluator: Fitness evaluation function
    chunk_size: Size of chunks for processing

  Returns:
    Tuple of (fitness_values, fitness_components)

  """
  from proteinsmc.utils import calculate_population_fitness

  if population.shape[0] <= chunk_size:
    return calculate_population_fitness(
      key,
      population,
      sequence_type,
      fitness_evaluator,
    )

  # Process population in chunks using the efficient chunk-based approach
  population_size = population.shape[0]
  num_chunks = (population_size + chunk_size - 1) // chunk_size
  chunk_keys = random.split(key, num_chunks)

  fitness_values_list = []
  fitness_components_list = []

  for i, chunk_key in enumerate(chunk_keys):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, population_size)
    chunk_population = population[start_idx:end_idx]

    chunk_fitness_vals, chunk_fitness_comps = calculate_population_fitness(
      chunk_key,
      chunk_population,
      sequence_type,
      fitness_evaluator,
    )

    fitness_values_list.append(chunk_fitness_vals)
    fitness_components_list.append(chunk_fitness_comps)

  # Concatenate results properly
  fitness_values = jnp.concatenate(fitness_values_list, axis=0)
  
  # fitness_components from each chunk: shape (n_fitness_functions, chunk_size)
  # We want final shape: (n_fitness_functions, population_size)
  fitness_components = jnp.concatenate(fitness_components_list, axis=1)

  return fitness_values, fitness_components


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
    fitness_values, fitness_components = chunked_fitness_evaluation(
      key_fitness,
      mutated_population,
      config.sequence_type,
      config.fitness_evaluator,
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
