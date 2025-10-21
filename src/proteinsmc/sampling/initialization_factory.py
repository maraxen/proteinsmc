"""Factory function for initializing sampler states.

This module provides a unified interface for creating initial states for different
sampling algorithms (SMC, MCMC, HMC, NUTS, Parallel Replica).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from blackjax import hmc, mcmc, nuts, smc
from blackjax.smc.base import SMCState as BaseSMCState
from blackjax.smc.partial_posteriors_path import PartialPosteriorsSMCState
from blackjax.smc.tempered import TemperedSMCState
from jax import vmap
from jaxtyping import PRNGKeyArray

from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.models.smc import SMCAlgorithm
from proteinsmc.utils.initiate import generate_template_population
from proteinsmc.utils.mutation import diversify_initial_sequences

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.types import BatchEvoSequence, EvoSequence, SequenceType

BlackjaxSMCState = BaseSMCState | PartialPosteriorsSMCState | TemperedSMCState


def initialize_sampler_state(  # noqa: PLR0913
  sampler_type: str,
  sequence_type: SequenceType,
  seed_sequence: EvoSequence,
  mutation_rate: Float | None,
  population_size: Int | None,
  algorithm: SMCAlgorithm | None,
  smc_algo_kwargs: dict | None,
  n_islands: Int | None,
  population_size_per_island: Int | None,
  island_betas: Float | None,
  diversification_ratio: Float | None,
  key: PRNGKeyArray,
  beta: Float | None,
  fitness_fn: StackedFitnessFn,
) -> SamplerState:
  """Initialize state for any sampler type.

  Args:
    sampler_type: Type of sampler ("SMC", "HMC", "MCMC", "NUTS", "ParallelReplica").
    sequence_type: Type of sequence ("protein" or "nucleotide").
    seed_sequence: Initial sequence string.
    mutation_rate: Mutation rate for SMC samplers.
    population_size: Number of particles in the population.
    algorithm: SMC algorithm variant to use.
    smc_algo_kwargs: Additional algorithm-specific arguments.
    n_islands: Number of parallel islands (for Parallel Replica SMC).
    population_size_per_island: Number of particles per island (for Parallel Replica SMC).
    island_betas: Beta values for each island (for Parallel Replica SMC).
    diversification_ratio: Mutation rate for initial diversification (for Parallel Replica SMC).
    key: JAX PRNG key.
    beta: Initial beta value for tempering (for SMC).
    fitness_fn: Fitness function to evaluate sequences.

  Returns:
    Initial state for the specified sampler type.

  Raises:
    ValueError: If the sampler type is not recognized.

  Example:
    >>> key = jax.random.PRNGKey(0)
    >>> seed_seq = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    >>> initial_state = initialize_sampler_state(
    ...     sampler_type="SMC",
    ...     sequence_type="protein",
    ...     seed_sequence=seed_seq,
    ...     mutation_rate=0.1,
    ...     population_size=100,
    ...     algorithm=SMCAlgorithm.BASE,
    ...     smc_algo_kwargs={},
    ...     n_islands=None,
    ...     population_size_per_island=None,
    ...     island_betas=None,
    ...     diversification_ratio=None,
    ...     key=key,
    ...     beta=1.0,
    ...     fitness_fn=fitness_fn,
    ... )

  """
  initial_population = generate_template_population(
    initial_sequence=seed_sequence,
    population_size=population_size,
    input_sequence_type=sequence_type,
    output_sequence_type=sequence_type,
  )
  if sampler_type in {"HMC", "MCMC", "NUTS"}:
    return _initialize_single_state(
      sampler_type=sampler_type,
      initial_population=initial_population,
      fitness_fn=fitness_fn,
      key=key,
    )

  if sampler_type == "SMC":
    if smc_algo_kwargs is None:
      smc_algo_kwargs = {}
    return _initialize_smc_state(
      initial_population=initial_population,
      algorithm=algorithm if algorithm is not None else SMCAlgorithm.BASE,
      beta=beta,
      mutation_rate=mutation_rate,
      smc_algo_kwargs=smc_algo_kwargs,
      key=key,
      fitness_fn=fitness_fn,
    )
  if sampler_type == "ParallelReplica":
    key_init_islands, key = jax.random.split(key)
    _n_islands = n_islands if n_islands is not None else jnp.array(4, dtype=jnp.int32)
    _population_size_per_island = (
      population_size_per_island
      if population_size_per_island is not None
      else jnp.array(50, dtype=jnp.int32)
    )
    initial_populations = jnp.tile(initial_population, (_n_islands, 1, 1))
    initial_populations = diversify_initial_sequences(
      key=key_init_islands,
      seed_sequences=initial_populations,
      mutation_rate=diversification_ratio,
      sequence_type=sequence_type,
    )
    initial_populations_stacked = initial_populations.reshape(
      _n_islands,
      _population_size_per_island,
      -1,
    )
    return _initialize_prsmc_state(
      initial_populations=initial_populations_stacked,
      n_islands=_n_islands,
      population_size_per_island=_population_size_per_island,
      island_betas=island_betas,
      key=key,
      fitness_fn=fitness_fn,
    )
  msg = f"Unsupported sampler type: {sampler_type}"
  raise ValueError(msg)


def _initialize_single_state(
  sampler_type: str,
  initial_population: BatchEvoSequence,
  fitness_fn: StackedFitnessFn,
  key: PRNGKeyArray,
) -> SamplerState:
  match sampler_type:
    case "HMC":
      init_fn = hmc.init
    case "MCMC":
      init_fn = mcmc.random_walk.init
    case "NUTS":
      init_fn = nuts.init
    case _:
      msg = f"Unsupported single-particle sampler type: {sampler_type}"
      raise ValueError(msg)

  # Initialize blackjax state
  key_fitness, key_next = jax.random.split(key)
  initial_sequence = initial_population[0]

  # Compute initial fitness
  initial_fitness = fitness_fn(initial_sequence, key_fitness, None)

  # Create a logdensity function for blackjax (uses combined fitness at index 0)
  def logdensity_fn(seq: EvoSequence) -> Float:
    return fitness_fn(seq, key_fitness, None)[0]

  blackjax_initial_state = init_fn(
    initial_sequence,
    logdensity_fn,
  )

  return SamplerState(
    sequence=initial_sequence,
    fitness=initial_fitness,
    key=key_next,
    blackjax_state=blackjax_initial_state,
    step=jnp.array(0, dtype=jnp.int32),
  )


def _initialize_blackjax_smc_state(
  key: PRNGKeyArray,
  algorithm: SMCAlgorithm,
  initial_population: BatchEvoSequence,
  update_params: dict | None = None,
  smc_algo_kwargs: dict | None = None,
) -> BlackjaxSMCState:
  """Initialize the Blackjax SMC state based on algorithm type.

  Args:
    algorithm: SMC algorithm variant to use.
    initial_population: Initial population of sequences.
    smc_algo_kwargs: Additional algorithm-specific arguments.
    key: JAX PRNG key.
    update_params: Update parameters for the SMC state.

  Returns:
    Initial Blackjax SMC state.

  Raises:
    NotImplementedError: If the algorithm is not yet implemented.
    ValueError: If the algorithm type is not recognized.

  """
  smc_algo_kwargs = smc_algo_kwargs or {}
  match algorithm:
    case (
      SMCAlgorithm.BASE
      | SMCAlgorithm.ANNEALING
      | SMCAlgorithm.PARALLEL_REPLICA
      | SMCAlgorithm.FROM_MCMC
    ):
      return smc.base.init(particles=initial_population, init_update_params=update_params or {})
    case SMCAlgorithm.ADAPTIVE_TEMPERED:
      return smc.adaptive_tempered.init(particles=initial_population)
    case SMCAlgorithm.INNER_MCMC:
      msg = "Inner MCMC SMC algorithm is not implemented yet."
      raise NotImplementedError(msg)
    case SMCAlgorithm.PARTIAL_POSTERIORS:
      return smc.partial_posteriors_path.init(
        particles=initial_population,
        num_datapoints=smc_algo_kwargs.get("num_datapoints", 1),
      )
    case SMCAlgorithm.PRETUNING:
      msg = "Pretuning SMC algorithm is not implemented yet."
      raise NotImplementedError(msg)
    case SMCAlgorithm.TEMPERED:
      return smc.tempered.init(particles=initial_population)
    case SMCAlgorithm.CUSTOM:
      if "custom_init_fn" in smc_algo_kwargs:
        return smc_algo_kwargs["custom_init_fn"](initial_population, key)
      msg = "Custom SMC algorithm requires a 'custom_init_fn' in smc_algo_kwargs."
      raise ValueError(msg)
    case _:
      msg = f"Unsupported SMC algorithm: {algorithm}"
      raise ValueError(msg)


def _initialize_smc_state(  # noqa: PLR0913
  initial_population: BatchEvoSequence,
  beta: Float,
  mutation_rate: Float,
  algorithm: SMCAlgorithm,
  smc_algo_kwargs: dict,
  key: PRNGKeyArray,
  fitness_fn: StackedFitnessFn,
) -> SamplerState:
  """Initialize the state for the SMC sampler.

  Args:
    initial_population: Initial population of sequences.
    beta: Initial beta value for tempering.
    algorithm: SMC algorithm variant to use.
    smc_algo_kwargs: Additional algorithm-specific arguments.
    key: JAX PRNG key.
    fitness_fn: Fitness function to evaluate sequences.
    mutation_rate: Mutation rate for SMC samplers.

  Returns:
    An initial SamplerState for SMC.

  """
  key_fitness, key_blackjax, key_next = jax.random.split(key, 3)
  update_params = {"mutation_rate": mutation_rate}

  blackjax_initial_state = _initialize_blackjax_smc_state(
    algorithm=algorithm,
    initial_population=initial_population,
    smc_algo_kwargs=smc_algo_kwargs,
    key=key_blackjax,
    update_params=update_params,
  )
  initial_fitness_batch = vmap(lambda seq: fitness_fn(seq, key_fitness, beta))(
    initial_population,
  )
  mean_fitness = jnp.mean(initial_fitness_batch, axis=0)

  return SamplerState(
    sequence=initial_population,
    fitness=mean_fitness,
    key=key_next,
    blackjax_state=blackjax_initial_state,
    step=jnp.array(0, dtype=jnp.int32),
    additional_fields={"beta": beta if beta is not None else jnp.array(1.0)},
  )


def _initialize_prsmc_state(  # noqa: PLR0913
  initial_populations: BatchEvoSequence,
  n_islands: Int,
  population_size_per_island: Int,
  island_betas: Float,
  key: PRNGKeyArray,
  fitness_fn: StackedFitnessFn,
) -> SamplerState:
  """Initialize the state for the Parallel Replica SMC sampler.

  Args:
    initial_populations: Initial populations for each island.
    n_islands: Number of islands.
    population_size_per_island: Population size per island.
    island_betas: List of beta values for each island.
    key: JAX PRNG key.
    fitness_fn: Fitness function to evaluate sequences.

  Returns:
    An initial SamplerState for PRSMC with stacked island data.

  Raises:
    ValueError: If the seed sequence contains invalid characters.

  """
  key_init_islands, _ = jax.random.split(key)

  # Initialize island-specific data
  island_keys = jax.random.split(key_init_islands, n_islands)
  island_betas_array = jnp.array(island_betas, dtype=jnp.float32)

  initial_weights = jnp.full(
    population_size_per_island,
    1.0 / population_size_per_island,
    dtype=jnp.float32,
  )
  initial_blackjax_states = vmap(
    lambda p: BaseSMCState(
      particles=p,
      weights=initial_weights,
      update_parameters=jnp.array(0.0, dtype=jnp.float32),
    ),
  )(initial_populations)

  # Compute initial fitness for each island (n_islands, population_size_per_island, n_fitness+1)
  key_fitness, _ = jax.random.split(key_init_islands)
  fitness_keys = jax.random.split(key_fitness, n_islands)
  initial_fitness_batch = vmap(
    lambda island_pop, k, beta: vmap(lambda seq: fitness_fn(seq, k, beta))(island_pop),
  )(initial_populations, fitness_keys, island_betas_array)

  # For PRSMC additional fields
  mean_fitness = jnp.mean(initial_fitness_batch[:, :, 0], axis=1)  # Mean over particles
  max_fitness = jnp.max(initial_fitness_batch[:, :, 0], axis=1)  # Max over particles

  # Extract update_parameters if available
  update_params = {}
  update_param_values = getattr(initial_blackjax_states, "update_parameters", None)
  if update_param_values is not None and isinstance(update_param_values, jax.Array):
    update_params = {"smc_update_param": update_param_values}

  # Construct initial island states
  return SamplerState(
    sequence=initial_populations,  # Shape: (n_islands, population_size_per_island, seq_len)
    fitness=initial_fitness_batch,  # Shape: (n_islands, population_size_per_island, n_fitness+1)
    key=island_keys,  # Shape: (n_islands, 2)
    blackjax_state=initial_blackjax_states,
    step=jnp.zeros(n_islands, dtype=jnp.int32),
    update_parameters=update_params,
    additional_fields={
      "beta": island_betas_array,
      "mean_fitness": mean_fitness,
      "max_fitness": max_fitness,
      "ess": jnp.zeros(n_islands, dtype=jnp.float32),
      "logZ_estimate": jnp.zeros(n_islands, dtype=jnp.float32),
    },
  )
