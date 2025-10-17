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
from jax import config, vmap
from jaxtyping import PRNGKeyArray

from proteinsmc.models.hmc import HMCState
from proteinsmc.models.mcmc import MCMCState
from proteinsmc.models.nuts import NUTSState
from proteinsmc.models.parallel_replica import (
  IslandState,
  PRSMCState,
)
from proteinsmc.models.smc import SMCAlgorithm, SMCState
from proteinsmc.utils.config_unpacker import with_config
from proteinsmc.utils.initiate import generate_template_population
from proteinsmc.utils.mutation import diversify_initial_sequences

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.types import BatchEvoSequence, EvoSequence, SequenceType

BlackjaxSMCState = BaseSMCState | PartialPosteriorsSMCState | TemperedSMCState


@with_config
def initialize_sampler_state(  # noqa: PLR0913
  sampler_type: str,
  sequence_type: SequenceType,
  seed_sequence: EvoSequence,
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
  *,
  track_lineage: bool = False,
) -> HMCState | MCMCState | NUTSState | SMCState | PRSMCState:
  """Make initialize state for any sampler type.

  Args:
    sampler_type: Type of sampler ("SMC", "HMC", "MCMC", "NUTS", "ParallelReplica").
    sequence_type: Type of sequence ("protein" or "nucleotide").
    seed_sequence: Initial sequence string.
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
    track_lineage: Whether to track particle lineage (for Parallel Replica SMC).

  Returns:
    Initial state for the specified sampler type.

  Raises:
    ValueError: If the config type is not recognized.

  Example:
    >>> config = SMCConfig(seed_sequence="ACGT", population_size=100, ...)
    >>> key = jax.random.PRNGKey(0)
    >>> initial_state = initialize_sampler_state(config, fitness_fn, key)

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
      smc_algo_kwargs=smc_algo_kwargs,
      key=key,
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
      track_lineage=track_lineage,
      key=key,
    )
  msg = f"Unsupported sampler config type: {type(config)}"
  raise ValueError(msg)


def _initialize_single_state(
  sampler_type: str,
  initial_population: BatchEvoSequence,
  fitness_fn: StackedFitnessFn,
  key: PRNGKeyArray,
) -> HMCState | MCMCState | NUTSState:
  match sampler_type:
    case "HMC":
      init_fn = hmc.init
      state_cls = HMCState
    case "MCMC":
      init_fn = mcmc.random_walk.init
      state_cls = MCMCState
    case "NUTS":
      init_fn = nuts.init
      state_cls = NUTSState
    case _:
      msg = f"Unsupported single-particle sampler type: {sampler_type}"
      raise ValueError(msg)
  blackjax_initial_state = init_fn(
    initial_population[0],
    fitness_fn,
  )
  return state_cls(
    sequence=initial_population[0],
    fitness=jnp.array(blackjax_initial_state.logdensity, dtype=jnp.float32),
    key=key,
    blackjax_state=blackjax_initial_state,
  )


def _initialize_blackjax_smc_state(
  algorithm: SMCAlgorithm,
  initial_population: BatchEvoSequence,
  smc_algo_kwargs: dict,
  key: PRNGKeyArray,
) -> BlackjaxSMCState:
  """Initialize the Blackjax SMC state based on algorithm type.

  Args:
    algorithm: SMC algorithm variant to use.
    initial_population: Initial population of sequences.
    smc_algo_kwargs: Additional algorithm-specific arguments.
    key: JAX PRNG key.

  Returns:
    Initial Blackjax SMC state.

  Raises:
    NotImplementedError: If the algorithm is not yet implemented.
    ValueError: If the algorithm type is not recognized.

  """
  match algorithm:
    case (
      SMCAlgorithm.BASE
      | SMCAlgorithm.ANNEALING
      | SMCAlgorithm.PARALLEL_REPLICA
      | SMCAlgorithm.FROM_MCMC
    ):
      return smc.base.init(particles=initial_population, init_update_params={})
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


def _initialize_smc_state(
  initial_population: BatchEvoSequence,
  beta: Float,
  algorithm: SMCAlgorithm,
  smc_algo_kwargs: dict,
  key: PRNGKeyArray,
) -> SMCState:
  """Initialize the state for the SMC sampler.

  Args:
    initial_population: Initial population of sequences.
    beta: Initial beta value for tempering.
    algorithm: SMC algorithm variant to use.
    smc_algo_kwargs: Additional algorithm-specific arguments.
    key: JAX PRNG key.

  Returns:
    An initial SMCState.

  """
  key, subkey = jax.random.split(key)

  blackjax_initial_state = _initialize_blackjax_smc_state(
    algorithm=algorithm,
    initial_population=initial_population,
    smc_algo_kwargs=smc_algo_kwargs,
    key=subkey,
  )

  return SMCState(
    population=initial_population,
    blackjax_state=blackjax_initial_state,
    beta=beta,
    key=subkey,
    step=jnp.array(0, dtype=jnp.int32),
  )


def _initialize_prsmc_state(  # noqa: PLR0913
  initial_populations: BatchEvoSequence,
  n_islands: Int,
  population_size_per_island: Int,
  island_betas: Float,
  key: PRNGKeyArray,
  *,
  track_lineage: bool = False,
) -> PRSMCState:
  """Initialize the state for the Parallel Replica SMC sampler.

  Args:
    initial_populations: Initial populations for each island.
    n_islands: Number of islands.
    population_size_per_island: Population size per island.
    island_betas: List of beta values for each island.
    key: JAX PRNG key.
    track_lineage: Whether to track lineage information.

  Returns:
    An initial PRSMCState.

  Raises:
    ValueError: If the seed sequence contains invalid characters.

  """
  key_init_islands, key_smc_loop = jax.random.split(key)

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

  # Initialize lineage tracking if enabled
  if track_lineage:
    island_indices = jnp.arange(n_islands)[:, None]
    particle_indices = jnp.arange(population_size_per_island)[None, :]
    global_ids = island_indices * population_size_per_island + particle_indices
    parent_ids = jnp.full_like(global_ids, -1)
    initial_lineage_arrays = jnp.transpose(
      jnp.stack([global_ids, parent_ids], axis=1),
      (0, 2, 1),
    )
  else:
    initial_lineage_arrays = None

  # Construct initial island states
  initial_island_states = IslandState(
    key=island_keys,
    beta=island_betas_array,
    logZ_estimate=jnp.zeros(n_islands, dtype=jnp.float32),
    mean_fitness=jnp.zeros(n_islands, dtype=jnp.float32),
    ess=jnp.zeros(n_islands, dtype=jnp.float32),
    blackjax_state=initial_blackjax_states,
    lineage=initial_lineage_arrays,
    step=jnp.zeros(n_islands, dtype=jnp.int32),
  )

  return PRSMCState(
    current_overall_state=initial_island_states,
    prng_key=key_smc_loop,
    total_swaps_attempted=jnp.array(0, dtype=jnp.int32),
    total_swaps_accepted=jnp.array(0, dtype=jnp.int32),
  )
