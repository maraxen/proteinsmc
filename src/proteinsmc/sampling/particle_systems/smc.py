"""Core JIT-compiled logic for the SMC sampler."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from blackjax import smc
from blackjax.smc import resampling
from blackjax.smc.base import SMCInfo
from blackjax.smc.base import step as smc_step
from jax import jit

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.mutation import MutationFn

from proteinsmc.models.smc import (
  BlackjaxSMCState,
  PopulationSequences,
  SMCAlgorithm,
  SMCConfig,
  SMCState,
)
from proteinsmc.utils.initiate import generate_template_population

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from proteinsmc.models.annealing import AnnealingFn
  from proteinsmc.models.fitness import StackedFitnessFn


def initialize_blackjax_state(
  config: SMCConfig,
  initial_population: PopulationSequences,
  key: PRNGKeyArray,
) -> BlackjaxSMCState:
  """Initialize the Blackjax SMC state."""
  match config.algorithm:
    case (
      SMCAlgorithm.BASE
      | SMCAlgorithm.ANNEALING
      | SMCAlgorithm.PARALLEL_REPLICA
      | SMCAlgorithm.FROM_MCMC
    ):
      return smc.base.init(
        particles=initial_population,
      )
    case SMCAlgorithm.ADAPTIVE_TEMPERED:
      return smc.adaptive_tempered.init()
    case SMCAlgorithm.INNER_MCMC:
      msg = "Inner MCMC SMC algorithm is not implemented yet."
      raise NotImplementedError(msg)
    case SMCAlgorithm.PARTIAL_POSTERIORS:
      return smc.partial_posteriors.init(
        particles=initial_population,
        num_datapoints=config.smc_algo_kwargs.get("num_datapoints", 1),
      )
    case SMCAlgorithm.PRETUNING:
      msg = "Pretuning SMC algorithm is not implemented yet."
      raise NotImplementedError(msg)
    case SMCAlgorithm.TEMPERED:
      return smc.tempered.init(
        particles=initial_population,
      )
    case SMCAlgorithm.CUSTOM:
      if "custom_init_fn" in config.smc_algo_kwargs:
        custom_init_fn: Callable[[PopulationSequences, PRNGKeyArray], BlackjaxSMCState] = (
          config.smc_algo_kwargs["custom_init_fn"]
        )
        return custom_init_fn(initial_population, key)
      msg = "Custom SMC algorithm requires a 'custom_init_fn' in smc_algo_kwargs."
      raise ValueError(msg)
    case _:
      msg = f"Unsupported SMC algorithm: {config.algorithm}"
      raise ValueError(msg)


def initialize_smc_state(
  config: SMCConfig,
  _fitness_function: StackedFitnessFn,
  key: PRNGKeyArray,
) -> SMCState:
  """Initialize the state for the SMC sampler."""
  key, subkey = jax.random.split(key)

  initial_population = generate_template_population(
    initial_sequence=config.seed_sequence,
    population_size=config.population_size,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )

  blackjax_initial_state = initialize_blackjax_state(
    config=config,
    initial_population=initial_population,
    key=subkey,
  )

  return SMCState(
    population=initial_population,
    blackjax_state=blackjax_initial_state,
    beta=0.0,
    key=subkey,
    step=0,
  )


def resample(config: SMCConfig, key: PRNGKeyArray, weights: Float, num_samples: int) -> Array:
  """Resampling function based on the configured approach."""
  match config.resampling_approach:
    case "systematic":
      return resampling.systematic(key, weights, num_samples)
    case "multinomial":
      return resampling.multinomial(key, weights, num_samples)
    case "stratified":
      return resampling.stratified(key, weights, num_samples)
    case "residual":
      return resampling.residual(key, weights, num_samples)
    case _:
      msg = f"Unknown resampling approach: {config.resampling_approach}"
      raise ValueError(msg)


def smc_loop_func(
  config: SMCConfig,
  weight_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
) -> Callable[[SMCState, PRNGKeyArray], tuple[SMCState, SMCInfo]]:
  """Create a JIT-compiled SMC loop function."""
  match config.algorithm:
    case SMCAlgorithm.BASE | SMCAlgorithm.ANNEALING | SMCAlgorithm.PARALLEL_REPLICA:
      return smc_step(
        rng_key=config.key,
        state=config.blackjax_state,
        weight_fn=weight_fn,
        update_fn=mutation_fn,
        resample_fn=partial(resample, config),
      )
    case SMCAlgorithm.ADAPTIVE_TEMPERED:
      msg = "Adaptive Tempered SMC algorithm is not implemented in the loop function."
      raise NotImplementedError(msg)
    case _:
      msg = f"SMC algorithm {config.algorithm} is not currently supported."
      raise NotImplementedError(msg)


@partial(jit, static_argnames=("config", "fitness_fn", "mutation_fn", "annealing_fn"))
def run_smc_loop(
  config: SMCConfig,
  initial_state: SMCState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: MutationFn,
  annealing_fn: AnnealingFn | None = None,
) -> tuple[SMCState, SMCInfo]:
  """JIT-compiled SMC loop."""

  def scan_body(carry_state: SMCState, i: Int) -> tuple[SMCState, SMCInfo]:
    current_beta = None if annealing_fn is None else annealing_fn(i, _context=None)  # type: ignore[call-arg]
    state_for_step = carry_state.replace(beta=current_beta)
    key_for_fitness_fn, key_for_blackjax = jax.random.split(state_for_step.key)

    def weight_fn(
      sequence: PopulationSequences,
    ) -> Array:
      """Weight function for the SMC step."""
      return fitness_fn(key_for_fitness_fn, sequence, current_beta)

    next_state, info = smc_loop_func(
      config=config,
      initial_state=state_for_step,  # type: ignore[arg-type]
      weight_fn=weight_fn,
      mutation_fn=mutation_fn,
    )
    return next_state, info

  final_state, collected_metrics = jax.lax.scan(
    scan_body,
    initial_state,
    jnp.arange(config.num_samples),
    length=config.num_samples,
  )

  return final_state, collected_metrics
