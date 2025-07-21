"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import blackjax
import jax
import jax.numpy as jnp

from proteinsmc.models.mcmc import MCMCConfig, MCMCState
from proteinsmc.utils.initiate import generate_template_population

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.types import EvoSequence


def initialize_mcmc_state(
  config: MCMCConfig,
  fitness_fn: StackedFitnessFn,
  key: PRNGKeyArray,
) -> MCMCState:
  """Initialize the state of the MCMC sampler using Blackjax."""
  initial_sequence = generate_template_population(
    initial_sequence=config.seed_sequence,
    population_size=1,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )
  blackjax_initial_state = blackjax.mcmc.random_walk.init(
    initial_sequence,
    fitness_fn,
  )
  return MCMCState(
    sequence=initial_sequence,
    fitness=jnp.array(blackjax_initial_state.logdensity, dtype=jnp.float32),
    key=key,
    blackjax_state=blackjax_initial_state,
  )


def run_mcmc_loop(
  config: MCMCConfig,
  initial_state: MCMCState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: Callable[[PRNGKeyArray, EvoSequence], EvoSequence],
) -> tuple[MCMCState, MCMCState]:
  """Run the MCMC sampling loop using Blackjax."""
  kernel = blackjax.mcmc.random_walk.build_rmh()

  def one_step(state: MCMCState, key: PRNGKeyArray) -> tuple[MCMCState, MCMCState]:
    """Perform one step of the MCMC sampler."""
    new_blackjax_state, _ = kernel(
      rng_key=key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
      transition_generator=mutation_fn,
    )
    new_state = state.replace(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=key,
      blackjax_state=new_blackjax_state,
    )
    return new_state, new_state

  keys = jax.random.split(initial_state.key, config.num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
