"""Implements the Hamiltonian Monte Carlo (HMC) sampling algorithm."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import blackjax
import jax
from jax import jit

from proteinsmc.models.hmc import HMCConfig, HMCState
from proteinsmc.utils.initiate import generate_template_population

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn

__all__ = ["initialize_hmc_state", "run_hmc_loop"]


def initialize_hmc_state(
  config: HMCConfig,
  fitness_fn: StackedFitnessFn,
  key: PRNGKeyArray,
) -> HMCState:
  """Initialize the state of the HMC sampler.

  Args:
      config: Configuration for the HMC sampler.
      fitness_fn: Fitness function to evaluate sequences.
      key: JAX PRNG key.

  Returns:
      An initial HMCState.

  """
  initial_sequence = generate_template_population(
    initial_sequence=config.seed_sequence,
    population_size=1,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )
  blackjax_initial_state = blackjax.hmc.init(
    initial_sequence,
    fitness_fn,
    step_size=config.step_size,
    num_integration_steps=config.num_leapfrog_steps,
  )
  return HMCState(
    sequence=initial_sequence,
    fitness=blackjax_initial_state.logdensity,
    key=key,
    blackjax_state=blackjax_initial_state,
  )


@partial(jit, static_argnames=("config", "fitness_fn", "mutation_fn"))
def run_hmc_loop(
  config: HMCConfig,
  initial_state: HMCState,
  fitness_fn: StackedFitnessFn,
  _mutation_fn: MutationFn | None = None,
) -> tuple[HMCState, HMCState]:
  """Run the Hamiltonian Monte Carlo (HMC) sampler loop.

  Args:
      config: HMC sampler configuration.
      initial_state: Initial sequence of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      mutation_fn: Mutation function to generate new sequences.

  Returns:
      A tuple containing the final state and the history of states.

  """
  kernel = blackjax.hmc.build_kernel()

  def one_step(state: HMCState, key: PRNGKeyArray) -> tuple[HMCState, HMCState]:
    """Perform one step of the HMC sampler."""
    new_blackjax_state, _ = kernel(
      rng_key=key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
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
