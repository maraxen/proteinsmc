"""Implements the Hamiltonian Monte Carlo (HMC) sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import blackjax
import jax

from proteinsmc.models.hmc import HMCState
from proteinsmc.utils.config_unpacker import with_config

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn


__all__ = ["run_hmc_loop"]


@with_config
def run_hmc_loop(
  num_samples: int,
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
    new_state = HMCState(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=key,
      blackjax_state=new_blackjax_state,
    )
    return new_state, new_state

  keys = jax.random.split(initial_state.key, num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
