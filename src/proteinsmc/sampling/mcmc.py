"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import blackjax
import jax

from proteinsmc.models.mcmc import MCMCState

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.types import EvoSequence


def run_mcmc_loop(
  num_samples: int,
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
    new_state = MCMCState(
      sequence=new_blackjax_state.position,  # pyright: ignore[reportArgumentType]
      fitness=new_blackjax_state.logdensity,  # pyright: ignore[reportArgumentType]
      key=key,
      blackjax_state=new_blackjax_state,
    )
    return new_state, new_state

  keys = jax.random.split(initial_state.key, num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
