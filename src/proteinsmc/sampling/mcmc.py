"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import blackjax
import jax

from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.types import EvoSequence


def run_mcmc_loop(
  num_samples: int,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: Callable[[PRNGKeyArray, EvoSequence], EvoSequence],
) -> tuple[SamplerState, SamplerState]:
  """Run the MCMC sampling loop using Blackjax."""
  kernel = blackjax.mcmc.random_walk.build_rmh()

  def one_step(state: SamplerState, key: PRNGKeyArray) -> tuple[SamplerState, SamplerState]:
    """Perform one step of the MCMC sampler."""
    new_blackjax_state, _ = kernel(
      rng_key=key,
      state=state.blackjax_state,  # pyright: ignore[reportArgumentType]
      logdensity_fn=fitness_fn,
      transition_generator=mutation_fn,
    )
    new_state = SamplerState(
      sequence=new_blackjax_state.position,  # pyright: ignore[reportArgumentType]
      fitness=new_blackjax_state.logdensity,  # pyright: ignore[reportArgumentType]
      key=key,
      blackjax_state=new_blackjax_state,
      step=state.step + 1,
    )
    return new_state, new_state

  keys = jax.random.split(initial_state.key, num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
