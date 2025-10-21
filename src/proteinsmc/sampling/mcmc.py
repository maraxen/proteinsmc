"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import blackjax
import jax
from jax.experimental import io_callback

from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.types import EvoSequence


def run_mcmc_loop(
  config,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: Callable[[PRNGKeyArray, EvoSequence], EvoSequence],
  io_callback: Callable,
  **kwargs,
) -> tuple[SamplerState, dict]:
  """Run the MCMC sampling loop using Blackjax."""
  kernel = blackjax.mcmc.random_walk.build_rmh()

  def one_step(i, state):
    """Perform one step of the MCMC sampler."""
    key, kernel_key, next_key = jax.random.split(state.key, 3)
    new_blackjax_state, mcmc_info = kernel(
      rng_key=kernel_key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
      transition_generator=mutation_fn,
    )
    new_state = SamplerState(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=next_key,
      blackjax_state=new_blackjax_state,
      step=i,
    )
    payload = {
      "sequence": new_state.sequence,
      "fitness": new_state.fitness,
      "step": new_state.step,
      "acceptance_rate": mcmc_info.acceptance_rate,
    }
    io_callback(payload)
    return new_state

  final_state = jax.lax.fori_loop(0, config.num_samples, one_step, initial_state)
  return final_state, {}
