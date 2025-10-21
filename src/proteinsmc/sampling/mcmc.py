"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import blackjax
import jax
import jax.numpy as jnp
from jax.experimental import io_callback as jax_io_callback

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
  io_callback: Callable | None = None,
) -> tuple[SamplerState, dict[str, jax.Array]]:
  """Run the MCMC sampling loop using Blackjax.

  Args:
      num_samples: Number of MCMC samples to generate.
      initial_state: Initial state of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      mutation_fn: Mutation function to generate new sequences.
      io_callback: Optional callback function for writing outputs.

  Returns:
      A tuple containing the final state and empty metrics dictionary.

  """
  kernel = blackjax.mcmc.random_walk.build_rmh()

  def body_fn(step_idx: int, state: SamplerState) -> SamplerState:
    """Perform one step of the MCMC sampler."""
    key = jax.random.fold_in(state.key, step_idx)
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
      step=jnp.array(step_idx + 1),
    )

    if io_callback is not None:
      jax_io_callback(
        io_callback,
        None,
        {"sequence": new_state.sequence, "fitness": new_state.fitness, "step": new_state.step},
      )

    return new_state

  final_state = jax.lax.fori_loop(0, num_samples, body_fn, initial_state)
  return final_state, {}
