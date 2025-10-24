"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWState
from jax.experimental import io_callback as jax_io_callback

from proteinsmc.models.sampler_base import SamplerOutput, SamplerState

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
  # Initialize blackjax RW state if not already present
  if initial_state.blackjax_state is None:
    # Compute initial logdensity
    initial_fitness = fitness_fn(initial_state.key, initial_state.sequence, None)

    rw_state = RWState(
      position=initial_state.sequence,
      logdensity=float(initial_fitness[0]),
    )
    initial_state = dataclasses.replace(initial_state, blackjax_state=rw_state)

  # Build the random walk Metropolis-Hastings kernel
  kernel = blackjax.mcmc.random_walk.build_rmh()

  def body_fn(step_idx: int, state: SamplerState) -> SamplerState:
    """Perform one step of the MCMC sampler."""
    key = jax.random.fold_in(state.key, step_idx)

    # Wrap fitness function for blackjax
    def logdensity_fn_step(position: jax.Array) -> jax.Array:
      """Logdensity function for this MCMC step."""
      fitness = fitness_fn(key, position, None)
      return fitness[0]  # Return combined fitness

    # Wrap mutation function to preserve dtype
    def transition_generator_wrapped(key_mutation: jax.Array, position: jax.Array) -> jax.Array:
      """Mutation function that preserves the dtype of the input."""
      mutated = mutation_fn(key_mutation, position)
      # Cast back to original dtype if needed
      return mutated.astype(position.dtype)

    # The kernel returns (new_state, info)
    new_blackjax_state, _ = kernel(
      rng_key=key,
      state=state.blackjax_state,  # pyright: ignore[reportArgumentType]
      logdensity_fn=logdensity_fn_step,
      transition_generator=transition_generator_wrapped,
    )

    # Recompute full fitness to maintain proper shape
    new_fitness = fitness_fn(key, new_blackjax_state.position, None)  # pyright: ignore[reportArgumentType]

    new_state = SamplerState(
      sequence=new_blackjax_state.position,  # pyright: ignore[reportArgumentType]
      key=key,
      blackjax_state=new_blackjax_state,
      step=jnp.array(step_idx + 1),
    )

    if io_callback is not None:
      output = SamplerOutput(
        step=jnp.array(step_idx + 1, dtype=jnp.int32),
        sequences=new_blackjax_state.position,  # type: ignore[arg-type]
        fitness=new_fitness,
        key=key,
      )
      jax_io_callback(
        io_callback,
        None,
        output,
      )

    return new_state

  final_state = jax.lax.fori_loop(0, num_samples, body_fn, initial_state)
  return final_state, {}
