"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.random_walk import RWState

from proteinsmc.models.sampler_base import SamplerOutput, SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.types import ArrayLike, EvoSequence, PRNGKey, ScalarFloat


def run_mcmc_loop(
  num_samples: int,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  mutation_fn: Callable[[PRNGKey, EvoSequence], EvoSequence],
  io_callback: Callable | None = None,
) -> tuple[SamplerState, dict[str, ArrayLike]]:
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
    initial_state = initial_state.replace(blackjax_state=rw_state)

  # Build the random walk Metropolis-Hastings kernel
  kernel = blackjax.mcmc.random_walk.build_rmh()

  def body_fn(carry: SamplerState, _inputs: None = None) -> tuple[SamplerState, SamplerOutput]:
    """Perform one step of the MCMC sampler.

    Args:
      carry: Current SamplerState.

    Returns:
      Tuple of updated SamplerState and SamplerOutput.

    """
    key, subkey = jax.random.split(carry.key)

    def logdensity_fn_step(position: ArrayLike) -> ScalarFloat:
      """Logdensity function for this MCMC step."""
      fitness = fitness_fn(key, position, None)
      return fitness[0]

    def transition_generator_wrapped(key_mutation: PRNGKey, position: ArrayLike) -> Array:
      """Mutation function that preserves the dtype of the input."""
      mutated = mutation_fn(key_mutation, position)
      return mutated.astype(position.dtype)

    new_blackjax_state, _ = kernel(
      rng_key=key,
      state=carry.blackjax_state,  # pyright: ignore[reportArgumentType]
      logdensity_fn=logdensity_fn_step,
      transition_generator=transition_generator_wrapped,
    )

    new_fitness = fitness_fn(key, new_blackjax_state.position, None)  # pyright: ignore[reportArgumentType]

    new_state = SamplerState(
      sequence=new_blackjax_state.position,  # pyright: ignore[reportArgumentType]
      key=subkey,
      blackjax_state=new_blackjax_state,
      step=jnp.array(carry.step + 1),
    )

    output = SamplerOutput(
      step=jnp.array(carry.step + 1, dtype=jnp.int32),
      sequences=new_blackjax_state.position,  # type: ignore[arg-type]
      fitness=new_fitness,
      key=subkey,
    )

    return new_state, output

  # Use scan to accumulate outputs
  final_state, outputs = jax.lax.scan(body_fn, initial_state, jnp.arange(num_samples))

  # If io_callback is provided, write outputs using Python for loop
  if io_callback is not None:
    for i in range(num_samples):
      # Extract single step output
      single_output = SamplerOutput(
        step=outputs.step[i],
        sequences=outputs.sequences[i],
        fitness=outputs.fitness[i],
        key=outputs.key[i],
      )
      io_callback(single_output)

  # Return final state and outputs as dict
  metrics = {
    "steps": outputs.step,
    "sequences": outputs.sequences,
    "fitness": outputs.fitness,
    "key": outputs.key,
  }
  return final_state, metrics
