"""A simplified implementation of the No-U-Turn Sampler (NUTS).

Note: A full, robust NUTS implementation is complex and typically relies on
advanced numerical methods and tree-building algorithms. This is a conceptual
placeholder for demonstration purposes and will not be a complete, production-ready
NUTS sampler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCState
from jax.experimental import io_callback as jax_io_callback

from proteinsmc.models.sampler_base import SamplerOutput, SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Float, Int

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn

__all__ = ["run_nuts_loop"]


def run_nuts_loop(  # noqa: PLR0913
  num_samples: Int,
  step_size: Float,
  num_doublings: Int,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  inverse_mass_matrix: jax.Array | None = None,
  _mutation_fn: MutationFn | None = None,
  io_callback: Callable | None = None,
) -> tuple[SamplerState, dict[str, jax.Array]]:
  """Run a NUTS sampling loop.

  This function demonstrates the basic idea of NUTS but lacks the full
  adaptivity and tree-building of a true NUTS implementation.

  Args:
      num_samples: Number of samples to generate.
      step_size: Step size for the NUTS kernel.
      num_doublings: Maximum number of tree doublings.
      initial_state: Initial position of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      inverse_mass_matrix: Inverse mass matrix for NUTS (optional).
      _mutation_fn: Mutation function (unused for NUTS).
      io_callback: Optional callback function for writing outputs.

  Returns:
      A tuple containing the final state and empty metrics dictionary.

  """
  # Initialize blackjax NUTS state if not already present (uses HMCState)
  if initial_state.blackjax_state is None:
    # Compute initial logdensity and gradient
    def logdensity_fn_wrapped(position: jax.Array) -> jax.Array:
      """Convert fitness_fn to blackjax's expected signature."""
      fitness = fitness_fn(initial_state.key, position, None)
      return fitness[0]

    initial_logdensity = logdensity_fn_wrapped(initial_state.sequence)
    logdensity_grad = jax.grad(logdensity_fn_wrapped)(initial_state.sequence)

    nuts_state = HMCState(
      position=initial_state.sequence,
      logdensity=float(initial_logdensity),
      logdensity_grad=logdensity_grad,
    )
    initial_state = initial_state.replace(blackjax_state=nuts_state)

  kernel = blackjax.nuts.build_kernel()

  def body_fn(step_idx: int, state: SamplerState) -> SamplerState:
    """Perform one step of the NUTS sampler."""
    key = jax.random.fold_in(state.key, step_idx)

    # Create default inverse mass matrix if not provided
    mass_matrix = inverse_mass_matrix
    if mass_matrix is None:
      # Use identity matrix with dimension matching the sequence
      dim = state.sequence.size
      mass_matrix = jnp.eye(dim)

    # Wrap fitness function for this step
    def logdensity_fn_step(position: jax.Array) -> jax.Array:
      """Logdensity function for this NUTS step."""
      fitness = fitness_fn(key, position, None)
      return fitness[0]  # Return combined fitness

    new_blackjax_state, info = kernel(
      rng_key=key,
      state=state.blackjax_state,
      logdensity_fn=logdensity_fn_step,
      step_size=step_size,
      max_num_doublings=num_doublings,
      inverse_mass_matrix=mass_matrix,
    )

    # Recompute full fitness to maintain proper shape
    new_fitness = fitness_fn(key, new_blackjax_state.position, None)

    new_state = SamplerState(
      sequence=new_blackjax_state.position,
      key=key,
      blackjax_state=new_blackjax_state,
      step=jnp.array(step_idx + 1),
    )

    if io_callback is not None:
      output = SamplerOutput(
        step=jnp.array(step_idx + 1, dtype=jnp.int32),
        sequences=new_blackjax_state.position,
        fitness=new_fitness,
        key=key,
        acceptance_probability=jnp.array(info.acceptance_probability),
        num_integration_steps=jnp.array(info.num_integration_steps, dtype=jnp.int32),
      )
      jax_io_callback(
        io_callback,
        None,
        output,
      )

    return new_state

  final_state = jax.lax.fori_loop(0, num_samples, body_fn, initial_state)
  return final_state, {}
