"""Implements the Hamiltonian Monte Carlo (HMC) sampling algorithm."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import blackjax
import jax
import jax.numpy as jnp
from blackjax.mcmc.hmc import HMCState
from jax.experimental import io_callback as jax_io_callback

from proteinsmc.models.sampler_base import SamplerOutput, SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn


__all__ = ["run_hmc_loop"]


def run_hmc_loop(  # noqa: PLR0913
  num_samples: int,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  step_size: float = 0.1,
  num_integration_steps: int = 10,
  inverse_mass_matrix: jax.Array | None = None,
  _mutation_fn: MutationFn | None = None,
  io_callback: Callable | None = None,
) -> tuple[SamplerState, dict[str, jax.Array]]:
  """Run the Hamiltonian Monte Carlo (HMC) sampler loop.

  Args:
      num_samples: Number of samples to draw.
      initial_state: Initial sequence of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      step_size: Step size for the integrator.
      num_integration_steps: Number of integration steps to take.
      inverse_mass_matrix: Inverse mass matrix for HMC (optional).
      _mutation_fn: Mutation function to generate new sequences (unused for HMC).
      io_callback: Optional callback function for writing outputs.

  Returns:
      A tuple containing the final state and empty metrics dictionary.

  """

  # Create a wrapper that converts fitness_fn to blackjax's expected signature
  # Blackjax expects a function that takes only position and returns (logdensity, logdensity_grad)
  def logdensity_fn_wrapped(position: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Convert fitness_fn to blackjax's expected signature."""
    # fitness_fn returns (combined_fitness, components_fitness)
    fitness = fitness_fn(initial_state.key, position, None)
    return fitness[0], fitness  # Return combined fitness and full fitness for auxiliary data

  # Initialize blackjax HMC state if not already present
  if initial_state.blackjax_state is None:
    # Compute initial logdensity and gradient
    initial_logdensity, _ = logdensity_fn_wrapped(initial_state.sequence)
    logdensity_grad = jax.grad(lambda seq: logdensity_fn_wrapped(seq)[0])(initial_state.sequence)

    hmc_state = HMCState(
      position=initial_state.sequence,
      logdensity=float(initial_logdensity),  # Combined fitness
      logdensity_grad=logdensity_grad,
    )
    # Create a new state with the blackjax_state set
    initial_state = dataclasses.replace(initial_state, blackjax_state=hmc_state)

  kernel = blackjax.hmc.build_kernel()

  def body_fn(step_idx: int, state: SamplerState) -> SamplerState:
    """Perform one step of the HMC sampler."""
    kernel_key = jax.random.fold_in(state.key, step_idx)

    # Create default inverse mass matrix if not provided
    mass_matrix = inverse_mass_matrix
    if mass_matrix is None:
      # Use identity matrix with dimension matching the sequence
      dim = state.sequence.size
      mass_matrix = jnp.eye(dim)

    # Call kernel with required parameters
    # Note: we need to wrap fitness_fn for each step with the current key
    def logdensity_fn_step(position: jax.Array) -> jax.Array:
      """Logdensity function for this HMC step."""
      fitness = fitness_fn(kernel_key, position, None)
      return fitness[0]  # Return combined fitness as JAX array

    new_blackjax_state, hmc_info = kernel(
      rng_key=kernel_key,
      state=state.blackjax_state,
      logdensity_fn=logdensity_fn_step,
      step_size=step_size,
      num_integration_steps=num_integration_steps,
      inverse_mass_matrix=mass_matrix,
    )

    # Recompute full fitness to maintain proper shape
    new_fitness = fitness_fn(kernel_key, new_blackjax_state.position, None)

    new_state = SamplerState(
      sequence=new_blackjax_state.position,
      key=kernel_key,
      blackjax_state=new_blackjax_state,
      step=jnp.array(step_idx + 1),
    )

    if io_callback is not None:
      output = SamplerOutput(
        step=jnp.array(step_idx + 1, dtype=jnp.int32),
        sequences=new_blackjax_state.position,
        fitness=new_fitness,
        key=kernel_key,
        acceptance_probability=jnp.array(hmc_info.acceptance_probability),
        num_integration_steps=jnp.array(num_integration_steps, dtype=jnp.int32),
      )
      jax_io_callback(
        io_callback,
        None,
        output,
      )

    return new_state

  final_state = jax.lax.fori_loop(0, num_samples, body_fn, initial_state)
  return final_state, {}
