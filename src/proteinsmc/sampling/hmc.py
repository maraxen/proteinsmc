"""Implements the Hamiltonian Monte Carlo (HMC) sampling algorithm."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import blackjax
import jax
import jax.numpy as jnp
from jax.experimental import io_callback as jax_io_callback

from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from blackjax.mcmc.hmc import HMCInfo

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn


__all__ = ["run_hmc_loop"]


@dataclasses.dataclass
class HMCOutput:
  state: SamplerState
  info: HMCInfo


def run_hmc_loop(
  num_samples: int,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  _mutation_fn: MutationFn | None = None,
  io_callback: Callable | None = None,
) -> tuple[SamplerState, dict[str, jax.Array]]:
  """Run the Hamiltonian Monte Carlo (HMC) sampler loop.

  Args:
      num_samples: Number of samples to draw.
      initial_state: Initial sequence of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      _mutation_fn: Mutation function to generate new sequences (unused for HMC).
      io_callback: Optional callback function for writing outputs.

  Returns:
      A tuple containing the final state and empty metrics dictionary.

  """
  kernel = blackjax.hmc.build_kernel()

  def body_fn(step_idx: int, state: SamplerState) -> SamplerState:
    """Perform one step of the HMC sampler."""
    kernel_key = jax.random.fold_in(state.key, step_idx)
    new_blackjax_state, hmc_info = kernel(
      rng_key=kernel_key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
    )
    new_state = SamplerState(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=kernel_key,
      blackjax_state=new_blackjax_state,
      step=jnp.array(step_idx + 1),
    )

    if io_callback is not None:
      output = HMCOutput(state=new_state, info=hmc_info)
      jax_io_callback(
        io_callback,
        None,
        {"state": output.state, "info": output.info},
      )

    return new_state

  final_state = jax.lax.fori_loop(0, num_samples, body_fn, initial_state)
  return final_state, {}
