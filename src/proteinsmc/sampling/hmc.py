"""Implements the Hamiltonian Monte Carlo (HMC) sampling algorithm."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import blackjax
import jax
from blackjax.mcmc.hmc import HMCInfo
from jax.experimental import io_callback

from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import PRNGKeyArray

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
  writer_callback: Callable | None = None,
) -> tuple[SamplerState, SamplerState]:
  """Run the Hamiltonian Monte Carlo (HMC) sampler loop.

  Args:
      config: HMC sampler configuration.
      num_samples: Number of samples to draw.
      initial_state: Initial sequence of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      mutation_fn: Mutation function to generate new sequences.
      writer_callback: Optional callback function for writing outputs.

  Returns:
      A tuple containing the final state and the history of states.

  """
  kernel = blackjax.hmc.build_kernel()

  def one_step(state: SamplerState, key: PRNGKeyArray) -> tuple[SamplerState, SamplerState]:
    """Perform one step of the HMC sampler."""
    kernel_key, next_key = jax.random.split(key)
    new_blackjax_state, hmc_info = kernel(
      rng_key=kernel_key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
    )
    new_state = SamplerState(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=next_key,
      blackjax_state=new_blackjax_state,
      step=state.step + 1,
    )
    if writer_callback is not None:
      io_callback(
        writer_callback,
        new_state,
        result_shape=SamplerState,
      )
      io_callback(
        writer_callback,
        hmc_info,
        result_shape=HMCInfo,
      )
    return new_state, new_state

  keys = jax.random.split(initial_state.key, num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
