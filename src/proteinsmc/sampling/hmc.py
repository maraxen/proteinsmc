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
  config,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  io_callback: Callable,
  **kwargs,
) -> tuple[SamplerState, dict]:
  """Run the Hamiltonian Monte Carlo (HMC) sampler loop.
  Args:
      config: HMC sampler configuration.
      initial_state: Initial state of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      io_callback: Callback function for writing outputs.
      **kwargs: Additional keyword arguments.
  Returns:
      A tuple containing the final state and a dictionary of metrics.
  """
  kernel = blackjax.hmc.build_kernel()

  def one_step(i, state):
    """Perform one step of the HMC sampler."""
    key, kernel_key, next_key = jax.random.split(state.key, 3)
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
      step=i,
    )

    payload = {
      "sequence": new_state.sequence,
      "fitness": new_state.fitness,
      "step": new_state.step,
      "acceptance_rate": hmc_info.acceptance_rate,
      "divergence": hmc_info.is_divergent,
    }
    io_callback(payload)
    return new_state

  final_state = jax.lax.fori_loop(0, config.num_samples, one_step, initial_state)
  return final_state, {}
