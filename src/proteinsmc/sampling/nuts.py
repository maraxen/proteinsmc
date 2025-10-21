"""A simplified implementation of the No-U-Turn Sampler (NUTS).

Note: A full, robust NUTS implementation is complex and typically relies on
advanced numerical methods and tree-building algorithms. This is a conceptual
placeholder for demonstration purposes and will not be a complete, production-ready
NUTS sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import blackjax
import jax
from blackjax.mcmc.nuts import NUTSInfo
from jax.experimental import io_callback

from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn

__all__ = ["run_nuts_loop"]


@dataclass
class NUTSOutput:
  state: SamplerState
  info: NUTSInfo


def run_nuts_loop(
  config,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  io_callback: Callable,
  **kwargs,
) -> tuple[SamplerState, dict]:
  """Run a NUTS sampling loop.
  Args:
      config: Configuration for the NUTS sampler.
      initial_state: Initial position of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      io_callback: Callback function for writing outputs.
      **kwargs: Additional keyword arguments.
  Returns:
      A tuple containing the final state and the history of states.
  """
  kernel = blackjax.nuts.build_kernel()

  def one_step(i, state):
    """Perform one step of the NUTS sampler."""
    key, kernel_key, next_key = jax.random.split(state.key, 3)
    new_blackjax_state, nuts_info = kernel(
      rng_key=kernel_key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
      step_size=config.step_size,
      max_num_doublings=config.num_doublings,
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
      "acceptance_rate": nuts_info.acceptance_rate,
      "divergence": nuts_info.is_divergent,
    }
    io_callback(payload)
    return new_state

  final_state = jax.lax.fori_loop(0, config.num_samples, one_step, initial_state)
  return final_state, {}
