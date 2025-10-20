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
  num_samples: Int,
  step_size: Float,
  num_doublings: Int,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
  _mutation_fn: MutationFn | None = None,
  writer_callback: Callable | None = None,
) -> tuple[SamplerState, SamplerState]:
  """Run a NUTS sampling loop.

  This function demonstrates the basic idea of NUTS but lacks the full
  adaptivity and tree-building of a true NUTS implementation.

  Args:
      config: Configuration for the NUTS sampler.
      initial_state: Initial position of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      key: JAX PRNG key for random number generation.

  Returns:
      A tuple containing the final state and the history of states.

  """
  kernel = blackjax.nuts.build_kernel()

  def one_step(state: SamplerState, key: PRNGKeyArray) -> tuple[SamplerState, SamplerState]:
    """Perform one step of the NUTS sampler."""
    new_blackjax_state, info = kernel(
      rng_key=key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
      step_size=step_size,
      max_num_doublings=num_doublings,
    )
    new_state = SamplerState(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=key,
      blackjax_state=new_blackjax_state,
    )

    if writer_callback:
      writer_callback(NUTSOutput(state=new_state, info=info))

    return new_state, new_state

  keys = jax.random.split(initial_state.key, num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
