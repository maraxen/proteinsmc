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
import jax.numpy as jnp
from jax.experimental import io_callback as jax_io_callback

from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from blackjax.mcmc.nuts import NUTSInfo
  from jaxtyping import Float, Int

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn

__all__ = ["run_nuts_loop"]


@dataclass
class NUTSOutput:
  state: SamplerState
  info: NUTSInfo  # type: ignore[name-defined]


def run_nuts_loop(  # noqa: PLR0913
  num_samples: Int,
  step_size: Float,
  num_doublings: Int,
  initial_state: SamplerState,
  fitness_fn: StackedFitnessFn,
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
      _mutation_fn: Mutation function (unused for NUTS).
      io_callback: Optional callback function for writing outputs.

  Returns:
      A tuple containing the final state and empty metrics dictionary.

  """
  kernel = blackjax.nuts.build_kernel()

  def body_fn(step_idx: int, state: SamplerState) -> SamplerState:
    """Perform one step of the NUTS sampler."""
    key = jax.random.fold_in(state.key, step_idx)
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
      step=jnp.array(step_idx + 1),
    )

    if io_callback is not None:
      output = NUTSOutput(state=new_state, info=info)
      jax_io_callback(
        io_callback,
        None,
        {"state": output.state, "info": output.info},
      )

    return new_state

  final_state = jax.lax.fori_loop(0, num_samples, body_fn, initial_state)
  return final_state, {}
