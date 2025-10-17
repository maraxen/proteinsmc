"""A simplified implementation of the No-U-Turn Sampler (NUTS).

Note: A full, robust NUTS implementation is complex and typically relies on
advanced numerical methods and tree-building algorithms. This is a conceptual
placeholder for demonstration purposes and will not be a complete, production-ready
NUTS sampler.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import blackjax
import jax
from jax import jit

from proteinsmc.models.nuts import NUTSState
from proteinsmc.utils.config_unpacker import with_config

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn

__all__ = ["run_nuts_loop"]


@with_config
@partial(jit, static_argnames=("num_samples", "step_size", "num_doublings", "fitness_fn"))
def run_nuts_loop(
  num_samples: int,
  step_size: float,
  num_doublings: int,
  initial_state: NUTSState,
  fitness_fn: StackedFitnessFn,
  _mutation_fn: MutationFn | None = None,
) -> tuple[NUTSState, NUTSState]:
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

  def one_step(state: NUTSState, key: PRNGKeyArray) -> tuple[NUTSState, NUTSState]:
    """Perform one step of the NUTS sampler."""
    new_blackjax_state, _ = kernel(
      rng_key=key,
      state=state.blackjax_state,
      logdensity_fn=fitness_fn,
      step_size=step_size,
      max_num_doublings=num_doublings,
    )
    new_state = NUTSState(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=key,
      blackjax_state=new_blackjax_state,
    )
    return new_state, new_state

  keys = jax.random.split(initial_state.key, num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
