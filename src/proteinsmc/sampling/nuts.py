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
import jax.numpy as jnp
from jax import jit

from proteinsmc.models.nuts import NUTSConfig, NUTSState
from proteinsmc.utils.initiate import generate_template_population

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFn
  from proteinsmc.models.mutation import MutationFn

__all__ = ["initialize_nuts_state", "run_nuts_loop"]


def initialize_nuts_state(
  config: NUTSConfig,
  fitness_fn: StackedFitnessFn,
  key: PRNGKeyArray,
) -> NUTSState:
  """Initialize the state of the NUTS sampler.

  Args:
      config: Configuration for the NUTS sampler.
      fitness_fn: Fitness function to evaluate sequences.
      key: JAX PRNG key.

  Returns:
      An initial NUTSState.

  """
  initial_sequence = generate_template_population(
    initial_sequence=config.seed_sequence,
    population_size=1,
    input_sequence_type=config.sequence_type,
    output_sequence_type=config.sequence_type,
  )
  blackjax_initial_state = blackjax.nuts.init(
    initial_sequence,
    fitness_fn,
    step_size=config.step_size,
    max_tree_depth=config.max_num_doublings,
  )
  return NUTSState(
    sequence=initial_sequence,
    fitness=jnp.array(blackjax_initial_state.logdensity, dtype=jnp.float32),
    key=key,
    blackjax_state=blackjax_initial_state,
  )


@partial(jit, static_argnames=("config", "fitness_fn"))
def run_nuts_loop(
  config: NUTSConfig,
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
      step_size=config.step_size,
      max_num_doublings=config.max_num_doublings,
    )
    new_state = state.replace(
      sequence=new_blackjax_state.position,
      fitness=new_blackjax_state.logdensity,
      key=key,
      blackjax_state=new_blackjax_state,
    )
    return new_state, new_state

  keys = jax.random.split(initial_state.key, config.num_samples)
  final_state, state_history = jax.lax.scan(one_step, initial_state, keys)
  return final_state, state_history
