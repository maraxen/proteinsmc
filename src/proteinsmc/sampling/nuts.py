"""A simplified implementation of the No-U-Turn Sampler (NUTS).

Note: A full, robust NUTS implementation is complex and typically relies on
advanced numerical methods and tree-building algorithms. This is a conceptual
placeholder for demonstration purposes and will not be a complete, production-ready
NUTS sampler.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, random

from proteinsmc.models.nuts import NUTSConfig, NUTSState

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFuncSignature
  from proteinsmc.models.types import EvoSequence

__all__ = ["initialize_nuts_state", "run_nuts_loop"]


def initialize_nuts_state(config: NUTSConfig) -> NUTSState:
  """Initialize the state of the NUTS sampler.

  Args:
      config: Configuration for the NUTS sampler.
      key: JAX PRNG key.

  Returns:
      An initial NUTSState.

  """
  initial_samples = jnp.array(config.seed_sequence, dtype=jnp.int32)
  return NUTSState(
    samples=initial_samples,
    fitness=jnp.array(0.0),
    key=jax.random.PRNGKey(config.prng_seed),
  )


@partial(jit, static_argnames=("config", "fitness_fn"))
def run_nuts_loop(
  config: NUTSConfig,
  initial_state: NUTSState,
  fitness_fn: StackedFitnessFuncSignature,
  key: PRNGKeyArray,
) -> tuple[NUTSState, NUTSState]:
  """Run a simplified conceptual NUTS sampler loop (placeholder).

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

  def leapfrog(
    current_q: EvoSequence,
    current_p: EvoSequence,
    fitness_fn: StackedFitnessFuncSignature,
    step_size: float,
    key: PRNGKeyArray,
  ) -> tuple[EvoSequence, EvoSequence]:
    """Perform a single leapfrog step."""
    grad_log_prob = jax.grad(lambda x: fitness_fn(x, key, None))  # type: ignore[arg-type]

    p_half = current_p + step_size * grad_log_prob(current_q) / 2.0
    next_q = current_q + step_size * p_half
    next_p = p_half + step_size * grad_log_prob(next_q) / 2.0

    return next_q, next_p

  def nuts_step(state: NUTSState, _i: Int) -> tuple[NUTSState, NUTSState]:
    """Perform a single NUTS step."""
    current_q = state.samples
    current_log_prob = state.fitness

    key_momentum, key_leapfrog, key_accept, key_next = random.split(state.key, 4)

    p0 = random.normal(key_momentum, shape=current_q.shape)

    q_new, p_new = current_q, p0

    for _i in range(config.num_leapfrog_steps):
      q_new, p_new = leapfrog(q_new, p_new, fitness_fn, config.step_size, key_leapfrog)

    proposed_log_prob = fitness_fn(q_new, key_leapfrog, None)  # type: ignore[arg-type]

    current_hamiltonian = -current_log_prob + 0.5 * jnp.sum(p0**2)
    proposed_hamiltonian = -proposed_log_prob + 0.5 * jnp.sum(p_new**2)

    acceptance_ratio = jnp.exp(current_hamiltonian - proposed_hamiltonian)
    accept = random.uniform(key_accept) < acceptance_ratio

    next_q = jnp.where(accept, q_new, current_q)
    next_log_prob = jnp.array(
      jnp.where(accept, proposed_log_prob, current_log_prob),
      dtype=jnp.float32,
    )

    next_state = NUTSState(samples=next_q, fitness=next_log_prob, key=key_next)
    return next_state, next_state

  def scan_body(state: NUTSState, i: Int) -> tuple[NUTSState, NUTSState]:
    """Scan body function for the NUTS loop."""
    step_key = random.fold_in(key, i)
    next_state, history_state = nuts_step(state, step_key)
    return next_state, history_state

  # Initialize with proper fitness evaluation
  key_init, key_loop = random.split(key)
  initial_fitness = fitness_fn(initial_state.samples, key_init, None)  # type: ignore[arg-type]
  state_with_fitness = initial_state.replace(fitness=initial_fitness)

  final_state, state_history = jax.lax.scan(
    scan_body,
    state_with_fitness,
    jnp.arange(config.num_samples),
  )

  return final_state, state_history
