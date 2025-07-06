"""
This module provides a simplified implementation of the No-U-Turn Sampler (NUTS).

Note: A full, robust NUTS implementation is complex and typically relies on
advanced numerical methods and tree-building algorithms. This is a conceptual
placeholder for demonstration purposes and will not be a complete, production-ready
NUTS sampler.
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit, random
from jaxtyping import PRNGKeyArray

from proteinsmc.utils.types import (
  EvoSequence,
  PopulationSequences,
  ScalarFloat,
)


@partial(
  jit,
  static_argnames=(
    "log_prob_fn",
    "num_samples",
    "warmup_steps",
    "num_chains",
    "step_size",
    "adapt_step_size",
    "num_leapfrog_steps",
  ),
)
def nuts_sampler(
  key: PRNGKeyArray,
  log_prob_fn: Callable[[EvoSequence], ScalarFloat],
  initial_position: EvoSequence,
  num_samples: int,
  warmup_steps: int,
  num_chains: int = 1,
  step_size: float = 0.1,
  adapt_step_size: bool = True,  # TODO(mar): Implement step size adaptation
  num_leapfrog_steps: int = 10,
) -> PopulationSequences:
  """
  A simplified conceptual NUTS sampler (placeholder).

  This function demonstrates the basic idea of NUTS but lacks the full
  adaptivity and tree-building of a true NUTS implementation.

  Args:
      key: JAX PRNG key.
      log_prob_fn: Log probability function of the target distribution.
      initial_position: Initial position of the sampler.
      num_samples: Number of samples to generate.
      warmup_steps: Number of warmup steps.
      num_chains: Number of parallel chains to run.
      step_size: Integration step size for leapfrog.
      adapt_step_size: Whether to adapt the step size during warmup.
      num_leapfrog_steps: Number of leapfrog steps to take.

  Returns:
      Array of samples.
  """

  def leapfrog(current_q, current_p, log_prob_fn, step_size):
    grad_log_prob = jax.grad(log_prob_fn)

    p_half = current_p + step_size * grad_log_prob(current_q) / 2.0

    next_q = current_q + step_size * p_half

    next_p = p_half + step_size * grad_log_prob(next_q) / 2.0

    return next_q, next_p

  def nuts_step(carry, _):
    current_q, current_p, current_log_prob, key = carry

    key_momentum, key_accept, key_nuts = random.split(key, 3)

    p0 = random.normal(key_momentum, shape=current_q.shape)

    q_new, p_new = current_q, p0
    for _ in range(num_leapfrog_steps):
      q_new, p_new = leapfrog(q_new, p_new, log_prob_fn, step_size)

    proposed_log_prob = log_prob_fn(q_new)

    current_hamiltonian = -current_log_prob + 0.5 * jnp.sum(current_p**2)
    proposed_hamiltonian = -proposed_log_prob + 0.5 * jnp.sum(p_new**2)

    acceptance_ratio = jnp.exp(current_hamiltonian - proposed_hamiltonian)
    accept = random.uniform(key_accept) < acceptance_ratio

    next_q = jnp.where(accept, q_new, current_q)
    next_log_prob = jnp.where(accept, proposed_log_prob, current_log_prob)

    return (next_q, p0, next_log_prob, key_nuts), next_q

  def run_chain(key, initial_position):
    initial_log_prob = log_prob_fn(initial_position)
    initial_momentum = random.normal(key, shape=initial_position.shape)

    _, samples = jax.lax.scan(
      nuts_step,
      (initial_position, initial_momentum, initial_log_prob, key),
      None,
      length=num_samples + warmup_steps,
    )
    return samples[warmup_steps:]

  keys = random.split(key, num_chains)
  if num_chains > 1:
    initial_positions = jnp.tile(initial_position, (num_chains, 1))
    samples = jax.vmap(run_chain)(keys, initial_positions)
  else:
    samples = run_chain(key, initial_position)[None, ...]

  return samples
