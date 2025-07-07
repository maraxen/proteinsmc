"""A simplified implementation of the No-U-Turn Sampler (NUTS).

Note: A full, robust NUTS implementation is complex and typically relies on
advanced numerical methods and tree-building algorithms. This is a conceptual
placeholder for demonstration purposes and will not be a complete, production-ready
NUTS sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import jit, random

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.utils.types import (
    EvoSequence,
    PopulationSequences,
    ScalarFloat,
  )


@dataclass(frozen=True)
class NUTSConfig:
  """Configuration for the NUTS sampler.

  Attributes:
      log_prob_fn: Function to compute the log probability of a sequence.
      step_size: Integration step size for leapfrog.
      num_samples: Number of samples to generate after warmup.
      warmup_steps: Number of warmup steps to adapt the step size.
      num_chains: Number of parallel chains to run.
      num_leapfrog_steps: Number of leapfrog steps to take.
      adapt_step_size: Whether to adapt the step size during warmup.

  """

  log_prob_fn: Callable[[EvoSequence], ScalarFloat]
  step_size: float
  warmup_steps: int
  num_samples: int
  num_chains: int
  num_leapfrog_steps: int
  adapt_step_size: bool = True

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[NUTSConfig],
    aux_data: dict,
    _children: tuple,
  ) -> NUTSConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


jax.tree_util.register_pytree_node_class(NUTSConfig)


@partial(
  jit,
  static_argnames=("config",),
)
def nuts_sampler(
  key: PRNGKeyArray,
  initial_position: EvoSequence,
  config: NUTSConfig,
) -> PopulationSequences:
  """Run a simplified conceptual NUTS sampler (placeholder).

  This function demonstrates the basic idea of NUTS but lacks the full
  adaptivity and tree-building of a true NUTS implementation.

  Args:
      key: JAX PRNG key.
      initial_position: Initial position of the sampler.
      config: Configuration for the NUTS sampler.

  Returns:
      Array of samples.

  """

  def leapfrog(
    current_q: EvoSequence,
    current_p: EvoSequence,
    log_prob_fn: Callable[[EvoSequence], ScalarFloat],
    step_size: float,
  ) -> tuple[EvoSequence, EvoSequence]:
    """Perform a single leapfrog step.

    Args:
        current_q: Current position.
        current_p: Current momentum.
        log_prob_fn: Log probability function.
        step_size: Integration step size for leapfrog.

    Returns:
        Tuple of (next position, next momentum).

    """
    grad_log_prob = jax.grad(log_prob_fn)

    p_half = current_p + step_size * grad_log_prob(current_q) / 2.0

    next_q = current_q + step_size * p_half

    next_p = p_half + step_size * grad_log_prob(next_q) / 2.0

    return next_q, next_p

  def nuts_step(
    carry: tuple[EvoSequence, EvoSequence, ScalarFloat, PRNGKeyArray],
    _: None,
  ) -> tuple[tuple[EvoSequence, EvoSequence, ScalarFloat, PRNGKeyArray], EvoSequence]:
    """Perform a single NUTS step.

    Args:
        carry: Tuple containing the current position, momentum, log probability,
                and PRNG key.
        _: Unused placeholder for the scan loop.

    Returns:
        Tuple of (next carry, next position).

    """
    current_q, current_p, current_log_prob, key = carry

    key_momentum, key_accept, key_nuts = random.split(key, 3)

    p0 = random.normal(key_momentum, shape=current_q.shape)

    q_new, p_new = current_q, p0

    for _i in range(config.num_leapfrog_steps):
      q_new, p_new = leapfrog(q_new, p_new, config.log_prob_fn, config.step_size)

    proposed_log_prob = config.log_prob_fn(q_new)

    current_hamiltonian = -current_log_prob + 0.5 * jnp.sum(current_p**2)
    proposed_hamiltonian = -proposed_log_prob + 0.5 * jnp.sum(p_new**2)

    acceptance_ratio = jnp.exp(current_hamiltonian - proposed_hamiltonian)
    accept = random.uniform(key_accept) < acceptance_ratio

    next_q = jnp.where(accept, q_new, current_q)
    next_log_prob = jnp.where(accept, proposed_log_prob, current_log_prob)

    return (next_q, p0, next_log_prob, key_nuts), next_q

  def run_chain(
    key: PRNGKeyArray,
    initial_position: EvoSequence,
  ) -> jax.Array:
    initial_log_prob = config.log_prob_fn(initial_position)
    initial_momentum = random.normal(key, shape=initial_position.shape)

    _, samples = jax.lax.scan(
      nuts_step,
      (initial_position, initial_momentum, initial_log_prob, key),
      None,
      length=config.num_samples + config.warmup_steps,
    )
    return samples[config.warmup_steps :]

  keys = random.split(key, config.num_chains)
  if config.num_chains > 1:
    initial_positions = jnp.tile(initial_position, (config.num_chains, 1))
    samples = jax.vmap(run_chain)(keys, initial_positions)
  else:
    samples = run_chain(key, initial_position)[None, ...]

  return samples
