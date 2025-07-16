"""Implements the Hamiltonian Monte Carlo (HMC) sampling algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import jit, random

if TYPE_CHECKING:
  from jaxtyping import Float, PRNGKeyArray

  from proteinsmc.models.types import EvoSequence


@dataclass(frozen=True)
class HMCSamplerConfig:
  """Configuration for Hamiltonian Monte Carlo (HMC) sampler."""

  step_size: float
  num_leapfrog_steps: int
  num_samples: int
  log_prob_fn: Callable[[EvoSequence], Float]

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[HMCSamplerConfig],
    aux_data: dict,
    _children: tuple,
  ) -> HMCSamplerConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


jax.tree_util.register_pytree_node_class(HMCSamplerConfig)


@partial(
  jit,
  static_argnames=("config",),
)
def hmc_sampler(
  key: PRNGKeyArray,
  initial_position: EvoSequence,
  config: HMCSamplerConfig,
) -> EvoSequence:
  """Run the Hamiltonian Monte Carlo (HMC) sampler.

  Args:
      key: JAX PRNG key.
      initial_position: Initial position of the sampler.
      num_samples: Number of samples to generate.
      log_prob_fn: Log probability function of the target distribution.
      config: HMC sampler configuration (step size, leapfrog steps).

  Returns:
      Array of samples of shape (num_samples, sequence_length).

  Raises:
      ValueError: If num_samples or num_leapfrog_steps is not positive.

  """
  if config.num_samples <= 0:
    msg = "num_samples must be positive."
    raise ValueError(msg)
  if config.num_leapfrog_steps <= 0:
    msg = "num_leapfrog_steps must be positive."
    raise ValueError(msg)

  def leapfrog(
    q: EvoSequence,
    p: EvoSequence,
    config: HMCSamplerConfig,
  ) -> tuple[EvoSequence, EvoSequence]:
    """Perform leapfrog integration for HMC.

    Args:
        q: Current position.
        p: Current momentum.
        log_prob_fn: Log probability function.
        config: HMC sampler configuration (step size, leapfrog steps).

    Returns:
        Tuple of (final position, final momentum).

    """
    grad_log_prob = jax.grad(config.log_prob_fn)

    def body_fn(
      _i: int,
      carry: tuple[EvoSequence, EvoSequence],
    ) -> tuple[EvoSequence, EvoSequence]:
      q, p = carry
      p_half = p + config.step_size * grad_log_prob(q) / 2.0
      q_new = q + config.step_size * p_half
      p_new = p_half + config.step_size * grad_log_prob(q_new) / 2.0
      return q_new, p_new

    final_q, final_p = jax.lax.fori_loop(0, config.num_leapfrog_steps, body_fn, (q, p))
    return final_q, final_p

  def hmc_step(
    carry: tuple[EvoSequence, Float, PRNGKeyArray],
    _: None,
  ) -> tuple[tuple[EvoSequence, Float, PRNGKeyArray], EvoSequence]:
    """Perform a single HMC step.

    Args:
        carry: Tuple of (current position, current log prob, PRNG key).
        _: Dummy argument for scan.

    Returns:
        Updated carry and the new sample.

    """
    current_q, current_log_prob, key = carry

    key_momentum, key_leapfrog, key_accept = random.split(key, 3)

    p0 = random.normal(key_momentum, shape=current_q.shape)

    q_new, p_new = leapfrog(current_q, p0, config)

    current_hamiltonian = -current_log_prob + 0.5 * jnp.sum(p0**2)
    proposed_log_prob = config.log_prob_fn(q_new)
    proposed_hamiltonian = -proposed_log_prob + 0.5 * jnp.sum(p_new**2)

    acceptance_ratio = jnp.exp(current_hamiltonian - proposed_hamiltonian)
    accept = random.uniform(key_accept) < acceptance_ratio

    next_q = jnp.where(accept, q_new, current_q)
    next_log_prob = jnp.where(accept, proposed_log_prob, current_log_prob)

    return (next_q, next_log_prob, key_accept), next_q

  initial_log_prob = config.log_prob_fn(initial_position)

  _, samples = jax.lax.scan(
    hmc_step,
    (initial_position, initial_log_prob, key),
    None,
    length=config.num_samples,
  )

  return samples
