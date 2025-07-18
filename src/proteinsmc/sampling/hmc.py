"""Implements the Hamiltonian Monte Carlo (HMC) sampling algorithm."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, random

from proteinsmc.models.hmc import HMCConfig, HMCState

if TYPE_CHECKING:
  from jaxtyping import Int

  from proteinsmc.models.fitness import StackedFitnessFuncSignature
  from proteinsmc.models.types import EvoSequence

__all__ = ["initialize_hmc_state", "run_hmc_loop"]


def initialize_hmc_state(config: HMCConfig) -> HMCState:
  """Initialize the state of the HMC sampler.

  Args:
      config: Configuration for the HMC sampler.
      key: JAX PRNG key.

  Returns:
      An initial HMCState.

  """
  key = jax.random.PRNGKey(config.prng_seed)
  initial_samples = jnp.array(config.seed_sequence, dtype=jnp.int32)
  return HMCState(samples=initial_samples, fitness=jnp.array(0.0), key=key)


@partial(jit, static_argnames=("config", "fitness_fn"))
def run_hmc_loop(
  config: HMCConfig,
  initial_state: HMCState,
  fitness_fn: StackedFitnessFuncSignature,
) -> tuple[HMCState, HMCState]:
  """Run the Hamiltonian Monte Carlo (HMC) sampler loop.

  Args:
      config: HMC sampler configuration.
      initial_state: Initial position of the sampler.
      fitness_fn: Fitness function to evaluate sequences.

  Returns:
      A tuple containing the final state and the history of states.

  """
  key = jax.random.PRNGKey(config.prng_seed)

  def leapfrog(
    q: EvoSequence,
    p: EvoSequence,
    config: HMCConfig,
  ) -> tuple[EvoSequence, EvoSequence]:
    """Perform leapfrog integration for HMC."""
    grad_log_prob = jax.grad(fitness_fn)

    def body_fn(
      _i: int,
      carry: tuple[EvoSequence, EvoSequence],
    ) -> tuple[EvoSequence, EvoSequence]:
      q, p = carry
      p_half = p + config.step_size * grad_log_prob(q) / 2.0
      q_new = q + config.step_size * p_half
      p_new = p_half + config.step_size * grad_log_prob(q_new) / 2.0
      return q_new, p_new

    final_q, final_p = jax.lax.fori_loop(
      0,
      config.num_leapfrog_steps,
      body_fn,
      (q, p),
    )
    return final_q, final_p

  def hmc_step(state: HMCState, _i: Int) -> tuple[HMCState, HMCState]:
    """Perform a single HMC step."""
    current_q = state.samples
    current_log_prob = state.fitness

    key_momentum, key_accept, key_next = random.split(key, 3)

    p0 = random.normal(key_momentum, shape=current_q.shape)

    q_new, p_new = leapfrog(current_q, p0, config)

    current_hamiltonian = -current_log_prob + 0.5 * jnp.sum(p0**2)
    proposed_log_prob = fitness_fn(sequence=q_new, _key=key, _context=None)  # type: ignore[arg-type]
    proposed_hamiltonian = -proposed_log_prob + 0.5 * jnp.sum(p_new**2)

    acceptance_ratio = jnp.exp(current_hamiltonian - proposed_hamiltonian)
    accept = random.uniform(key_accept) < acceptance_ratio

    next_q = jnp.where(accept, q_new, current_q)
    next_log_prob = jnp.asarray(
      jnp.where(accept, proposed_log_prob, current_log_prob),
      dtype=jnp.float32,
    )

    next_state = HMCState(samples=next_q, fitness=next_log_prob, key=key_next)
    return next_state, next_state

  final_state, state_history = jax.lax.scan(hmc_step, initial_state, jnp.arange(config.num_samples))

  return final_state, state_history
