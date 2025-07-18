"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import jit, random

from proteinsmc.models.mcmc import MCMCConfig, MCMCState

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from proteinsmc.models.fitness import StackedFitnessFuncSignature
  from proteinsmc.models.types import EvoSequence

__all__ = ["initialize_mcmc_state", "make_random_mutation_proposal_fn", "run_mcmc_loop"]


def initialize_mcmc_state(config: MCMCConfig) -> MCMCState:
  """Initialize the state of the MCMC sampler."""
  key = jax.random.PRNGKey(config.prng_seed)
  initial_samples = jnp.array(config.seed_sequence, dtype=jnp.int32)
  return MCMCState(samples=initial_samples, fitness=jnp.array(0.0), key=key)


def make_random_mutation_proposal_fn(
  n_states: int,
) -> Callable[[PRNGKeyArray, EvoSequence], EvoSequence]:
  """Create a proposal function that generates a new sequence by randomly mutating one position.

  Args:
      n_states: The number of possible states (e.g., amino acids) for each position in the sequence.

  Returns:
      A function that takes a PRNG key and an EvoSequence, and returns a new EvoSequence with one
      position mutated.

  """

  @jit
  def proposal_fn(key: PRNGKeyArray, seq: EvoSequence) -> EvoSequence:
    seq_new = seq.copy()
    pos = random.randint(key, (), 0, seq.shape[0])
    new_val = random.randint(key, (), 0, n_states)
    return seq_new.at[pos].set(new_val)

  return proposal_fn


@partial(jit, static_argnames=("config", "fitness_fn", "proposal_fn"))
def run_mcmc_loop(
  config: MCMCConfig,
  initial_state: MCMCState,
  fitness_fn: StackedFitnessFuncSignature,
  proposal_fn: Callable[[PRNGKeyArray, EvoSequence], EvoSequence],
) -> tuple[MCMCState, MCMCState]:
  """Run the Metropolis-Hastings MCMC sampler loop.

  Args:
      config: Configuration for the MCMC sampler.
      initial_state: Initial state of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      proposal_fn: Proposal function to generate new states.

  Returns:
      A tuple containing the final state and the history of states.

  """

  def body_fn(state: MCMCState, _i: Int) -> tuple[MCMCState, MCMCState]:
    current_state = state.samples

    key_proposal, key_accept, key_log_prob, key_next = random.split(state.key, 4)

    proposed_state = proposal_fn(key_proposal, current_state)

    current_log_prob = fitness_fn(current_state, key_log_prob, _context=None)  # type: ignore[arg-type]
    proposed_log_prob = fitness_fn(proposed_state, key_log_prob, _context=None)  # type: ignore[arg-type]

    acceptance_ratio = jnp.exp(proposed_log_prob - current_log_prob)

    accept = random.uniform(key_accept) < acceptance_ratio

    next_state = jnp.where(accept, proposed_state, current_state)
    fitness = jnp.where(accept, proposed_log_prob, current_log_prob)

    next_mcmc_state = MCMCState(samples=next_state, fitness=fitness, key=key_next)
    return next_mcmc_state, next_mcmc_state

  final_state, state_history = jax.lax.scan(body_fn, initial_state, jnp.arange(config.num_samples))

  return final_state, state_history
