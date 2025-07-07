"""Implements Metropolis-Hastings MCMC sampling algorithm."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import jit, random

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.utils.types import EvoSequence, PopulationSequences, ScalarFloat


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


@partial(jit, static_argnames=("num_samples", "log_prob_fn", "proposal_fn"))
def mcmc_sampler(
  key: PRNGKeyArray,
  initial_state: EvoSequence,
  num_samples: int,
  log_prob_fn: Callable[[EvoSequence], ScalarFloat],
  proposal_fn: Callable[[PRNGKeyArray, EvoSequence], EvoSequence],
) -> PopulationSequences:
  """Run the Metropolis-Hastings MCMC sampler.

  Args:
      key: JAX PRNG key.
      initial_state: Initial state of the sampler.
      num_samples: Number of samples to generate.
      log_prob_fn: Log probability function of the target distribution.
      proposal_fn: Proposal function to generate new states.

  Returns:
      Array of samples.

  """

  def body_fn(
    i: int,
    state_and_samples: tuple[EvoSequence, PopulationSequences],
  ) -> tuple[EvoSequence, PopulationSequences]:
    current_state, samples = state_and_samples

    key_proposal, key_accept = random.split(random.fold_in(key, i))

    proposed_state = proposal_fn(key_proposal, current_state)

    current_log_prob = log_prob_fn(current_state)
    proposed_log_prob = log_prob_fn(proposed_state)

    acceptance_ratio = jnp.exp(proposed_log_prob - current_log_prob)

    accept = random.uniform(key_accept) < acceptance_ratio

    next_state = jnp.where(accept, proposed_state, current_state)
    samples = samples.at[i].set(next_state)

    return next_state, samples

  samples = jnp.zeros((num_samples, *initial_state.shape))
  _, final_samples = jax.lax.fori_loop(0, num_samples, body_fn, (initial_state, samples))

  return final_samples
