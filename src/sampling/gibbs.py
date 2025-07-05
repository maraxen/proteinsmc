"""
This module implements the Gibbs sampling algorithm for general and protein sequence models.
"""

from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import jit, random

from ..utils import FitnessEvaluator, calculate_population_fitness
from ..utils.types import (
  EvoSequence,
  PopulationSequences,
  ScalarFloat,
)


@partial(
  jit,
  static_argnames=(
    "fitness_evaluator",
    "evolve_as",
  ),
)
def make_sequence_log_prob_fn(
  fitness_evaluator: FitnessEvaluator, evolve_as: Literal["protein", "nucleotide"]
) -> Callable[[EvoSequence], ScalarFloat]:
  """
  Returns a log-probability function for a sequence array using the fitness evaluator.
  """

  def log_prob_fn(seq: jax.Array) -> jax.Array:
    seq_batch = seq if seq.ndim == 2 else seq[None, :]
    fitness, _ = calculate_population_fitness(
      random.PRNGKey(0), seq_batch, evolve_as, fitness_evaluator
    )
    # Return log-prob as fitness (or sum if batch)
    return fitness[0] if seq.ndim == 1 else fitness

  return log_prob_fn


def make_gibbs_update_fns(
  sequence_length: int, n_states: int, evolve_as: Literal["protein", "nucleotide"]
) -> list[Callable[[jax.Array, EvoSequence, Callable[[EvoSequence], ScalarFloat]], EvoSequence]]:
  """
  Returns a list of update functions for each position in the sequence.
  Each update function samples a new value for one position conditioned on the rest.
  """

  def update_fn_factory(pos):
    def update_fn(key, seq, log_prob_fn):
      # For each possible state at position pos, compute log-prob
      proposals = jnp.tile(seq, (n_states, 1))
      proposals = proposals.at[:, pos].set(jnp.arange(n_states))
      log_probs = jax.vmap(log_prob_fn)(proposals)
      probs = jax.nn.softmax(log_probs)
      new_val = jax.random.choice(key, n_states, p=probs)
      return seq.at[pos].set(new_val)

    return update_fn

  return [update_fn_factory(pos) for pos in range(sequence_length)]


@partial(jit, static_argnames=("num_samples",))
def gibbs_sampler(
  key: jax.Array,
  initial_state: EvoSequence,
  num_samples: int,
  log_prob_fn: Callable[[EvoSequence], ScalarFloat],
  update_fns: list[
    Callable[[jax.Array, EvoSequence, Callable[[EvoSequence], ScalarFloat]], EvoSequence]
  ],
) -> PopulationSequences:
  """
  This function runs the Gibbs sampler.

  Args:
      key: JAX PRNG key.
      initial_state: Initial state of the sampler.
      num_samples: Number of samples to generate.
      log_prob_fn: Log probability function of the target distribution.
      update_fns: List of functions, each updating one component of the state.

  Returns:
      Array of samples.
  """

  def body_fn(i, state_and_samples):
    current_state, samples = state_and_samples

    new_state = current_state
    for j, update_fn in enumerate(update_fns):
      key_comp, _ = random.split(random.fold_in(key, i * len(update_fns) + j))
      new_state = update_fn(key_comp, new_state, log_prob_fn)

    samples = samples.at[i].set(new_state)
    return new_state, samples

  samples = jnp.zeros((num_samples,) + initial_state.shape, dtype=initial_state.dtype)
  _, final_samples = jax.lax.fori_loop(0, num_samples, body_fn, (initial_state, samples))

  return final_samples
