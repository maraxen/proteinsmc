"""Implements the Gibbs sampling algorithm for general and protein sequence models."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, random

from proteinsmc.models.gibbs import GibbsConfig, GibbsUpdateFn
from proteinsmc.models.sampler_base import SamplerState

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.models.types import EvoSequence
  from proteinsmc.utils.fitness import FitnessFn

__all__ = ["initialize_gibbs_state", "make_gibbs_update_fns", "run_gibbs_loop"]


def initialize_gibbs_state(config: GibbsConfig) -> SamplerState:
  """Initialize the state of the Gibbs sampler."""
  key = jax.random.PRNGKey(config.prng_seed)
  initial_samples = jnp.array(config.seed_sequence, dtype=jnp.int8)
  return SamplerState(
    sequence=initial_samples,
    fitness=jnp.array(0.0),
    key=key,
  )


def make_gibbs_update_fns(
  sequence_length: int,
  n_states: int,
) -> tuple[
  GibbsUpdateFn,
  ...,
]:
  """Return update functions for each sequence position.

  Each update function samples a new value for one position conditioned on the rest.

  Args:
      sequence_length: Length of the sequence.
      n_states: Number of possible states per position.

  Returns:
      Tuple of update functions, one for each sequence position.

  Raises:
      ValueError: If sequence_length or n_states is not positive.

  """
  if sequence_length <= 0 or n_states <= 0:
    msg = "sequence_length and n_states must be positive."
    raise ValueError(msg)

  def update_fn_factory(
    pos: int,
  ) -> GibbsUpdateFn:
    def update_fn(
      key: PRNGKeyArray,
      seq: EvoSequence,
      log_prob_fn: Callable[[PRNGKeyArray, EvoSequence], Float],
    ) -> EvoSequence:
      proposal_key, new_val_key = random.split(key)
      proposal_keys = random.split(random.fold_in(proposal_key, pos), n_states)
      proposals = jnp.tile(seq, (n_states, 1))
      proposals = proposals.at[:, pos].set(jnp.arange(n_states))
      log_probs = jax.vmap(log_prob_fn)(proposal_keys, proposals)
      probs = jax.nn.softmax(log_probs)
      new_val = jax.random.choice(new_val_key, n_states, p=probs)
      return seq.at[pos].set(new_val)

    return update_fn  # type: ignore[return-value]

  update_fns = ()
  for pos in range(sequence_length):
    update_fns += (update_fn_factory(pos),)
  return update_fns  # type: ignore[return-value]


@partial(jit, static_argnames=("config", "fitness_fn", "update_fns"))
def run_gibbs_loop(
  config: GibbsConfig,
  initial_state: SamplerState,
  fitness_fn: FitnessFn,
  update_fns: tuple[
    GibbsUpdateFn,
    ...,
  ],
) -> tuple[SamplerState, SamplerState]:
  """Run the Gibbs sampler loop.

  Args:
      config: Configuration for the Gibbs sampler.
      initial_state: Initial state of the sampler.
      fitness_fn: Fitness function to evaluate sequences.
      update_fns: Tuple of update functions, each updating one component of the state.

  Returns:
      A tuple containing the final state and the history of states.

  """

  def body_fn(state: SamplerState, _i: Int) -> tuple[SamplerState, SamplerState]:
    current_state = state.sequence

    new_state = current_state
    for j, update_fn in enumerate(update_fns):
      key_comp, _ = random.split(random.fold_in(state.key, j))
      new_state = update_fn(key_comp, new_state, fitness_fn)  # type: ignore[call-arg]

    fitness = fitness_fn(
      _key=state.key,  # type: ignore[arg-type]
      sequence=new_state,
      _context=None,
    )
    _, key_next = random.split(state.key)
    next_state = SamplerState(sequence=new_state, fitness=fitness, key=key_next)
    return next_state, next_state

  final_state, state_history = jax.lax.scan(body_fn, initial_state, jnp.arange(config.num_samples))
  return final_state, state_history
