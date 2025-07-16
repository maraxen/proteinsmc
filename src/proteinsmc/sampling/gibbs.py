"""Implements the Gibbs sampling algorithm for general and protein sequence models."""

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
class GibbsSamplerOutput:
  """Data structure to hold the output of the Gibbs sampler.

  Attributes:
      samples: Array of sampled sequences.
      final_fitness: Fitness value of the final state.

  """

  samples: EvoSequence
  final_fitness: Float

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children = (self.samples, self.final_fitness)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, _aux_data: dict, children: tuple) -> GibbsSamplerOutput:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      samples=children[0],
      final_fitness=children[1],
    )


jax.tree_util.register_pytree_node_class(GibbsSamplerOutput)


def make_gibbs_update_fns(
  sequence_length: int,
  n_states: int,
) -> tuple[
  Callable[
    [jax.Array, EvoSequence, Callable[[PRNGKeyArray, EvoSequence], Float]],
    EvoSequence,
  ],
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
  ) -> Callable[
    [jax.Array, EvoSequence, Callable[[PRNGKeyArray, EvoSequence], Float]],
    EvoSequence,
  ]:
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


@partial(jit, static_argnames=("num_samples", "log_prob_fn", "update_fns"))
def gibbs_sampler(
  key: PRNGKeyArray,
  initial_state: EvoSequence,
  num_samples: int,
  log_prob_fn: Callable[[PRNGKeyArray, EvoSequence], Float],
  update_fns: tuple[
    Callable[
      [jax.Array, EvoSequence, Callable[[PRNGKeyArray, EvoSequence], Float]],
      EvoSequence,
    ],
    ...,
  ],
) -> GibbsSamplerOutput:
  """Run the Gibbs sampler.

  Args:
      key: JAX PRNG key.
      initial_state: Initial state of the sampler.
      num_samples: Number of samples to generate.
      log_prob_fn: Log probability function of the target distribution.
      update_fns: Tuple of update functions, each updating one component of the state.

  Returns:
      Array of samples of shape (num_samples, sequence_length).

  Raises:
      ValueError: If num_samples is not positive.

  """
  if num_samples <= 0:
    msg = "num_samples must be positive."
    raise ValueError(msg)

  def body_fn(
    i: int,
    state_and_samples: tuple[EvoSequence, EvoSequence, Float],
  ) -> tuple[EvoSequence, EvoSequence, Float]:
    current_state, samples, fitness = state_and_samples

    new_state = current_state
    for j, update_fn in enumerate(update_fns):
      key_comp, _ = random.split(random.fold_in(key, i * len(update_fns) + j))
      new_state = update_fn(key_comp, new_state, log_prob_fn)

    samples = samples.at[i].set(new_state)
    return (
      new_state,
      samples,
      jnp.where(
        i == 0,
        log_prob_fn(key, new_state),
        fitness,
      ),
    )

  samples = jnp.zeros((num_samples, *initial_state.shape), dtype=initial_state.dtype)
  _, samples, final_fitness = jax.lax.fori_loop(
    0,
    num_samples,
    body_fn,
    (initial_state, samples, 0.0),
  )

  return samples
