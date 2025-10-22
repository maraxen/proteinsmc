"""Fitness functions for evaluating sequence populations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, PRNGKeyArray

from proteinsmc.scoring import cai, combine, esm, mpnn

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, PRNGKeyArray

  from proteinsmc.models.fitness import (
    CombineFn,
    FitnessEvaluator,
    FitnessFn,
    StackedFitnessFn,
  )
  from proteinsmc.models.translation import TranslateFuncSignature
  from proteinsmc.models.types import EvoSequence


FITNESS_FUNCTIONS: dict[str, Callable[..., FitnessFn]] = {
  "cai": cai.make_cai_score,
  "mpnn": mpnn.make_mpnn_score,
  "esm": esm.make_esm_score,
}

COMBINE_FUNCTIONS: dict[str, Callable[..., CombineFn]] = {
  "sum": combine.make_sum_combine,
  "weighted_sum": combine.make_weighted_combine,
}


def get_fitness_function(
  evaluator_config: FitnessEvaluator,
  n_states: int,
  translate_func: TranslateFuncSignature,
  batch_size: int | None = None,
) -> StackedFitnessFn:
  """Create a single, JIT-compatible fitness function."""
  score_fns: list[FitnessFn] = []
  for func_config in evaluator_config.fitness_functions:
    if func_config.name not in FITNESS_FUNCTIONS:
      error_msg = f"Unknown fitness function: {func_config.name}"
      raise ValueError(error_msg)
    make_fn = FITNESS_FUNCTIONS[func_config.name]
    score_fns.append(make_fn(**func_config.kwargs))
  needs_translation = evaluator_config.needs_translation(n_states)

  needs_translation_tuple = tuple(bool(x) for x in needs_translation)

  combine_config = evaluator_config.combine_fn
  if combine_config.name not in COMBINE_FUNCTIONS:
    error_msg = f"Unknown combine function: {combine_config.name}"
    raise ValueError(error_msg)
  make_combine_fn = COMBINE_FUNCTIONS[combine_config.name]
  combine_fn: CombineFn = make_combine_fn(**combine_config.kwargs)

  def make_score_branch(score_fn: FitnessFn, needs_trans: bool) -> Callable:  # noqa: FBT001
    """Create a branch function for a specific score function index."""

    def branch_fn(args: tuple[EvoSequence, PRNGKeyArray, Array | None]) -> Array:
      """Branch function for computing scores."""
      sequence, key_i, _context = args
      seq = translate_func(sequence, key_i, _context) if needs_trans else sequence
      keys_i = jax.random.split(key_i, seq.shape[0])

      # Use vmap to vectorize over the population dimension
      # score_fn expects (sequence, key, context) per individual
      # If _context is None, pass None for each individual
      # Otherwise, broadcast context to all individuals
      if batch_size is None:
        # No chunking - use vmap
        vmapped_score = jax.vmap(lambda s, k: score_fn(s, k, _context))
        return vmapped_score(seq, keys_i)

      # With chunking - use lax.map over vmapped function
      def score_single(s: EvoSequence, k: PRNGKeyArray) -> Array:
        return score_fn(s, k, _context)

      return jax.lax.map(
        lambda inputs: score_single(inputs[0], inputs[1]),
        (seq, keys_i),
        batch_size=batch_size,
      )

    return branch_fn

  # Build score branches using list comprehension instead of jax.lax.map
  score_branches = [
    make_score_branch(score_fn, needs_trans)
    for score_fn, needs_trans in zip(score_fns, needs_translation_tuple, strict=True)
  ]

  @jit
  def final_fitness_fn(
    key: PRNGKeyArray,
    sequence: EvoSequence,
    _context: Array | None = None,
  ) -> Array:
    keys = jax.random.split(key, len(score_fns) + 1)

    def compute_score(i: int) -> Array:
      """Compute scores for the i-th fitness function."""
      return jax.lax.switch(i, score_branches, (sequence, keys[i], _context))

    all_scores = jax.lax.map(compute_score, jnp.arange(len(score_fns)), batch_size=batch_size)

    keys_for_combiner = jax.random.split(keys[-1], all_scores.shape[1])

    # Vmap combine_fn over population, but broadcast _context
    # in_axes=(0, 0, None) means vmap over first two args, broadcast third
    combined_fitness = jax.vmap(combine_fn, in_axes=(0, 0, None))(
      all_scores.T, keys_for_combiner, _context
    )

    # Reshape combined_fitness to (1, population_size) for concatenation
    # all_scores has shape (n_fitness_functions, population_size)
    # Result should be (1 + n_fitness_functions, population_size)
    combined_fitness_reshaped = jnp.expand_dims(combined_fitness, axis=0)

    return jnp.concatenate(
      [combined_fitness_reshaped, all_scores],
      axis=0,
    )

  return final_fitness_fn
