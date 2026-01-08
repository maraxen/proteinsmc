"""Fitness functions for evaluating sequence populations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit

from proteinsmc.scoring import cai, combine, esm, mpnn, nk

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Int, PRNGKeyArray

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
  "nk": nk.make_nk_score,
}

COMBINE_FUNCTIONS: dict[str, Callable[..., CombineFn]] = {
  "sum": combine.make_sum_combine,
  "weighted_sum": combine.make_weighted_combine,
}


def get_fitness_function(
  evaluator_config: FitnessEvaluator,
  n_states: int | Int,
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

    def branch_fn(args: tuple[PRNGKeyArray, EvoSequence, Array | None]) -> Array:
      """Branch function for computing scores."""
      key_i, sequence, _context = args
      if needs_trans:
        seq, _ = translate_func(sequence, key_i, _context)
      else:
        seq = sequence
      # Call score_fn directly on the single sequence (outer vmap handles population)
      return score_fn(key_i, jnp.asarray(seq), _context)

    return branch_fn

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

    # Optimize for single fitness function case (no switch needed)
    if len(score_fns) == 1:
      all_scores = score_branches[0]((keys[0], sequence, _context))[jnp.newaxis]
    else:

      def compute_score(i: Array | int) -> Array:
        """Compute scores for the i-th fitness function."""
        return jax.lax.switch(i, score_branches, (keys[i], sequence, _context))

      # If batch_size is None, use vmap instead of lax.map to avoid dynamic shape issues
      if batch_size is None:
        all_scores = jax.vmap(compute_score)(jnp.arange(len(score_fns)))
      else:
        all_scores = jax.lax.map(compute_score, jnp.arange(len(score_fns)), batch_size=batch_size)

    # Combine the scores from different fitness functions
    # all_scores has shape (n_fitness_functions,) for a single sequence
    combined_fitness = combine_fn(all_scores, keys[-1], _context)

    # Return stacked: [combined, score1, score2, ...]
    return jnp.concatenate([jnp.array([combined_fitness]), all_scores])

  return final_fitness_fn
