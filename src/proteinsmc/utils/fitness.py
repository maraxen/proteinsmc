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

      return jax.lax.map(
        score_fn,
        (seq, keys_i, _context),
        batch_size=batch_size,
      )

    return branch_fn

  score_branches = jax.lax.map(make_score_branch, (score_fns, needs_translation_tuple))

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

    combined_fitness = jax.vmap(combine_fn)(all_scores.T, keys_for_combiner, _context)

    return jnp.stack(
      [combined_fitness, all_scores],
      axis=0,
    )

  return final_fitness_fn
