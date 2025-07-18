"""Fitness functions for evaluating sequence populations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, PRNGKeyArray

from proteinsmc.scoring import cai, combine, mpnn
from proteinsmc.utils.vmap_utils import chunked_vmap

if TYPE_CHECKING:
  from jaxtyping import Array, PRNGKeyArray

  from proteinsmc.models.fitness import (
    CombineFuncSignature,
    FitnessEvaluator,
    FitnessFuncSignature,
    StackedFitnessFuncSignature,
  )
  from proteinsmc.models.translation import TranslateFuncSignature
  from proteinsmc.models.types import EvoSequence


FITNESS_FUNCTIONS: dict[str, Callable[..., FitnessFuncSignature]] = {
  "cai": cai.make_cai_score,
  "mpnn": mpnn.make_mpnn_score,
}

COMBINE_FUNCTIONS: dict[str, Callable[..., CombineFuncSignature]] = {
  "sum": combine.make_sum_combine,
  "weighted_sum": combine.make_weighted_combine,
}


def get_fitness_function(
  evaluator_config: FitnessEvaluator,
  n_states: int,
  translate_func: TranslateFuncSignature,
  chunk_size: int | None = None,
) -> StackedFitnessFuncSignature:
  """Create a single, JIT-compatible fitness function."""
  score_fns: list[FitnessFuncSignature] = []
  for func_config in evaluator_config.fitness_functions:
    if func_config.name not in FITNESS_FUNCTIONS:
      error_msg = f"Unknown fitness function: {func_config.name}"
      raise ValueError(error_msg)
    make_fn = FITNESS_FUNCTIONS[func_config.name]
    score_fns.append(make_fn(**func_config.kwargs))
  needs_translation = evaluator_config.needs_translation(n_states)

  combine_config = evaluator_config.combine_fn
  if combine_config.name not in COMBINE_FUNCTIONS:
    error_msg = f"Unknown combine function: {combine_config.name}"
    raise ValueError(error_msg)
  make_combine_fn = COMBINE_FUNCTIONS[combine_config.name]
  combine_fn: CombineFuncSignature = make_combine_fn(**combine_config.kwargs)

  @jit
  def final_fitness_fn(
    key: PRNGKeyArray,
    sequence: EvoSequence,
    _context: Array | None = None,
  ) -> tuple[Array, Array]:
    keys = jax.random.split(key, len(score_fns) + 1)

    all_scores = []
    for i, score_fn in enumerate(score_fns):
      sequence = (
        translate_func(sequence=sequence, _key=keys[i], _context=_context)  # type: ignore[call-arg]
        if needs_translation[i]
        else sequence
      )
      if chunk_size is not None:
        vmapped_scorer = chunked_vmap(
          score_fn,
          (sequence, keys[i], _context),
          in_axes=(0, 0, None),
          chunk_size=chunk_size,
        )
      else:
        vmapped_scorer = vmap(score_fn, in_axes=(0, 0, None))
      scores = vmapped_scorer(jax.random.split(keys[i], sequence.shape[0]), sequence, _context)  # type: ignore[arg-type]
      all_scores.append(scores)

    fitness_components = jnp.stack(all_scores, axis=0)

    if chunk_size is not None:
      vmapped_combiner = chunked_vmap(
        combine_fn,
        (keys[-1], fitness_components.T, _context),
        in_axes=(0, 0, None),
        chunk_size=chunk_size,
      )
    else:
      vmapped_combiner = vmap(combine_fn, in_axes=(0, 0, 0))
    combined_fitness = vmapped_combiner(keys[-1], fitness_components.T, _context)  # type: ignore[arg-type]

    return combined_fitness, fitness_components

  return final_fitness_fn
