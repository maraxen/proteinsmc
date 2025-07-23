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
}

COMBINE_FUNCTIONS: dict[str, Callable[..., CombineFn]] = {
  "sum": combine.make_sum_combine,
  "weighted_sum": combine.make_weighted_combine,
}


def get_fitness_function(
  evaluator_config: FitnessEvaluator,
  n_states: int,
  translate_func: TranslateFuncSignature,
  chunk_size: int | None = None,
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

  combine_config = evaluator_config.combine_fn
  if combine_config.name not in COMBINE_FUNCTIONS:
    error_msg = f"Unknown combine function: {combine_config.name}"
    raise ValueError(error_msg)
  make_combine_fn = COMBINE_FUNCTIONS[combine_config.name]
  combine_fn: CombineFn = make_combine_fn(**combine_config.kwargs)

  @jit
  def final_fitness_fn(
    key: PRNGKeyArray,
    sequence: EvoSequence,
    _context: Array | None = None,
  ) -> Array:
    keys = jax.random.split(key, len(score_fns) + 1)

    def score_body(
      i: int,
      carry: tuple[list[Array], EvoSequence],
    ) -> tuple[list[Array], EvoSequence]:
      all_scores, sequence = carry
      score_fn = score_fns[i]
      seq = (
        translate_func(sequence=sequence, _key=keys[i], _context=_context)  # type: ignore[call-arg]
        if needs_translation[i]
        else sequence
      )
      if chunk_size is not None:
        vmapped_scorer = chunked_vmap(
          score_fn,
          (seq, keys[i], _context),
          in_axes=(0, 0, None),
          chunk_size=chunk_size,
        )
      else:
        vmapped_scorer = vmap(score_fn, in_axes=(0, 0, None))
      scores = vmapped_scorer(jax.random.split(keys[i], seq.shape[0]), seq, _context)  # type: ignore[arg-type]
      all_scores = [*all_scores, scores]
      return all_scores, sequence

    all_scores, _ = jax.lax.fori_loop(
      0,
      len(score_fns),
      score_body,
      ([], sequence),
    )

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

    if (
      _context is not None
    ):  # TODO(mar): assume _context is our beta and fitness is logprob, this should be reworked
      combined_fitness = combined_fitness * _context

    return jnp.stack(
      [combined_fitness, *fitness_components],
      axis=0,
    )

  return final_fitness_fn
