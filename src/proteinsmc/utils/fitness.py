"""Fitness functions for evaluating sequence populations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, PRNGKeyArray

from proteinsmc.scoring import cai, combine, mpnn
from proteinsmc.utils.translation import reverse_translate, translate

if TYPE_CHECKING:
  from jaxtyping import Array, Int, PRNGKeyArray

  from proteinsmc.models.fitness import CombineFuncSignature, FitnessEvaluator, FitnessFuncSignature
  from proteinsmc.models.types import EvoSequence, NucleotideSequence, ProteinSequence


FITNESS_FUNCTIONS: dict[str, Callable[..., FitnessFuncSignature]] = {
  "cai": cai.make_cai_score,
  "mpnn": mpnn.make_mpnn_score,
}

COMBINE_FUNCTIONS: dict[str, Callable[..., CombineFuncSignature]] = {
  "sum": combine.make_sum_combine,
  "weighted_sum": combine.make_weighted_combine,
}


@partial(jit, static_argnames=("sequence_type",))
def _get_sequences(
  sequences: EvoSequence,
  sequence_type: str,
) -> tuple[NucleotideSequence, ProteinSequence]:
  """Extract nucleotide and amino acid sequences from the population."""
  if sequence_type == "nucleotide":
    vmapped_translate = vmap(translate, in_axes=(0,))
    aa_seqs = vmapped_translate(sequences)
    return sequences, aa_seqs
  if sequence_type == "amino_acid":
    vmapped_reverse_translate = vmap(reverse_translate, in_axes=(0,))
    nuc_seqs = vmapped_reverse_translate(sequences)
    return nuc_seqs, sequences
  msg = f"Unknown sequence type: {sequence_type}"
  raise ValueError(msg)


def get_fitness_function(
  evaluator_config: FitnessEvaluator,
  chunk_size: int | None = None,
) -> Callable:
  """Create a single, JIT-compatible fitness function."""
  score_fns: list[FitnessFuncSignature] = []
  for func_config in evaluator_config.fitness_functions:
    if func_config.name not in FITNESS_FUNCTIONS:
      error_msg = f"Unknown fitness function: {func_config.name}"
      raise ValueError(error_msg)
    make_fn = FITNESS_FUNCTIONS[func_config.name]
    score_fns.append(make_fn(**func_config.kwargs))
  score_fns_nucleotide = [
    evaluator_config.fitness_functions[i].input_type == "nucleotide"
    for i in range(len(evaluator_config.fitness_functions))
  ]

  combine_config = evaluator_config.combine_fn
  if combine_config.name not in COMBINE_FUNCTIONS:
    error_msg = f"Unknown combine function: {combine_config.name}"
    raise ValueError(error_msg)
  make_combine_fn = COMBINE_FUNCTIONS[combine_config.name]
  combine_fn: CombineFuncSignature = make_combine_fn(**combine_config.kwargs)

  @partial(jit, static_argnames=("sequence_type",))
  def final_fitness_fn(
    key: PRNGKeyArray,
    sequence: EvoSequence,
    sequence_type: str,
    _context: Array | None = None,
  ) -> tuple[Array, Array]:
    nuc_seq, aa_seq = _get_sequences(sequence, sequence_type)
    keys = jax.random.split(key, len(score_fns) + 1)

    all_scores = []
    for i, score_fn in enumerate(score_fns):
      sequence = nuc_seq if score_fns_nucleotide[i] else aa_seq
      vmapped_scorer = vmap(score_fn, in_axes=(0, 0, None))
      scores = vmapped_scorer(jax.random.split(keys[i], sequence.shape[0]), sequence, _context)  # type: ignore[arg-type]
      all_scores.append(scores)

    fitness_components = jnp.stack(all_scores, axis=0)

    vmapped_combiner = vmap(combine_fn, in_axes=(0, 0, 0))
    combined_fitness = vmapped_combiner(keys[-1], fitness_components.T, _context)  # type: ignore[arg-type]

    return combined_fitness, fitness_components

  @partial(jit, static_argnames=("sequence_type",))
  def chunked_final_fitness_fn(
    key: PRNGKeyArray,
    sequence: EvoSequence,
    sequence_type: str,
    _context: Array | None = None,
  ) -> tuple[Array, Array]:
    """Chunked version of the final fitness function."""
    if chunk_size is None:
      msg = "chunk_size must be specified for chunked_final_fitness_fn."
      raise ValueError(msg)

    num_sequences = sequence.shape[0]
    num_chunks = (num_sequences + chunk_size - 1) // chunk_size

    def body_fun(
      i: Int,
      carry: tuple[PRNGKeyArray, Array, Array],
    ) -> tuple[PRNGKeyArray, Array, Array]:
      key, out_fitness, out_components = carry
      start = i * chunk_size
      end = jnp.minimum((i + 1) * chunk_size, num_sequences)
      seq_chunk = jax.tree_util.tree_map(lambda x: x[start:end], sequence)
      key_chunk, key = jax.random.split(key)
      fitness, components = final_fitness_fn(key_chunk, seq_chunk, sequence_type, _context)
      out_fitness = out_fitness.at[start:end].set(fitness)
      out_components = out_components.at[:, start:end].set(components)
      return (key, out_fitness, out_components)

    # Preallocate output arrays
    dummy_fitness, dummy_components = final_fitness_fn(
      key,
      jax.tree_util.tree_map(lambda x: x[:1], sequence),
      sequence_type,
      _context,
    )
    out_fitness = jnp.zeros((num_sequences,) + dummy_fitness.shape[1:], dtype=dummy_fitness.dtype)
    out_components = jnp.zeros(
      (dummy_components.shape[0], num_sequences) + dummy_components.shape[2:],
      dtype=dummy_components.dtype,
    )

    init_carry = (key, out_fitness, out_components)
    final_carry = jax.lax.fori_loop(0, num_chunks, body_fun, init_carry)
    _, out_fitness, out_components = final_carry
    return out_fitness, out_components

  if chunk_size is not None:
    return chunked_final_fitness_fn
  return final_fitness_fn
