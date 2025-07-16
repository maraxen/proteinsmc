"""Fitness functions for evaluating sequence populations."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax.numpy as jnp
from jax import jit, vmap

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from proteinsmc.models.fitness import (
    FitnessEvaluator,
    FitnessFunction,
  )
  from proteinsmc.models.types import (
    EvoSequence,
    NucleotideSequence,
    ProteinSequence,
  )


from proteinsmc.scoring.registry import COMBINE_REGISTRY, FITNESS_REGISTRY

from .translation import reverse_translate, translate
from .vmap_utils import chunked_vmap


def _get_seqs(
  nuc_seq: NucleotideSequence | None,
  aa_seq: ProteinSequence | None,
) -> tuple[NucleotideSequence, ProteinSequence]:
  """Get nucleotide and amino acid sequences for a sequence or population.

  Args:
      nuc_seq: JAX array of nucleotide sequences (integer encoded).
      aa_seq: JAX array of amino acid sequences (integer encoded in ColabDesign's scheme).

  """
  if nuc_seq is not None and aa_seq is not None:
    return nuc_seq, aa_seq

  if nuc_seq is None and aa_seq is None:
    msg = "Either nucleotide or amino acid sequence must be provided."
    raise ValueError(msg)

  if nuc_seq is not None:
    if nuc_seq.shape[1] % 3 != 0:
      msg = "Nucleotide sequences must have a length that is a multiple of 3."
      raise ValueError(msg)
    vmapped_translate = vmap(translate, in_axes=(0,))
    aa_seq, _ = vmapped_translate(nuc_seq)

  elif aa_seq is not None:
    vmapped_reverse_translate = vmap(reverse_translate, in_axes=(0,), out_axes=0)
    nuc_seq, _ = vmapped_reverse_translate(aa_seq)

  if not isinstance(nuc_seq, jax.Array) or not isinstance(aa_seq, jax.Array):
    msg = "Nucleotide and amino acid sequences must be JAX arrays."
    raise TypeError(msg)

  return nuc_seq, aa_seq


def dispatch_fitness_function(
  key: PRNGKeyArray,
  nuc_seq: NucleotideSequence,
  aa_seq: ProteinSequence,
  fitness_function: FitnessFunction,
  _context: Array | None = None,
  fitness_args: dict[str, Any] | None = None,
) -> Float:
  """Dispatch a fitness function based on the sequence type."""
  if fitness_function.input_type not in ["nucleotide", "protein"]:
    msg = f"Invalid input_type '{fitness_function.input_type}' for '{fitness_function.func}'."
    raise ValueError(msg)

  if fitness_args is None:
    fitness_args = {}

  runtime_fitness_func = fitness_function(registry=FITNESS_REGISTRY, **fitness_args)
  vmapped_func = vmap(
    runtime_fitness_func,
    in_axes=(fitness_function.key_split, 0, *fitness_function.context_tuple),
  )

  if fitness_function.input_type == "nucleotide":
    return vmapped_func(key, nuc_seq, _context)  # type: ignore[call-arg]
  if fitness_function.input_type == "protein":
    return vmapped_func(key, aa_seq, _context)  # type: ignore[call-arg]

  msg = f"Invalid input_type '{fitness_function.input_type}' for '{fitness_function.func}'."
  raise ValueError(msg)


@partial(
  jit,
  static_argnames=["sequence_type", "fitness_evaluator", "fitness_kwargs", "combine_kwargs"],
)
def calculate_fitness(
  key: PRNGKeyArray,
  sequence: EvoSequence,
  sequence_type: Literal["nucleotide", "protein"],
  fitness_evaluator: FitnessEvaluator,
  _context: Array | None = None,
  fitness_kwargs: dict[str, Any] | None = None,
  combine_kwargs: dict[str, Any] | None = None,
) -> tuple[Float, Float]:
  """Calculate fitness for a population using configurable fitness functions.

  Returns:
    Tuple of (combined_fitness, individual_fitness_components)

  """
  if fitness_kwargs is None:
    fitness_kwargs = {}
  if combine_kwargs is None:
    combine_kwargs = {}

  nuc_seq, aa_seq = _get_seqs(
    nuc_seq=None if sequence_type == "protein" else sequence,
    aa_seq=None if sequence_type == "nucleotide" else sequence,
  )
  keys_for_functions = jax.random.split(key, len(fitness_evaluator.fitness_functions))

  results = [
    dispatch_fitness_function(
      func_key,
      nuc_seq,
      aa_seq,
      fitness_func,
      _context,
      fitness_kwargs,
    )
    for func_key, fitness_func in zip(keys_for_functions, fitness_evaluator.fitness_functions)
  ]

  if fitness_evaluator.combine_func is not None:
    combined_fitness = fitness_evaluator.combine_func(registry=COMBINE_REGISTRY, **combine_kwargs)(
      jnp.array(results),
    )
  else:
    combined_fitness = jnp.sum(jnp.array(results), axis=0)

  fitness_components = jnp.stack(results)

  return combined_fitness, fitness_components


def make_sequence_log_prob_fn(
  fitness_evaluator: FitnessEvaluator,
  sequence_type: Literal["protein", "nucleotide"],
) -> Callable:
  """Create a JIT-compiled log-probability function."""

  @jit
  def log_prob_fn(seq: jax.Array) -> jax.Array:
    seq_batch = jnp.atleast_2d(seq)
    fitness_batch, _ = calculate_fitness(
      jax.random.PRNGKey(0),
      seq_batch,
      sequence_type,
      fitness_evaluator,
    )
    return fitness_batch[0] if seq.ndim == 1 else fitness_batch

  return log_prob_fn


def chunked_calculate_population_fitness(
  key: PRNGKeyArray,
  population: EvoSequence,
  fitness_evaluator: FitnessEvaluator,
  sequence_type: Literal["nucleotide", "protein"],
  _context: Array | None = None,
  fitness_kwargs: dict[str, Any] | None = None,
  combine_kwargs: dict[str, Any] | None = None,
  chunk_size: int = 32,
) -> tuple[Array, Array]:
  """Calculate population fitness in chunks using the general chunked_vmap utility."""
  keys = jax.random.split(key, population.shape[0])

  def make_calculate_fitness(
    key: PRNGKeyArray,
    seq: EvoSequence,
    _context: Array | None = None,
  ) -> tuple[Float, Float]:
    """Calculate fitness for a single sequence."""
    return calculate_fitness(
      key,
      seq,
      sequence_type,
      fitness_evaluator,
      _context,
      fitness_kwargs,
      combine_kwargs,
    )

  fitness_values, components = chunked_vmap(
    func=make_calculate_fitness,
    data=(
      keys,
      population,
      _context,
    ),
    chunk_size=chunk_size,
    in_axes=(0, 0, 0),
  )

  return fitness_values, components.T
