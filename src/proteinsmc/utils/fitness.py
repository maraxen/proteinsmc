"""Fitness functions for evaluating sequence populations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

import jax
import jax.numpy as jnp
from jax import jit, vmap

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.utils.types import (
    EvoSequence,
    FitnessWeights,
    FunctionFloats,
    PopulationSequenceFloats,
    PopulationSequences,
    StackedPopulationSequenceFloats,
  )

from .translation import reverse_translate, translate


@dataclass(frozen=True)
class FitnessFunction:
  """Data structure for managing fitness function metadata."""

  func: Callable[[PRNGKeyArray, EvoSequence], PopulationSequenceFloats]
  input_type: Literal["nucleotide", "protein"]
  name: str

  def __post_init__(self) -> None:
    """Validate the fitness function metadata."""
    if not callable(self.func):
      msg = f"Fitness function {self.func} is not callable."
      raise TypeError(msg)
    if self.input_type not in ["nucleotide", "protein"]:
      msg = f"Invalid input_type '{self.input_type}'. Expected 'nucleotide' or 'protein'."
      raise ValueError(
        msg,
      )
    if not isinstance(self.name, str):
      msg = "name must be a string."
      raise TypeError(msg)

  def __hash__(self) -> int:
    """Hash the fitness function based on its properties."""
    return hash((self.func, self.input_type, frozenset(self.name)))

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children = ()
    aux_data = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data: dict, _children: tuple) -> FitnessFunction:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass(frozen=True)
class FitnessEvaluator:
  """Manager for collection of fitness functions."""

  fitness_functions: tuple[FitnessFunction, ...]
  combine_func: Callable[[StackedPopulationSequenceFloats], PopulationSequenceFloats] | None = None

  def __post_init__(self) -> None:
    """Validate the fitness evaluator configuration."""
    if not self.fitness_functions:
      msg = "At least one fitness function must be provided."
      raise ValueError(msg)

  def get_functions_by_type(
    self,
    input_type: Literal["nucleotide", "protein"],
  ) -> list[FitnessFunction]:
    """Get active fitness functions that accept the specified input type."""
    return [f for f in self.fitness_functions if f.input_type == input_type]

  def __hash__(self) -> int:
    """Hash the fitness evaluator based on its properties."""
    return hash(tuple(sorted(self.fitness_functions, key=lambda f: f.name)))

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children = ()
    aux_data = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data: dict, _children: tuple) -> FitnessEvaluator:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


jax.tree_util.register_pytree_node_class(FitnessEvaluator)
jax.tree_util.register_pytree_node_class(FitnessFunction)


def make_fitness_function(
  func: Callable[[PRNGKeyArray, EvoSequence], PopulationSequenceFloats],
  **kwargs: Any,  # noqa: ANN401
) -> Callable[[PRNGKeyArray, EvoSequence], PopulationSequenceFloats]:
  """Create a fitness function with the specified properties.

  Args:
      func: The function to be used as the fitness function.
      **kwargs: Additional keyword arguments for the fitness function.

  Returns:
      A FitnessFunction instance.

  """

  @jit
  def fitness_func(key: PRNGKeyArray, sequences: EvoSequence) -> PopulationSequenceFloats:
    """Call the provided fitness function."""
    return func(key, sequences, **kwargs)

  return fitness_func


def make_combine_fitness_function(
  func: Callable[[StackedPopulationSequenceFloats], FunctionFloats],
  **kwargs: Any,  # noqa: ANN401
) -> Callable:
  """Create a function to combine fitness scores.

  Needed for keeping sampling compatible with JAX's JIT compilation.

  This function allows for flexible combination of fitness scores from
  different fitness functions. It can be used to sum, average, or apply
  any custom combination logic to the fitness scores.

  To use this function, you need to provide a callable `func` that
  takes a dictionary of fitness scores and returns a single combined score.

  The `kwargs` can be used to pass additional parameters to the combining function, but this needs
  to be done before incorporating the function into the `FitnessEvaluator`.

  Args:
      func: The function to combine fitness scores.
      **kwargs: Additional keyword arguments for the combining function.

  Returns:
      A callable that combines fitness scores.

  """

  @jit
  def combine_func(
    fitness_components: StackedPopulationSequenceFloats,
  ) -> StackedPopulationSequenceFloats:
    """Combine individual fitness scores into a single score."""
    return func(fitness_components, **kwargs)

  return combine_func


def _get_seqs(
  sequence_type: Literal["nucleotide", "protein"],
  population: PopulationSequences,
) -> tuple[PopulationSequences, PopulationSequences]:
  """Get nucleotide and amino acid sequences from the population.

  Args:
      sequence_type: Type of sequences in the population, either "nucleotide" or "protein".
      population: Population of sequences, either nucleotide or protein.

  """
  if sequence_type not in ["nucleotide", "protein"]:
    msg = f"Invalid sequence_type '{sequence_type}'. Expected 'nucleotide' or 'protein'."
    raise ValueError(
      msg,
    )

  aa_seqs = None
  nuc_seqs = None
  if sequence_type == "nucleotide":
    nuc_seqs = population
    if population.shape[1] % 3 != 0:
      msg = "Nucleotide sequences must have a length that is a multiple of 3."
      raise ValueError(msg)
    vmapped_translate = vmap(translate, in_axes=(0,))
    aa_seqs, _ = vmapped_translate(population)

  elif sequence_type == "protein":
    aa_seqs = population
    vmapped_reverse_translate = vmap(reverse_translate, in_axes=(0,), out_axes=0)
    nuc_seqs, valid_translation = vmapped_reverse_translate(population)

  return nuc_seqs, aa_seqs


def dispatch_fitness_function(
  key: PRNGKeyArray,
  nuc_seqs: PopulationSequences,
  aa_seqs: PopulationSequences,
  fitness_function: FitnessFunction,
) -> PopulationSequenceFloats:
  """Dispatch a fitness function based on the sequence type."""
  vmapped_func = vmap(
    fitness_function.func,
    in_axes=(None, 0),
  )  # TODO(mar): do we want different keys within the batch?

  if fitness_function.input_type == "nucleotide":
    return vmapped_func(key, nuc_seqs)
  if fitness_function.input_type == "protein":
    return vmapped_func(key, aa_seqs)

  msg = f"Invalid input_type '{fitness_function.input_type}' for '{fitness_function.name}'."
  raise ValueError(msg)


@partial(jit, static_argnames=["sequence_type", "fitness_evaluator"])
def calculate_population_fitness(
  key: PRNGKeyArray,
  population: PopulationSequences,
  sequence_type: Literal["nucleotide", "protein"],
  fitness_evaluator: FitnessEvaluator,
) -> tuple[PopulationSequenceFloats, FunctionFloats]:
  """Calculate fitness for a population using configurable fitness functions.

  Returns:
    Tuple of (combined_fitness, individual_fitness_components)

  """
  nuc_seqs, aa_seqs = _get_seqs(sequence_type, population)
  keys_for_functions = jax.random.split(key, len(fitness_evaluator.fitness_functions))

  results = [
    dispatch_fitness_function(
      func_key,
      nuc_seqs,
      aa_seqs,
      fitness_func,
    )
    for func_key, fitness_func in zip(keys_for_functions, fitness_evaluator.fitness_functions)
  ]

  if fitness_evaluator.combine_func is not None:
    combined_fitness = fitness_evaluator.combine_func(jnp.array(results))
  else:
    combined_fitness = jnp.sum(jnp.array(results), axis=0)

  fitness_components = jnp.stack(results)

  return combined_fitness, fitness_components


@partial(jit)
def combine_fitness_scores(
  fitness_components: FunctionFloats,
  fitness_weights: FitnessWeights | None = None,
) -> PopulationSequenceFloats:
  """Combine individual fitness scores into a single score using weights.

  Args:
      fitness_components: Dictionary of individual fitness scores.
      fitness_weights: Optional weights for combining fitness scores.

  Returns:
      Combined fitness score.

  """
  if fitness_weights is not None:
    combined_fitness = jnp.tensordot(
      fitness_weights,
      fitness_components,
      axes=1,
    )
  else:
    combined_fitness = jnp.sum(fitness_components, axis=0)

  return combined_fitness


def make_sequence_log_prob_fn(
  fitness_evaluator: FitnessEvaluator,
  sequence_type: Literal["protein", "nucleotide"],
) -> Callable:
  """Create a JIT-compiled log-probability function."""

  @jit
  def log_prob_fn(seq: jax.Array) -> jax.Array:
    seq_batch = jnp.atleast_2d(seq)
    fitness_batch, _ = calculate_population_fitness(
      jax.random.PRNGKey(0),
      seq_batch,
      sequence_type,
      fitness_evaluator,
    )
    return fitness_batch[0] if seq.ndim == 1 else fitness_batch

  return log_prob_fn
