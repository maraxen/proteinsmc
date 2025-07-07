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
    FitnessWeights,
    PopulationSequenceFloats,
    PopulationSequences,
  )

from .translation import reverse_translate, translate


@dataclass(frozen=True)
class FitnessFunction:
  """Data structure for managing fitness function metadata."""

  func: Callable
  input_type: Literal["nucleotide", "protein"]
  args: dict[str, Any]
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

  def __hash__(self) -> int:
    """Hash the fitness function based on its properties."""
    return hash((self.func, self.input_type, frozenset(self.args.items()), self.name))

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

  fitness_functions: list[FitnessFunction]
  combine_func: Callable | None = None
  combine_func_args: dict[str, Any] | None = None

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


@partial(jit, static_argnames=["sequence_type", "fitness_evaluator"])
def calculate_population_fitness(
  key: PRNGKeyArray,
  population: PopulationSequences,
  sequence_type: Literal["nucleotide", "protein"],
  fitness_evaluator: FitnessEvaluator,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Calculate fitness for a population using configurable fitness functions.

  Returns:
    Tuple of (combined_fitness, individual_fitness_components)

  """
  nuc_seqs, aa_seqs = _get_seqs(sequence_type, population)
  fitness_components = {}
  main_scores = {}
  keys_for_functions = jax.random.split(key, len(fitness_evaluator.fitness_functions))

  for func_key, fitness_func in zip(keys_for_functions, fitness_evaluator.fitness_functions):
    if fitness_func.input_type == "nucleotide":
      input_seqs = nuc_seqs
    elif fitness_func.input_type == "protein":
      input_seqs = aa_seqs
    else:
      msg = f"Invalid input_type '{fitness_func.input_type}' for fitness function"
      f"'{fitness_func.name}'."
      raise ValueError(
        msg,
      )

    def create_fitness_evaluation(func: Callable, args: dict[str, Any]) -> Callable:
      def single_fitness_evaluation(seq_key: PRNGKeyArray, seq: jax.Array) -> jax.Array:
        return func(seq_key, seq, **args)

      return single_fitness_evaluation

    evaluation_func = create_fitness_evaluation(fitness_func.func, fitness_func.args)
    vmapped_fitness = vmap(evaluation_func, in_axes=(0, 0))
    func_keys = jax.random.split(func_key, population.shape[0])

    result = vmapped_fitness(func_keys, input_seqs)
    main_scores[fitness_func.name] = result

  if fitness_evaluator.combine_func is not None:
    combine_args = fitness_evaluator.combine_func_args or {}
    combined_fitness = fitness_evaluator.combine_func(main_scores, **combine_args)
  else:
    combined_fitness = jnp.sum(jnp.array(list(main_scores.values())), axis=0)

  fitness_components = dict(main_scores.items())

  return combined_fitness, fitness_components


@partial(jit)
def combine_fitness_scores(
  fitness_components: dict,
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
      jnp.array(list(fitness_components.values())),
      axes=1,
    )
  else:
    combined_fitness = jnp.sum(jnp.array(list(fitness_components.values())), axis=0)

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
