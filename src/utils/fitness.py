from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import PRNGKeyArray

from .translate import translate
from .types import (
  FitnessWeights,
  PopulationSequenceFloats,
  PopulationSequences,
)


@dataclass
class FitnessFunction:
  """Data structure for managing fitness function metadata."""

  func: Callable
  input_type: Literal["nucleotide", "protein"]
  args: dict[str, Any]
  name: str
  is_active: bool = True

  def __post_init__(self):
    if not callable(self.func):
      raise ValueError(f"Fitness function {self.func} is not callable.")
    if self.input_type not in ["nucleotide", "protein"]:
      raise ValueError(
        f"Invalid input_type '{self.input_type}'. Expected 'nucleotide' or 'protein'."
      )


@dataclass
class FitnessEvaluator:
  """Manager for collection of fitness functions."""

  fitness_functions: list[FitnessFunction]
  combine_func: Callable | None = None
  combine_func_args: dict[str, Any] | None = None

  def __post_init__(self):
    if not self.fitness_functions:
      raise ValueError("At least one fitness function must be provided.")

  def get_active_functions(self) -> list[FitnessFunction]:
    """Get only the active fitness functions."""
    return [f for f in self.fitness_functions if f.is_active]

  def get_functions_by_type(
    self, input_type: Literal["nucleotide", "protein"]
  ) -> list[FitnessFunction]:
    """Get active fitness functions that accept the specified input type."""
    return [f for f in self.get_active_functions() if f.input_type == input_type]


def calculate_population_fitness(
  key: PRNGKeyArray,
  population: PopulationSequences,
  sequence_type: Literal["nucleotide", "protein"],
  fitness_evaluator: FitnessEvaluator,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Calculate fitness for a population using configurable fitness functions.

  Args:
    key: PRNG key for random operations
    population: Population of sequences to evaluate
    sequence_type: Type of sequences in population ("nucleotide" or "protein")
    fitness_evaluator: FitnessEvaluator containing fitness functions and configuration

  Returns:
    Tuple of (combined_fitness, individual_fitness_components)
  """
  if sequence_type not in ["nucleotide", "protein"]:
    raise ValueError(
      f"Invalid sequence_type '{sequence_type}'. Expected 'nucleotide' or 'protein'."
    )

  aa_seqs = None
  has_x_flags = None
  if sequence_type == "nucleotide":
    if population.shape[1] % 3 != 0:
      raise ValueError("Nucleotide sequences must have a length that is a multiple of 3.")
    vmapped_translate = vmap(translate, in_axes=(0,))
    aa_seqs, has_x_flags = vmapped_translate(population)

  fitness_components = {}
  active_functions = fitness_evaluator.get_active_functions()
  keys_for_functions = jax.random.split(key, len(active_functions))

  for func_key, fitness_func in zip(keys_for_functions, active_functions):
    if fitness_func.input_type == "nucleotide":
      if sequence_type != "nucleotide":
        raise ValueError(
          f"Function {fitness_func.name} requires nucleotide input but got {sequence_type}"
        )
      input_seqs = population
    else:
      if sequence_type == "nucleotide":
        input_seqs = aa_seqs
      else:
        input_seqs = population

    def create_fitness_evaluation(func: Callable, args: dict[str, Any]) -> Callable:
      def single_fitness_evaluation(seq_key: PRNGKeyArray, seq: jax.Array) -> jax.Array:
        return func(seq_key, seq, **args)

      return single_fitness_evaluation

    evaluation_func = create_fitness_evaluation(fitness_func.func, fitness_func.args)
    vmapped_fitness = vmap(evaluation_func, in_axes=(0, 0))
    func_keys = jax.random.split(func_key, population.shape[0])

    fitness_components[fitness_func.name] = vmapped_fitness(func_keys, input_seqs)

  if fitness_evaluator.combine_func is not None:
    combine_args = fitness_evaluator.combine_func_args or {}
    combined_fitness = fitness_evaluator.combine_func(fitness_components, **combine_args)
  else:
    # Default: sum all fitness components
    combined_fitness = jnp.sum(jnp.array(list(fitness_components.values())), axis=0)

  return combined_fitness, fitness_components


@partial(jit)
def combine_fitness_scores(
  fitness_components: dict,
  fitness_weights: Optional[FitnessWeights] = None,
) -> PopulationSequenceFloats:
  """Combines individual fitness scores into a single score using weights.

  Args:
      fitness_components: Dictionary of individual fitness scores.
      fitness_weights: Optional weights for combining fitness scores.

  Returns:
      Combined fitness score.
  """
  if fitness_weights is not None:
    combined_fitness = jnp.tensordot(
      fitness_weights, jnp.array(list(fitness_components.values())), axes=1
    )
  else:
    combined_fitness = jnp.sum(jnp.array(list(fitness_components.values())), axis=0)

  return combined_fitness
