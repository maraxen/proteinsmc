"""Data structures for fitness evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

from proteinsmc.models.types import EvoSequence

StackedFitness = Float[Array, "1+n_fitness_functions"]
"""Type alias for stacked fitness scores, including combined score on the 0 batch dimension."""
NeedsTranslation = Bool[Array, "n_fitness_functions"]
FitnessFn = Callable[[PRNGKeyArray | None, EvoSequence, PyTree | Array | None], Float]
CombineFn = Callable[[Array, PRNGKeyArray | None, PyTree | Array | None], Float]
StackedFitnessFn = Callable[
  [PRNGKeyArray | None, EvoSequence, PyTree | Array | None],
  StackedFitness,
]


@dataclass(frozen=True)
class FitnessFunction:
  """Represents a single fitness function configuration."""

  name: str
  n_states: int
  kwargs: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    """Validate the fitness function configuration."""
    if not isinstance(self.name, str):
      msg = "name must be a string."
      raise TypeError(msg)
    if not isinstance(self.n_states, int):
      msg = "n_states must be an integer."
      raise TypeError(msg)
    if not isinstance(self.kwargs, dict):
      msg = "kwargs must be a dict."
      raise TypeError(msg)


@dataclass(frozen=True)
class CombineFunction:
  """Represents a function to combine fitness scores."""

  name: str
  kwargs: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    """Validate the combine function configuration."""
    if not isinstance(self.name, str):
      msg = "name must be a string."
      raise TypeError(msg)
    if not isinstance(self.kwargs, dict):
      msg = "kwargs must be a dict."
      raise TypeError(msg)


@dataclass(frozen=True)
class FitnessEvaluator:
  """Manages multiple fitness functions and their combination."""

  fitness_functions: tuple[FitnessFunction, ...]
  combine_fn: CombineFunction = field(
    default_factory=lambda: CombineFunction(name="sum"),
  )

  def __post_init__(self) -> None:
    """Validate the fitness evaluator configuration."""
    if not self.fitness_functions:
      msg = "At least one fitness function must be provided."
      raise ValueError(msg)

  def get_functions_by_states(
    self,
    n_states: int,
  ) -> list[FitnessFunction]:
    """Get active fitness functions that accept the specified number of states."""
    return [f for f in self.fitness_functions if f.n_states == n_states]

  def needs_translation(self, n_states: int | Int) -> NeedsTranslation:
    """Check if any fitness function requires sequence translation."""
    return jnp.where(
      jnp.array([f.n_states != n_states for f in self.fitness_functions]),
      1,
      0,
    )
