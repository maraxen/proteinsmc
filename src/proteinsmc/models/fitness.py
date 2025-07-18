"""Data structures for fitness evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypedDict, Unpack

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

if TYPE_CHECKING:
  from proteinsmc.models.types import EvoSequence

NeedsTranslation = Bool[Array, "n_fitness_functions"]


class FitnessKwargs(TypedDict):
  """TypedDict for the parameters of a fitness function."""

  sequence: EvoSequence
  _key: PRNGKeyArray | None
  _context: Array | None


FitnessFuncSignature = Callable[[Unpack[FitnessKwargs]], Float]
StackedFitnessFuncSignature = Callable[[Unpack[FitnessKwargs]], tuple[Array, Array]]


class CombineKwargs(TypedDict):
  """TypedDict for the parameters of a combine function."""

  scores: Array
  _key: PRNGKeyArray | None
  _context: Array | None


CombineFuncSignature = Callable[[Unpack[CombineKwargs]], Float]


@dataclass(frozen=True)
class FitnessFunction:
  """Represents a single fitness function configuration."""

  name: str
  n_states: int
  kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CombineFunction:
  """Represents a function to combine fitness scores."""

  name: str
  kwargs: dict[str, Any] = field(default_factory=dict)


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

  def needs_translation(self, n_states: int) -> NeedsTranslation:
    """Check if any fitness function requires sequence translation."""
    return jnp.where(
      jnp.array([f.n_states != n_states for f in self.fitness_functions]),
      True,
      False,
    )
