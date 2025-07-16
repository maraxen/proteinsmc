"""Data structures for SMC sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (  # Added Any, Callable for clarity
  TYPE_CHECKING,
  Any,
  Callable,
  Literal,
  TypedDict,
  Unpack,
)

import jax

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from .types import EvoSequence

from jaxtyping import Array, Float

from .base import RegisteredFunction, Registry, RegistryItem

FitnessScores = Float[jax.Array, "fitness_functions"]


class FitnessKwargs(TypedDict):
  """TypedDict for the parameters of a fitness function."""

  _key: PRNGKeyArray | None
  sequence: EvoSequence
  _context: Array | None


FitnessFuncSignature = Callable[[Unpack[FitnessKwargs]], Float]


@dataclass(frozen=True)
class FitnessRegistryItem(RegistryItem):
  """Represents a single fitness function in the registry."""

  method_factory: Callable[..., FitnessFuncSignature]
  input_type: Literal["protein", "nucleotide"] = "protein"

  def __post_init__(self) -> None:
    """Validate the fitness function metadata."""
    if not callable(self.method_factory):
      msg = f"Fitness method {self.method_factory} is not callable."
      raise TypeError(msg)
    if self.input_type not in ["nucleotide", "protein"]:
      msg = f"Invalid input_type '{self.input_type}'. Expected 'nucleotide' or 'protein'."
      raise ValueError(msg)
    if not isinstance(self.name, str):
      msg = "name must be a string."
      raise TypeError(msg)

  def __call__(
    self: FitnessRegistryItem,
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
  ) -> FitnessFuncSignature:
    """Return the callable fitness function."""
    return self.method_factory(*args, **kwargs)


@dataclass
class FitnessRegistry(Registry):
  """Registry for fitness functions."""

  def __post_init__(self) -> None:
    """Initialize the registry."""
    super().__post_init__()
    if not all(isinstance(item, FitnessRegistryItem) for item in self.items.values()):
      msg = "All items in FitnessRegistry must be instances of FitnessRegistryItem."
      raise TypeError(msg)


@dataclass(frozen=True)
class FitnessFunction(RegisteredFunction):
  """Represents a single fitness function."""

  input_type: Literal["protein", "nucleotide"] = "protein"

  def __post_init__(self) -> None:
    """Validate the fitness function metadata."""
    super().__post_init__()
    if self.input_type not in ["nucleotide", "protein"]:
      msg = f"Invalid input_type '{self.input_type}'. Expected 'nucleotide' or 'protein'."
      raise ValueError(
        msg,
      )

  def __call__(
    self,
    fitness_registry: FitnessRegistry,
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
  ) -> FitnessFuncSignature:
    """Call the fitness function from the registry."""
    if self.func not in fitness_registry:
      msg = f"Fitness function {self.func} is not registered."
      raise ValueError(msg)
    return fitness_registry.get(self.func)(*args, **kwargs)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.func,
      self.input_type,
      self.required_args,
      self.required_kwargs,
    )
    return children, {}

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> FitnessFunction:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      func=children[0],
      input_type=children[1],
      required_args=children[2],
      required_kwargs=children[3],
      **aux_data,
    )


class CombineKwargs(TypedDict):
  """TypedDict for the parameters of a combine function."""

  fitness_scores: FitnessScores
  _context: Array | None


CombineFuncSignature = Callable[[Unpack[CombineKwargs]], Float]


@dataclass(frozen=True)
class CombineFunction(RegisteredFunction):
  """Represents a function to combine fitness scores."""


class CombineRegistryItem(RegistryItem):
  """Represents a single combine function in the registry."""

  method_factory: Callable[..., CombineFuncSignature]
  func: CombineFunction

  def __post_init__(self) -> None:
    """Validate the combine function metadata."""
    super().__post_init__()

  def __call__(self, *args: Any, **kwargs: Any) -> CombineFuncSignature:  # noqa: ANN401
    """Return the callable combine function."""
    return self.method_factory(*args, **kwargs)


@dataclass
class CombineRegistry(Registry):
  """Registry for combine functions."""

  _registry: dict[str, CombineRegistryItem] = field(default_factory=dict)

  def get(self, name: str) -> CombineRegistryItem:
    """Get a registered combine function by name."""
    if name not in self._registry:
      msg = f"Combine function {name} is not registered."
      raise KeyError(msg)
    return self._registry[name]


@dataclass(frozen=True)
class FitnessEvaluator:
  """Manages multiple fitness functions and their combination."""

  fitness_functions: tuple[FitnessFunction, ...]
  combine_func: CombineFunction = field(
    default_factory=lambda: CombineFunction(func="sum"),
  )
  fitness_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
  combine_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)

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

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (self.fitness_functions, self.combine_func)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, _aux_data: dict, children: tuple) -> FitnessEvaluator:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      fitness_functions=children[0],
      combine_func=children[1],
    )

  def get_score_fns(
    self,
    fitness_registry: FitnessRegistry,
  ) -> tuple[FitnessFuncSignature, ...]:
    """Evaluate fitness scores using the registered functions."""
    return tuple(
      fitness_function(registry=fitness_registry, **self.fitness_kwargs[fitness_function.func])
      for fitness_function in self.fitness_functions
    )

  def combine(
    self,
    combine_registry: CombineRegistry,
  ) -> CombineFuncSignature:
    """Combine fitness scores using the registered combine function."""
    if self.combine_func.func not in combine_registry:
      msg = f"Combine function {self.combine_func.func} is not registered."
      raise ValueError(msg)

    return self.combine_func(
      registry=combine_registry,
      **self.combine_kwargs.get(self.combine_func.func, {}),
    )


jax.tree_util.register_pytree_node_class(FitnessFunction)
jax.tree_util.register_pytree_node_class(FitnessEvaluator)
