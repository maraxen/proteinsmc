"""Provides common base classes and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
  Any,
  Callable,
  Literal,
  Protocol,
)

import jax
from jaxtyping import Array, Float, Int

from proteinsmc.models.fitness import FitnessEvaluator
from proteinsmc.models.memory import MemoryConfig

NucleotideSequence = Int[Array, "nucleotide_sequence_length"]
ProteinSequence = Int[Array, "protein_sequence_length"]
EvoSequence = NucleotideSequence | ProteinSequence
PerGenerationMetrics = Float[Array, "generations"]


@dataclass(frozen=True)
class RegistryItem(ABC):
  """Base class for registry items."""

  method_factory: Callable[..., Callable]
  name: str

  def __post_init__(self) -> None:
    """Validate the registry item metadata."""
    if not isinstance(self.name, str):
      msg = "name must be a string."
      raise TypeError(msg)
    if not callable(self.method_factory):
      msg = f"Method factory {self.method_factory} is not callable."
      raise TypeError(msg)

  @abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> Callable:  # noqa: ANN401
    """Abstract method to be implemented by subclasses."""


@dataclass
class Registry(ABC):
  """Base class for registry items."""

  items: dict[str, RegistryItem]

  def __post_init__(self) -> None:
    """Initialize the registry."""
    if not isinstance(self.items, dict):
      msg = "items must be a dictionary."
      raise TypeError(msg)
    for item in self.items.values():
      if not isinstance(item, RegistryItem):
        msg = f"All items must be instances of RegistryItem, got {type(item)}."
        raise TypeError(msg)

  def __contains__(self, name: str) -> bool:
    """Check if an item is registered."""
    return name in self.items

  def __getitem__(self, name: str) -> RegistryItem:
    """Get a registered item by name."""
    if name not in self.items:
      msg = f"Item {name} is not registered."
      raise KeyError(msg)
    return self.items[name]

  def register(self, item: RegistryItem) -> None:
    """Register a new item."""
    if item.name in self.items:
      msg = f"Item {item.name} is already registered."
      raise ValueError(msg)
    self.items[item.name] = item

  def get(self, name: str) -> RegistryItem:
    """Get a registered item by name."""
    if name not in self.items:
      msg = f"Item {name} is not registered."
      raise KeyError(msg)
    return self.items[name]


@dataclass(frozen=True)
class RegisteredFunction:
  """Represents a registered function with metadata."""

  func: str
  key_split: Literal[0, None] = None
  context_tuple: tuple[int | None, ...] = field(default_factory=lambda: (None,))
  required_args: tuple[type, ...] = field(default_factory=tuple)
  required_kwargs: dict[str, type] = field(default_factory=dict)

  def __post_init__(self) -> None:
    """Validate the registered function metadata."""
    if not isinstance(self.func, str):
      msg = f"Registered function {self.func} is not a string."
      raise TypeError(msg)
    if not isinstance(self.context_tuple, tuple):
      msg = "context_tuple must be a tuple."
      raise TypeError(msg)
    if not isinstance(self.required_args, tuple):
      msg = "required_args must be a tuple."
      raise TypeError(msg)
    if not all(isinstance(arg, type) for arg in self.required_args):
      msg = "All required_args must be types."
      raise TypeError(msg)
    if not isinstance(self.required_kwargs, dict):
      msg = "required_kwargs must be a dictionary."
      raise TypeError(msg)
    if not all(isinstance(v, type) for v in self.required_kwargs.values()):
      msg = "All required_kwargs values must be types."
      raise TypeError(msg)

  def __call__(self, registry: Registry, *args: Any, **kwargs: Any) -> Callable:  # noqa: ANN401
    """Call the registered function from the registry."""
    if not isinstance(registry, Registry):
      msg = f"Expected registry to be an instance of Registry, got {type(registry)}."
      raise TypeError(msg)
    if self.func not in registry:
      msg = f"Function {self.func} is not registered."
      raise ValueError(msg)
    if not all(isinstance(arg, req_type) for arg, req_type in zip(args, self.required_args)):
      msg = f"Arguments {args} do not match required types {self.required_args}."
      raise TypeError(msg)
    if not all(isinstance(kwargs.get(k), v) for k, v in self.required_kwargs.items()):
      msg = f"Keyword arguments {kwargs} do not match required types {self.required_kwargs}."
      raise TypeError(msg)
    return registry.get(self.func)(*args, **kwargs)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.func,
      self.required_args,
      self.required_kwargs,
    )
    return children, {}

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> RegisteredFunction:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      func=children[0],
      required_args=children[1],
      required_kwargs=children[2],
      **aux_data,
    )


@dataclass(frozen=True)
class BaseSamplerConfig:
  """Base configuration for samplers.

  All sampler configurations should inherit from this.
  """

  seed_sequence: str
  generations: int
  n_states: int
  mutation_rate: float
  diversification_ratio: float
  sequence_type: Literal["protein", "nucleotide"]
  fitness_evaluator: FitnessEvaluator
  memory_config: MemoryConfig

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.seed_sequence, str):
      msg = "seed_sequence must be a string."
      raise TypeError(msg)
    if not isinstance(self.n_states, int):
      msg = "n_states must be an integer."
      raise TypeError(msg)
    if not isinstance(self.generations, int):
      msg = "generations must be an integer."
      raise TypeError(msg)
    if not isinstance(self.mutation_rate, float):
      msg = "mutation_rate must be a float."
      raise TypeError(msg)
    if not isinstance(self.diversification_ratio, float):
      msg = "diversification_ratio must be a float."
      raise TypeError(msg)
    if not isinstance(self.fitness_evaluator, FitnessEvaluator):
      msg = "fitness_evaluator must be a FitnessEvaluator instance."
      raise TypeError(msg)
    if not isinstance(self.memory_config, MemoryConfig):
      msg = "memory_config must be a MemoryConfig instance."
      raise TypeError(msg)

  def __post_init__(self) -> None:
    """Validate the common configuration fields."""
    if self.n_states <= 0:
      msg = "n_states must be positive."
      raise ValueError(msg)
    if self.generations <= 0:
      msg = "generations must be positive."
      raise ValueError(msg)
    if not (0.0 <= self.mutation_rate <= 1.0):
      msg = "mutation_rate must be in [0.0, 1.0]."
      raise ValueError(msg)
    if not (0.0 <= self.diversification_ratio <= 1.0):
      msg = "diversification_ratio must be in [0.0, 1.0]."
      raise ValueError(msg)
    if self.sequence_type not in ("protein", "nucleotide"):
      msg = "sequence_type must be 'protein' or 'nucleotide'."
      raise ValueError(msg)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.seed_sequence,
      self.generations,
      self.n_states,
      self.mutation_rate,
      self.diversification_ratio,
      self.sequence_type,
      self.fitness_evaluator,
      self.memory_config,
    )
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> BaseSamplerConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      seed_sequence=children[0],
      generations=children[1],
      n_states=children[2],
      mutation_rate=children[3],
      diversification_ratio=children[4],
      sequence_type=children[5],
      fitness_evaluator=children[6],
      memory_config=children[7],
      **aux_data,
    )

  @property
  def additional_config_fields(self) -> dict[str, str]:
    """Return additional fields for the configuration that are not part of the PyTree."""
    return {}


class SamplerOutputProtocol(Protocol):
  """Protocol for sampler output objects, for generic data extraction."""

  @property
  def input_configs(self) -> BaseSamplerConfig | tuple[BaseSamplerConfig, ...]:
    """Return the input configuration(s) for the sampler output.

    This can be a single config or a tuple of configs if batched.
    """
    ...

  @property
  def per_gen_stats_metrics(self) -> dict[str, str]:
    """Return a mapping from generic metric name to attribute name for per-generation stats."""
    ...

  @property
  def summary_stats_metrics(self) -> dict[str, str]:
    """Return a mapping from generic metric name to attribute name for summary stats."""
    ...

  @property
  def output_type_name(self) -> str:
    """Return the string name of the output type (e.g., 'SMC', 'ParallelReplicaSMC')."""
    ...


jax.tree_util.register_pytree_node_class(BaseSamplerConfig)
