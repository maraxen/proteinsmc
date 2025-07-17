"""Provides common base classes and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
  Any,
  Callable,
  Literal,
)

from jaxtyping import Array, Float, Int

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
