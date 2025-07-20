"""Data structures for SMC sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax


@dataclass(frozen=True)
class MemoryConfig:
  """Configuration for memory efficiency and performance optimization."""

  population_chunk_size: int = field(default=64)
  """Size of the population chunk for processing."""
  enable_chunked_vmap: bool = field(default=True)
  device_memory_fraction: float = field(default=0.8)
  auto_tuning_config: AutoTuningConfig = field(default_factory=lambda: AutoTuningConfig())

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.population_chunk_size, int):
      msg = "population_chunk_size must be an integer."
      raise TypeError(msg)
    if not isinstance(self.enable_chunked_vmap, bool):
      msg = "enable_chunked_vmap must be a boolean."
      raise TypeError(msg)
    if not isinstance(self.device_memory_fraction, float):
      msg = "device_memory_fraction must be a float."
      raise TypeError(msg)
    if not isinstance(self.auto_tuning_config, AutoTuningConfig):
      msg = "auto_tuning_config must be an AutoTuningConfig instance."
      raise TypeError(msg)

  def __post_init__(self) -> None:
    """Validate the memory configuration."""
    self._validate_types()
    if not (0.0 < self.device_memory_fraction <= 1.0):
      msg = "device_memory_fraction must be in (0.0, 1.0]."
      raise ValueError(msg)
    if self.population_chunk_size <= 0:
      msg = "population_chunk_size must be positive."
      raise ValueError(msg)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.population_chunk_size,
      self.enable_chunked_vmap,
      self.device_memory_fraction,
      self.auto_tuning_config,
    )
    aux_data = {}  # aux_data is empty as all varying fields are children
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> MemoryConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      population_chunk_size=children[0],
      enable_chunked_vmap=children[1],
      device_memory_fraction=children[2],
      auto_tuning_config=children[3],
      **aux_data,
    )


@dataclass(frozen=True)
class AutoTuningConfig:
  """Configuration for automatic chunk size tuning."""

  enable_auto_tuning: bool = field(default=True)
  probe_chunk_sizes: tuple[int, ...] = field(default=(16, 32, 64, 128, 256))
  max_probe_iterations: int = field(default=3)
  memory_safety_factor: float = field(default=0.8)
  performance_tolerance: float = field(default=0.1)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.enable_auto_tuning,
      self.probe_chunk_sizes,
      self.max_probe_iterations,
      self.memory_safety_factor,
      self.performance_tolerance,
    )
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> AutoTuningConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      enable_auto_tuning=children[0],
      probe_chunk_sizes=children[1],
      max_probe_iterations=children[2],
      memory_safety_factor=children[3],
      performance_tolerance=children[4],
      **aux_data,
    )


jax.tree_util.register_pytree_node_class(MemoryConfig)
jax.tree_util.register_pytree_node_class(AutoTuningConfig)
