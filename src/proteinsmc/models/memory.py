"""Data structures for SMC sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryConfig:
  """Configuration for memory efficiency and performance optimization."""

  batch_size: int = field(default=64)
  """Size of the batch for processing."""
  enable_batched_computation: bool = field(default=True)
  device_memory_fraction: float = field(default=0.8)
  auto_tuning_config: AutoTuningConfig = field(default_factory=lambda: AutoTuningConfig())

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.batch_size, int):
      msg = "batch_size must be an integer."
      raise TypeError(msg)
    if not isinstance(self.enable_batched_computation, bool):
      msg = "enable_batched_computation must be a boolean."
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
    if self.batch_size <= 0:
      msg = "batch_size must be positive."
      raise ValueError(msg)


@dataclass(frozen=True)
class AutoTuningConfig:
  """Configuration for automatic chunk size tuning."""

  enable_auto_tuning: bool = field(default=True)
  probe_chunk_sizes: tuple[int, ...] = field(default=(16, 32, 64, 128, 256))
  max_probe_iterations: int = field(default=3)
  memory_safety_factor: float = field(default=0.8)
  performance_tolerance: float = field(default=0.1)
