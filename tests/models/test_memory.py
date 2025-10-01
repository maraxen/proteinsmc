"""Unit tests for MemoryConfig data model.

Tests cover initialization, type checking, and edge cases for MemoryConfig.
"""
import pytest
from proteinsmc.models import MemoryConfig, AutoTuningConfig


def test_memory_config_initialization():
  """Test MemoryConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_memory_config_initialization()
  """
  config = MemoryConfig(
    batch_size=16,
    enable_batched_computation=True,
    device_memory_fraction=0.8,
    auto_tuning_config=AutoTuningConfig(
      enable_auto_tuning=False,
      probe_chunk_sizes=(8, 16, 32),
      max_probe_iterations=2,
      memory_safety_factor=0.8,
      performance_tolerance=0.1,
    ),
  )
  assert config.batch_size == 16
  assert config.enable_batched_computation is True
  assert config.device_memory_fraction == 0.8
  assert isinstance(config.auto_tuning_config, AutoTuningConfig)
