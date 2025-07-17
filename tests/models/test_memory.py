"""Tests for memory configuration classes."""

from __future__ import annotations

import chex
import jax
import pytest

from proteinsmc.models.memory import AutoTuningConfig, MemoryConfig


class TestAutoTuningConfig:
  """Test cases for AutoTuningConfig."""

  def test_init_default_values(self) -> None:
    """Test AutoTuningConfig initialization with default values."""
    config = AutoTuningConfig()
    assert config.enable_auto_tuning is True
    assert config.probe_chunk_sizes == (16, 32, 64, 128, 256)
    assert config.max_probe_iterations == 3
    assert config.memory_safety_factor == 0.8
    assert config.performance_tolerance == 0.1

  def test_init_custom_values(self) -> None:
    """Test AutoTuningConfig initialization with custom values."""
    config = AutoTuningConfig(
      enable_auto_tuning=False,
      probe_chunk_sizes=(8, 16, 32),
      max_probe_iterations=5,
      memory_safety_factor=0.9,
      performance_tolerance=0.05,
    )
    assert config.enable_auto_tuning is False
    assert config.probe_chunk_sizes == (8, 16, 32)
    assert config.max_probe_iterations == 5
    assert config.memory_safety_factor == 0.9
    assert config.performance_tolerance == 0.05

  def test_pytree_registration(self) -> None:
    """Test that AutoTuningConfig is properly registered as a PyTree."""
    config = AutoTuningConfig(
      enable_auto_tuning=True,
      probe_chunk_sizes=(16, 32),
      max_probe_iterations=2,
      memory_safety_factor=0.7,
      performance_tolerance=0.2,
    )

    # Test that it can be used in JAX transformations
    def process_config(c: AutoTuningConfig) -> int:
      return c.max_probe_iterations

    jitted_process = jax.jit(process_config)
    result = jitted_process(config)
    chex.assert_trees_all_close(result, 2)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(config)
    unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

    assert config.enable_auto_tuning == unflattened_config.enable_auto_tuning
    assert config.probe_chunk_sizes == unflattened_config.probe_chunk_sizes
    assert config.max_probe_iterations == unflattened_config.max_probe_iterations
    assert config.memory_safety_factor == unflattened_config.memory_safety_factor
    assert config.performance_tolerance == unflattened_config.performance_tolerance


class TestMemoryConfig:
  """Test cases for MemoryConfig."""

  def test_init_default_values(self) -> None:
    """Test MemoryConfig initialization with default values."""
    config = MemoryConfig()
    assert config.population_chunk_size == 64
    assert config.enable_chunked_vmap is True
    assert config.device_memory_fraction == 0.8
    assert isinstance(config.auto_tuning_config, AutoTuningConfig)

  def test_init_custom_values(self) -> None:
    """Test MemoryConfig initialization with custom values."""
    auto_config = AutoTuningConfig(enable_auto_tuning=False)
    config = MemoryConfig(
      population_chunk_size=128,
      enable_chunked_vmap=False,
      device_memory_fraction=0.6,
      auto_tuning_config=auto_config,
    )
    assert config.population_chunk_size == 128
    assert config.enable_chunked_vmap is False
    assert config.device_memory_fraction == 0.6
    assert config.auto_tuning_config == auto_config

  def test_validation_invalid_device_memory_fraction_low(self) -> None:
    """Test validation fails for device_memory_fraction <= 0."""
    with pytest.raises(ValueError, match="device_memory_fraction must be in \\(0.0, 1.0\\]"):
      MemoryConfig(device_memory_fraction=0.0)

  def test_validation_invalid_device_memory_fraction_high(self) -> None:
    """Test validation fails for device_memory_fraction > 1."""
    with pytest.raises(ValueError, match="device_memory_fraction must be in \\(0.0, 1.0\\]"):
      MemoryConfig(device_memory_fraction=1.5)

  def test_validation_invalid_population_chunk_size(self) -> None:
    """Test validation fails for non-positive population_chunk_size."""
    with pytest.raises(ValueError, match="population_chunk_size must be positive"):
      MemoryConfig(population_chunk_size=0)

  def test_validation_invalid_population_chunk_size_negative(self) -> None:
    """Test validation fails for negative population_chunk_size."""
    with pytest.raises(ValueError, match="population_chunk_size must be positive"):
      MemoryConfig(population_chunk_size=-10)

  def test_type_validation_population_chunk_size(self) -> None:
    """Test type validation for population_chunk_size."""
    with pytest.raises(TypeError, match="population_chunk_size must be an integer"):
      MemoryConfig(population_chunk_size=64.5)  # type: ignore

  def test_type_validation_enable_chunked_vmap(self) -> None:
    """Test type validation for enable_chunked_vmap."""
    with pytest.raises(TypeError, match="enable_chunked_vmap must be a boolean"):
      MemoryConfig(enable_chunked_vmap="true")  # type: ignore

  def test_type_validation_device_memory_fraction(self) -> None:
    """Test type validation for device_memory_fraction."""
    with pytest.raises(TypeError, match="device_memory_fraction must be a float"):
      MemoryConfig(device_memory_fraction="0.8")  # type: ignore

  def test_type_validation_auto_tuning_config(self) -> None:
    """Test type validation for auto_tuning_config."""
    with pytest.raises(TypeError, match="auto_tuning_config must be an AutoTuningConfig instance"):
      MemoryConfig(auto_tuning_config="not_config")  # type: ignore

  def test_pytree_registration(self) -> None:
    """Test that MemoryConfig is properly registered as a PyTree."""
    auto_config = AutoTuningConfig(max_probe_iterations=5)
    config = MemoryConfig(
      population_chunk_size=32,
      enable_chunked_vmap=False,
      device_memory_fraction=0.7,
      auto_tuning_config=auto_config,
    )

    # Test that it can be used in JAX transformations
    def process_config(c: MemoryConfig) -> int:
      return c.population_chunk_size

    jitted_process = jax.jit(process_config)
    result = jitted_process(config)
    chex.assert_trees_all_close(result, 32)

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(config)
    unflattened_config = jax.tree_util.tree_unflatten(treedef, leaves)

    assert config.population_chunk_size == unflattened_config.population_chunk_size
    assert config.enable_chunked_vmap == unflattened_config.enable_chunked_vmap
    assert config.device_memory_fraction == unflattened_config.device_memory_fraction
    assert (
      config.auto_tuning_config.max_probe_iterations
      == unflattened_config.auto_tuning_config.max_probe_iterations
    )

  def test_edge_case_device_memory_fraction_boundary(self) -> None:
    """Test boundary values for device_memory_fraction."""
    # Test that exactly 1.0 is allowed
    config = MemoryConfig(device_memory_fraction=1.0)
    assert config.device_memory_fraction == 1.0

    # Test that very small positive value is allowed
    config = MemoryConfig(device_memory_fraction=0.001)
    assert config.device_memory_fraction == 0.001

  def test_edge_case_population_chunk_size_boundary(self) -> None:
    """Test boundary values for population_chunk_size."""
    # Test that 1 is allowed
    config = MemoryConfig(population_chunk_size=1)
    assert config.population_chunk_size == 1

    # Test large values
    config = MemoryConfig(population_chunk_size=10000)
    assert config.population_chunk_size == 10000
