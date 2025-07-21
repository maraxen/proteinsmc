"""Tests for memory configuration data structures."""

from __future__ import annotations

import jax
import pytest

from proteinsmc.models.memory import AutoTuningConfig, MemoryConfig


class TestAutoTuningConfig:
  """Test the AutoTuningConfig dataclass."""

  def test_default_initialization(self) -> None:
    """Test AutoTuningConfig with default values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the default values do not match expected values.

    Example:
        >>> test_default_initialization()

    """
    config = AutoTuningConfig()
    assert config.enable_auto_tuning is True
    assert config.probe_chunk_sizes == (16, 32, 64, 128, 256)
    assert config.max_probe_iterations == 3
    assert config.memory_safety_factor == 0.8
    assert config.performance_tolerance == 0.1

  def test_custom_initialization(self) -> None:
    """Test AutoTuningConfig with custom values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the custom values do not match expected values.

    Example:
        >>> test_custom_initialization()

    """
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

  def test_tree_flatten_unflatten(self) -> None:
    """Test JAX PyTree flatten/unflatten functionality.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the flattened and unflattened config does not match.

    Example:
        >>> test_tree_flatten_unflatten()

    """
    config = AutoTuningConfig(
      enable_auto_tuning=False,
      probe_chunk_sizes=(64, 128),
      max_probe_iterations=2,
    )
    children, aux_data = config.tree_flatten()
    reconstructed = AutoTuningConfig.tree_unflatten(aux_data, children)
    assert config == reconstructed

  def test_jax_transformations(self) -> None:
    """Test that AutoTuningConfig works with JAX transformations.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JAX transformations fail.

    Example:
        >>> test_jax_transformations()

    """
    config = AutoTuningConfig()

    def dummy_fn(cfg: AutoTuningConfig) -> int:
      return cfg.max_probe_iterations

    # Test with jit
    jitted_fn = jax.jit(dummy_fn)
    result = jitted_fn(config)
    assert result == 3

    # Test tree_map
    mapped_config = jax.tree_util.tree_map(lambda x: x, config)
    assert mapped_config == config


class TestMemoryConfig:
  """Test the MemoryConfig dataclass."""

  def test_default_initialization(self) -> None:
    """Test MemoryConfig with default values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the default values do not match expected values.

    Example:
        >>> test_default_initialization()

    """
    config = MemoryConfig()
    assert config.population_chunk_size == 64
    assert config.enable_chunked_vmap is True
    assert config.device_memory_fraction == 0.8
    assert isinstance(config.auto_tuning_config, AutoTuningConfig)

  def test_custom_initialization(self) -> None:
    """Test MemoryConfig with custom values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the custom values do not match expected values.

    Example:
        >>> test_custom_initialization()

    """
    auto_config = AutoTuningConfig(enable_auto_tuning=False)
    config = MemoryConfig(
      population_chunk_size=128,
      enable_chunked_vmap=False,
      device_memory_fraction=0.9,
      auto_tuning_config=auto_config,
    )
    assert config.population_chunk_size == 128
    assert config.enable_chunked_vmap is False
    assert config.device_memory_fraction == 0.9
    assert config.auto_tuning_config == auto_config

  def test_validation_device_memory_fraction_bounds(self) -> None:
    """Test device_memory_fraction validation bounds.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If validation does not raise ValueError for invalid bounds.

    Example:
        >>> test_validation_device_memory_fraction_bounds()

    """
    # Test lower bound
    with pytest.raises(ValueError, match="device_memory_fraction must be in"):
      MemoryConfig(device_memory_fraction=0.0)

    with pytest.raises(ValueError, match="device_memory_fraction must be in"):
      MemoryConfig(device_memory_fraction=-0.1)

    # Test upper bound
    with pytest.raises(ValueError, match="device_memory_fraction must be in"):
      MemoryConfig(device_memory_fraction=1.1)

    # Test valid edge case
    config = MemoryConfig(device_memory_fraction=1.0)
    assert config.device_memory_fraction == 1.0

  def test_validation_population_chunk_size(self) -> None:
    """Test population_chunk_size validation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If validation does not raise ValueError for invalid chunk size.

    Example:
        >>> test_validation_population_chunk_size()

    """
    with pytest.raises(ValueError, match="population_chunk_size must be positive"):
      MemoryConfig(population_chunk_size=0)

    with pytest.raises(ValueError, match="population_chunk_size must be positive"):
      MemoryConfig(population_chunk_size=-1)

  def test_type_validation(self) -> None:
    """Test type validation for all fields.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If type validation does not raise TypeError for invalid types.

    Example:
        >>> test_type_validation()

    """
    # Test population_chunk_size type
    with pytest.raises(TypeError, match="population_chunk_size must be an integer"):
      MemoryConfig(population_chunk_size=64.0)  # type: ignore[arg-type]

    # Test enable_chunked_vmap type
    with pytest.raises(TypeError, match="enable_chunked_vmap must be a boolean"):
      MemoryConfig(enable_chunked_vmap=1)  # type: ignore[arg-type]

    # Test device_memory_fraction type
    with pytest.raises(TypeError, match="device_memory_fraction must be a float"):
      MemoryConfig(device_memory_fraction=1)  # type: ignore[arg-type]

    # Test auto_tuning_config type
    with pytest.raises(TypeError, match="auto_tuning_config must be an AutoTuningConfig"):
      MemoryConfig(auto_tuning_config="invalid")  # type: ignore[arg-type]


  def test_jax_transformations(self) -> None:
    """Test that MemoryConfig works with JAX transformations.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JAX transformations fail.

    Example:
        >>> test_jax_transformations()

    """
    config = MemoryConfig()

    def dummy_fn(cfg: MemoryConfig) -> int:
      return cfg.population_chunk_size

    # Test with jit
    jitted_fn = jax.jit(dummy_fn)
    result = jitted_fn(config)
    assert result == 64

    # Test tree_map
    mapped_config = jax.tree_util.tree_map(lambda x: x, config)
    assert mapped_config == config

  def test_nested_pytree_operations(self) -> None:
    """Test nested PyTree operations with AutoTuningConfig.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If nested PyTree operations fail.

    Example:
        >>> test_nested_pytree_operations()

    """
    config = MemoryConfig()

    def extract_nested_value(cfg: MemoryConfig) -> bool:
      return cfg.auto_tuning_config.enable_auto_tuning

    jitted_fn = jax.jit(extract_nested_value)
    result = jitted_fn(config)
    assert result is True

  def test_immutability(self) -> None:
    """Test that MemoryConfig instances are immutable.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the dataclass is not frozen (immutable).

    Example:
        >>> test_immutability()

    """
    config = MemoryConfig()
    with pytest.raises(AttributeError):
      config.population_chunk_size = 128  # type: ignore[misc]

  def test_pytree_registration(self) -> None:
    """Test that classes are properly registered as PyTree nodes.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If PyTree registration fails.

    Example:
        >>> test_pytree_registration()

    """
    config = MemoryConfig()
    auto_config = AutoTuningConfig()

    # Test that tree_leaves works
    leaves_config = jax.tree_util.tree_leaves(config)
    leaves_auto = jax.tree_util.tree_leaves(auto_config)

    assert len(leaves_config) > 0
    assert len(leaves_auto) > 0

    # Test that tree_structure works
    structure_config = jax.tree_util.tree_structure(config)
    structure_auto = jax.tree_util.tree_structure(auto_config)

    assert structure_config is not None
    assert structure_auto is not None
