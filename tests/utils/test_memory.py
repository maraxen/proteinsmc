"""Tests for memory management and auto-tuning utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_trees_all_close

from proteinsmc.models.memory import AutoTuningConfig
from proteinsmc.utils.memory import (
  BenchmarkResult,
  auto_tune_chunk_size,
  benchmark_chunk_size,
  create_test_population,
  estimate_memory_usage,
  get_device_memory_mb,
  suggest_chunk_size_heuristic,
)


class TestMemoryEstimation:
  """Test memory estimation functions."""

  def test_estimate_memory_usage_basic(self) -> None:
    """Test basic memory estimation."""
    # 100 sequences * 50 length * 4 bytes * 3 overhead = ~60KB
    memory_mb = estimate_memory_usage(
      population_size=100,
      sequence_length=50,
      dtype_size_bytes=4,
    )
    expected_mb = (100 * 50 * 4 * 3.0) / (1024 * 1024)
    assert abs(memory_mb - expected_mb) < 0.01

  def test_estimate_memory_usage_large_population(self) -> None:
    """Test memory estimation for large populations."""
    memory_mb = estimate_memory_usage(
      population_size=10000,
      sequence_length=200,
      dtype_size_bytes=4,
    )
    # Should be ~23 MB (10000 * 200 * 4 * 3 / 1024 / 1024)
    assert memory_mb > 20  # More than 20MB
    assert memory_mb < 30  # Less than 30MB

  def test_estimate_memory_usage_different_dtypes(self) -> None:
    """Test memory estimation with different dtype sizes."""
    memory_float32 = estimate_memory_usage(100, 50, dtype_size_bytes=4)
    memory_float64 = estimate_memory_usage(100, 50, dtype_size_bytes=8)
    # Float64 should use twice the memory
    assert abs(memory_float64 - 2 * memory_float32) < 0.01

  def test_get_device_memory_mb(self) -> None:
    """Test device memory retrieval."""
    memory_mb = get_device_memory_mb()
    # Should return a reasonable value (at least 1GB)
    assert memory_mb >= 1024
    # Should not be absurdly large (less than 1TB)
    assert memory_mb <= 1024 * 1024


class TestChunkSizeHeuristics:
  """Test chunk size heuristic functions."""

  def test_suggest_chunk_size_heuristic_disabled(self) -> None:
    """Test heuristic with auto-tuning disabled."""
    config = AutoTuningConfig(enable_auto_tuning=False)
    chunk_size = suggest_chunk_size_heuristic(
      population_size=1000,
      sequence_length=100,
      config=config,
    )
    # Should return minimum of population_size and 64
    assert chunk_size == 64

  def test_suggest_chunk_size_heuristic_enabled(self) -> None:
    """Test heuristic with auto-tuning enabled."""
    config = AutoTuningConfig(
      enable_auto_tuning=True,
      probe_chunk_sizes=(16, 32, 64, 128),
    )
    chunk_size = suggest_chunk_size_heuristic(
      population_size=1000,
      sequence_length=100,
      config=config,
    )
    # Should return one of the probe sizes
    assert chunk_size in config.probe_chunk_sizes

  def test_suggest_chunk_size_small_population(self) -> None:
    """Test heuristic with small population."""
    config = AutoTuningConfig(
      enable_auto_tuning=True,
      probe_chunk_sizes=(16, 32, 64, 128),
    )
    chunk_size = suggest_chunk_size_heuristic(
      population_size=20,
      sequence_length=100,
      config=config,
    )
    # Should not exceed population size
    assert chunk_size <= 20
    assert chunk_size == 16  # Closest probe size


class TestBenchmarking:
  """Test benchmarking functions."""

  def test_benchmark_chunk_size_success(self) -> None:
    """Test successful benchmark."""

    def test_func(x: jax.Array) -> jax.Array:
      return x * 2

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(100, 50, key)

    result = benchmark_chunk_size(
      func=test_func,
      test_data=test_data,
      chunk_size=32,
      num_trials=2,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.success
    assert result.chunk_size == 32
    assert result.avg_time_per_batch > 0
    assert result.memory_usage_mb > 0
    assert result.error_message is None

  def test_benchmark_chunk_size_with_static_args(self) -> None:
    """Test benchmark with static arguments."""

    def test_func(x: jax.Array, scale: float) -> jax.Array:
      return x * scale

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(100, 50, key)

    result = benchmark_chunk_size(
      func=test_func,
      test_data=test_data,
      chunk_size=32,
      num_trials=2,
      static_args={"scale": 2.5},
    )

    assert result.success
    assert result.chunk_size == 32

  def test_benchmark_chunk_size_failure(self) -> None:
    """Test benchmark with failing function."""

    def failing_func(x: jax.Array) -> jax.Array:
      raise ValueError("Intentional failure")

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(100, 50, key)

    result = benchmark_chunk_size(
      func=failing_func,
      test_data=test_data,
      chunk_size=32,
      num_trials=1,
    )

    assert not result.success
    assert result.avg_time_per_batch == float("inf")
    assert result.memory_usage_mb == float("inf")
    assert result.error_message is not None
    assert "Intentional failure" in result.error_message


class TestAutoTuning:
  """Test auto-tuning functions."""

  def test_auto_tune_chunk_size_disabled(self) -> None:
    """Test auto-tuning with disabled config."""

    def test_func(x: jax.Array) -> jax.Array:
      return x * 2

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(1000, 50, key)

    config = AutoTuningConfig(
      enable_auto_tuning=False,
      probe_chunk_sizes=(16, 32, 64, 128),
    )

    chunk_size = auto_tune_chunk_size(
      func=test_func,
      test_data=test_data,
      config=config,
    )

    # Should use heuristic
    assert chunk_size in config.probe_chunk_sizes

  def test_auto_tune_chunk_size_enabled(self) -> None:
    """Test auto-tuning with enabled config."""

    def test_func(x: jax.Array) -> jax.Array:
      return jnp.sum(x, axis=-1)

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(1000, 50, key)

    config = AutoTuningConfig(
      enable_auto_tuning=True,
      probe_chunk_sizes=(16, 32, 64, 128),
      max_probe_iterations=2,
    )

    chunk_size = auto_tune_chunk_size(
      func=test_func,
      test_data=test_data,
      config=config,
    )

    # Should return one of the probe sizes
    assert chunk_size in config.probe_chunk_sizes
    # Should be reasonable
    assert 16 <= chunk_size <= 128

  def test_auto_tune_chunk_size_with_static_args(self) -> None:
    """Test auto-tuning with static arguments."""

    def test_func(x: jax.Array, multiplier: int) -> jax.Array:
      return x * multiplier

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(500, 30, key)

    config = AutoTuningConfig(
      enable_auto_tuning=True,
      probe_chunk_sizes=(16, 32, 64),
      max_probe_iterations=2,
    )

    chunk_size = auto_tune_chunk_size(
      func=test_func,
      test_data=test_data,
      config=config,
      static_args={"multiplier": 3},
    )

    assert chunk_size in config.probe_chunk_sizes

  def test_auto_tune_chunk_size_small_population(self) -> None:
    """Test auto-tuning with population smaller than probe sizes."""

    def test_func(x: jax.Array) -> jax.Array:
      return x * 2

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(20, 50, key)

    config = AutoTuningConfig(
      enable_auto_tuning=True,
      probe_chunk_sizes=(16, 32, 64, 128),
      max_probe_iterations=2,
    )

    chunk_size = auto_tune_chunk_size(
      func=test_func,
      test_data=test_data,
      config=config,
    )

    # Should not exceed population size
    assert chunk_size <= 20

  def test_auto_tune_chunk_size_all_fail(self) -> None:
    """Test auto-tuning when all benchmarks fail."""

    def failing_func(x: jax.Array) -> jax.Array:
      raise MemoryError("Out of memory")

    key = jax.random.PRNGKey(0)
    test_data = create_test_population(1000, 50, key)

    config = AutoTuningConfig(
      enable_auto_tuning=True,
      probe_chunk_sizes=(16, 32, 64),
      max_probe_iterations=1,
    )

    chunk_size = auto_tune_chunk_size(
      func=failing_func,
      test_data=test_data,
      config=config,
    )

    # Should fall back to heuristic
    assert chunk_size in config.probe_chunk_sizes


class TestTestPopulationCreation:
  """Test test population creation."""

  def test_create_test_population_shape(self) -> None:
    """Test that test population has correct shape."""
    key = jax.random.PRNGKey(42)
    population = create_test_population(100, 50, key)

    assert population.shape == (100, 50)
    assert population.dtype == jnp.int32

  def test_create_test_population_values(self) -> None:
    """Test that test population has valid values."""
    key = jax.random.PRNGKey(42)
    population = create_test_population(100, 50, key)

    # All values should be in valid range [0, 20)
    assert jnp.all(population >= 0)
    assert jnp.all(population < 20)

  def test_create_test_population_determinism(self) -> None:
    """Test that same key produces same population."""
    key = jax.random.PRNGKey(42)
    population1 = create_test_population(100, 50, key)
    population2 = create_test_population(100, 50, key)

    assert_trees_all_close(population1, population2)

  def test_create_test_population_randomness(self) -> None:
    """Test that different keys produce different populations."""
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(43)
    population1 = create_test_population(100, 50, key1)
    population2 = create_test_population(100, 50, key2)

    # Should not be identical
    assert not jnp.all(population1 == population2)
