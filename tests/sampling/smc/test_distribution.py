"""Tests for device distribution strategies."""

import jax
import jax.numpy as jnp
import pytest

from proteinsmc.sampling.smc.distribution import (
  DeviceStrategy,
  estimate_island_memory_usage,
  get_island_distribution_strategy,
  validate_island_distribution,
)


def test_device_strategy_creation():
  """Test creating DeviceStrategy instances."""
  strategy = DeviceStrategy(
    strategy="direct",
    islands_per_device=1,
    active_devices=4,
    chunk_size=64,
  )
  
  assert strategy.strategy == "direct"
  assert strategy.islands_per_device == 1
  assert strategy.active_devices == 4
  assert strategy.chunk_size == 64


def test_get_island_distribution_strategy_direct():
  """Test distribution strategy when n_islands <= n_devices."""
  n_devices = jax.device_count()
  n_islands = min(n_devices, 2)  # Ensure we don't exceed available devices
  
  strategy = get_island_distribution_strategy(
    n_islands=n_islands,
    population_size_per_island=100,
    sequence_length=50,
  )
  
  assert strategy.strategy == "direct"
  assert strategy.islands_per_device == 1
  assert strategy.active_devices == n_islands
  assert strategy.chunk_size > 0


def test_get_island_distribution_strategy_batched():
  """Test distribution strategy when n_islands > n_devices."""
  n_devices = jax.device_count()
  n_islands = n_devices + 2  # Force batched strategy
  
  strategy = get_island_distribution_strategy(
    n_islands=n_islands,
    population_size_per_island=50,
    sequence_length=25,
  )
  
  assert strategy.strategy == "batched"
  assert strategy.islands_per_device >= 1
  assert strategy.active_devices == n_devices
  assert strategy.chunk_size > 0


def test_estimate_island_memory_usage():
  """Test memory usage estimation."""
  memory_mb = estimate_island_memory_usage(
    population_size_per_island=100,
    sequence_length=50,
    islands_per_device=2,
  )
  
  assert memory_mb > 0
  assert isinstance(memory_mb, float)
  
  # Test with different parameters
  memory_mb_larger = estimate_island_memory_usage(
    population_size_per_island=200,
    sequence_length=100,
    islands_per_device=2,
  )
  
  assert memory_mb_larger > memory_mb  # Larger populations should use more memory


def test_validate_island_distribution_feasible():
  """Test validation of feasible island distribution."""
  strategy = DeviceStrategy(
    strategy="direct",
    islands_per_device=1,
    active_devices=2,
    chunk_size=32,
  )
  
  result = validate_island_distribution(
    population_size_per_island=50,
    sequence_length=25,
    strategy=strategy,
    available_memory_mb=8 * 1024,  # 8GB
  )
  
  assert result["feasible"] is True
  assert isinstance(result["estimated_memory_mb"], (int, float)) and result["estimated_memory_mb"] > 0
  assert isinstance(result["safe_limit_mb"], (int, float)) and result["safe_limit_mb"] > 0
  assert isinstance(result["recommendation"], str) and "Configuration is feasible" in result["recommendation"]


def test_validate_island_distribution_infeasible():
  """Test validation of infeasible island distribution."""
  strategy = DeviceStrategy(
    strategy="batched",
    islands_per_device=20,  # Many islands per device
    active_devices=1,
    chunk_size=32,
  )
  
  result = validate_island_distribution(
    population_size_per_island=2000,  # Very large population
    sequence_length=1000,  # Very long sequences
    strategy=strategy,
    available_memory_mb=100,  # Very limited memory (100MB)
  )
  
  assert result["feasible"] is False
  estimated_mem = result["estimated_memory_mb"]
  safe_limit = result["safe_limit_mb"]
  assert isinstance(estimated_mem, (int, float)) and isinstance(safe_limit, (int, float))
  assert estimated_mem > safe_limit
  recommendation = result["recommendation"]
  assert isinstance(recommendation, str) and "Consider reducing population size" in recommendation


def test_memory_estimation_scaling():
  """Test that memory estimation scales correctly."""
  base_memory = estimate_island_memory_usage(
    population_size_per_island=100,
    sequence_length=50,
    islands_per_device=1,
  )
  
  # Double population size
  double_pop_memory = estimate_island_memory_usage(
    population_size_per_island=200,
    sequence_length=50,
    islands_per_device=1,
  )
  
  # Double islands per device
  double_islands_memory = estimate_island_memory_usage(
    population_size_per_island=100,
    sequence_length=50,
    islands_per_device=2,
  )
  
  # Both should approximately double the memory usage
  assert abs(double_pop_memory - 2 * base_memory) / base_memory < 0.1
  assert abs(double_islands_memory - 2 * base_memory) / base_memory < 0.1


def test_strategy_memory_awareness():
  """Test that strategy selection is memory-aware."""
  # Small memory scenario
  strategy_small = get_island_distribution_strategy(
    n_islands=8,
    population_size_per_island=1000,
    sequence_length=500,
    available_memory_mb=1024,  # 1GB
  )
  
  # Large memory scenario
  strategy_large = get_island_distribution_strategy(
    n_islands=8,
    population_size_per_island=1000,
    sequence_length=500,
    available_memory_mb=16 * 1024,  # 16GB
  )
  
  # With more memory, we might be able to fit more islands per device
  # The exact behavior depends on the algorithm, but we test that it runs
  assert strategy_small.islands_per_device >= 1
  assert strategy_large.islands_per_device >= 1


def test_chunk_size_selection():
  """Test that chunk sizes are reasonable."""
  strategy = get_island_distribution_strategy(
    n_islands=4,
    population_size_per_island=100,
    sequence_length=50,
  )
  
  # Chunk size should be reasonable (not too small, not larger than population)
  assert 16 <= strategy.chunk_size <= 100
  
  # Test with larger population
  strategy_large = get_island_distribution_strategy(
    n_islands=4,
    population_size_per_island=1000,
    sequence_length=50,
  )
  
  assert strategy_large.chunk_size > 0
  assert strategy_large.chunk_size <= 1000


def test_device_strategy_frozen():
  """Test that DeviceStrategy is frozen (immutable)."""
  strategy = DeviceStrategy(
    strategy="direct",
    islands_per_device=1,
    active_devices=2,
    chunk_size=32,
  )
  
  # Should not be able to modify fields (frozen dataclass)
  with pytest.raises(AttributeError):
    strategy.chunk_size = 64  # type: ignore[misc]
