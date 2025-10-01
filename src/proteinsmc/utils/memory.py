"""Auto-tuning utilities for memory and performance optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from proteinsmc.utils.jax_utils import chunked_map

if TYPE_CHECKING:
  from jaxtyping import Array, PRNGKeyArray, PyTree

  from proteinsmc.models.memory import AutoTuningConfig


@dataclass
class BenchmarkResult:
  """Results from a performance benchmark."""

  chunk_size: int
  avg_time_per_batch: float
  memory_usage_mb: float
  success: bool
  error_message: str | None = None


def estimate_memory_usage(
  population_size: int,
  sequence_length: int,
  dtype_size_bytes: int = 4,
) -> float:
  """Estimate memory usage for a population in MB.

  Args:
    population_size: Number of sequences in population
    sequence_length: Length of each sequence
    dtype_size_bytes: Size of each element in bytes

  Returns:
    Estimated memory usage in MB

  """
  base_memory = population_size * sequence_length * dtype_size_bytes
  overhead_factor = 3.0  # Conservative estimate
  total_bytes = base_memory * overhead_factor
  return total_bytes / (1024 * 1024)


def get_device_memory_mb() -> float:
  """Get available device memory in MB."""
  try:
    devices = jax.devices()
    if devices and hasattr(devices[0], "memory_stats"):
      stats = devices[0].memory_stats()
      return stats.get("bytes_limit", 8 * 1024**3) / (1024**2)

    device_type = devices[0].device_kind if devices else "cpu"
    if device_type == "gpu":
      return 16 * 1024
    if device_type == "tpu":
      return 32 * 1024
    return 8 * 1024  # noqa: TRY300
  except (AttributeError, IndexError, KeyError):
    return 8 * 1024


def suggest_chunk_size_heuristic(
  population_size: int,
  sequence_length: int,
  config: AutoTuningConfig,
) -> int:
  """Suggest chunk size using simple heuristics.

  Args:
    population_size: Size of the population to process
    sequence_length: Length of sequences
    config: Auto-tuning configuration

  Returns:
    Suggested chunk size

  """
  if not config.enable_auto_tuning:
    return min(population_size, 64)

  available_memory_mb = get_device_memory_mb() * config.memory_safety_factor
  max_chunk_by_memory = int(available_memory_mb / estimate_memory_usage(1, sequence_length))

  min_chunk_for_efficiency = max(16, population_size // 32)

  device_type = jax.devices()[0].device_kind if jax.devices() else "cpu"
  if device_type == "tpu":
    device_multiplier = 2.0
  elif device_type == "gpu":
    device_multiplier = 1.0
  else:
    device_multiplier = 0.5

  suggested = int(
    min(
      max_chunk_by_memory,
      min_chunk_for_efficiency * device_multiplier,
      population_size,
    ),
  )

  return min(config.probe_chunk_sizes, key=lambda x: abs(x - suggested))


def benchmark_chunk_size(
  func: Callable,
  test_data: PyTree[Array],
  chunk_size: int,
  num_trials: int = 3,
  static_args: PyTree | None = None,
) -> BenchmarkResult:
  """Benchmark a specific chunk size.

  Args:
    func: Function to benchmark
    test_data: Representative test data
    chunk_size: Chunk size to test
    num_trials: Number of trials to average
    static_args: Static arguments for the function

  Returns:
    Benchmark results

  """
  try:
    _ = chunked_map(func, test_data, chunk_size, static_args=static_args)

    times = []
    for _ in range(num_trials):
      start_time = time.time()
      result = chunked_map(func, test_data, chunk_size, static_args=static_args)
      jax.block_until_ready(result)
      end_time = time.time()
      times.append(end_time - start_time)

    avg_time = sum(times) / len(times)

    test_leaves = jax.tree_util.tree_leaves(test_data)
    sequence_length = test_leaves[0].shape[1] if len(test_leaves[0].shape) > 1 else 1
    estimated_memory = estimate_memory_usage(chunk_size, sequence_length)

    return BenchmarkResult(
      chunk_size=chunk_size,
      avg_time_per_batch=avg_time,
      memory_usage_mb=estimated_memory,
      success=True,
    )

  except (RuntimeError, ValueError, MemoryError) as e:
    return BenchmarkResult(
      chunk_size=chunk_size,
      avg_time_per_batch=float("inf"),
      memory_usage_mb=float("inf"),
      success=False,
      error_message=str(e),
    )


def auto_tune_chunk_size(
  func: Callable,
  test_data: PyTree[Array],
  config: AutoTuningConfig,
  static_args: PyTree | None = None,
) -> int:
  """Auto-tune chunk size for optimal performance.

  Args:
    func: Function to optimize chunk size for
    test_data: Representative test data for benchmarking
    config: Auto-tuning configuration
    static_args: Static arguments for the function

  Returns:
    Optimal chunk size

  """
  if not config.enable_auto_tuning:
    test_leaves = jax.tree_util.tree_leaves(test_data)
    population_size = test_leaves[0].shape[0]
    sequence_length = test_leaves[0].shape[1] if len(test_leaves[0].shape) > 1 else 1
    return suggest_chunk_size_heuristic(population_size, sequence_length, config)

  test_leaves = jax.tree_util.tree_leaves(test_data)
  population_size = test_leaves[0].shape[0]
  sequence_length = test_leaves[0].shape[1] if len(test_leaves[0].shape) > 1 else 1
  heuristic_size = suggest_chunk_size_heuristic(population_size, sequence_length, config)

  valid_sizes = [s for s in config.probe_chunk_sizes if s <= population_size]
  if not valid_sizes:
    return heuristic_size

  results = []
  for chunk_size in valid_sizes:
    result = benchmark_chunk_size(
      func,
      test_data,
      chunk_size,
      config.max_probe_iterations,
      static_args,
    )
    results.append(result)

    if not result.success:
      break

  successful_results = [r for r in results if r.success]
  if not successful_results:
    return heuristic_size

  optimal = min(successful_results, key=lambda r: r.avg_time_per_batch)

  heuristic_result = next(
    (r for r in successful_results if r.chunk_size == heuristic_size),
    None,
  )
  if heuristic_result and heuristic_result.avg_time_per_batch <= (
    optimal.avg_time_per_batch * (1 + config.performance_tolerance)
  ):
    return heuristic_size

  return optimal.chunk_size


def create_test_population(
  population_size: int,
  sequence_length: int,
  key: PRNGKeyArray,
) -> Array:
  """Create a test population for benchmarking.

  Args:
    population_size: Size of test population
    sequence_length: Length of sequences
    key: PRNG key

  Returns:
    Test population array

  """
  return jax.random.randint(
    key,
    shape=(population_size, sequence_length),
    minval=0,
    maxval=20,
    dtype=jnp.int32,
  )
