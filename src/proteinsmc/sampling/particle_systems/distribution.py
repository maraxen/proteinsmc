"""Device distribution strategies for parallel replica SMC using pmap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Array, PyTree

from proteinsmc.utils import distribute


@dataclass(frozen=True)
class DeviceStrategy:
  """Strategy for distributing islands across devices."""

  strategy: Literal["direct", "batched"]
  islands_per_device: int
  active_devices: int
  chunk_size: int


def get_island_distribution_strategy(
  n_islands: int,
  population_size_per_island: int,
  sequence_length: int,
  available_memory_mb: float = 16 * 1024,
  safety_factor: float = 0.8,
) -> DeviceStrategy:
  """Get optimal strategy for distributing islands across devices.

  Args:
    n_islands: Number of islands to distribute
    population_size_per_island: Population size per island
    sequence_length: Length of sequences
    available_memory_mb: Available memory per device in MB
    safety_factor: Safety factor for memory usage

  Returns:
    DeviceStrategy with optimal configuration

  """
  n_devices = jax.device_count()

  base_memory_per_island = population_size_per_island * sequence_length * 4
  overhead_factor = 4.0
  memory_per_island_mb = (base_memory_per_island * overhead_factor) / (1024 * 1024)

  safe_memory_limit = available_memory_mb * safety_factor
  max_islands_per_device = max(1, int(safe_memory_limit / memory_per_island_mb))

  if n_islands <= n_devices:
    return DeviceStrategy(
      strategy="direct",
      islands_per_device=1,
      active_devices=min(n_islands, n_devices),
      chunk_size=min(population_size_per_island, 64),  # Conservative chunk size
    )

  optimal_islands_per_device = min(
    max_islands_per_device,
    (n_islands + n_devices - 1) // n_devices,
  )

  return DeviceStrategy(
    strategy="batched",
    islands_per_device=optimal_islands_per_device,
    active_devices=n_devices,
    chunk_size=min(population_size_per_island, 32),
  )


def distribute_islands_across_devices(
  island_step_fn: Callable[[Array, PyTree], Array],
  islands_data: Array,
  step_config: PyTree,
  strategy: DeviceStrategy,
) -> Array:
  """Distribute island processing across devices using pmap.

  Args:
    island_step_fn: Function to process islands
    islands_data: Island data to process
    step_config: Configuration for island processing
    strategy: Distribution strategy

  Returns:
    Processed island data

  """
  n_islands = islands_data.shape[0]

  if strategy.strategy == "direct" and n_islands <= strategy.active_devices:
    return distribute(
      island_step_fn,
      islands_data,
      chunk_size=strategy.chunk_size,
      static_args=step_config,
    )

  return _batched_island_processing(
    island_step_fn,
    islands_data,
    step_config,
    strategy,
  )


def _batched_island_processing(
  island_step_fn: Callable[[Array, PyTree], Array],
  islands_data: Array,
  step_config: PyTree,
  strategy: DeviceStrategy,
) -> Array:
  """Process islands in batches when n_islands > n_devices.

  Args:
    island_step_fn: Function to process islands
    islands_data: Island data to process
    step_config: Configuration for island processing
    strategy: Distribution strategy

  Returns:
    Processed island data

  """
  n_islands = islands_data.shape[0]
  n_devices = strategy.active_devices
  islands_per_device = strategy.islands_per_device

  total_capacity = n_devices * islands_per_device
  n_batches = (n_islands + total_capacity - 1) // total_capacity

  if n_batches == 1:
    padded_data = _pad_islands_for_devices(islands_data, strategy)
    processed_padded = distribute(
      island_step_fn,
      padded_data,
      chunk_size=strategy.chunk_size,
      static_args=step_config,
    )
    return processed_padded[:n_islands]  # Trim padding

  processed_islands = []
  for batch_idx in range(n_batches):
    start_idx = batch_idx * total_capacity
    end_idx = min(start_idx + total_capacity, n_islands)
    batch_data = islands_data[start_idx:end_idx]

    padded_batch = _pad_islands_for_devices(batch_data, strategy)
    processed_batch = distribute(
      island_step_fn,
      padded_batch,
      chunk_size=strategy.chunk_size,
      static_args=step_config,
    )

    actual_batch_size = end_idx - start_idx
    processed_islands.append(processed_batch[:actual_batch_size])

  return jnp.concatenate(processed_islands, axis=0)


def _pad_islands_for_devices(islands_data: Array, strategy: DeviceStrategy) -> Array:
  """Pad island data to fit device distribution strategy.

  Args:
    islands_data: Island data with shape (n_islands, ...)
    strategy: Distribution strategy

  Returns:
    Padded data suitable for device distribution

  """
  n_islands = islands_data.shape[0]
  target_size = strategy.active_devices * strategy.islands_per_device

  if n_islands >= target_size:
    return islands_data[:target_size]

  pad_width = [(0, target_size - n_islands)] + [(0, 0)] * (islands_data.ndim - 1)
  return jnp.pad(islands_data, pad_width, mode="edge")


def estimate_island_memory_usage(
  population_size_per_island: int,
  sequence_length: int,
  islands_per_device: int,
  dtype_size_bytes: int = 4,
) -> float:
  """Estimate memory usage for island processing.

  Args:
    population_size_per_island: Population size for each island
    sequence_length: Length of sequences
    islands_per_device: Number of islands per device
    dtype_size_bytes: Size of each element in bytes

  Returns:
    Estimated memory usage in MB per device

  """
  base_memory_per_island = population_size_per_island * sequence_length * dtype_size_bytes

  overhead_factor = 4.0  # Conservative estimate for SMC operations
  memory_per_island = base_memory_per_island * overhead_factor

  # Total memory per device
  return (memory_per_island * islands_per_device) / (1024 * 1024)


def validate_island_distribution(
  population_size_per_island: int,
  sequence_length: int,
  strategy: DeviceStrategy,
  available_memory_mb: float = 16 * 1024,
  safety_factor: float = 0.8,
) -> dict[str, bool | float | str]:
  """Validate that island distribution is feasible.

  Args:
    population_size_per_island: Population size per island
    sequence_length: Sequence length
    strategy: Distribution strategy
    available_memory_mb: Available memory per device
    safety_factor: Safety factor for memory usage

  Returns:
    Validation results

  """
  estimated_memory = estimate_island_memory_usage(
    population_size_per_island,
    sequence_length,
    strategy.islands_per_device,
  )

  safe_limit = available_memory_mb * safety_factor
  is_feasible = estimated_memory <= safe_limit

  return {
    "feasible": is_feasible,
    "estimated_memory_mb": estimated_memory,
    "safe_limit_mb": safe_limit,
    "strategy": strategy.strategy,
    "islands_per_device": strategy.islands_per_device,
    "recommendation": (
      "Configuration is feasible"
      if is_feasible
      else "Consider reducing population size or using more devices"
    ),
  }
