"""Parallel Map Utilities for JAX."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, PyTree


from proteinsmc.utils import jax_utils


def distribute(
  core_fn: Callable,
  data: PyTree[Array],
  chunk_size: int,
  static_args: PyTree | None = None,
) -> Array:
  """Distribute a computation across all available JAX devices.

  This function shards data, uses pmap to run a chunked vmap on each device,
  and consolidates the results.

  Args:
      core_fn: The scientific logic function to be applied to a single data point.
      data: A PyTree of arrays to be processed.
      key: A JAX PRNG key.
      chunk_size: The micro-batch size for processing on each device.
      static_args: A PyTree of static arguments for core_fn.

  Returns:
      The consolidated results from all devices.

  """
  num_devices = jax.device_count()

  def shard_array(x: Array) -> Array:
    """Shard an array across devices."""
    return x.reshape((num_devices, -1) + x.shape[1:])

  sharded_data = jax.tree_util.tree_map(shard_array, data)

  @partial(jax.pmap, axis_name="devices")
  def pmapped_worker(data: PyTree[Array]) -> Array:
    return jax_utils.chunked_map(core_fn, data, chunk_size, static_args=static_args)

  sharded_results = pmapped_worker(sharded_data)

  def unshard_array(x: Array) -> Array:
    return x.reshape((-1,) + x.shape[2:])

  return jax.tree_util.tree_map(unshard_array, sharded_results)
