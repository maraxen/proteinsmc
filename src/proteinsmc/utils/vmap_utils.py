"""Utilities for chunked vmap processing in JAX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import lax

if TYPE_CHECKING:
  from jaxtyping import Array, PyTree


def chunked_vmap(
  func: Callable,
  data: PyTree[Array],
  chunk_size: int,
  static_args: PyTree | None = None,
) -> PyTree[Array]:
  """Process a large batch of data in smaller chunks to conserve memory.

  This function uses jax.lax.scan to iterate over a vmapped function,
  ensuring that only one chunk is in memory at any given time.

  Args:
    func: The function to apply to each element of the data. Must be vmappable.
    data: The data to be processed. This can be a single array or a PyTree of arrays.
          All arrays must have a leading axis of the same size.
    chunk_size: The size of each chunk to process.
    static_args: A PyTree of static arguments to be passed to `func` on each call.
                  These are not chunked or mapped over.

  Returns:
    The result of applying the function to the data, concatenated back into
    a single PyTree of arrays.

  """
  if not jax.tree_util.tree_leaves(data):
    return data
  num_entries = jax.tree_util.tree_leaves(data)[0].shape[0]
  num_chunks = (num_entries + chunk_size - 1) // chunk_size

  def pad_and_reshape(x: Array) -> Array:
    """Pad and reshape the input array to fit into chunks."""
    padded_size = num_chunks * chunk_size
    pad_width = ((0, padded_size - num_entries),) + ((0, 0),) * (x.ndim - 1)
    padded_x = jnp.pad(x, pad_width, mode="constant")
    return padded_x.reshape((num_chunks, chunk_size) + x.shape[1:])

  chunked_data = jax.tree_util.tree_map(pad_and_reshape, data)

  vmapped_func = jax.vmap(fun=func, in_axes=(0, None) if static_args else 0)

  def scan_body(_: None, chunk: Array) -> tuple[None, Array]:
    """Process a single chunk of data."""
    result_chunk = vmapped_func(chunk, static_args) if static_args else vmapped_func(chunk)
    return _, result_chunk

  # --- Run the Scan and Un-pad ---
  _, result_chunked = lax.scan(scan_body, None, chunked_data)

  def unpad_and_flatten(res_chunk: Array) -> Array:
    """Unpad the result chunk and flatten it."""
    flat_res = res_chunk.reshape((-1,) + res_chunk.shape[2:])
    return flat_res[:num_entries]

  return jax.tree_util.tree_map(unpad_and_flatten, result_chunked)
