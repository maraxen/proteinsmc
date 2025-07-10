"""Utilities for chunked vmap processing in JAX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import lax

if TYPE_CHECKING:
  from jaxtyping import Array, PyTree


def chunked_vmap(  # noqa: PLR0913
  func: Callable,
  data: PyTree[Array],
  chunk_size: int,
  in_axes: int | tuple[int, ...] | None = 0,
  out_axes: int | tuple[int, ...] | None = 0,
  concat_axis: int = 0,
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
    in_axes: Axes to map over for the input data (passed to jax.vmap).
    out_axes: Axes to map over for the output data (passed to jax.vmap).
    concat_axis: Axis along which to concatenate the results.
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
    if concat_axis != 0:
      padded_x = jnp.moveaxis(padded_x, concat_axis, 0)
    return padded_x.reshape((num_chunks, chunk_size) + padded_x.shape[1:])

  chunked_data = jax.tree_util.tree_map(pad_and_reshape, data)

  vmap_in_axes = (0, None) if static_args is not None else 0
  if in_axes is not None:
    vmap_in_axes = (in_axes, None) if static_args is not None else in_axes

  vmap_out_axes = out_axes

  vmapped_func = jax.vmap(
    func,
    in_axes=vmap_in_axes,
    out_axes=vmap_out_axes,
  )

  def scan_body(_: None, chunk: PyTree[Array]) -> tuple[None, PyTree[Array]]:
    """Process a single chunk of data."""
    # Unpack the chunk PyTree into positional arguments for the vmapped function.
    args = chunk if isinstance(chunk, tuple) else (chunk,)
    if static_args is not None:
      result_chunk = vmapped_func(*args, static_args)
    else:
      result_chunk = vmapped_func(*args)
    return _, result_chunk

  _, result_chunked = lax.scan(scan_body, None, chunked_data)

  def unpad_and_flatten(res_chunk: Array) -> Array:
    """Unpad the result chunk and flatten it."""
    if concat_axis != 0:
      res_chunk = jnp.moveaxis(res_chunk, 0, concat_axis)
    flat_res = res_chunk.reshape((-1,) + res_chunk.shape[2:])
    return flat_res[:num_entries]

  return jax.tree_util.tree_map(unpad_and_flatten, result_chunked)
