"""Utilities for chunked vmap processing in JAX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import lax

if TYPE_CHECKING:
  from typing import Any, Sequence

  from jaxtyping import Array, PyTree


def chunked_vmap(  # noqa: PLR0913
  func: Callable,
  data: PyTree[Array],
  chunk_size: int,
  in_axes: int | Sequence[Any] | None = 0,
  out_axes: int | tuple[int, ...] | None = 0,
  static_args: dict | None = None,
) -> PyTree[Array]:
  """Process a large batch of data in smaller chunks to conserve memory.

  This function is a wrapper around `jax.vmap` that uses `jax.lax.scan`
  to iterate over the function, ensuring that only one chunk is in memory
  at any given time. It is intended for vectorizing a function that acts
  on a single item.

  Args:
    func: The function to apply to each element of the data. Must be vmappable.
    data: The data to be processed. This can be a single array or a PyTree of
          arrays. All arrays must have a leading axis of the same size.
    chunk_size: The size of each chunk to process.
    in_axes: Axes to map over for the input data (passed to jax.vmap).
    out_axes: Axes to map over for the output data (passed to jax.vmap).
    static_args: A dictionary of static keyword arguments to be passed to `func`
                 on each call. These are not chunked or mapped over.

  Returns:
    The result of applying the function to the data, concatenated back into
    a single PyTree of arrays.

  """
  if not jax.tree_util.tree_leaves(data):
    return jax.tree_util.tree_map(lambda x: jnp.array([], dtype=x.dtype), data)

  num_entries = jax.tree_util.tree_leaves(data)[0].shape[0]
  if num_entries == 0:
    return jax.tree_util.tree_map(lambda x: jnp.array([], dtype=x.dtype), data)

  num_chunks = (num_entries + chunk_size - 1) // chunk_size
  padded_size = num_chunks * chunk_size

  def pad_and_reshape(x: Array) -> Array:
    """Pad array to be divisible by chunk_size and reshape."""
    pad_width = ((0, padded_size - x.shape[0]),) + ((0, 0),) * (x.ndim - 1)
    padded_x = jnp.pad(x, pad_width, mode="constant")
    return padded_x.reshape((num_chunks, chunk_size) + x.shape[1:])

  chunked_data = jax.tree_util.tree_map(pad_and_reshape, data)

  if static_args is not None:
    if not isinstance(static_args, dict):
      msg = f"static_args must be a dictionary, but got {type(static_args)}"
      raise TypeError(msg)

    def func_to_vmap(*dynamic_args: Array) -> Array:
      return func(*dynamic_args, **static_args)
  else:

    def func_to_vmap(*dynamic_args: Array) -> Array:
      return func(*dynamic_args)

  vmapped_func = jax.vmap(
    func_to_vmap,
    in_axes=in_axes,
    out_axes=out_axes,
  )

  def scan_body(_carry: None, chunk: Array) -> tuple[None, Array]:
    """Process one chunk, which contains only the dynamic data."""
    return _carry, vmapped_func(*chunk) if isinstance(chunk, tuple) else vmapped_func(chunk)

  _, result_chunked = lax.scan(scan_body, None, chunked_data)

  def unpad_and_flatten(res_chunk: Array) -> Array:
    """Unpad the result chunk and flatten it back to a single array."""
    flat_res = res_chunk.reshape((padded_size,) + res_chunk.shape[2:])
    return flat_res[:num_entries]

  return jax.tree_util.tree_map(unpad_and_flatten, result_chunked)
