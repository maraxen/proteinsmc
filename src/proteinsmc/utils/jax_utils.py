from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import jax.tree_util
from jax import lax

if TYPE_CHECKING:
  from jaxtyping import Array, PyTree


def chunked_map(
  func: Callable,
  data: PyTree[Array],
  chunk_size: int,
  static_args: dict | None = None,
) -> PyTree[Array]:
  """Process a large batch of data in smaller chunks to conserve memory.

  This function is a wrapper around `jax.lax.map` with a `batch_size` argument,
  and provides a way to pass static arguments to the mapped function.

  Args:
    func: The function to apply to each element of the data. Must be mappable.
    data: The data to be processed. This can be a single array or a PyTree of
          arrays. All arrays must have a leading axis of the same size.
    chunk_size: The size of each chunk to process. This is passed as `batch_size`
                to `jax.lax.map`.
    static_args: A dictionary of static keyword arguments to be passed to `func`
                 on each call. These are not chunked or mapped over.

  Returns:
    The result of applying the function to the data, concatenated back into
    a single PyTree of arrays.

  """
  if not jax.tree_util.tree_leaves(data):
    return jax.tree_util.tree_map(lambda x: jnp.array([], dtype=x.dtype), data)

  kwargs = static_args or {}
  if static_args is not None and not isinstance(static_args, dict):
    msg = f"static_args must be a dictionary, but got {type(static_args)}"
    raise TypeError(msg)

  def func_to_map(x):
    if isinstance(x, tuple):
      return func(*x, **kwargs)
    return func(x, **kwargs)

  return lax.map(func_to_map, data, batch_size=chunk_size)
