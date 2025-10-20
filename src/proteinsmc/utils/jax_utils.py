"""Utilities for JAX-based operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.tree_util
from jax import lax

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Int, PRNGKeyArray, PyTree

  from proteinsmc.models.types import UUIDArray


import jax


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

  def func_to_map(x: PyTree[Array]) -> PyTree[Array]:
    if isinstance(x, tuple):
      return func(*x, **kwargs)
    return func(x, **kwargs)

  return lax.map(func_to_map, data, batch_size=chunk_size)


def generate_jax_uuid(key: PRNGKeyArray) -> tuple[UUIDArray, PRNGKeyArray]:
  """Generate a UUID using JAX's random number generator.

  Args:
      key: A JAX PRNG key.

  Returns:
      A tuple containing:
        - The generated UUID as a JAX array of uint8.
        - The updated PRNG key.

  """
  new_key, subkey = jax.random.split(key)
  uuid_array = jax.random.randint(
    key=subkey,
    shape=(16,),
    minval=0,
    maxval=256,
    dtype=jnp.uint8,
  )
  return uuid_array, new_key


def generate_jax_hash(key: PRNGKeyArray, data: Int) -> tuple[UUIDArray, PRNGKeyArray]:
  """Generate a hash-based UUID using JAX's random number generator and input data.

  Args:
      key: A JAX PRNG key.
      data: An integer input to hash.

  Returns:
      A tuple containing:
        - The generated hash-based UUID as a JAX array of uint8.
        - The updated PRNG key.

  """
  new_key, subkey = jax.random.split(key)
  hash_value = jax.random.fold_in(subkey, data)
  hash_bytes = jnp.array(
    [(hash_value >> (i * 8)) & 0xFF for i in range(16)],
    dtype=jnp.uint8,
  )
  return hash_bytes, new_key
