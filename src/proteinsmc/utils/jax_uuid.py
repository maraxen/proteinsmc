"""Utility functions for working with UUIDs in JAX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Int, PRNGKeyArray

  from proteinsmc.models.types import UUIDArray


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
