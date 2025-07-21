"""General use JAX utility functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pad_array(array: jax.Array, max_len: int, value: int = 0) -> jax.Array:
  """Pad the first dimension of an array to a max_len."""
  pad_amount = max_len - array.shape[0]
  pad_shape = [(0, pad_amount)] + [(0, 0)] * (array.ndim - 1)
  return jnp.pad(array, pad_shape, mode="constant", constant_values=value)
