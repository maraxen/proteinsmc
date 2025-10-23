"""Utilities for standardized PRNG key management across samplers.

This module provides helper functions for consistent key splitting and threading
to ensure reproducibility and proper randomness in JAX-based samplers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray


def split_key_for_sampler(
  key: PRNGKeyArray,
  n_splits: int,
) -> tuple[PRNGKeyArray, ...]:
  """Split a PRNG key into child keys for sampler operations.

  This function provides a standardized way to split keys with clear documentation
  of what each child key is used for. The last key in the returned tuple is always
  the "next" key to be stored in the sampler state for the next iteration.

  Args:
    key: Parent PRNG key to split.
    n_splits: Number of child keys needed (excluding the "next" key).

  Returns:
    Tuple of (child_key_1, ..., child_key_n, next_key) where:
    - child_key_i: Keys for current iteration operations (mutation, fitness, etc.).
    - next_key: Key to store in state for next iteration.

  Raises:
    ValueError: If n_splits < 1.

  Example:
    >>> import jax
    >>> key = jax.random.PRNGKey(42)
    >>> key_fitness, key_mutation, next_key = split_key_for_sampler(key, 2)
    >>> # Use key_fitness for fitness evaluation
    >>> # Use key_mutation for mutation operations
    >>> # Store next_key in state for next iteration

  """
  if n_splits < 1:
    msg = f"n_splits must be >= 1, got {n_splits}"
    raise ValueError(msg)

  return tuple(jax.random.split(key, n_splits + 1))


def split_key_batched(
  key: PRNGKeyArray,
  batch_size: int,
) -> PRNGKeyArray:
  """Split a key into a batch of independent keys.

  Useful for parallel operations like PRSMC islands or population-based samplers.

  Args:
    key: Parent PRNG key.
    batch_size: Number of independent keys needed.

  Returns:
    Array of shape (batch_size, 2) containing independent PRNG keys.

  Example:
    >>> import jax
    >>> key = jax.random.PRNGKey(42)
    >>> island_keys = split_key_batched(key, n_islands=4)
    >>> # island_keys.shape == (4, 2)

  """
  return jax.random.split(key, batch_size)
