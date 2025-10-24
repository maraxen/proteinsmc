"""Tests for PRNG key management utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from proteinsmc.utils.key_management import split_key_batched, split_key_for_sampler


def test_split_key_for_sampler_basic() -> None:
  """Test basic key splitting functionality.

  Verifies that split_key_for_sampler correctly splits a key into the requested
  number of child keys plus one next key.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If any keys are equal or have incorrect shapes.

  Example:
    >>> test_split_key_for_sampler_basic()

  """
  key = jax.random.PRNGKey(42)
  key1, key2, next_key = split_key_for_sampler(key, 2)

  # All keys should be different
  assert not jnp.array_equal(key1, key2)
  assert not jnp.array_equal(key1, next_key)
  assert not jnp.array_equal(key2, next_key)

  # All should be valid PRNGKeys (shape (2,) for jax.random.PRNGKey)
  assert key1.shape == (2,)
  assert key2.shape == (2,)
  assert next_key.shape == (2,)


def test_split_key_for_sampler_reproducibility() -> None:
  """Test that key splitting is deterministic.

  The same parent key should always produce the same child keys.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If keys from same parent are not identical.

  Example:
    >>> test_split_key_for_sampler_reproducibility()

  """
  key = jax.random.PRNGKey(123)

  keys1 = split_key_for_sampler(key, 3)
  keys2 = split_key_for_sampler(key, 3)

  for k1, k2 in zip(keys1, keys2, strict=True):
    assert jnp.array_equal(k1, k2)


def test_split_key_for_sampler_single_split() -> None:
  """Test splitting with n_splits=1.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If child key equals next key or shapes are incorrect.

  Example:
    >>> test_split_key_for_sampler_single_split()

  """
  key = jax.random.PRNGKey(42)
  child_key, next_key = split_key_for_sampler(key, 1)

  assert not jnp.array_equal(child_key, next_key)
  assert child_key.shape == (2,)


def test_split_key_for_sampler_invalid() -> None:
  """Test that invalid n_splits raises ValueError.

  Args:
    None

  Returns:
    None

  Raises:
    ValueError: If n_splits is less than 1 (expected behavior).

  Example:
    >>> test_split_key_for_sampler_invalid()

  """
  key = jax.random.PRNGKey(42)

  with pytest.raises(ValueError, match="n_splits must be >= 1"):
    split_key_for_sampler(key, 0)

  with pytest.raises(ValueError, match="n_splits must be >= 1"):
    split_key_for_sampler(key, -1)


def test_split_key_batched_shape() -> None:
  """Test batched key splitting produces correct shape.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If batched keys have incorrect shape.

  Example:
    >>> test_split_key_batched_shape()

  """
  key = jax.random.PRNGKey(42)
  batch_size = 5

  batched_keys = split_key_batched(key, batch_size)

  assert batched_keys.shape == (batch_size, 2)


def test_split_key_batched_independence() -> None:
  """Test that batched keys are independent.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If any two keys in the batch are equal.

  Example:
    >>> test_split_key_batched_independence()

  """
  key = jax.random.PRNGKey(42)
  batched_keys = split_key_batched(key, 10)

  # Each key should be unique
  for i in range(10):
    for j in range(i + 1, 10):
      assert not jnp.array_equal(batched_keys[i], batched_keys[j])


def test_split_key_batched_reproducibility() -> None:
  """Test batched splitting is deterministic.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If same parent key produces different batched keys.

  Example:
    >>> test_split_key_batched_reproducibility()

  """
  key = jax.random.PRNGKey(123)

  batch1 = split_key_batched(key, 4)
  batch2 = split_key_batched(key, 4)

  assert jnp.array_equal(batch1, batch2)
