"""Tests for blackjax utility functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.utils.blackjax_utils import make_blackjax_log_prob_fn


class TestMakeBlackjaxLogProbFn:
  """Test the make_blackjax_log_prob_fn function."""

  def test_basic_functionality(self, rng_key) -> None:
    """Test that the function correctly wraps a fitness function.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If the output format is incorrect.

    Example:
        >>> test_basic_functionality(jax.random.PRNGKey(42))

    """
    # Create a simple mock fitness function
    def mock_fitness_fn(key, sequence, context):
      # Return combined fitness and components
      combined = jnp.sum(sequence.astype(jnp.float32))
      components = jnp.array([combined / 2, combined / 2])
      return jnp.stack([combined, *components])

    # Create a sample sequence
    sequence = jnp.array([0, 1, 2, 3], dtype=jnp.int8)

    # Wrap the fitness function
    blackjax_fn = make_blackjax_log_prob_fn(mock_fitness_fn, rng_key)

    # Call the wrapped function
    result = blackjax_fn(sequence)

    # Verify the result structure
    assert isinstance(result, jax.Array)
    assert_shape(result, (3,))  # combined + 2 components
    assert jnp.isfinite(result).all()

  def test_with_different_sequence_lengths(self, rng_key) -> None:
    """Test with sequences of varying lengths.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If the function fails with different sequence lengths.

    Example:
        >>> test_with_different_sequence_lengths(jax.random.PRNGKey(42))

    """
    def mock_fitness_fn(key, sequence, context):
      combined = jnp.sum(sequence.astype(jnp.float32))
      return jnp.array([combined, combined / 2])

    blackjax_fn = make_blackjax_log_prob_fn(mock_fitness_fn, rng_key)

    # Test with different lengths
    for length in [1, 5, 10, 20]:
      sequence = jnp.arange(length, dtype=jnp.int8)
      result = blackjax_fn(sequence)
      assert isinstance(result, jax.Array)
      assert jnp.isfinite(result).all()

  def test_deterministic_with_same_key(self, rng_key) -> None:
    """Test that the same key produces deterministic results.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If results are not deterministic.

    Example:
        >>> test_deterministic_with_same_key(jax.random.PRNGKey(42))

    """
    def mock_fitness_fn(key, sequence, context):
      # Add some randomness based on key
      noise = jax.random.normal(key, shape=())
      combined = jnp.sum(sequence.astype(jnp.float32)) + noise
      return jnp.array([combined, combined / 2])

    sequence = jnp.array([0, 1, 2, 3], dtype=jnp.int8)

    # Create two instances with the same key
    blackjax_fn_1 = make_blackjax_log_prob_fn(mock_fitness_fn, rng_key)
    blackjax_fn_2 = make_blackjax_log_prob_fn(mock_fitness_fn, rng_key)

    result_1 = blackjax_fn_1(sequence)
    result_2 = blackjax_fn_2(sequence)

    # Results should be identical with the same key
    assert jnp.allclose(result_1, result_2)

  def test_jit_compatibility(self, rng_key) -> None:
    """Test that the wrapped function is JIT-compatible.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_jit_compatibility(jax.random.PRNGKey(42))

    """
    def mock_fitness_fn(key, sequence, context):
      combined = jnp.sum(sequence.astype(jnp.float32))
      return jnp.array([combined, combined / 2])

    blackjax_fn = make_blackjax_log_prob_fn(mock_fitness_fn, rng_key)
    jitted_fn = jax.jit(blackjax_fn)

    sequence = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    result_regular = blackjax_fn(sequence)
    result_jitted = jitted_fn(sequence)

    # Both should produce the same result
    assert jnp.allclose(result_regular, result_jitted)

  def test_with_empty_sequence(self, rng_key) -> None:
    """Test with an empty sequence.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If empty sequence handling fails.

    Example:
        >>> test_with_empty_sequence(jax.random.PRNGKey(42))

    """
    def mock_fitness_fn(key, sequence, context):
      combined = jnp.sum(sequence.astype(jnp.float32))
      return jnp.array([combined, combined / 2])

    blackjax_fn = make_blackjax_log_prob_fn(mock_fitness_fn, rng_key)

    sequence = jnp.array([], dtype=jnp.int8)
    result = blackjax_fn(sequence)

    assert isinstance(result, jax.Array)
    assert jnp.isfinite(result).all()
