"""Tests for the combine scoring module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.scoring.combine import make_sum_combine, make_weighted_combine


class TestSumCombine:
  """Test the sum_combine function factory."""

  def test_make_sum_combine_returns_callable(self) -> None:
    """Test that make_sum_combine returns a callable function.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the returned object is not callable.

    Example:
        >>> test_make_sum_combine_returns_callable()

    """
    combine_fn = make_sum_combine()
    assert callable(combine_fn)

  def test_sum_combine_basic(self) -> None:
    """Test basic sum combination of fitness scores.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the sum is not calculated correctly.

    Example:
        >>> test_sum_combine_basic()

    """
    combine_fn = make_sum_combine()

    # Create a simple array of scores
    scores = jnp.array([1.0, 2.0, 3.0])

    result = combine_fn(scores, None, None)

    # Should sum to 6.0
    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert jnp.allclose(result, 6.0)

  def test_sum_combine_2d_array(self) -> None:
    """Test sum combination with 2D array (sums along axis 0).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the sum is not calculated correctly.

    Example:
        >>> test_sum_combine_2d_array()

    """
    combine_fn = make_sum_combine()

    # Create a 2D array of scores (3 fitness functions, 4 samples)
    scores = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5], [0.2, 0.3, 0.4, 0.5]])

    result = combine_fn(scores, None, None)

    # Should sum along axis 0
    expected = jnp.array([1.7, 3.8, 5.9, 8.0])
    assert isinstance(result, jax.Array)
    assert_shape(result, (4,))
    assert jnp.allclose(result, expected)

  def test_sum_combine_with_negative_values(self) -> None:
    """Test sum combination with negative fitness scores.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If negative values are not handled correctly.

    Example:
        >>> test_sum_combine_with_negative_values()

    """
    combine_fn = make_sum_combine()

    scores = jnp.array([-1.0, 2.0, -3.0])

    result = combine_fn(scores, None, None)

    assert jnp.allclose(result, -2.0)

  def test_sum_combine_is_jittable(self) -> None:
    """Test that sum_combine function is JIT-compatible.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_sum_combine_is_jittable()

    """
    combine_fn = make_sum_combine()
    scores = jnp.array([1.0, 2.0, 3.0])

    # Should execute without error
    result = combine_fn(scores, None, None)

    assert jnp.allclose(result, 6.0)


class TestWeightedCombine:
  """Test the weighted_combine function factory."""

  def test_make_weighted_combine_returns_callable(self) -> None:
    """Test that make_weighted_combine returns a callable function.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the returned object is not callable.

    Example:
        >>> test_make_weighted_combine_returns_callable()

    """
    combine_fn = make_weighted_combine()
    assert callable(combine_fn)

  def test_weighted_combine_without_weights(self) -> None:
    """Test weighted combination without explicit weights (defaults to sum).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the default sum is not calculated correctly.

    Example:
        >>> test_weighted_combine_without_weights()

    """
    combine_fn = make_weighted_combine(fitness_weights=None)
    key = jax.random.PRNGKey(0)

    scores = jnp.array([1.0, 2.0, 3.0])

    result = combine_fn(scores, key, None)

    # Should sum to 6.0 when no weights provided
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, 6.0)

  def test_weighted_combine_with_weights(self) -> None:
    """Test weighted combination with explicit weights.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the weighted sum is not calculated correctly.

    Example:
        >>> test_weighted_combine_with_weights()

    """
    weights = jnp.array([0.5, 0.3, 0.2])
    combine_fn = make_weighted_combine(fitness_weights=weights)
    key = jax.random.PRNGKey(0)

    scores = jnp.array([10.0, 20.0, 30.0])

    result = combine_fn(scores, key, None)

    # Should be: 0.5*10 + 0.3*20 + 0.2*30 = 5 + 6 + 6 = 17
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, 17.0)

  def test_weighted_combine_2d_with_weights(self) -> None:
    """Test weighted combination with 2D array and weights.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the weighted sum is not calculated correctly.

    Example:
        >>> test_weighted_combine_2d_with_weights()

    """
    weights = jnp.array([0.5, 0.3, 0.2])
    combine_fn = make_weighted_combine(fitness_weights=weights)
    key = jax.random.PRNGKey(0)

    # Shape: (3 fitness functions, 2 samples)
    scores = jnp.array([[10.0, 20.0], [5.0, 10.0], [2.0, 4.0]])

    result = combine_fn(scores, key, None)

    # For each sample: weight[0]*score[0] + weight[1]*score[1] + weight[2]*score[2]
    # Sample 1: 0.5*10 + 0.3*5 + 0.2*2 = 5 + 1.5 + 0.4 = 6.9
    # Sample 2: 0.5*20 + 0.3*10 + 0.2*4 = 10 + 3 + 0.8 = 13.8
    expected = jnp.array([6.9, 13.8])

    assert isinstance(result, jax.Array)
    assert_shape(result, (2,))
    assert jnp.allclose(result, expected)

  def test_weighted_combine_with_zero_weights(self) -> None:
    """Test weighted combination with some zero weights.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If zero weights are not handled correctly.

    Example:
        >>> test_weighted_combine_with_zero_weights()

    """
    weights = jnp.array([1.0, 0.0, 0.0])
    combine_fn = make_weighted_combine(fitness_weights=weights)
    key = jax.random.PRNGKey(0)

    scores = jnp.array([5.0, 10.0, 15.0])

    result = combine_fn(scores, key, None)

    # Should only use first score
    assert jnp.allclose(result, 5.0)

  def test_weighted_combine_normalization(self) -> None:
    """Test weighted combination with non-normalized weights.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If non-normalized weights are not handled correctly.

    Example:
        >>> test_weighted_combine_normalization()

    """
    # Weights don't sum to 1
    weights = jnp.array([2.0, 1.0, 1.0])
    combine_fn = make_weighted_combine(fitness_weights=weights)
    key = jax.random.PRNGKey(0)

    scores = jnp.array([1.0, 1.0, 1.0])

    result = combine_fn(scores, key, None)

    # Should be: 2*1 + 1*1 + 1*1 = 4
    assert jnp.allclose(result, 4.0)

  def test_weighted_combine_is_jittable(self) -> None:
    """Test that weighted_combine function is JIT-compatible.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_weighted_combine_is_jittable()

    """
    weights = jnp.array([0.5, 0.5])
    combine_fn = make_weighted_combine(fitness_weights=weights)
    key = jax.random.PRNGKey(0)
    scores = jnp.array([2.0, 4.0])

    # Should execute without error
    result = combine_fn(scores, key, None)

    # 0.5*2 + 0.5*4 = 3.0
    assert jnp.allclose(result, 3.0)

  def test_weighted_combine_with_negative_scores(self) -> None:
    """Test weighted combination with negative fitness scores.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If negative scores are not handled correctly.

    Example:
        >>> test_weighted_combine_with_negative_scores()

    """
    weights = jnp.array([0.5, 0.5])
    combine_fn = make_weighted_combine(fitness_weights=weights)
    key = jax.random.PRNGKey(0)

    scores = jnp.array([-2.0, 4.0])

    result = combine_fn(scores, key, None)

    # 0.5*(-2) + 0.5*4 = -1 + 2 = 1.0
    assert jnp.allclose(result, 1.0)
