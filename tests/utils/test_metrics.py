"""Tests for metrics utility functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.utils.metrics import (
  calculate_logZ_increment,
  calculate_position_entropy,
  safe_weighted_mean,
  shannon_entropy,
)


class TestSafeWeightedMean:
  """Test the safe_weighted_mean function."""

  def test_safe_weighted_mean_basic(self) -> None:
    """Test basic weighted mean calculation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the weighted mean is not calculated correctly.

    Example:
        >>> test_safe_weighted_mean_basic()

    """
    metric = jnp.array([1.0, 2.0, 3.0])
    weights = jnp.array([0.5, 0.3, 0.2])
    valid_mask = jnp.array([True, True, True])
    sum_valid_w = jnp.sum(weights)

    result = safe_weighted_mean(metric, weights, valid_mask, sum_valid_w)

    # Expected: (1.0*0.5 + 2.0*0.3 + 3.0*0.2) / 1.0 = (0.5 + 0.6 + 0.6) / 1.0 = 1.7
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, 1.7)

  def test_safe_weighted_mean_with_invalid_mask(self) -> None:
    """Test weighted mean with some invalid values masked.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If masking is not handled correctly.

    Example:
        >>> test_safe_weighted_mean_with_invalid_mask()

    """
    metric = jnp.array([1.0, 2.0, 3.0])
    weights = jnp.array([0.5, 0.3, 0.2])
    valid_mask = jnp.array([True, True, False])  # Last value is invalid
    sum_valid_w = jnp.sum(weights * valid_mask)

    result = safe_weighted_mean(metric, weights, valid_mask, sum_valid_w)

    # Expected: (1.0*0.5 + 2.0*0.3) / 0.8 = 1.1 / 0.8 = 1.375
    assert jnp.allclose(result, 1.375)

  def test_safe_weighted_mean_all_invalid(self) -> None:
    """Test weighted mean when all values are invalid.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If NaN is not returned for all invalid values.

    Example:
        >>> test_safe_weighted_mean_all_invalid()

    """
    metric = jnp.array([1.0, 2.0, 3.0])
    weights = jnp.array([0.5, 0.3, 0.2])
    valid_mask = jnp.array([False, False, False])
    sum_valid_w = jnp.sum(weights * valid_mask)

    result = safe_weighted_mean(metric, weights, valid_mask, sum_valid_w)

    # Should return NaN when no valid values
    assert jnp.isnan(result)

  def test_safe_weighted_mean_zero_weights(self) -> None:
    """Test weighted mean with zero total weight.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If zero weights are not handled correctly.

    Example:
        >>> test_safe_weighted_mean_zero_weights()

    """
    metric = jnp.array([1.0, 2.0, 3.0])
    weights = jnp.array([0.0, 0.0, 0.0])
    valid_mask = jnp.array([True, True, True])
    sum_valid_w = jnp.array(0.0)

    result = safe_weighted_mean(metric, weights, valid_mask, sum_valid_w)

    # Should return NaN when total weight is zero
    assert jnp.isnan(result)

  def test_safe_weighted_mean_type_check(self) -> None:
    """Test type checking in safe_weighted_mean.

    Args:
        None

    Returns:
        None

    Raises:
        TypeError: If inputs are not JAX arrays.

    Example:
        >>> test_safe_weighted_mean_type_check()

    """
    metric = [1.0, 2.0, 3.0]  # Not a JAX array
    weights = jnp.array([0.5, 0.3, 0.2])
    valid_mask = jnp.array([True, True, True])
    sum_valid_w = jnp.array(1.0)

    with pytest.raises(TypeError, match="Expected metric to be a JAX array"):
      safe_weighted_mean(metric, weights, valid_mask, sum_valid_w)


class TestCalculateLogZIncrement:
  """Test the calculate_logZ_increment function."""

  def test_calculate_logZ_increment_basic(self) -> None:
    """Test basic log Z increment calculation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the log Z increment is not calculated correctly.

    Example:
        >>> test_calculate_logZ_increment_basic()

    """
    log_weights = jnp.array([0.0, 0.0, 0.0])  # Equal weights
    population_size = 3

    result = calculate_logZ_increment(log_weights, population_size)

    # log(sum(exp(0))) - log(3) = log(3) - log(3) = 0
    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert jnp.allclose(result, 0.0, atol=1e-6)

  def test_calculate_logZ_increment_with_varying_weights(self) -> None:
    """Test log Z increment with varying weights.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If varying weights are not handled correctly.

    Example:
        >>> test_calculate_logZ_increment_with_varying_weights()

    """
    log_weights = jnp.array([-1.0, 0.0, 1.0])
    population_size = 3

    result = calculate_logZ_increment(log_weights, population_size)

    # Should be: log(exp(-1) + exp(0) + exp(1)) - log(3)
    expected = jnp.log(jnp.exp(-1.0) + jnp.exp(0.0) + jnp.exp(1.0)) - jnp.log(3.0)
    assert jnp.allclose(result, expected, atol=1e-6)

  def test_calculate_logZ_increment_empty_weights(self) -> None:
    """Test log Z increment with empty log weights array.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If empty array is not handled correctly.

    Example:
        >>> test_calculate_logZ_increment_empty_weights()

    """
    log_weights = jnp.array([])
    population_size = 0

    result = calculate_logZ_increment(log_weights, population_size)

    # Should return -inf for empty array
    assert jnp.isneginf(result)

  def test_calculate_logZ_increment_with_neginf(self) -> None:
    """Test log Z increment with negative infinity weights.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If -inf weights are not handled correctly.

    Example:
        >>> test_calculate_logZ_increment_with_neginf()

    """
    log_weights = jnp.array([-jnp.inf, -jnp.inf, 0.0])
    population_size = 3

    result = calculate_logZ_increment(log_weights, population_size)

    # Should handle -inf weights by treating them as zero contribution
    # Only the last weight contributes
    assert isinstance(result, jax.Array)
    assert jnp.isfinite(result)

  def test_calculate_logZ_increment_zero_population(self) -> None:
    """Test log Z increment with zero population size.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If zero population is not handled correctly.

    Example:
        >>> test_calculate_logZ_increment_zero_population()

    """
    log_weights = jnp.array([0.0, 0.0])
    population_size = 0

    result = calculate_logZ_increment(log_weights, population_size)

    # Should return -inf for zero population
    assert jnp.isneginf(result)


class TestCalculatePositionEntropy:
  """Test the calculate_position_entropy function."""

  def test_calculate_position_entropy_uniform(self) -> None:
    """Test position entropy with uniform distribution.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If uniform distribution entropy is not maximal.

    Example:
        >>> test_calculate_position_entropy_uniform()

    """
    # Four different values, each appearing once
    seqs = jnp.array([0, 1, 2, 3])

    result = calculate_position_entropy(seqs)

    # Maximum entropy for 4 states is log(4)
    expected = jnp.log(4.0)
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, expected, atol=1e-6)

  def test_calculate_position_entropy_constant(self) -> None:
    """Test position entropy with constant (no variation).

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If constant entropy is not zero.

    Example:
        >>> test_calculate_position_entropy_constant()

    """
    # All same value
    seqs = jnp.array([0, 0, 0, 0])

    result = calculate_position_entropy(seqs)

    # No entropy when all values are the same
    assert jnp.allclose(result, 0.0, atol=1e-6)

  def test_calculate_position_entropy_mixed(self) -> None:
    """Test position entropy with mixed distribution.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If mixed distribution entropy is not correct.

    Example:
        >>> test_calculate_position_entropy_mixed()

    """
    # Two 0s and two 1s
    seqs = jnp.array([0, 0, 1, 1])

    result = calculate_position_entropy(seqs)

    # Entropy = -2*(0.5*log(0.5)) = log(2)
    expected = jnp.log(2.0)
    assert jnp.allclose(result, expected, atol=1e-6)


class TestShannonEntropy:
  """Test the shannon_entropy function."""

  def test_shannon_entropy_basic(self) -> None:
    """Test basic Shannon entropy calculation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If Shannon entropy is not calculated correctly.

    Example:
        >>> test_shannon_entropy_basic()

    """
    # 3 sequences of length 4, all same
    seqs = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int8)

    result = shannon_entropy(seqs)

    # All positions have no variation, so entropy should be 0
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, 0.0, atol=1e-6)

  def test_shannon_entropy_variable(self) -> None:
    """Test Shannon entropy with variable sequences.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If variable entropy is not positive.

    Example:
        >>> test_shannon_entropy_variable()

    """
    # Different sequences
    seqs = jnp.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]], dtype=jnp.int8)

    result = shannon_entropy(seqs)

    # Should have positive entropy
    assert result > 0.0

  def test_shannon_entropy_empty(self) -> None:
    """Test Shannon entropy with empty sequences.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If empty sequences are not handled correctly.

    Example:
        >>> test_shannon_entropy_empty()

    """
    seqs = jnp.array([], dtype=jnp.int8).reshape(0, 0)

    result = shannon_entropy(seqs)

    # Should return 0 for empty sequences
    assert jnp.allclose(result, 0.0)

  def test_shannon_entropy_single_sequence(self) -> None:
    """Test Shannon entropy with a single sequence.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If single sequence entropy is not zero.

    Example:
        >>> test_shannon_entropy_single_sequence()

    """
    seqs = jnp.array([[0, 1, 2, 3]], dtype=jnp.int8)

    result = shannon_entropy(seqs)

    # No variation across positions with single sequence
    assert jnp.allclose(result, 0.0, atol=1e-6)

  def test_shannon_entropy_zero_population(self) -> None:
    """Test Shannon entropy with zero population.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If zero population is not handled correctly.

    Example:
        >>> test_shannon_entropy_zero_population()

    """
    seqs = jnp.array([], dtype=jnp.int8).reshape(0, 4)

    result = shannon_entropy(seqs)

    # Should return 0 for zero population
    assert jnp.allclose(result, 0.0)
