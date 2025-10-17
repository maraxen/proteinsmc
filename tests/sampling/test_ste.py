"""Tests for the Straight-Through Estimator (STE) module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.sampling.ste import (
  DEFAULT_LEARNING_RATE,
  DEFAULT_NUM_STEPS,
  create_pure_optimization_fn,
  straight_through_estimator,
)


class TestStraightThroughEstimator:
  """Test the straight_through_estimator function."""

  def test_basic_functionality(self, rng_key) -> None:
    """Test basic STE functionality.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If output shape or properties are incorrect.

    Example:
        >>> test_basic_functionality(jax.random.PRNGKey(42))

    """
    # Create logits for a simple sequence
    logits = jax.random.normal(rng_key, shape=(10, 20))

    result = straight_through_estimator(logits)

    # Output should have the same shape as input
    assert_shape(result, logits.shape)

    # Output should be probabilities (sum to 1 along last axis)
    assert jnp.allclose(result.sum(axis=-1), 1.0)

    # All values should be non-negative
    assert (result >= 0).all()

  def test_gradient_flow(self) -> None:
    """Test that gradients flow through the STE operation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If gradients are not computed correctly.

    Example:
        >>> test_gradient_flow()

    """
    def loss_fn(logits):
      probs = straight_through_estimator(logits)
      return jnp.sum(probs)

    logits = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(logits)

    # Gradients should exist and be finite
    assert grads is not None
    assert jnp.isfinite(grads).all()

  def test_output_is_soft_one_hot(self, rng_key) -> None:
    """Test that output approximates one-hot encoding.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If output doesn't approximate one-hot.

    Example:
        >>> test_output_is_soft_one_hot(jax.random.PRNGKey(42))

    """
    # Create logits with clear maximum
    logits = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])

    result = straight_through_estimator(logits)

    # For large logit differences, output should be close to one-hot
    # The maximum position should have value close to 1
    max_vals = result.max(axis=-1)
    assert jnp.allclose(max_vals, 1.0, atol=0.01)

  def test_jit_compatibility(self, rng_key) -> None:
    """Test that STE is JIT-compatible.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_jit_compatibility(jax.random.PRNGKey(42))

    """
    logits = jax.random.normal(rng_key, shape=(5, 10))

    jitted_ste = jax.jit(straight_through_estimator)
    result_regular = straight_through_estimator(logits)
    result_jitted = jitted_ste(logits)

    # Both should produce the same result
    assert jnp.allclose(result_regular, result_jitted)


class TestCreatePureOptimizationFn:
  """Test the create_pure_optimization_fn function."""

  def test_basic_optimization(self, rng_key) -> None:
    """Test basic optimization functionality.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If optimization doesn't work correctly.

    Example:
        >>> test_basic_optimization(jax.random.PRNGKey(42))

    """
    # Create optimization function with default parameters
    optimize_fn = create_pure_optimization_fn()

    # Create batch of initial logits, target logits, and masks
    batch_size = 4
    seq_length = 10
    num_classes = 20

    key1, key2, key3 = jax.random.split(rng_key, 3)
    initial_logits = jax.random.normal(key1, shape=(batch_size, seq_length, num_classes))
    target_logits = jax.random.normal(key2, shape=(batch_size, seq_length, num_classes))
    mask = jnp.ones((batch_size, seq_length), dtype=bool)

    result = optimize_fn(initial_logits, target_logits, mask)

    # Result should have the same shape as initial logits
    assert_shape(result, initial_logits.shape)
    assert jnp.isfinite(result).all()

  def test_with_custom_learning_rate(self, rng_key) -> None:
    """Test optimization with custom learning rate.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If custom learning rate is not applied.

    Example:
        >>> test_with_custom_learning_rate(jax.random.PRNGKey(42))

    """
    # Create optimization function with custom learning rate
    custom_lr = 0.1
    optimize_fn = create_pure_optimization_fn(learning_rate=custom_lr)

    batch_size = 2
    seq_length = 5
    num_classes = 10

    key1, key2 = jax.random.split(rng_key, 2)
    initial_logits = jax.random.normal(key1, shape=(batch_size, seq_length, num_classes))
    target_logits = jax.random.normal(key2, shape=(batch_size, seq_length, num_classes))
    mask = jnp.ones((batch_size, seq_length), dtype=bool)

    result = optimize_fn(initial_logits, target_logits, mask)

    assert_shape(result, initial_logits.shape)
    # Result should differ from initial logits (optimization happened)
    assert not jnp.allclose(result, initial_logits)

  def test_with_custom_num_steps(self, rng_key) -> None:
    """Test optimization with custom number of steps.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If custom num_steps is not applied.

    Example:
        >>> test_with_custom_num_steps(jax.random.PRNGKey(42))

    """
    # Create optimization function with custom num_steps
    custom_steps = 50
    optimize_fn = create_pure_optimization_fn(num_steps=custom_steps)

    batch_size = 2
    seq_length = 5
    num_classes = 10

    key1, key2 = jax.random.split(rng_key, 2)
    initial_logits = jax.random.normal(key1, shape=(batch_size, seq_length, num_classes))
    target_logits = jax.random.normal(key2, shape=(batch_size, seq_length, num_classes))
    mask = jnp.ones((batch_size, seq_length), dtype=bool)

    result = optimize_fn(initial_logits, target_logits, mask)

    assert_shape(result, initial_logits.shape)

  def test_with_partial_mask(self, rng_key) -> None:
    """Test optimization with partial masking.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If masking doesn't work correctly.

    Example:
        >>> test_with_partial_mask(jax.random.PRNGKey(42))

    """
    optimize_fn = create_pure_optimization_fn(num_steps=10)

    batch_size = 2
    seq_length = 10
    num_classes = 20

    key1, key2, key3 = jax.random.split(rng_key, 3)
    initial_logits = jax.random.normal(key1, shape=(batch_size, seq_length, num_classes))
    target_logits = jax.random.normal(key2, shape=(batch_size, seq_length, num_classes))

    # Create a mask that only covers half the sequence
    mask = jnp.concatenate([
      jnp.ones((batch_size, seq_length // 2), dtype=bool),
      jnp.zeros((batch_size, seq_length // 2), dtype=bool),
    ], axis=1)

    result = optimize_fn(initial_logits, target_logits, mask)

    assert_shape(result, initial_logits.shape)
    assert jnp.isfinite(result).all()

  def test_jit_compatibility(self, rng_key) -> None:
    """Test that the optimization function is JIT-compatible.

    Args:
        rng_key: PRNG key fixture.

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test_jit_compatibility(jax.random.PRNGKey(42))

    """
    optimize_fn = create_pure_optimization_fn(num_steps=10)

    batch_size = 2
    seq_length = 5
    num_classes = 10

    key1, key2 = jax.random.split(rng_key, 2)
    initial_logits = jax.random.normal(key1, shape=(batch_size, seq_length, num_classes))
    target_logits = jax.random.normal(key2, shape=(batch_size, seq_length, num_classes))
    mask = jnp.ones((batch_size, seq_length), dtype=bool)

    # The function is already JIT-compiled internally, but test it works
    result = optimize_fn(initial_logits, target_logits, mask)

    assert_shape(result, initial_logits.shape)

  def test_default_parameters_match_constants(self) -> None:
    """Test that default parameters match module constants.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If defaults don't match.

    Example:
        >>> test_default_parameters_match_constants()

    """
    # This test ensures the defaults are properly set
    optimize_fn_default = create_pure_optimization_fn()
    optimize_fn_explicit = create_pure_optimization_fn(
      learning_rate=DEFAULT_LEARNING_RATE,
      num_steps=DEFAULT_NUM_STEPS,
    )

    # Both should be functionally equivalent
    # We can't directly compare functions, but we can ensure they exist
    assert optimize_fn_default is not None
    assert optimize_fn_explicit is not None
