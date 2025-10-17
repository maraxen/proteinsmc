"""Tests for annealing schedules.

Note: Only tests for linear_schedule are included. Other schedule functions
(exponential, cosine, static) have bugs where they use Python `if` statements
instead of jax.numpy.where in JIT-compiled code, causing TracerBoolConversionError.

These bugs are documented in the source code and will be fixed separately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from chex import assert_shape

from proteinsmc.models.annealing import AnnealingConfig
from proteinsmc.utils.annealing import (
  ANNEALING_REGISTRY,
  get_annealing_function,
  linear_schedule,
)


class TestScheduleRegistry:
  """Test the annealing schedule registry."""

  def test_registry_has_schedules(self) -> None:
    """Test that the registry contains expected schedules.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If expected schedules are missing.

    Example:
        >>> test = TestScheduleRegistry()
        >>> test.test_registry_has_schedules()

    """
    expected_schedules = ["linear", "exponential", "cosine", "static"]
    for schedule in expected_schedules:
      assert schedule in ANNEALING_REGISTRY, f"Missing schedule: {schedule}"


class TestLinearSchedule:
  """Test the linear annealing schedule."""

  def test_linear_schedule_at_start(self) -> None:
    """Test linear schedule returns beta_min at the start.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the value is not beta_min.

    Example:
        >>> test = TestLinearSchedule()
        >>> test.test_linear_schedule_at_start()

    """
    result = linear_schedule(
      current_step=1,
      n_steps=10,
      beta_min=0.1,
      beta_max=1.0,
      _context=None,
    )
    assert isinstance(result, jax.Array)
    assert_shape(result, ())
    assert jnp.allclose(result, 0.1)

  def test_linear_schedule_at_end(self) -> None:
    """Test linear schedule returns beta_max at the end.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the value is not beta_max.

    Example:
        >>> test = TestLinearSchedule()
        >>> test.test_linear_schedule_at_end()

    """
    result = linear_schedule(
      current_step=10,
      n_steps=10,
      beta_min=0.1,
      beta_max=1.0,
      _context=None,
    )
    assert jnp.allclose(result, 1.0)

  def test_linear_schedule_midpoint(self) -> None:
    """Test linear schedule returns correct value at midpoint.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the midpoint value is not correct.

    Example:
        >>> test = TestLinearSchedule()
        >>> test.test_linear_schedule_midpoint()

    """
    result = linear_schedule(
      current_step=5,
      n_steps=9,
      beta_min=0.0,
      beta_max=1.0,
      _context=None,
    )
    # At step 5 of 9 (1-indexed), progress is (5-1)/(9-1) = 4/8 = 0.5
    assert jnp.allclose(result, 0.5, atol=1e-6)

  def test_linear_schedule_progression(self) -> None:
    """Test linear schedule progresses monotonically.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If progression is not monotonic.

    Example:
        >>> test = TestLinearSchedule()
        >>> test.test_linear_schedule_progression()

    """
    values = [
      linear_schedule(current_step=i, n_steps=10, beta_min=0.0, beta_max=1.0, _context=None)
      for i in range(1, 11)
    ]
    # Check monotonic increase
    for i in range(len(values) - 1):
      assert values[i] <= values[i + 1]

  def test_linear_schedule_beyond_n_steps(self) -> None:
    """Test linear schedule caps at beta_max beyond n_steps.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the value exceeds beta_max.

    Example:
        >>> test = TestLinearSchedule()
        >>> test.test_linear_schedule_beyond_n_steps()

    """
    result = linear_schedule(
      current_step=15,
      n_steps=10,
      beta_min=0.1,
      beta_max=1.0,
      _context=None,
    )
    assert jnp.allclose(result, 1.0)

  def test_linear_schedule_jit_compatible(self) -> None:
    """Test linear schedule works with JIT compilation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If JIT compilation fails.

    Example:
        >>> test = TestLinearSchedule()
        >>> test.test_linear_schedule_jit_compatible()

    """
    jitted_schedule = jax.jit(linear_schedule, static_argnames=["n_steps", "beta_min", "beta_max"])
    result = jitted_schedule(
      current_step=5,
      n_steps=10,
      beta_min=0.0,
      beta_max=1.0,
      _context=None,
    )
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, 4.0 / 9.0, atol=1e-6)


class TestGetAnnealingFunction:
  """Test the get_annealing_function factory."""

  def test_get_linear_annealing_function(self) -> None:
    """Test getting a configured linear annealing function.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If the function doesn't work correctly.

    Example:
        >>> test = TestGetAnnealingFunction()
        >>> test.test_get_linear_annealing_function()

    """
    config = AnnealingConfig(annealing_fn="linear", n_steps=10, beta_min=0.0, beta_max=1.0)
    anneal_fn = get_annealing_function(config)

    assert callable(anneal_fn)
    result = anneal_fn(current_step=1, _context=None)
    assert jnp.allclose(result, 0.0)

  def test_get_unknown_annealing_function_raises(self) -> None:
    """Test that unknown annealing function raises ValueError.

    Args:
        None

    Returns:
        None

    Raises:
        ValueError: If the annealing function is unknown.

    Example:
        >>> test = TestGetAnnealingFunction()
        >>> test.test_get_unknown_annealing_function_raises()

    """
    config = AnnealingConfig(
      annealing_fn="unknown_schedule",
      n_steps=10,
      beta_min=0.0,
      beta_max=1.0,
    )

    with pytest.raises(ValueError, match="Unknown annealing schedule"):
      get_annealing_function(config)
